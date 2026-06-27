"""MVBench accuracy when video tokens are kept by *attention score* and the rest
are masked out of attention.

This is an attention-oracle ceiling. For each clip we run one teacher forward and
read the language->video attention (how much the post-video text tokens attend to
each video token) at a band of critical decoder layers. At a retention ratio rho
we keep the ``k = max(1, round(rho * tokens))`` highest-scoring video tokens and
make the rest invisible to **every** decoder layer: their key columns in the
additive attention mask are set to -inf from layer 0 onward, so no query -- and in
particular not the answer token -- ever attends to them.

Unlike FastV (which prunes the sequence at a layer K and so saves compute, but
lets all tokens leak through layers 0..K first), the sequence length never changes
and the dropped tokens are blocked at *all* depths. This isolates the one question
"are these the right k tokens to keep?" -- a selection-quality ceiling, not a
speedup. The scores themselves need the full forward, so this is not deployable;
that is the point of a ceiling.

Selection is a global top-k by language->video attention: at each rho we keep the
k highest-scoring video tokens and mask the rest.

Self-contained single-forward MC eval on OpenGVLab/MVBench using the official
mvbench.ipynb prompt (``Best option:(``; the next token is the option letter).
Eager attention is required so attentions are returned and the additive 4D mask is
materialized for the hooks to edit; the model's own pre-LLM gating stays off. The
full (no-mask) model accuracy comes for free from the teacher forward.

Example:
    python mvbench_masked_baselines.py --data_root ~/MVBench \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --rhos 0.1 0.25 0.5 \
        --score_layers 12 13 14 15 16 --tasks "Action Sequence" --official_sampling
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# task -> (json file, video subdir under <data_root>/video, data_type, has_temporal_bound)
DATA_LIST = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}

SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question."
)

ANSWER_PROMPT = "Best option:("  # official mvbench.ipynb answer cue; the next token is the letter


def frame_indices(bound, fps, max_frame, num_segments, first_idx=0):
    """Even-count frame indices for the frame-folder (tvqa) task."""
    start, end = (bound[0], bound[1]) if bound else (-1e5, 1e5)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    n = max(2, round(num_segments / 2) * 2)
    return np.linspace(start_idx, end_idx, n).round().astype(int)


def get_index(bound, fps, max_frame, num_segments, first_idx=0):
    """Official mvbench.ipynb sampler: the midpoint index of each of num_segments
    equal temporal segments within [start, end] seconds (or the whole clip)."""
    start, end = (bound[0], bound[1]) if bound else (-1e5, 1e5)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + seg_size / 2 + np.round(seg_size * idx))
                     for idx in range(num_segments)])


def official_frames(path, data_type, has_bound, record, num_segments):
    """Reference mvbench.ipynb frames: a fixed num_segments PIL images at segment
    midpoints. Video clips are decoded with decord (avg fps); the frame-folder task
    indexes 1-based <idx>.jpg names at fps=3."""
    bound = (record["start"], record["end"]) if has_bound else None
    if data_type == "frame":
        names = sorted(os.listdir(path))
        idxs = get_index(bound, 3, len(names), num_segments, first_idx=1)
        return [Image.open(os.path.join(path, f"{i:05d}.jpg")).convert("RGB") for i in idxs]
    from decord import VideoReader, cpu  # lazy: only needed for --official_sampling
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idxs = get_index(bound, float(vr.get_avg_fps()), len(vr) - 1, num_segments, first_idx=0)
    return [Image.fromarray(f) for f in vr.get_batch(idxs).asnumpy()]


def make_prompt(path, data_type, has_bound, record, text, max_frames, max_pixels, fps,
                official=False, num_segments=16):
    if official:
        vid = {"type": "video", "video": official_frames(path, data_type, has_bound, record, num_segments)}
    elif data_type == "frame":
        bound = (record["start"], record["end"]) if has_bound else None
        names = sorted(os.listdir(path))
        idxs = frame_indices(bound, 3, len(names), max_frames, first_idx=1)
        vid = {"type": "video", "video": [os.path.join(path, f"{i:05d}.jpg") for i in idxs]}
    else:
        vid = {"type": "video", "video": path, "fps": fps, "max_frames": max_frames}
        if has_bound:
            vid["video_start"] = record["start"]
            vid["video_end"] = record["end"]
    if max_pixels is not None:
        vid["max_pixels"] = max_pixels
    return [{"role": "user", "content": [vid, {"type": "text", "text": text}]}]


def build_prompt(record):
    """Official mvbench.ipynb prompt: system + question + lettered options. The
    ``Best option:(`` answer cue is appended to the rendered chat in the loop."""
    letters = [chr(ord("A") + i) for i in range(len(record["candidates"]))]
    opts = "".join(f"({L}) {c}\n" for L, c in zip(letters, record["candidates"]))
    text = f"{SYSTEM_PROMPT}\nQuestion: {record['question']}\nOptions:\n{opts}"
    gt_idx = record["candidates"].index(record["answer"])
    return text, letters, gt_idx


def letter_token_ids(processor, letters):
    """First-token id candidates per option letter (with/without leading space)."""
    tok = processor.tokenizer
    out = []
    for L in letters:
        cands = set()
        for form in (L, f" {L}"):
            ids = tok.encode(form, add_special_tokens=False)
            if ids:
                cands.add(ids[0])
        out.append(sorted(cands))
    return out


@torch.no_grad()
def logits_to_pred(logits, letter_ids):
    last = logits[0, -1]
    opt = [max(last[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    return int(np.argmax(opt))


# --------------------------------------------------------------------------- #
# Attention scoring (language -> video), replicating experiments/common.py
# --------------------------------------------------------------------------- #
def query_rows(input_ids, video_cols, source, device):
    """Which query rows to aggregate attention over.

    * "language" -- post-video text tokens (the paper's teacher signal): every
                    non-video position after the video block (question/options/cue).
    * "all"      -- every query row.
    * "last"     -- only the final (answer-cue) position (FastV's last-token signal).
    """
    seq_len = input_ids.shape[0]
    if source == "last":
        return torch.tensor([seq_len - 1], device=device)
    if source == "all":
        return torch.arange(seq_len, device=device)
    if source == "language":
        last_vid = int(video_cols.max())
        pos = torch.arange(seq_len, device=device)
        is_video = torch.zeros(seq_len, dtype=torch.bool, device=device)
        is_video[video_cols] = True
        return pos[(pos > last_vid) & ~is_video]
    raise ValueError(f"unknown attention source {source!r}")


def attention_scores(attentions, score_layers, rows, video_cols):
    """One score per video token: language->video attention averaged over heads,
    over the selected query rows, then over the requested layers. ``attentions`` is
    the tuple of per-layer (B, heads, q, k) tensors from output_attentions=True."""
    per_layer = []
    for L in score_layers:
        attn = attentions[L][0]               # (heads, q, k)
        block = attn[:, rows][:, :, video_cols]  # (heads, |rows|, |vid|)
        per_layer.append(block.mean(dim=0).mean(dim=0))  # (|vid|,)
    return torch.stack(per_layer, dim=0).mean(dim=0).float()  # (n_video,)


def select_topk(scores, k):
    """Local indices of the k highest-scoring video tokens (sorted ascending)."""
    idx = torch.topk(scores, k=min(k, scores.numel())).indices
    return torch.sort(idx).values


class KeyMasker:
    """Mask given key columns out of attention in EVERY decoder layer.

    For every layer, the dropped video-token columns of the additive 4D attention
    mask are set to -inf, so no query at any depth (and hence not the answer token)
    can attend to those tokens -- vision is restricted to the kept set throughout.
    The sequence is never shortened: this measures selection quality, not speed.
    Hooks are (de)registered per forward via the context-manager protocol."""

    def __init__(self, layers, drop_cols):
        self.layers = layers
        self.drop_cols = drop_cols  # LongTensor of video-token sequence positions to block
        self.handles = []

    def _mask_hook(self, module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None and len(args) >= 2:
            am = args[1]
        if torch.is_tensor(am):  # additive 4D float mask (eager): block the dropped columns
            am[..., self.drop_cols] = torch.finfo(am.dtype).min

    def __enter__(self):
        for layer in self.layers:
            self.handles.append(layer.register_forward_pre_hook(self._mask_hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main():
    p = argparse.ArgumentParser(
        description="MVBench accuracy when video tokens are kept by attention score and the rest "
                    "are masked out of attention in every layer (attention-oracle selection ceiling).")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.25, 0.5],
                   help="Retention ratios: fraction of video tokens kept per clip.")
    p.add_argument("--score_layers", type=int, nargs="+", default=[12, 13, 14, 15, 16],
                   help="Decoder layers whose language->video attention defines the score.")
    p.add_argument("--score_source", default="language", choices=["language", "last", "all"],
                   help="Query rows the attention is read from: language = post-video text tokens "
                        "(paper signal); last = final answer-cue position; all = every row.")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model (no-mask) reference.")
    p.add_argument("--max_frames", type=int, default=16, help="upper cap on frames per clip.")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for video sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="Use the reference mvbench.ipynb sampler (fixed --num_segments frames at "
                        "segment midpoints) for leaderboard-comparable numbers, instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="results_mvbench_attn_oracle.json", help="Path to dump metrics JSON.")
    args = p.parse_args()

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    # eager attention guarantees returned attentions + a materialized additive 4D mask.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    layers = model.model.language_model.layers
    video_token_id = model.config.video_token_id
    n_layers = len(layers)
    bad = [L for L in args.score_layers if not 0 <= L < n_layers]
    if bad:
        raise ValueError(f"score_layers {bad} out of range; model has {n_layers} layers (0..{n_layers - 1}).")

    # (column name, rho). 'full' = no masking; then attn top-k at each rho.
    settings = [] if args.no_full else [("full", None)]
    for r in args.rhos:
        settings.append((f"attn@{r}", r))   # e.g. attn@0.25
    col = [name for name, _ in settings]

    correct = {t: {c: 0 for c in col} for t in tasks}
    seen = {t: 0 for t in tasks}
    kept_vid = {c: 0 for c in col}
    vid_total = 0

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")
    n = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=task):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = build_prompt(rec)
                prompt = make_prompt(path, data_type, has_bound, rec, text,
                                     args.max_frames, args.max_pixels, args.fps,
                                     official=args.official_sampling, num_segments=args.num_segments)
                # append the official answer cue so the next token is the option letter
                chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) + ANSWER_PROMPT
                imgs, vids = process_vision_info(prompt)
                inputs = processor(text=[chat], images=imgs, videos=vids, return_tensors="pt").to(model.device)
                letter_ids = letter_token_ids(processor, letters)
                video_cols = (inputs["input_ids"][0] == video_token_id).nonzero(as_tuple=False).flatten()
                if video_cols.numel() == 0:
                    tqdm.write(f"skip [{task}] {rec.get('video')}: no video tokens")
                    continue
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            n_video = video_cols.numel()
            n_frames = int(inputs["video_grid_thw"][0][0].item())
            vid_total += n_video
            device = video_cols.device

            # Teacher forward: full-model logits (no-mask reference) + attentions for scoring.
            with torch.no_grad():
                out = model(**inputs, output_attentions=need_scores, use_cache=False)
            full_logits = out.logits
            scores = None
            if need_scores:
                rows = query_rows(inputs["input_ids"][0], video_cols, args.score_source, device)
                scores = attention_scores(out.attentions, args.score_layers, rows, video_cols)
            del out  # free the attention tensors before the masked forwards

            for name, rho, mode in settings:
                if rho is None:  # full model -- reuse the teacher forward
                    correct[task][name] += int(logits_to_pred(full_logits, letter_ids) == gt_idx)
                    kept_vid[name] += n_video
                    continue
                k = min(n_video, max(1, int(round(rho * n_video))))
                local = kept_local_indices(mode, scores, k, n_frames, device)
                keep = torch.zeros(n_video, dtype=torch.bool, device=device)
                keep[local] = True
                drop_cols = video_cols[~keep]
                with torch.no_grad(), KeyMasker(layers, drop_cols):
                    logits = model(**inputs, use_cache=False).logits
                correct[task][name] += int(logits_to_pred(logits, letter_ids) == gt_idx)
                kept_vid[name] += int(keep.sum())
            seen[task] += 1
            n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (json/ and video/).")

    valid = [t for t in tasks if seen[t]]
    setting_mean = {c: float(np.mean([correct[t][c] / seen[t] for t in valid])) for c in col}
    setting_micro = {c: sum(correct[t][c] for t in valid) / n for c in col}

    print(f"\nEvaluated {n} samples across {len(valid)} task(s)\n")
    print(f"{'task':<26}{'n':>5}" + "".join(f"{c:>10}" for c in col))
    print("-" * (31 + 10 * len(col)))
    for t in valid:
        accs = "".join(f"{correct[t][c] / seen[t]:>10.4f}" for c in col)
        print(f"{t:<26}{seen[t]:>5}{accs}")
    print("-" * (31 + 10 * len(col)))
    print(f"{'mean (per-task)':<26}{'':>5}" + "".join(f"{setting_mean[c]:>10.4f}" for c in col))
    print(f"{'micro (per-sample)':<26}{n:>5}" + "".join(f"{setting_micro[c]:>10.4f}" for c in col))

    # per-setting accuracy / kept-token budget
    print(f"\n{'setting':<10}{'acc':>9}{'vid%':>8}{'vid_tok':>9}")
    print("-" * 36)
    for name in col:
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        print(f"{name:<10}{setting_mean[name]:>9.4f}{vid_pct:>7.1f}%{kept_vid[name] / n:>9.0f}")

    result = {
        "experiment": "mvbench_attention_oracle_masked",
        "method": "keep top-k video tokens by language->video attention; mask the rest out of "
                  "attention in every layer (selection-quality ceiling)",
        "model_name": args.model_name,
        "data_root": args.data_root,
        "sampling": ("official" if args.official_sampling else "fps"),
        "num_frames": (args.num_segments if args.official_sampling else None),
        "fps": (None if args.official_sampling else args.fps),
        "max_frames": args.max_frames,
        "rhos": args.rhos,
        "baselines": args.baselines,
        "score_layers": args.score_layers,
        "score_source": args.score_source,
        "seed": args.seed,
        "settings": col,
        "tasks": valid,
        "n_evaluated": n,
        "per_task": {t: {"n": seen[t], **{c: correct[t][c] / seen[t] for c in col}} for t in valid},
        "table": {
            name: {
                "accuracy_mean": setting_mean[name],
                "accuracy_micro": setting_micro[name],
                "kept_vid_pct": (kept_vid[name] / vid_total * 100 if vid_total else 0.0),
                "vid_tok": kept_vid[name] / n,
            }
            for name in col
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
