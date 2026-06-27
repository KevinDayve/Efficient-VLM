"""Accuracy when ALL video tokens are dropped *after decoder layer K*.

Self-contained, single-forward MC eval on OpenGVLab/MVBench using the official
mvbench.ipynb prompt (``Best option:(``). For each layer K we run the decoder
normally through layer K, then for every layer after K mask *all* video tokens
out of attention -- so the answer can use vision only up to layer K. This probes
how deep the model actually needs image tokens (the motivation being that deep
layers barely attend to them).

For a single forward whose only output is the answer token's logits, masking the
video tokens out of all layers > K delivers the *same* logits as physically
removing them (no surviving token ever attends to them), without the bookkeeping
of resizing hidden states / masks / RoPE mid-stack. Eager attention is used so
the additive 4D mask is materialized for the hooks to edit; the model's own
pre-LLM gating stays off.

Example:
    python layer_drop.py --data_root ~/MVBench \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --tasks "Action Sequence" --layers 2 5 10 20 --official_sampling
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


class LayerKDropper:
    """Installs pre-hooks that mask ALL video tokens out of attention after layer K.

    For every layer > K, the video-token columns of the additive 4D attention mask
    are set to -inf, so no later layer (and hence not the answer token) can attend
    to any video token -- vision is available only through layers 0..K. Hooks are
    (de)registered per forward via the context-manager protocol."""

    def __init__(self, layers, video_cols, layer_k):
        self.layers = layers
        self.video_cols = video_cols  # LongTensor of video-token sequence positions
        self.k = layer_k
        self.handles = []

    def _mask_hook(self, module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None and len(args) >= 2:
            am = args[1]
        if torch.is_tensor(am):  # additive 4D float mask (eager): block all video columns
            am[..., self.video_cols] = torch.finfo(am.dtype).min

    def __enter__(self):
        for idx in range(self.k + 1, len(self.layers)):
            self.handles.append(self.layers[idx].register_forward_pre_hook(self._mask_hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def make_plot(path, layer_ks, acc, full, valid, correct, seen):
    """Accuracy vs K (drop-after-layer) line, full model as a dashed reference."""
    import matplotlib
    matplotlib.use("Agg")  # headless: write PNG, no display
    import matplotlib.pyplot as plt

    ys = [acc[f"L{k}"] for k in layer_ks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(valid) > 1:  # faint per-task curves behind the mean
        for t in valid:
            ax.plot(layer_ks, [correct[t][f"L{k}"] / seen[t] for k in layer_ks],
                    color="gray", alpha=0.3, lw=1)
    ax.plot(layer_ks, ys, "o-", color="C0", label="drop video after layer K (mean over tasks)")
    ax.axhline(full, ls="--", color="C3", label=f"full model = {full:.3f}")
    ax.set_xlabel("K  (all video tokens dropped after layer K)")
    ax.set_ylabel("MVBench accuracy (mean over tasks)")
    ax.set_title("Accuracy vs vision depth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Accuracy when all video tokens are dropped after layer K.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[2, 5, 10, 20],
                   help="Decoder layer indices K after which ALL video tokens are dropped.")
    p.add_argument("--layer_step", type=int, default=None,
                   help="Sweep K across ALL layers at this step instead of --layers: drop after "
                        "each step-layer block, i.e. K = step-1, 2*step-1, ... (step=4 -> after "
                        "layers 3, 7, 11, ... so vision is available through 0-3, 0-7, 0-11, ...).")
    p.add_argument("--max_frames", type=int, default=16, help="upper cap on frames per clip.")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for video sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="Use the reference mvbench.ipynb sampler (fixed --num_segments frames at "
                        "segment midpoints) instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="results_layer_drop.json", help="Path to dump metrics JSON.")
    p.add_argument("--plot", default=None, help="Path for the accuracy-vs-K PNG (default: --out with .png).")
    args = p.parse_args()

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    # eager attention guarantees a materialized additive 4D mask the hooks can edit.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    layers = model.model.language_model.layers
    video_token_id = model.config.video_token_id
    n_layers = len(layers)
    # --layer_step sweeps every step-th boundary across all layers; else use --layers.
    if args.layer_step:
        layer_ks = list(range(args.layer_step - 1, n_layers - 1, args.layer_step))
    else:
        layer_ks = sorted(set(args.layers))
    bad = [k for k in layer_ks if not 0 <= k < n_layers - 1]
    if bad:
        raise ValueError(f"layer indices {bad} out of range; model has {n_layers} layers (use 0..{n_layers - 2}).")

    # (column, K). 'full' = no dropping; 'L{k}' = all video tokens dropped after layer k
    # (vision available through layers 0..k).
    settings = [("full", None)] + [(f"L{k}", k) for k in layer_ks]
    col = [c for c, _ in settings]

    correct = {t: {c: 0 for c in col} for t in tasks}
    seen = {t: 0 for t in tasks}
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
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            for name, k in settings:
                with torch.no_grad():
                    if k is None:  # full model, no hooks
                        logits = model(**inputs, use_cache=False).logits
                    else:  # mask all video tokens out of layers > k
                        with LayerKDropper(layers, video_cols, k):
                            logits = model(**inputs, use_cache=False).logits
                correct[task][name] += int(logits_to_pred(logits, letter_ids) == gt_idx)
            seen[task] += 1
            n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (json/ and video/).")

    valid = [t for t in tasks if seen[t]]
    acc = {c: float(np.mean([correct[t][c] / seen[t] for t in valid])) for c in col}  # mean over tasks
    full = acc["full"]

    print(f"\nEvaluated {n} samples across {len(valid)} task(s)\n")
    print(f"{'task':<26}{'n':>5}" + "".join(f"{c:>10}" for c in col))
    print("-" * (31 + 10 * len(col)))
    for t in valid:
        print(f"{t:<26}{seen[t]:>5}" + "".join(f"{correct[t][c] / seen[t]:>10.4f}" for c in col))
    print("-" * (31 + 10 * len(col)))
    print(f"{'mean (per-task)':<26}{'':>5}" + "".join(f"{acc[c]:>10.4f}" for c in col))

    # accuracy drop vs the full model, per setting
    print(f"\n{'setting':<10}{'acc':>9}{'drop':>9}")
    print("-" * 28)
    for c in col:
        print(f"{c:<10}{acc[c]:>9.4f}{full - acc[c]:>9.4f}")

    result = {
        "experiment": "mvbench_drop_all_video_after_layer_k",
        "model_name": args.model_name,
        "data_root": args.data_root,
        "sampling": ("official" if args.official_sampling else "fps"),
        "num_frames": (args.num_segments if args.official_sampling else None),
        "fps": (None if args.official_sampling else args.fps),
        "layers": layer_ks,
        "layer_step": args.layer_step,
        "n_layers": n_layers,
        "tasks": valid,
        "n_evaluated": n,
        "full_accuracy": full,
        "per_task": {t: {"n": seen[t], **{c: correct[t][c] / seen[t] for c in col}} for t in valid},
        "table": {c: {"accuracy": acc[c], "drop": full - acc[c]} for c in col},
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")

    plot_path = args.plot or os.path.splitext(args.out)[0] + ".png"
    try:
        make_plot(plot_path, layer_ks, acc, full, valid, correct, seen)
        print(f"Saved -> {plot_path}")
    except Exception as e:  # plotting is best-effort; results are already saved
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
