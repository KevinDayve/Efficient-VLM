"""Content-free video-token reduction baselines on OpenGVLab/MVBench.

Most video tokens are unnecessary. This measures how MVBench accuracy holds up
when, at a retention ratio rho, we keep only ``k = round(rho * tokens)`` video
tokens per clip chosen *without looking at content*:

  * random        -- k random tokens (chance baseline at the budget)
  * uniform        -- k evenly-spaced tokens over the flattened (frame-major)
                      sequence (a space-time diagonal sweep)
  * uniform_strat  -- even per-frame quota of evenly-spaced spatial tokens
                      (a uniform spatial grid replicated across frames)
  * regvar         -- content-free regularly-varying (Pareto) per-frame budget
                      drawn by inverse-transform sampling: the heavy-tailed shape
                      of the Pareto-adaptive budget spent on arbitrary frames (the
                      content-free null for adaptive budgeting; --rv_alpha sets the
                      tail index)
  * first / last   -- the first / last k tokens in order (the earliest / latest
                      frames; prefix / suffix position baselines)

Both reuse the model's native pre-LLM gating (``token_selection_mode``), so the
decoder runs on a genuinely shorter sequence -- no scorer checkpoint required.
The full (no-drop) model is reported as the reference, and ``--blind`` adds a
text-only baseline (zero vision tokens, the model's language prior). Same data layout, frame
sampling and option-letter logit readout as the experiment suite (one forward
per sample, argmax over the option-letter tokens -- no autoregressive decoding).

Setup (run once):
    pip install -U "huggingface_hub[cli]" decord pillow
    hf download OpenGVLab/MVBench --repo-type dataset --local-dir ~/MVBench
    cd ~/MVBench/video && for z in $(find . -name '*.zip'); do unzip -n -q "$z"; done

Example:
    python inference.py \
        --data_root ~/MVBench \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --rhos 0.25 0.5 0.75 --baselines random uniform \
        --tasks "Action Sequence" "Scene Transition"
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
    # EgoSchema (long-form egocentric, 500-question Subset): built by make_egoschema_json.py.
    # subdir="" -> path = <data_root>/video/<uuid>.mp4 (symlink <root>/video -> videos/videos).
    "EgoSchema": ("egoschema.json", "", "video", False),
}

SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question."
)

# Official mvbench.ipynb forces the answer with this assistant-turn prefix; the very next
# token is the option letter, so the logit readout still works (no generate+parse needed).
ANSWER_PREFIX = "Best option:("

TEMPORAL_PATCH_SIZE = 2  # Qwen2.5-VL pairs adjacent frames; the sampled count must be even.


def frame_indices(bound, fps, max_frame, num_segments, first_idx=0):
    """Frame indices for the frame-folder (tvqa) task, matching qwen_vl_utils."""
    start, end = (bound[0], bound[1]) if bound else (-1e5, 1e5)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    n = max(TEMPORAL_PATCH_SIZE, round(num_segments / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
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


def make_mvbench_prompt(path, data_type, has_bound, record, text, max_frames, max_pixels,
                        fps, official=False, num_segments=16, min_pixels=None):
    """Qwen chat prompt whose video item drives process_vision_info (the project sampler).

    With ``official=True`` the video item is an explicit list of ``num_segments``
    midpoint-sampled PIL frames (the reference mvbench.ipynb protocol) instead of
    qwen_vl_utils fps sampling."""
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
    if min_pixels is not None:
        vid["min_pixels"] = min_pixels
    return [{"role": "user", "content": [vid, {"type": "text", "text": text}]}]


def build_prompt(record):
    """MVBench option block + the letters present, and the ground-truth index.

    Matches the reference mvbench.ipynb user turn exactly: the options block is rstripped
    and followed by ``Only give the best option.`` (the answer is then forced by the
    ``ANSWER_PREFIX`` assistant prefix at chat-build time)."""
    letters = [chr(ord("A") + i) for i in range(len(record["candidates"]))]
    opts = "".join(f"({L}) {c}\n" for L, c in zip(letters, record["candidates"]))
    text = (f"{SYSTEM_PROMPT}\nQuestion: {record['question']}\nOptions:\n{opts.rstrip()}"
            "\nOnly give the best option.")
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
def predict(model, inputs, letter_ids):
    last = model(**inputs).logits[0, -1]
    opt = [max(last[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    return int(np.argmax(opt))


def video_text_counts(model, inputs):
    ids = inputs["input_ids"][0]
    n_video = int((ids == model.config.video_token_id).sum())
    return n_video, int(ids.numel()) - n_video


def kept_video_count(model, inputs, keep_ratio):
    """Video tokens the gating keeps (deterministic: max(1, round(rho*tokens)) per clip)."""
    n_video, _ = video_text_counts(model, inputs)
    if keep_ratio is None:
        return n_video
    merge = model.config.vision_config.spatial_merge_size**2
    kept = sum(max(1, round(keep_ratio * (int(row.prod()) // merge))) for row in inputs["video_grid_thw"])
    return min(kept, n_video)


def main():
    p = argparse.ArgumentParser(
        description="Content-free video-token reduction baselines (random / uniform) on MVBench.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="Retention ratios: fraction of video tokens kept per clip.")
    p.add_argument("--baselines", nargs="*", default=["random", "uniform", "uniform_strat"],
                   choices=["random", "uniform", "uniform_strat", "regvar", "first", "last"],
                   help="Content-free selection modes to run at each rho (same token budget). "
                        "random = k random tokens; uniform = k evenly-spaced tokens over the "
                        "flattened sequence; uniform_strat = even per-frame quota of evenly-spaced "
                        "spatial tokens (a uniform grid replicated across frames); regvar = a "
                        "regularly-varying (Pareto, --rv_alpha) per-frame budget via inverse-transform "
                        "sampling, then evenly-spaced spatial tokens (content-free null for adaptive "
                        "budgeting); first/last = the first/last k tokens in order (the earliest/latest "
                        "frames -- prefix/suffix position baselines).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random / regvar selection baselines.")
    p.add_argument("--rv_alpha", type=float, default=2.0,
                   help="Pareto tail index for the 'regvar' baseline (smaller = heavier tail / more "
                        "concentrated per-frame budget).")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model (no-drop) reference.")
    p.add_argument("--blind", action="store_true",
                   help="Also run a text-only (no video) baseline: the question+options with ZERO "
                        "vision tokens, measuring the model's language prior.")
    p.add_argument("--max_frames", type=int, default=16, help="upper cap on frames per clip.")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for video sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="Use the reference mvbench.ipynb sampler (fixed --num_segments frames at "
                        "segment midpoints) for leaderboard-comparable numbers, instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--out", default="results_mvbench_baselines.json", help="Path to dump metrics JSON.")
    p.add_argument("--responses_out", default=None,
                   help="Optional path to dump per-sample responses JSON (predicted option per "
                        "setting, ground truth, correctness). Defaults to <out>_responses.json.")
    args = p.parse_args()

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    torch.manual_seed(args.seed)  # reproducible "random" selection
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.model.token_gating_rv_alpha = args.rv_alpha  # Pareto tail index for the regvar baseline
    # No scorer checkpoint needed: random/uniform gating activates on token_selection_mode
    # alone (token_scorer stays None); we just toggle keep_ratio + mode per setting.

    # (column name, rho, selection mode). full = no dropping; blind = no video tokens
    # at all (text-only); then each rho x baseline.
    label = {"random": "rand", "uniform": "unif", "uniform_strat": "ustr", "regvar": "rvar",
             "first": "frst", "last": "last"}
    settings = [] if args.no_full else [("full", None, None)]
    if args.blind:
        settings.append(("blind", None, "blind"))
    for r in args.rhos:
        for b in args.baselines:
            settings.append((f"{label[b]}@{r}", r, b))   # e.g. rand@0.25, ustr@0.25
    col = [name for name, _, _ in settings]
    need_video = any(m != "blind" for _, _, m in settings)

    correct = {t: {c: 0 for c in col} for t in tasks}
    seen = {t: 0 for t in tasks}
    kept_vid = {c: 0 for c in col}
    vid_total = 0
    n = 0
    responses = []  # per-sample predictions for every setting

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

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
                letter_ids = letter_token_ids(processor, letters)
                inputs = None
                if need_video:
                    prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                                 args.max_frames, args.max_pixels, args.fps,
                                                 official=args.official_sampling,
                                                 num_segments=args.num_segments)
                    chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    chat += ANSWER_PREFIX  # force the answer; the option letter is the next token after '('
                    imgs, vids = process_vision_info(prompt)
                    inputs = processor(text=[chat], images=imgs, videos=vids, return_tensors="pt").to(model.device)
                blind_inputs = None
                if args.blind:  # text-only prompt: same question/options, no video item
                    bmsg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                    bchat = processor.apply_chat_template(bmsg, tokenize=False, add_generation_prompt=True)
                    bchat += ANSWER_PREFIX
                    blind_inputs = processor(text=[bchat], return_tensors="pt").to(model.device)
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            if need_video:
                vid_total += video_text_counts(model, inputs)[0]
            preds = {}  # setting -> predicted option index for this sample
            for name, rho, mode in settings:
                if mode == "blind":  # no video tokens at all -- language prior
                    preds[name] = predict(model, blind_inputs, letter_ids)
                    correct[task][name] += int(preds[name] == gt_idx)
                    continue
                if rho is None:
                    model.model.disable_token_gating()
                else:
                    model.model.token_keep_ratio = rho
                    model.model.token_selection_mode = mode
                preds[name] = predict(model, inputs, letter_ids)
                correct[task][name] += int(preds[name] == gt_idx)
                kept_vid[name] += kept_video_count(model, inputs, rho)
            responses.append({
                "task": task,
                "video": rec.get("video"),
                "question": rec["question"],
                "candidates": rec["candidates"],
                "answer": rec["answer"],
                "gt_letter": letters[gt_idx],
                "predictions": {
                    name: {"letter": letters[preds[name]],
                           "text": rec["candidates"][preds[name]],
                           "correct": preds[name] == gt_idx}
                    for name in col
                },
            })
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
        "experiment": "mvbench_content_free_baselines",
        "model_name": args.model_name,
        "data_root": args.data_root,
        "sampling": ("official" if args.official_sampling else "fps"),
        "num_frames": (args.num_segments if args.official_sampling else None),
        "fps": (None if args.official_sampling else args.fps),
        "max_frames": args.max_frames,
        "rhos": args.rhos,
        "baselines": args.baselines,
        "rv_alpha": args.rv_alpha,
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

    resp_out = args.responses_out or f"{os.path.splitext(args.out)[0]}_responses.json"
    with open(resp_out, "w") as fh:
        json.dump({
            "model_name": args.model_name,
            "settings": col,
            "n_evaluated": n,
            "responses": responses,
        }, fh, indent=2)
    print(f"Saved responses ({len(responses)}) -> {resp_out}")


if __name__ == "__main__":
    main()
