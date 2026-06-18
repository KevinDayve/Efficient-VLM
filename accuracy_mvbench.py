"""MC accuracy on OpenGVLab/MVBench with the native Qwen2.5-VL token gating.

MVBench is 20 video-MC tasks. Annotations live in ``<data_root>/json/<task>.json``
(a list of ``{"video", "question", "candidates", "answer"[, "start", "end"]}``) and
the clips live under ``<data_root>/video/<source_subdir>/``. The official task ->
(json, video_subdir, data_type, has_temporal_bound) mapping is reproduced in
``DATA_LIST`` below; frames are sampled exactly as the reference MVBench loader
(uniform segments, fps=3 for the frame-folder task, respecting start/end bounds).

For every sample we run ONE forward (no autoregressive decoding) and read the
next-token logits at the final position, restricted to the option-letter tokens --
the same readout as accuracy.py. The full model is the accuracy reference; each
retention ratio rho reuses the attached scorer (we just flip ``token_keep_ratio``).

Setup (run once on the remote machine):
    pip install -U "huggingface_hub[cli]" decord pillow
    hf download OpenGVLab/MVBench --repo-type dataset --local-dir ~/MVBench
    # extract every per-source video archive in place
    cd ~/MVBench/video && for z in $(find . -name '*.zip'); do unzip -n -q "$z"; done

Example:
    python accuracy_mvbench.py \
        --data_root ~/MVBench \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --scorer_ckpt checkpoints/scorer_best.pt \
        --rhos 0.25 0.5 0.75
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
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


def frame_indices(bound, fps, max_frame, num_segments, first_idx=0):
    """Uniform segment-center sampling, identical to the reference MVBench loader."""
    start, end = (bound[0], bound[1]) if bound else (-1e5, 1e5)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + seg / 2 + np.round(seg * i)) for i in range(num_segments)])


def read_video(path, bound, num_segments):
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idxs = frame_indices(bound, float(vr.get_avg_fps()), len(vr) - 1, num_segments, first_idx=0)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idxs]


def read_frames(path, bound, num_segments, fps=3):
    """Frame-folder task (tvqa): images are named 00001.jpg.. at 3 fps."""
    names = sorted(os.listdir(path))
    idxs = frame_indices(bound, fps, len(names), num_segments, first_idx=1)
    return [Image.open(os.path.join(path, f"{i:05d}.jpg")).convert("RGB") for i in idxs]


def load_frames(video_root, subdir, data_type, record, num_segments):
    path = os.path.join(video_root, subdir, record["video"])
    bound = (record["start"], record["end"]) if "start" in record else None
    if data_type == "frame":
        return read_frames(path, bound, num_segments)
    return read_video(path, bound, num_segments)


def build_prompt(record):
    """MVBench option block + the letters present, and the ground-truth index."""
    letters = [chr(ord("A") + i) for i in range(len(record["candidates"]))]
    opts = "".join(f"({L}) {c}\n" for L, c in zip(letters, record["candidates"]))
    text = (f"{SYSTEM_PROMPT}\nQuestion: {record['question']}\nOptions:\n{opts}"
            "Answer with the option's letter (A, B, C, ...) directly.")
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


def build_inputs(processor, model, frames, text, max_pixels):
    """Feed pre-sampled frames straight to the processor as one video clip."""
    if max_pixels is not None:
        processor.image_processor.max_pixels = max_pixels
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video = np.stack([np.asarray(f) for f in frames])  # (T, H, W, 3)
    inputs = processor(text=[chat], videos=[video], return_tensors="pt", fps=2.0)
    return inputs.to(model.device)


def main():
    p = argparse.ArgumentParser(description="Gated MC accuracy on OpenGVLab/MVBench.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/ (see module docstring).")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--scorer_ckpt", default="checkpoints/scorer_best.pt")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--k_min", type=int, default=1)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--max_frames", type=int, default=16, help="frames sampled per clip (MVBench default 16).")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model baseline.")
    args = p.parse_args()

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.load_token_scorer(args.scorer_ckpt, keep_ratio=args.rhos[0], hidden_dim=args.hidden_dim,
                            k_min=args.k_min, temp=args.temp, beta_max=args.beta_max)

    settings = ([("full", None)] if not args.no_full else []) + [(f"rho={r}", r) for r in args.rhos]
    # per-task and aggregate accuracy
    correct = {t: {name: 0 for name, _ in settings} for t in tasks}
    seen = {t: 0 for t in tasks}
    kept_vid = {name: 0 for name, _ in settings}
    vid_total = 0
    text_total = 0
    n = 0

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
                bound_rec = rec if has_bound else {k: v for k, v in rec.items() if k not in ("start", "end")}
                frames = load_frames(video_dir, subdir, data_type, bound_rec, args.max_frames)
                text, letters, gt_idx = build_prompt(rec)
                inputs = build_inputs(processor, model, frames, text, args.max_pixels)
                letter_ids = letter_token_ids(processor, letters)
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            n_video, n_text = video_text_counts(model, inputs)
            vid_total += n_video
            text_total += n_text
            for name, rho in settings:
                if rho is None:
                    model.disable_token_gating()
                else:
                    model.model.token_keep_ratio = rho
                correct[task][name] += int(predict(model, inputs, letter_ids) == gt_idx)
                kept_vid[name] += kept_video_count(model, inputs, rho)
            seen[task] += 1
            n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (json/ and video/).")

    text_avg = text_total / n
    print(f"\nEvaluated {n} samples across {len(tasks)} task(s)  "
          f"(avg video tokens/clip = {vid_total / n:.0f}, avg text tokens = {text_avg:.0f})\n")

    # per-task accuracy table (one column per setting)
    col = [name for name, _ in settings]
    print(f"{'task':<26}{'n':>5}" + "".join(f"{c:>10}" for c in col))
    print("-" * (31 + 10 * len(col)))
    for task in tasks:
        if not seen[task]:
            continue
        accs = "".join(f"{correct[task][c] / seen[task]:>10.4f}" for c in col)
        print(f"{task:<26}{seen[task]:>5}{accs}")
    # MVBench headline number is the mean over per-task accuracies
    print("-" * (31 + 10 * len(col)))
    valid = [t for t in tasks if seen[t]]
    mean_row = "".join(f"{np.mean([correct[t][c] / seen[t] for t in valid]):>10.4f}" for c in col)
    micro_row = "".join(f"{sum(correct[t][c] for t in valid) / n:>10.4f}" for c in col)
    print(f"{'mean (per-task)':<26}{'':>5}{mean_row}")
    print(f"{'micro (per-sample)':<26}{n:>5}{micro_row}")

    print(f"\n{'setting':<10}{'vid%':>8}{'vid_tok':>9}{'text_tok':>10}")
    print("-" * 37)
    for name, _ in settings:
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        print(f"{name:<10}{vid_pct:>7.1f}%{kept_vid[name] / n:>9.0f}{text_avg:>10.0f}")


if __name__ == "__main__":
    main()
