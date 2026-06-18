"""MC accuracy on a local NExT-QA val.jsonl with the native Qwen2.5-VL gating.

Each record is read in the train/val layout:
    {"messages": [...user text with options inlined...],
     "video": {"path": "./NExTVideo/<grp>/<id>.mp4", "num_frames": 8},
     "gt": "A", "all_choices": ["A",...], "index2ans": {...}}

For every sample we run ONE forward (no autoregressive decoding) and read the
next-token logits at the final position, restricted to the option-letter tokens --
the same readout as the experiment suite. The full model is the accuracy reference;
each retention ratio rho reuses the attached scorer (we just flip ``token_keep_ratio``)
so the scorer checkpoint is read once.

Example:
    python accuracy.py \
        --data_file ~/NeXTVideo/val.jsonl \
        --video_root ~/NeXTVideo \
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
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_records(data_file, max_samples):
    with open(os.path.expanduser(data_file)) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    return recs[:max_samples] if max_samples else recs


def user_text(record):
    """The user text turn (question with options inlined + the answer instruction)."""
    for m in record["messages"]:
        for it in m["content"]:
            if it.get("type") == "text" and it.get("text"):
                return it["text"]
    raise ValueError("no text turn in record")


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
def predict(model, processor, inputs, letter_ids):
    last = model(**inputs).logits[0, -1]
    opt = [max(last[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    return int(np.argmax(opt))


def video_text_counts(model, inputs):
    """Original video-placeholder token count and text (non-video) token count."""
    ids = inputs["input_ids"][0]
    n_video = int((ids == model.config.video_token_id).sum())
    return n_video, int(ids.numel()) - n_video


def kept_video_count(model, inputs, keep_ratio):
    """Video tokens the gating keeps (deterministic: max(1, round(rho*tokens)) per clip)."""
    n_video, _ = video_text_counts(model, inputs)
    if keep_ratio is None:  # full model keeps everything
        return n_video
    merge = model.config.vision_config.spatial_merge_size**2
    kept = sum(max(1, round(keep_ratio * (int(row.prod()) // merge))) for row in inputs["video_grid_thw"])
    return min(kept, n_video)


def main():
    p = argparse.ArgumentParser(description="Gated MC accuracy on a NExT-QA val.jsonl.")
    p.add_argument("--data_file", required=True)
    p.add_argument("--video_root", required=True, help="Dir the record's relative video.path resolves against.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--scorer_ckpt", default="checkpoints/scorer_best.pt")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--k_min", type=int, default=1)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model baseline.")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    # Attach the scorer once; we toggle token_keep_ratio per rho instead of reloading.
    model.load_token_scorer(args.scorer_ckpt, keep_ratio=args.rhos[0], hidden_dim=args.hidden_dim,
                            k_min=args.k_min, temp=args.temp, beta_max=args.beta_max)

    records = load_records(args.data_file, args.max_samples)
    settings = ([("full", None)] if not args.no_full else []) + [(f"rho={r}", r) for r in args.rhos]
    correct = {name: 0 for name, _ in settings}
    kept_vid = {name: 0 for name, _ in settings}
    vid_total = 0
    text_total = 0
    n = 0

    for rec in tqdm(records):
        try:
            video_path = os.path.normpath(os.path.join(os.path.expanduser(args.video_root), rec["video"]["path"]))
            nframes = rec["video"].get("num_frames", args.max_frames)
            choices = rec["all_choices"]
            gt_idx = choices.index(rec["gt"])
            video_item = {"type": "video", "video": video_path, "nframes": nframes}
            if args.max_pixels is not None:
                video_item["max_pixels"] = args.max_pixels
            messages = [{"role": "user", "content": [video_item, {"type": "text", "text": user_text(rec)}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(messages)
            inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
            letter_ids = letter_token_ids(processor, choices)
        except Exception as e:  # missing video / decode error -> skip
            tqdm.write(f"skip {rec.get('video', {}).get('path')}: {e}")
            continue

        n_video, n_text = video_text_counts(model, inputs)
        vid_total += n_video
        text_total += n_text
        for name, rho in settings:
            if rho is None:
                model.disable_token_gating()
            else:
                model.model.token_keep_ratio = rho  # re-enable / change ratio without reloading
            correct[name] += int(predict(model, processor, inputs, letter_ids) == gt_idx)
            kept_vid[name] += kept_video_count(model, inputs, rho)
        n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --video_root / --data_file.")

    text_avg = text_total / n
    print(f"\nEvaluated {n} samples  (avg video tokens/clip = {vid_total / n:.0f}, "
          f"avg text tokens = {text_avg:.0f})\n")
    print(f"{'setting':<10}{'acc':>9}{'vid%':>8}{'vid_tok':>9}{'text_tok':>10}")
    print("-" * 46)
    for name, _ in settings:
        acc = correct[name] / n
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        vid_avg = kept_vid[name] / n
        print(f"{name:<10}{acc:>9.4f}{vid_pct:>7.1f}%{vid_avg:>9.0f}{text_avg:>10.0f}")


if __name__ == "__main__":
    main()
