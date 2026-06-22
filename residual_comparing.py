"""
Temporal-redundancy tail-index diagnostic.
==========================================
AI2's STTS regresses a per-patch redundancy target with plain MSE:
    target_i(t) = 1 - cos( patch_i(t), patch_i(t-1) )
i.e. "how novel is this patch versus where it was last frame".

Question: is that redundancy signal heavy-tailed (Frechet, xi > 0) the way the
language->video attention scores were (xi ~ 0.28)? If yes, MSE-to-the-mean
underweights the rare high-novelty patches that matter most -- the same failure
mode we diagnosed for importance-MSE -- and a rank/EVT-aware treatment is a
principled improvement over their MSE. If xi ~ 0, redundancy is well-behaved
and their MSE is appropriate (no wedge there).

We compute the redundancy signal directly from the SAME merged video features
the scorer consumes, so the distribution matches the deployment quantity.
Reuses einmahlHaan from efficient_vlm.utils (the DEdH moment estimator).

Run:
    python redundancy_tail_check.py \
        --data_file /home/ubuntu/NeXTVideo/val.jsonl \
        --video_root /home/ubuntu/NeXTVideo \
        --max_frames 16 --limit 60
"""

import os
import json
import random
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

from efficient_vlm.utils import einmahlHaan


def make_prompt(record, video_root, max_frames, max_pixels=None):
    rel = record["video"]["path"]
    abs_path = os.path.normpath(os.path.join(video_root, rel))
    content = []
    for msg in record["messages"]:
        if msg["role"] != "user":
            continue
        for item in msg["content"]:
            if item.get("type") == "video":
                vid = {"type": "video", "video": abs_path, "nframes": max_frames}
                if max_pixels is not None:
                    vid["max_pixels"] = max_pixels
                content.append(vid)
            else:
                content.append({"type": "text", "text": item["text"]})
    return [{"role": "user", "content": content}]


def redundancy_signal(video_embeds: torch.Tensor, n_frames: int) -> torch.Tensor:
    """video_embeds: (n_video, d) merged tokens, laid out frame-major
    (frame 0's tokens, then frame 1's, ...). Returns per-patch novelty
    1 - cos(patch_i(t), patch_i(t-1)) for t >= 1, flattened. Frame 0 has
    no predecessor so it is skipped (matches AI2's t-1 formulation)."""
    n_video, d = video_embeds.shape
    per_frame = n_video // n_frames
    x = video_embeds[: per_frame * n_frames].view(n_frames, per_frame, d)
    cur = x[1:]            # (T-1, per_frame, d)
    prev = x[:-1]          # (T-1, per_frame, d)
    cos = F.cosine_similarity(cur, prev, dim=-1)   # (T-1, per_frame)
    novelty = (1.0 - cos).flatten()
    return novelty.clamp(min=0)   # numerical guard; cos in [-1,1] -> novelty in [0,2]


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name, use_fast=True)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    with open(args.data_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    random.Random(args.seed).shuffle(records)
    records = records[: args.limit]

    xi_hist, median_hist = [], []
    for i, rec in enumerate(records):
        prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels)
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(prompt)
        inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if video_positions.numel() == 0:
            continue
        video_grid_thw = inputs["video_grid_thw"].to(device)
        n_frames = int(video_grid_thw[0][0].item())
        pix = inputs["pixel_values_videos"].to(device)
        embeds = model.get_video_features(pix, video_grid_thw).pooler_output
        embeds = torch.cat(embeds, dim=0).float()          # (n_video, d)

        nov = redundancy_signal(embeds, n_frames)
        if nov.numel() < 50:
            continue
        xi_hist.append(einmahlHaan(nov))
        median_hist.append(nov.median().item())
        if (i + 1) % 10 == 0:
            print(f"[{i+1}] running median xi_hat (redundancy): "
                  f"{np.median(xi_hist):.4f}  (n={len(xi_hist)})  "
                  f"median novelty: {np.median(median_hist):.4e}")

    print("\n==== REDUNDANCY-SIGNAL TAIL INDEX ====")
    print(f"  videos: {len(xi_hist)}")
    print(f"  median xi_hat: {np.median(xi_hist):.4f}")
    print(f"  IQR xi_hat:   [{np.percentile(xi_hist,25):.4f}, {np.percentile(xi_hist,75):.4f}]")
    print(f"  median novelty (1-cos): {np.median(median_hist):.4e}")
    print("\n  Compare to your attention-score xi_hat (~0.28).")
    print("  xi > ~0.15  -> heavy-tailed -> AI2's MSE underweights the novel tail (WEDGE)")
    print("  xi ~  0     -> well-behaved -> MSE is appropriate, no wedge here")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())