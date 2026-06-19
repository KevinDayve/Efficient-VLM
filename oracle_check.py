"""
Gradient-oracle diagnostic for video-LLM token selection.
=========================================================
Tests whether the LITE oracle principle transfers from ViT action-recognition
to a video-LLM, BEFORE training any scorer.

The oracle (GradCAM-style, LITE Eq. 2): for the correct answer letter, compute
    g = d log p(answer) / d (merged video token)        # one backward pass
    value_i = ReLU( sum_d  g_{i,d} * activation_{i,d} )  # per-token saliency
then min-max to [0,1]. This uses the TRUE label (privileged), so it is a
ceiling, exactly as in LITE.

Two checks, neither needs a trained scorer:
  H4  -- tail index xi_hat of the oracle values (DEdH / einmahlHaan).
         Compare to attention (+0.28, useless) and redundancy (-0.28, useful).
         User's registered bet: NOT Pareto (xi <= ~0).
  CEILING -- top-k-by-oracle vs uniform on answer accuracy. If the oracle
         (which cheats) cannot beat uniform, the whole direction is dead.

NOTE the mild circularity of the ceiling test: we differentiate log p(answer)
and then keep the tokens that most raise it. That is the point of an oracle
(upper bound, privileged label) and is what LITE does. Read it as "is there a
small token subset that carries the answer at all", not as an achievable score.

Run (8 frames + grad checkpointing -- backward is memory-heavy, do not push frames):
    python oracle_check.py \
        --data_file /home/ubuntu/NeXTVideo/val.jsonl \
        --video_root /home/ubuntu/NeXTVideo \
        --max_frames 8 --retention 0.1 --limit 50
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
    abs_path = os.path.normpath(os.path.join(video_root, record["video"]["path"]))
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


def options_and_answer(record):
    choices = record.get("all_choices") or sorted(record.get("index2ans", {}).keys())
    gt = record.get("gt")
    if not choices or gt is None:
        return None, None
    gt = gt.strip().upper()
    if gt not in choices:
        return None, None
    return choices, choices.index(gt)


def select_uniform(n_video, n_frames, k, device):
    per = n_video // n_frames
    kp = max(1, k // n_frames)
    idx = []
    for t in range(n_frames):
        base = t * per
        step = max(1, per // kp)
        idx.extend(list(range(base, base + per, step))[:kp])
    return torch.tensor(sorted(idx[:k]), device=device, dtype=torch.long)


def select_by_scores(scores, n_frames, k):
    n_video = scores.numel()
    per = n_video // n_frames
    kp = max(1, k // n_frames)
    idx = []
    for t in range(n_frames):
        seg = scores[t * per:(t + 1) * per]
        top = torch.topk(seg, min(kp, seg.numel())).indices + t * per
        idx.append(top)
    keep = torch.cat(idx).sort().values
    return keep[:k] if keep.numel() > k else keep


def build_full_positions(model, input_ids, video_positions, video_grid_thw, attention_mask):
    mm = torch.zeros_like(input_ids)
    mm[0, video_positions] = 2
    pos, _ = model.model.get_rope_index(
        input_ids, mm_token_type_ids=mm,
        video_grid_thw=video_grid_thw, attention_mask=attention_mask)
    return pos  # (3,1,S)


@torch.no_grad()
def score_answer(model, inputs_embeds, position_ids, attention_mask,
                 video_positions, keep_video, letter_token_ids):
    """First-token log-prob of each candidate letter, on the kept-token seq."""
    kept_abs = video_positions[keep_video]
    keep_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
    keep_mask[video_positions] = False
    keep_mask[kept_abs] = True
    e = inputs_embeds[:, keep_mask, :]
    p = position_ids[:, :, keep_mask]
    a = attention_mask[:, keep_mask]
    out = model(inputs_embeds=e, position_ids=p, attention_mask=a, use_cache=False)
    lp = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return torch.tensor([lp[t].item() for t in letter_token_ids])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)              # freeze params: only input embeds carry grad
    model.gradient_checkpointing_enable()    # backward is memory-heavy; recompute activations
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    with open(args.data_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    random.Random(args.seed).shuffle(records)
    records = records[: args.limit]

    xi_hist, frac_zero_hist, skew_hist = [], [], []
    oracle_correct = uniform_correct = total = 0

    for rec in records:
        choices, correct_idx = options_and_answer(rec)
        if choices is None:
            continue
        letter_token_ids = [processor.tokenizer(c, add_special_tokens=False).input_ids[0]
                            for c in choices]
        gt_token = letter_token_ids[correct_idx]

        prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels)
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(prompt)
        inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if video_positions.numel() == 0:
            continue
        n_video = video_positions.numel()
        video_grid_thw = inputs["video_grid_thw"].to(device)
        n_frames = int(video_grid_thw[0][0].item())
        pix = inputs["pixel_values_videos"].to(device)
        k = max(n_frames, int(round(args.retention * n_video)))

        # ---- merged video tokens, with grad ----
        ve = model.get_video_features(pix, video_grid_thw).pooler_output
        ve = torch.cat(ve, dim=0).to(device)            # (n_video, d)
        ve = ve.detach().requires_grad_(True)

        inputs_embeds = model.get_input_embeddings()(input_ids).clone()  # (1,S,d)
        inputs_embeds[0, video_positions] = ve.to(inputs_embeds.dtype)

        position_ids = build_full_positions(model, input_ids, video_positions,
                                            video_grid_thw, attn)

        # ---- forward + backward to get the GradCAM oracle ----
        out = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                    attention_mask=attn, use_cache=False)
        target = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)[gt_token]
        model.zero_grad(set_to_none=True)
        if ve.grad is not None:
            ve.grad = None
        target.backward()
        g = ve.grad.float()                              # (n_video, d)
        oracle = F.relu((g * ve.detach().float()).sum(dim=-1))   # (n_video,) GradCAM
        # min-max to [0,1]
        omin, omax = oracle.min(), oracle.max()
        oracle_n = (oracle - omin) / (omax - omin + 1e-8)

        # ---- H4: tail index + shape ----
        if oracle.numel() >= 50:
            xi_hist.append(einmahlHaan(oracle.detach()))
            frac_zero_hist.append((oracle <= 1e-8).float().mean().item())
            o = oracle.detach()
            skew_hist.append((((o - o.mean()) / (o.std() + 1e-8)) ** 3).mean().item())

        # ---- CEILING: oracle-arm vs uniform-arm downstream ----
        inputs_embeds_d = inputs_embeds.detach()
        keep_oracle = select_by_scores(oracle.detach(), n_frames, k)
        keep_uniform = select_uniform(n_video, n_frames, k, device)
        lp_or = score_answer(model, inputs_embeds_d, position_ids, attn,
                            video_positions, keep_oracle, letter_token_ids)
        lp_un = score_answer(model, inputs_embeds_d, position_ids, attn,
                            video_positions, keep_uniform, letter_token_ids)
        oracle_correct += int(lp_or.argmax().item() == correct_idx)
        uniform_correct += int(lp_un.argmax().item() == correct_idx)
        total += 1

        del out, g, ve, inputs_embeds
        if total % 10 == 0:
            print(f"[{total}] xi_hat(oracle) median: {np.median(xi_hist):.4f} | "
                  f"oracle acc: {oracle_correct/total:.3f} | uniform acc: {uniform_correct/total:.3f}")

    print("\n==== GRADIENT-ORACLE DIAGNOSTIC ====")
    print(f"  samples: {total}")
    print(f"  H4  median xi_hat(oracle): {np.median(xi_hist):.4f}   "
          f"IQR [{np.percentile(xi_hist,25):.4f}, {np.percentile(xi_hist,75):.4f}]")
    print(f"      median frac at zero (post-ReLU): {np.median(frac_zero_hist):.3f}")
    print(f"      median skew: {np.median(skew_hist):.3f}")
    print(f"  CEILING  oracle acc: {oracle_correct/total:.4f}   "
          f"uniform acc: {uniform_correct/total:.4f}   "
          f"gap: {(oracle_correct-uniform_correct)/total:+.4f}")
    print("\n  Read:")
    print("   xi > ~0.15 + positive skew + high frac-zero -> Pareto-like (LITE transfers)")
    print("   xi ~ 0, low skew                            -> NOT Pareto (your bet)")
    print("   CEILING gap > 0  -> a token subset carries the answer; a scorer has something to chase")
    print("   CEILING gap ~ 0  -> even the cheating oracle ties uniform; direction is dead")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--retention", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())