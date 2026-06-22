"""
oracle_correlation.py -- is the gradient oracle predictable from feature-only signals?

The MLP scorer plateaued at chance recall. Two explanations: (a) the signal is in
the features but BCE/the MLP failed to extract it, or (b) the answer-criticality
the oracle captures is simply NOT a function of the forward features. This script
tests (b) directly and without the self-referential linear-probe (which regresses
v_i onto a quantity built from v_i).

For each video we compute the gradient oracle s_i = ReLU(<g_i, v_i>) (min-max'd),
then correlate its per-token RANKING against three signals that are independent of
how the oracle is built:

  * token_norm   : ||v_i||              (is the oracle just keeping high-norm tokens?)
  * query_cosine : max_t cos(v_i, q_t)  (semantic relevance to the question text;
                    this is the attention-like signal we already saw ties uniform)
  * novelty      : 1 - cos(v_i, v_i^{prev frame})  (temporal redundancy, AI2's target)

Read:
  * If the oracle ranking correlates with NONE of these (Spearman ~ 0, recall@25 ~
    chance), the oracle's structure is invisible to feature-only signals -> not
    feature-predictable -> no loss change rescues the scorer (the "information"
    verdict). That is the negative-transfer finding.
  * If it correlates strongly with query_cosine, the oracle is ~semantic relevance
    (which we showed ties uniform) -- surprising, would need rethinking.
  * If it correlates with norm or novelty, there is a cheap feature signal a scorer
    could learn -> the MLP/loss was the bottleneck, CE is worth trying.

Run:
    python oracle_correlation.py \
        --data_file /home/ubuntu/NeXTVideo/val.jsonl \
        --video_root /home/ubuntu/NeXTVideo \
        --max_frames 8 --limit 200
"""

import os
import json
import random
import argparse
import warnings

import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

# reuse the exact data + position helpers training/eval use
from train_oracle import make_prompt, options_and_answer, build_full_positions


def topk_overlap(a, b, frac):
    """recall@frac: fraction of b's top-(frac) that also appear in a's top-(frac)."""
    k = max(1, int(frac * a.numel()))
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    return len(sa & sb) / k


def safe_spearman(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).statistic
    return r if r == r else 0.0  # guard NaN


def query_cosine_signal(feats, query_embeds):
    """max cosine similarity of each video token to any query (text) token.
    feats: (N, d) video tokens; query_embeds: (Q, d) text-token embeddings."""
    if query_embeds.numel() == 0:
        return torch.zeros(feats.shape[0], device=feats.device)
    vn = F.normalize(feats, dim=-1)            # (N, d)
    qn = F.normalize(query_embeds, dim=-1)     # (Q, d)
    sims = vn @ qn.t()                         # (N, Q)
    return sims.max(dim=1).values              # (N,)


def novelty_signal(feats, n_frames):
    """1 - cos(token_i(t), token_i(t-1)), frame-major layout. Frame 0 -> 0."""
    N, d = feats.shape
    per = N // n_frames
    x = feats[: per * n_frames].view(n_frames, per, d)
    cos = F.cosine_similarity(x[1:], x[:-1], dim=-1)   # (T-1, per)
    nov = torch.zeros(n_frames, per, device=feats.device)
    nov[1:] = (1.0 - cos)
    return nov.reshape(-1)[: per * n_frames]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    special_ids = set(processor.tokenizer.all_special_ids)

    with open(args.data_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    random.Random(args.seed).shuffle(records)
    records = records[: args.limit]

    sig_names = ["token_norm", "query_cosine", "novelty"]
    rho = {s: [] for s in sig_names}        # Spearman(oracle, signal)
    rec25 = {s: [] for s in sig_names}      # recall@25 of signal vs oracle
    n_done = 0

    for rec in records:
        choices, correct_idx = options_and_answer(rec)
        if choices is None:
            continue
        gt_token = processor.tokenizer(choices[correct_idx], add_special_tokens=False).input_ids[0]

        prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels)
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(prompt)
        inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        video_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if video_pos.numel() < 50:
            continue
        video_grid_thw = inputs["video_grid_thw"].to(device)
        n_frames = int(video_grid_thw[0][0].item())
        pix = inputs["pixel_values_videos"].to(device)

        # ---- gradient oracle (same as train_oracle) ----
        with torch.no_grad():
            ve = model.get_video_features(pix, video_grid_thw).pooler_output
            ve = torch.cat(ve, dim=0).to(device)
        ve = ve.detach().requires_grad_(True)
        inputs_embeds = model.get_input_embeddings()(input_ids).clone()
        inputs_embeds[0, video_pos] = ve.to(inputs_embeds.dtype)
        position_ids = build_full_positions(model, input_ids, video_pos, video_grid_thw, attn)
        out = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                    attention_mask=attn, use_cache=False)
        L_ans = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)[gt_token]
        if ve.grad is not None:
            ve.grad = None
        L_ans.backward()
        g = ve.grad.float()
        oracle = F.relu((g * ve.detach().float()).sum(dim=-1))     # (N,)
        feats = ve.detach().float()
        del out, inputs_embeds, g, L_ans
        model.zero_grad(set_to_none=True)

        # ---- feature-only signals (no grad, no label) ----
        with torch.no_grad():
            # query/text tokens = non-video, non-special positions
            is_video = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=device)
            is_video[video_pos] = True
            text_mask = ~is_video
            for sid in special_ids:
                text_mask &= (input_ids[0] != sid)
            query_embeds = model.get_input_embeddings()(input_ids)[0][text_mask].float()

            signals = {
                "token_norm":   feats.norm(dim=-1),
                "query_cosine": query_cosine_signal(feats, query_embeds),
                "novelty":      novelty_signal(feats, n_frames),
            }
            for s in sig_names:
                rho[s].append(safe_spearman(oracle, signals[s]))
                rec25[s].append(topk_overlap(signals[s], oracle, 0.25))

        n_done += 1
        if n_done % 25 == 0:
            line = " | ".join(f"{s}: rho {np.median(rho[s]):+.3f} R@25 {np.median(rec25[s]):.3f}"
                              for s in sig_names)
            print(f"[{n_done}] {line}")

    chance = 0.25
    print("\n==== ORACLE vs FEATURE-ONLY SIGNALS ====")
    print(f"  videos: {n_done}   (recall@25 chance baseline ~ {chance:.2f})")
    for s in sig_names:
        print(f"  {s:<13} median Spearman {np.median(rho[s]):+.3f}   "
              f"IQR [{np.percentile(rho[s],25):+.3f}, {np.percentile(rho[s],75):+.3f}]   "
              f"recall@25 {np.median(rec25[s]):.3f}")
    print("\n  Read:")
    print("   all signals ~0 Spearman, R@25 ~ chance  -> oracle NOT feature-predictable")
    print("        => loss change won't help; this is the negative-transfer finding.")
    print("   strong query_cosine corr                -> oracle ~ semantic relevance (which ties uniform)")
    print("   norm/novelty corr clearly > chance      -> a cheap feature signal exists; CE worth trying")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())