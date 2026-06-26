"""
Gradient-oracle diagnostic for video-LLM token selection.
=========================================================
Tests whether the LITE oracle principle transfers from ViT action-recognition
to a video-LLM, BEFORE training any scorer -- and whether a LABEL-FREE
"self-oracle" preserves the ceiling (the only branch left open after the
feature-scorer was shown to fail: the oracle's signal lives in the gradient,
not the features, so compute it directly at inference rather than predicting it).

Oracles (GradCAM-style, LITE Eq. 2):
    g = d log p(letter) / d (merged video token)        # one backward pass
    value_i = ReLU( sum_d  g_{i,d} * activation_{i,d} )  # per-token saliency
  * TRUE-LABEL oracle: letter = gold answer. Privileged ceiling (uses the label).
  * SELF oracle:       letter = model's OWN argmax prediction. Label-free, so it
                       is computable at inference -- this is the deployable signal.

Checks (no trained scorer needed):
  H4      -- tail index xi_hat of the true-label oracle (DEdH / einmahlHaan).
  CEILING -- top-k-by-{oracle, self_oracle} vs uniform on answer accuracy.
             oracle gap > 0       : a token subset carries the answer (privileged).
             self_oracle gap > 0  : that subset is recoverable WITHOUT the label
                                    -> a deployable training-free method exists.
             self_oracle gap ~ 0  : the label was doing the work; branch closed.

The self-oracle's quality is bounded by how often the model's prediction is right
(when pred == gold, self-oracle == true oracle). We log model accuracy so the
self_oracle gap can be read against it.

NOTE the mild circularity of the ceiling test: we differentiate log p(letter)
and then keep the tokens that most raise it. That is the point of an oracle
(upper bound), and is what LITE does. Read it as "is there a small token subset
that carries the answer", not as an achievable score.

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

from efficient_vlm.utils import einmahlHaan, hill_tail_index


def make_prompt(record, video_root, max_frames, max_pixels=None, fps=2.0):
    abs_path = os.path.normpath(os.path.join(video_root, record["video"]["path"]))
    content = []
    for msg in record["messages"]:
        if msg["role"] != "user":
            continue
        for item in msg["content"]:
            if item.get("type") == "video":
                vid = {"type": "video", "video": abs_path, "fps": fps, "max_frames": max_frames}
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


def oracle_from_letter(model, base_embeds, ve_feats, video_positions,
                       position_ids, attn, letter_token_id):
    """Build a FRESH ve leaf + inputs_embeds, run one forward, backward on
    log p(letter_token_id), and return the GradCAM oracle ReLU(<g, ve>).

    Each call owns its own graph and leaf, so calling this twice per sample
    (gold letter, then predicted letter) gives two fully independent
    forward->backward->free cycles: no retain_graph, no shared buffers, and
    peak memory stays at a SINGLE backward (critical on tight GPUs).

    base_embeds : (1,S,d) text embeddings with video slots present but to be
                  overwritten (detached; carries no grad).
    ve_feats    : (n_video, d) merged video features (detached); the source
                  values for the fresh leaf.
    """
    ve = ve_feats.detach().clone().requires_grad_(True)      # fresh leaf
    inputs_embeds = base_embeds.clone()
    inputs_embeds[0, video_positions] = ve.to(inputs_embeds.dtype)

    out = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                attention_mask=attn, use_cache=False)
    logits_last = out.logits[0, -1, :].float()
    target = F.log_softmax(logits_last, dim=-1)[letter_token_id]
    model.zero_grad(set_to_none=True)
    target.backward()
    g = ve.grad.float()
    oracle = F.relu((g * ve.detach().float()).sum(dim=-1))
    res = oracle.detach(), logits_last.detach()
    del out, inputs_embeds, ve, g, target            # free this call's graph
    return res


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


def plot_oracle_distribution(pooled_saliency, survivals, xi_hist, hill_hist, out_path):
    """Three-panel figure motivating the Dekkers-Einmahl-de Haan tail index.

    (1) pooled saliency histogram (mass near 0, long right tail);
    (2) log-log empirical survival P(X>x) -- a straight tail is the Pareto
        signature the estimator detects;
    (3) per-sample tail index: Demahl (sign-aware) vs Hill (always >= 0), with
        the gamma=0 light-tail line and the logged median marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not pooled_saliency:
        print("  [plot] no usable samples to plot; skipping.")
        return
    pooled_all = np.concatenate(pooled_saliency)
    xi_med = float(np.median(xi_hist)) if xi_hist else float("nan")
    hill_med = float(np.median(hill_hist)) if hill_hist else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.hist(pooled_all, bins=80, color="#4C72B0", log=True)
    ax.set_xlabel("oracle saliency  relu(grad . feature)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"(1) Saliency distribution\n{pooled_all.size:,} positive tokens, "
                 f"{len(pooled_saliency)} samples")

    ax = axes[1]
    for xs, surv in survivals[:40]:        # cap lines so the panel stays legible
        ax.loglog(xs, surv, color="#999999", alpha=0.25, linewidth=0.8)
    xs = np.sort(pooled_all)
    ax.loglog(xs, 1.0 - np.arange(xs.size) / xs.size, color="#C44E52",
              linewidth=2.0, label="pooled")
    ax.set_xlabel("saliency x  (log)")
    ax.set_ylabel("P(X > x)  (log)")
    ax.set_title("(2) Survival function\nstraight tail => heavy / Pareto")
    ax.legend()

    ax = axes[2]
    edges = np.linspace(min(xi_hist + hill_hist + [0.0]),
                        max(xi_hist + hill_hist + [0.0]), 30)
    ax.hist(xi_hist, bins=edges, alpha=0.6, color="#4C72B0",
            label=f"Demahl  (med {xi_med:.2f})")
    ax.hist(hill_hist, bins=edges, alpha=0.6, color="#DD8452",
            label=f"Hill  (med {hill_med:.2f})")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1, label="gamma = 0 (light tail)")
    ax.axvline(xi_med, color="#4C72B0", linestyle=":", linewidth=1.5)
    ax.set_xlabel("estimated tail index  gamma (xi_hat)")
    ax.set_ylabel("number of samples")
    ax.set_title("(3) Tail index per sample\nDemahl is sign-aware; Hill >= 0")
    ax.legend()

    fig.suptitle("Gradient-oracle saliency: motivation for the Dekkers-Einmahl-de Haan tail index",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved -> {out_path}")


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
    # Accumulated only when --plot: pooled positive saliency, per-sample survival
    # curves, and the (sign-blind) Hill index for the Demahl-vs-Hill panel.
    pooled_saliency, survivals, hill_hist = [], [], []
    oracle_correct = self_correct = uniform_correct = 0
    model_pred_correct = 0          # how often the full-model prediction is right
    self_matches_gold = 0           # how often self-oracle letter == gold letter
    total = 0

    for rec in records:
        choices, correct_idx = options_and_answer(rec)
        if choices is None:
            continue
        letter_token_ids = [processor.tokenizer(c, add_special_tokens=False).input_ids[0]
                            for c in choices]
        gt_token = letter_token_ids[correct_idx]

        prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels, args.fps)
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

        # ---- merged video tokens (detached source values for the leaves) ----
        with torch.no_grad():
            ve_feats = model.get_video_features(pix, video_grid_thw).pooler_output
            ve_feats = torch.cat(ve_feats, dim=0).to(device)   # (n_video, d), detached
            base_embeds = model.get_input_embeddings()(input_ids)  # (1,S,d), detached
            base_embeds = base_embeds.clone()
            base_embeds[0, video_positions] = ve_feats.to(base_embeds.dtype)

        position_ids = build_full_positions(model, input_ids, video_positions,
                                            video_grid_thw, attn)

        # ---- TRUE-LABEL oracle (own forward+backward on gold letter) ----
        oracle, logits_last = oracle_from_letter(
            model, base_embeds, ve_feats, video_positions, position_ids, attn, gt_token)

        # model's own predicted letter (restricted to the option letters) ----
        opt_logits = torch.tensor([logits_last[t].item() for t in letter_token_ids])
        pred_idx = int(opt_logits.argmax().item())
        pred_token = letter_token_ids[pred_idx]
        model_pred_correct += int(pred_idx == correct_idx)
        self_matches_gold += int(pred_token == gt_token)

        # ---- SELF oracle (own forward+backward on the model's predicted letter) ----
        # Fully independent call: its own fresh leaf + graph, freed on return.
        # Peak memory stays at one backward; no retain_graph needed.
        self_oracle, _ = oracle_from_letter(
            model, base_embeds, ve_feats, video_positions, position_ids, attn, pred_token)

        # ---- H4: tail index + shape (true-label oracle) ----
        if oracle.numel() >= 50:
            xi_hist.append(einmahlHaan(oracle))
            frac_zero_hist.append((oracle <= 1e-8).float().mean().item())
            o = oracle
            skew_hist.append((((o - o.mean()) / (o.std() + 1e-8)) ** 3).mean().item())
            if args.plot:
                x = oracle.flatten().float().cpu().numpy()
                x = np.sort(x[x > 1e-8])
                if x.size >= 12:
                    pooled_saliency.append(x)
                    survivals.append((x, 1.0 - np.arange(x.size) / x.size))
                    h = hill_tail_index(oracle)
                    if h == h:
                        hill_hist.append(h)

        # ---- CEILING: oracle / self_oracle / uniform downstream ----
        # base_embeds is already detached (built under no_grad); score directly.
        keep_oracle = select_by_scores(oracle, n_frames, k)
        keep_self = select_by_scores(self_oracle, n_frames, k)
        keep_uniform = select_uniform(n_video, n_frames, k, device)
        lp_or = score_answer(model, base_embeds, position_ids, attn,
                            video_positions, keep_oracle, letter_token_ids)
        lp_self = score_answer(model, base_embeds, position_ids, attn,
                            video_positions, keep_self, letter_token_ids)
        lp_un = score_answer(model, base_embeds, position_ids, attn,
                            video_positions, keep_uniform, letter_token_ids)
        oracle_correct += int(lp_or.argmax().item() == correct_idx)
        self_correct += int(lp_self.argmax().item() == correct_idx)
        uniform_correct += int(lp_un.argmax().item() == correct_idx)
        total += 1

        del oracle, self_oracle, base_embeds, ve_feats
        if total % 10 == 0:
            print(f"[{total}] xi {np.median(xi_hist):.3f} | "
                  f"oracle {oracle_correct/total:.3f} | self {self_correct/total:.3f} | "
                  f"uniform {uniform_correct/total:.3f} | "
                  f"model_acc {model_pred_correct/total:.3f}")

    print("\n==== GRADIENT-ORACLE DIAGNOSTIC (with label-free self-oracle) ====")
    print(f"  samples: {total}")
    print(f"  H4  median xi_hat(oracle): {np.median(xi_hist):.4f}   "
          f"IQR [{np.percentile(xi_hist,25):.4f}, {np.percentile(xi_hist,75):.4f}]")
    print(f"      median frac at zero (post-ReLU): {np.median(frac_zero_hist):.3f}")
    print(f"      median skew: {np.median(skew_hist):.3f}")
    print(f"  model prediction accuracy (full):   {model_pred_correct/total:.4f}")
    print(f"  self-oracle letter == gold letter:  {self_matches_gold/total:.4f}")
    print(f"  CEILING  oracle:      {oracle_correct/total:.4f}   "
          f"gap vs uniform {(oracle_correct-uniform_correct)/total:+.4f}")
    print(f"           self_oracle: {self_correct/total:.4f}   "
          f"gap vs uniform {(self_correct-uniform_correct)/total:+.4f}")
    print(f"           uniform:     {uniform_correct/total:.4f}")
    print("\n  Read:")
    print("   oracle gap > 0       -> a token subset carries the answer (privileged ceiling)")
    print("   self_oracle gap > 0  -> recoverable WITHOUT the label -> deployable method exists")
    print("   self_oracle gap ~ 0  -> the label was doing the work; this branch is closed")
    print("   (self-oracle quality is bounded by model prediction accuracy above)")

    if args.plot:
        plot_oracle_distribution(pooled_saliency, survivals, xi_hist, hill_hist, args.plot_out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames per clip; fps sampling clamps to the clip length below this.")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for video sampling (qwen_vl_utils default 2.0); short clips yield fewer frames instead of being skipped.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--retention", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true",
                   help="Also save a 3-panel figure of the oracle saliency distribution "
                        "(histogram, log-log survival, Demahl-vs-Hill tail index).")
    p.add_argument("--plot_out", default="figs/oracle_distribution.png",
                   help="Output path for the --plot figure.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())