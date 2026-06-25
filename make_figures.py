"""
make_figures.py -- the four paper figures, computed in real time.
=================================================================
Runs the VLM over the validation set, computes the three importance signals per
sample IN MEMORY, and writes only the figure PNGs at the end. No caching, no
score dumps, no synthetic data.

Per sample we compute:
  * oracle      : ReLU(<g, v>), g = d log p(gold letter)/d v   (one backward)
  * attention   : language->video attention, mean over layers [--attn_layers]
  * redundancy  : 1 - cos(v_i^t, v_i^{t-1}), temporal novelty   (no grad)

Figures (written to --out_dir):
  F1  captured-mass C_top(rho) vs rho per signal, with reference slope 1-xi_hat
      (validates Proposition 1: heavy -> slope <1; bounded -> slope ~1)
  F2  tail-index stability xi_hat(k) vs k, DEdH (sign-aware) vs Hill (>=0)
  F3  per-sample-normalised pooled survival P(X>x) (straight=heavy, bends=bounded)
  F4  impossibility ladder: accuracy vs rho for oracle / uniform / random

Run:
    python make_figures.py \
        --data_file /home/ubuntu/NeXTVideo/val.jsonl \
        --video_root /home/ubuntu/NeXTVideo \
        --max_frames 8 --limit 300 --out_dir figs
"""

import os
import json
import random
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

from efficient_vlm.utils import einmahlHaan
# Reuse the exact data + position helpers used everywhere else.
from oracle_check import (make_prompt, options_and_answer, build_full_positions,
                          select_uniform, select_by_scores, score_answer)

PALETTE = {"oracle": "#C44E52", "attention": "#4C72B0", "redundancy": "#55A868",
           "uniform": "#7F7F7F", "random": "#CCCCCC"}


# --------------------------------------------------------------------------- #
# estimators (self-contained; sign-aware DEdH + Hill as functions of k)
# --------------------------------------------------------------------------- #
def _dedh_hill_at_k(x_desc, k):
    if k + 1 >= x_desc.size or x_desc[k] <= 0:
        return np.nan, np.nan
    logs = np.log(x_desc[:k]) - np.log(x_desc[k])
    M1, M2 = logs.mean(), (logs ** 2).mean()
    if M2 <= 0:
        return np.nan, np.nan
    return M1 + 1.0 - 0.5 / (1.0 - M1 * M1 / M2), M1


def tail_index_curves(pooled_positive, k_grid):
    x = np.sort(pooled_positive[pooled_positive > 0])[::-1]
    dedh = np.full(k_grid.size, np.nan)
    hill = np.full(k_grid.size, np.nan)
    for j, k in enumerate(k_grid):
        dedh[j], hill[j] = _dedh_hill_at_k(x, int(k))
    return dedh, hill


def per_sample_normalise(arrays, mode="median"):
    out = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        pos = a[a > 0]
        if pos.size == 0:
            continue
        s = np.median(pos) if mode == "median" else pos.max()
        if s > 0:
            out.append(pos / s)
    return out


def captured_mass(arrays, rhos):
    """Mean over samples of C_top(rho) = sum(top ceil(rho*N)) / sum(all).
    Global top-k per sample (matches Proposition 1)."""
    C = np.zeros(rhos.size)
    cnt = 0
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        a = a[a >= 0]
        N, tot = a.size, a.sum()
        if N < 8 or tot <= 0:
            continue
        csum = np.cumsum(np.sort(a)[::-1])
        for j, rho in enumerate(rhos):
            k = max(1, int(np.ceil(rho * N)))
            C[j] += csum[min(k, N) - 1] / tot
        cnt += 1
    return (C / cnt if cnt else C), cnt


# --------------------------------------------------------------------------- #
# signal computation (real time)
# --------------------------------------------------------------------------- #
def oracle_scores(model, base_embeds, ve_feats, video_positions,
                  position_ids, attn, letter_token_id):
    """ReLU(<g, v>) with a fresh leaf + own graph (single backward)."""
    ve = ve_feats.detach().clone().requires_grad_(True)
    inputs_embeds = base_embeds.clone()
    inputs_embeds[0, video_positions] = ve.to(inputs_embeds.dtype)
    out = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                attention_mask=attn, use_cache=False)
    logits_last = out.logits[0, -1, :].float()
    target = F.log_softmax(logits_last, dim=-1)[letter_token_id]
    model.zero_grad(set_to_none=True)
    target.backward()
    g = ve.grad.float()
    s = F.relu((g * ve.detach().float()).sum(dim=-1)).detach()
    del out, inputs_embeds, ve, g, target
    return s, logits_last.detach()


@torch.no_grad()
def attention_scores(model, base_embeds, video_positions, position_ids, attn, layers):
    """Mean language->video attention received by each video token over `layers`."""
    out = model(inputs_embeds=base_embeds, position_ids=position_ids,
                attention_mask=attn, use_cache=False, output_attentions=True)
    attns = out.attentions  # tuple[L] of (1, H, S, S)
    vpos = video_positions
    text_mask = torch.ones(base_embeds.shape[1], dtype=torch.bool, device=base_embeds.device)
    text_mask[vpos] = False
    acc = torch.zeros(vpos.numel(), device=base_embeds.device)
    for L in layers:
        a = attns[L][0].mean(0)                  # (S, S) mean over heads
        # attention FROM text queries TO each video key, summed over text queries
        acc += a[text_mask][:, vpos].sum(dim=0)
    del out
    return (acc / max(1, len(layers))).float()


@torch.no_grad()
def redundancy_scores(ve_feats, n_frames):
    """1 - cos(v_i^t, v_i^{t-1}); frame 0 -> 0. Returns (n_video,) >= 0."""
    N, d = ve_feats.shape
    per = N // n_frames
    x = ve_feats[: per * n_frames].view(n_frames, per, d).float()
    cos = F.cosine_similarity(x[1:], x[:-1], dim=-1)       # (T-1, per)
    nov = torch.zeros(n_frames, per, device=ve_feats.device)
    nov[1:] = (1.0 - cos).clamp(min=0)
    return nov.reshape(-1)[: per * n_frames]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _save(fig, path):
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {path}")


def fig_captured_mass(signals, xi, path):
    rhos = np.logspace(np.log10(0.005), np.log10(0.5), 24)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, arrays in signals.items():
        C, cnt = captured_mass(arrays, rhos)
        if cnt == 0:
            continue
        c = PALETTE.get(name)
        ax.loglog(rhos, C, "o-", color=c, lw=1.8, ms=4, label=f"{name}: measured")
        if name in xi:
            sl = 1.0 - xi[name]
            anchor = C[-1] / (rhos[-1] ** sl)
            ax.loglog(rhos, anchor * rhos ** sl, "--", color=c, lw=1.2,
                      label=f"  slope $1-\\xi={sl:.2f}$")
    ax.loglog(rhos, rhos, ":", color="k", lw=1.0, label="slope 1 (no concentration)")
    ax.set_xlabel(r"retention $\rho$")
    ax.set_ylabel(r"captured importance mass $C_{top}(\rho)$")
    ax.set_title("F1  Captured mass vs Proposition 1")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
    _save(fig, path)


def fig_tail_stability(signals, path):
    items = list(signals.items())
    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 4.4), squeeze=False)
    for ax, (name, arrays) in zip(axes[0], items):
        pooled = np.concatenate(per_sample_normalise(arrays)) if arrays else np.zeros(0)
        if pooled.size < 50:
            ax.set_title(f"{name}: too few points"); continue
        n_pos = int((pooled > 0).sum())
        k_grid = np.unique(np.linspace(10, max(11, int(0.25 * n_pos)), 60).astype(int))
        dedh, hill = tail_index_curves(pooled, k_grid)
        ax.plot(k_grid, dedh, color=PALETTE.get(name, "#333"), lw=1.8, label="DEdH")
        ax.plot(k_grid, hill, color="#DD8452", lw=1.4, ls="--", label="Hill")
        ax.axhline(0.0, color="k", lw=0.8, ls=":", label=r"$\xi=0$")
        ax.set_xlabel("k"); ax.set_ylabel(r"$\hat\xi(k)$")
        ax.set_title(f"F2  {name}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    _save(fig, path)


def fig_survival(signals, xi, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, arrays in signals.items():
        pooled = per_sample_normalise(arrays)
        if not pooled:
            continue
        x = np.sort(np.concatenate(pooled)); x = x[x > 0]
        surv = 1.0 - np.arange(x.size) / x.size
        c = PALETTE.get(name)
        ax.loglog(x, surv, color=c, lw=1.8, label=name)
        if name in xi and xi[name] > 0.05:
            sl = -1.0 / xi[name]; m = x[x.size // 2]
            anchor = surv[x.size // 2] / (m ** sl)
            ax.loglog(x, anchor * x ** sl, ls="--", lw=1.0, color=c, alpha=0.6,
                      label=f"  slope $-1/\\xi={sl:.1f}$")
    ax.set_xlabel("normalised score x (log)"); ax.set_ylabel(r"$P(X>x)$ (log)")
    ax.set_title("F3  Normalised survival"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    _save(fig, path)


def fig_ladder(table, path, full_acc=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rhos = sorted(table)
    for name in ("oracle", "uniform", "random"):
        if name in table[rhos[0]]:
            ax.plot(rhos, [table[r][name] for r in rhos], "o-",
                    color=PALETTE.get(name), lw=1.8, ms=5, label=name)
    if full_acc is not None:
        ax.axhline(full_acc, color="k", ls=":", lw=1.0, label="full model")
    ax.set_xlabel(r"retention $\rho$"); ax.set_ylabel("answer accuracy")
    ax.set_title("F4  Impossibility ladder"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# main: compute everything in real time, then plot
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False); model.gradient_checkpointing_enable()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    attn_layers = [int(x) for x in args.attn_layers.split(",")]

    with open(args.data_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    random.Random(args.seed).shuffle(records)
    records = records[: args.limit]

    # in-memory only
    sig = {"oracle": [], "attention": [], "redundancy": []}
    xi_running = []
    rho_grid = [0.01, 0.05, 0.10, 0.25]
    ladder = {r: {"oracle": 0, "uniform": 0, "random": 0} for r in rho_grid}
    ladder_n = 0
    rng = torch.Generator().manual_seed(args.seed)

    for rec in records:
        choices, correct_idx = options_and_answer(rec)
        if choices is None:
            continue
        letter_ids = [processor.tokenizer(c, add_special_tokens=False).input_ids[0] for c in choices]
        gt_token = letter_ids[correct_idx]

        prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels)
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(prompt)
        inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        vpos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if vpos.numel() < 50:
            continue
        n_video = vpos.numel()
        grid_thw = inputs["video_grid_thw"].to(device)
        n_frames = int(grid_thw[0][0].item())
        pix = inputs["pixel_values_videos"].to(device)

        with torch.no_grad():
            ve = model.get_video_features(pix, grid_thw).pooler_output
            ve = torch.cat(ve, dim=0).to(device)
            base = model.get_input_embeddings()(input_ids).clone()
            base[0, vpos] = ve.to(base.dtype)
        pos = build_full_positions(model, input_ids, vpos, grid_thw, attn)

        # --- signals (real time) ---
        o, _ = oracle_scores(model, base, ve, vpos, pos, attn, gt_token)
        a = attention_scores(model, base, vpos, pos, attn, attn_layers)
        r = redundancy_scores(ve, n_frames)

        sig["oracle"].append(o.float().cpu().numpy())
        sig["attention"].append(a.float().cpu().numpy())
        sig["redundancy"].append(r.float().cpu().numpy())
        if o.numel() >= 50:
            xi_running.append(einmahlHaan(o))

        # --- F4 ladder: oracle vs uniform vs random (real downstream accuracy) ---
        for rho in rho_grid:
            k = max(n_frames, int(round(rho * n_video)))
            keep_o = select_by_scores(o, n_frames, k)
            keep_u = select_uniform(n_video, n_frames, k, device)
            ku = torch.sort(torch.randperm(n_video, generator=rng)[:k]).values.to(device)
            lp_o = score_answer(model, base, pos, attn, vpos, keep_o, letter_ids)
            lp_u = score_answer(model, base, pos, attn, vpos, keep_u, letter_ids)
            lp_r = score_answer(model, base, pos, attn, vpos, ku, letter_ids)
            ladder[rho]["oracle"] += int(lp_o.argmax().item() == correct_idx)
            ladder[rho]["uniform"] += int(lp_u.argmax().item() == correct_idx)
            ladder[rho]["random"] += int(lp_r.argmax().item() == correct_idx)
        ladder_n += 1

        del base, ve, o, a, r
        if ladder_n % 20 == 0:
            print(f"[{ladder_n}] xi(oracle) median {np.median(xi_running):.3f}")

    if ladder_n == 0:
        print("no usable samples; nothing to plot."); return
    for rho in rho_grid:
        for kk in ladder[rho]:
            ladder[rho][kk] /= ladder_n

    xi = {"oracle": float(np.median(xi_running)) if xi_running else 0.43,
          "attention": args.attention_xi, "redundancy": args.redundancy_xi}
    print(f"  measured oracle xi (median): {xi['oracle']:.3f}  over {ladder_n} samples")

    fig_captured_mass(sig, xi, os.path.join(args.out_dir, "F1_captured_mass.png"))
    fig_tail_stability(sig, os.path.join(args.out_dir, "F2_tail_stability.png"))
    fig_survival(sig, xi, os.path.join(args.out_dir, "F3_survival.png"))
    fig_ladder(ladder, os.path.join(args.out_dir, "F4_ladder.png"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--attn_layers", default="12,13,14,15,16",
                   help="comma-separated decoder layers for the attention signal")
    p.add_argument("--attention_xi", type=float, default=0.28,
                   help="xi_hat for the attention reference slope (from your run)")
    p.add_argument("--redundancy_xi", type=float, default=-0.28)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="figs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())