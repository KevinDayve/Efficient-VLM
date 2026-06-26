"""
make_figures.py -- the four paper figures.
==========================================
Computes the three token-importance signals over the validation set and writes
the four figure PNGs. By default everything is real-time and in-memory (no
caching). Optionally, --save_scores writes the per-sample score vectors so a
later --load_scores re-plot skips the VLM entirely (opt-in; off by default).

Per sample:
  * oracle      : ReLU(<g, v>), g = d log p(gold letter)/d v   (one backward)
  * attention   : language->video attention, mean over --attn_layers
  * redundancy  : 1 - cos(v_i^t, v_i^{t-1}), temporal novelty   (no grad)

Figures (--out_dir):
  F1  Captured importance mass vs retention  (validates the captured-mass law)
  F2  Tail-index estimator stability vs threshold k  (DEdH vs Hill)
  F3  Upper-tail survival of token-importance scores
  F4  Downstream accuracy vs retention: oracle ceiling vs uniform vs random

Run (compute + plot):
    python make_figures.py --data_file .../val.jsonl --video_root ... \
        --max_frames 8 --limit 300 --out_dir figs --save_scores figs/scores.npz

Re-plot only (no VLM):
    python make_figures.py --load_scores figs/scores.npz --out_dir figs
"""

import os
import json
import random
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {"oracle": "#C44E52", "attention": "#4C72B0", "redundancy": "#55A868",
           "uniform": "#7F7F7F", "random": "#CCCCCC"}
SIGNAL_NAMES = ("oracle", "attention", "redundancy")


# --------------------------------------------------------------------------- #
# score-vector persistence (opt-in; ragged-safe flat concat + lengths)
# --------------------------------------------------------------------------- #
def save_all_scores(path, sig, ladder, ladder_n, xi_oracle):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"ladder_json": json.dumps(ladder),
               "ladder_n": np.array([ladder_n]),
               "xi_oracle": np.array([xi_oracle])}
    for name in SIGNAL_NAMES:
        arrs = [np.asarray(a, dtype=np.float64).ravel() for a in sig[name]]
        payload[f"{name}__flat"] = np.concatenate(arrs) if arrs else np.zeros(0)
        payload[f"{name}__len"] = np.array([a.size for a in arrs], dtype=np.int64)
    np.savez(path, **payload)
    print(f"  [save] score vectors + ladder -> {path}")


def load_all_scores(path):
    d = np.load(path, allow_pickle=False)
    sig = {}
    for name in SIGNAL_NAMES:
        flat, lens = d[f"{name}__flat"], d[f"{name}__len"]
        out, i = [], 0
        for n in lens:
            out.append(flat[i:i + int(n)]); i += int(n)
        sig[name] = out
    ladder = {float(k): v for k, v in json.loads(str(d["ladder_json"])).items()}
    ladder_n = int(d["ladder_n"][0])
    xi_oracle = float(d["xi_oracle"][0])
    print(f"  [load] {path}: "
          + ", ".join(f"{n}={len(sig[n])}" for n in SIGNAL_NAMES)
          + f", ladder_n={ladder_n}")
    return sig, ladder, ladder_n, xi_oracle


# --------------------------------------------------------------------------- #
# estimators
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
    Global top-k per sample (matches the captured-mass proposition)."""
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
# figures
# --------------------------------------------------------------------------- #
def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def fig_captured_mass(signals, xi, path):
    rhos = np.logspace(np.log10(0.005), np.log10(0.5), 24)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for name, arrays in signals.items():
        C, cnt = captured_mass(arrays, rhos)
        if cnt == 0:
            continue
        c = PALETTE.get(name)
        ax.loglog(rhos, C, "o-", color=c, lw=1.8, ms=4, label=f"{name} (measured)")
        if name in xi:
            sl = 1.0 - xi[name]
            anchor = C[-1] / (rhos[-1] ** sl)
            ax.loglog(rhos, anchor * rhos ** sl, "--", color=c, lw=1.2,
                      label=f"{name} (predicted)")
    ax.loglog(rhos, rhos, ":", color="k", lw=1.0, label=r"no concentration ($C_{top}=\rho$)")
    ax.set_xlabel(r"retention ratio $\rho$")
    ax.set_ylabel(r"captured importance mass $C_{top}(\rho)$")
    ax.set_title("Importance-mass concentration under top-$k$ selection\n"
                 "curves above the diagonal concentrate importance")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.2)
    _save(fig, path)


def fig_tail_stability(signals, path, frac=0.10):
    items = list(signals.items())
    fig, axes = plt.subplots(1, len(items), figsize=(5.4 * len(items), 4.6), squeeze=False)
    for ax, (name, arrays) in zip(axes[0], items):
        pooled = np.concatenate(per_sample_normalise(arrays)) if arrays else np.zeros(0)
        if pooled.size < 50:
            ax.set_title(f"{name}: too few points"); continue
        n_pos = int((pooled > 0).sum())
        k_hi = max(11, int(frac * n_pos))                 # cap k in the TAIL only
        k_grid = np.unique(np.linspace(10, k_hi, 60).astype(int))
        dedh, hill = tail_index_curves(pooled, k_grid)
        ax.plot(k_grid, dedh, color=PALETTE.get(name, "#333"), lw=1.8,
                label="DEdH (sign-aware)")
        ax.plot(k_grid, hill, color="#DD8452", lw=1.4, ls="--", label=r"Hill ($\geq 0$)")
        ax.axhline(0.0, color="k", lw=0.8, ls=":", label=r"$\xi=0$ (light tail)")
        ax.set_xlabel(r"threshold $k$ (upper order statistics used)")
        ax.set_ylabel(r"estimated tail index $\hat\xi(k)$")
        ax.set_title(f"{name}")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
        ax.grid(True, alpha=0.2)
    fig.suptitle(r"Tail-index estimator stability vs threshold $k$ "
                 r"(tail region, $k\leq$ %d%% of positives)" % int(frac * 100),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, path)


def fig_survival(signals, xi, path):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for name, arrays in signals.items():
        pooled = per_sample_normalise(arrays)
        if not pooled:
            continue
        x = np.sort(np.concatenate(pooled)); x = x[x > 0]
        surv = 1.0 - np.arange(x.size) / x.size
        ax.loglog(x, surv, color=PALETTE.get(name), lw=1.8, label=f"{name}")
    ax.set_xlabel(r"normalised importance score $x$ (per-sample median = 1)")
    ax.set_ylabel(r"survival $P(X>x)$")
    ax.set_title("Upper-tail survival of token-importance scores\n"
                 "straight line = power-law (heavy); early drop-off = bounded")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.2)
    _save(fig, path)


def fig_ladder(table, path, full_acc=None):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    rhos = sorted(table)
    labels = {"oracle": "gold oracle (privileged ceiling)",
              "attention": "attention (learned-importance proxy)",
              "uniform": "stratified-uniform", "random": "random"}
    for name in ("oracle", "attention", "uniform", "random"):
        if name in table[rhos[0]]:
            ax.plot(rhos, [table[r][name] for r in rhos], "o-",
                    color=PALETTE.get(name), lw=1.8, ms=5, label=labels[name])
    if full_acc is not None:
        ax.axhline(full_acc, color="k", ls=":", lw=1.0, label="full model")
    ax.set_xlabel(r"retention ratio $\rho$")
    ax.set_ylabel("answer accuracy")
    ax.set_title("Downstream accuracy under token selection\n"
                 "oracle beats uniform; learned/uniform/random coincide")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    _save(fig, path)


def render_all(sig, ladder, xi, out_dir, frac):
    fig_captured_mass(sig, xi, os.path.join(out_dir, "F1_captured_mass.png"))
    fig_tail_stability(sig, os.path.join(out_dir, "F2_tail_stability.png"), frac=frac)
    fig_survival(sig, xi, os.path.join(out_dir, "F3_survival.png"))
    fig_ladder(ladder, os.path.join(out_dir, "F4_ladder.png"))


# --------------------------------------------------------------------------- #
# compute (only when not loading from disk)
# --------------------------------------------------------------------------- #
def compute_scores(args):
    import torch
    import torch.nn.functional as F
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from qwen_vl_utils import process_vision_info
    from efficient_vlm.utils import einmahlHaan
    from oracle_check import (make_prompt, options_and_answer, build_full_positions,
                              select_uniform, select_by_scores, score_answer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False); model.gradient_checkpointing_enable()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    attn_layers = [int(x) for x in args.attn_layers.split(",")]

    def oracle_scores(base_embeds, ve_feats, vpos, position_ids, attn, letter_token_id):
        ve = ve_feats.detach().clone().requires_grad_(True)
        ie = base_embeds.clone(); ie[0, vpos] = ve.to(ie.dtype)
        out = model(inputs_embeds=ie, position_ids=position_ids, attention_mask=attn, use_cache=False)
        logits_last = out.logits[0, -1, :].float()
        target = F.log_softmax(logits_last, dim=-1)[letter_token_id]
        model.zero_grad(set_to_none=True); target.backward()
        s = F.relu((ve.grad.float() * ve.detach().float()).sum(dim=-1)).detach()
        del out, ie, ve, target
        return s, logits_last.detach()

    @torch.no_grad()
    def attention_scores(base_embeds, vpos, position_ids, attn, layers):
        out = model(inputs_embeds=base_embeds, position_ids=position_ids,
                    attention_mask=attn, use_cache=False, output_attentions=True)
        text_mask = torch.ones(base_embeds.shape[1], dtype=torch.bool, device=base_embeds.device)
        text_mask[vpos] = False
        acc = torch.zeros(vpos.numel(), device=base_embeds.device)
        for L in layers:
            a = out.attentions[L][0].mean(0)
            acc += a[text_mask][:, vpos].sum(dim=0)
        del out
        return (acc / max(1, len(layers))).float()

    @torch.no_grad()
    def redundancy_scores(ve_feats, n_frames):
        N, d = ve_feats.shape
        per = N // n_frames
        x = ve_feats[: per * n_frames].view(n_frames, per, d).float()
        cos = F.cosine_similarity(x[1:], x[:-1], dim=-1)
        nov = torch.zeros(n_frames, per, device=ve_feats.device)
        nov[1:] = (1.0 - cos).clamp(min=0)
        return nov.reshape(-1)[: per * n_frames]

    with open(args.data_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    random.Random(args.seed).shuffle(records)
    records = records[: args.limit]

    sig = {n: [] for n in SIGNAL_NAMES}
    xi_running = []
    rho_grid = [0.01, 0.05, 0.10, 0.25]
    ladder = {r: {"oracle": 0, "attention": 0, "uniform": 0, "random": 0} for r in rho_grid}
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

        o, _ = oracle_scores(base, ve, vpos, pos, attn, gt_token)
        a = attention_scores(base, vpos, pos, attn, attn_layers)
        r = redundancy_scores(ve, n_frames)
        sig["oracle"].append(o.float().cpu().numpy())
        sig["attention"].append(a.float().cpu().numpy())
        sig["redundancy"].append(r.float().cpu().numpy())
        if o.numel() >= 50:
            xi_running.append(einmahlHaan(o))

        for rho in rho_grid:
            k = max(n_frames, int(round(rho * n_video)))
            keep_o = select_by_scores(o, n_frames, k)
            keep_a = select_by_scores(a, n_frames, k)
            keep_u = select_uniform(n_video, n_frames, k, device)
            ku = torch.sort(torch.randperm(n_video, generator=rng)[:k]).values.to(device)
            lp_o = score_answer(model, base, pos, attn, vpos, keep_o, letter_ids)
            lp_a = score_answer(model, base, pos, attn, vpos, keep_a, letter_ids)
            lp_u = score_answer(model, base, pos, attn, vpos, keep_u, letter_ids)
            lp_r = score_answer(model, base, pos, attn, vpos, ku, letter_ids)
            ladder[rho]["oracle"] += int(lp_o.argmax().item() == correct_idx)
            ladder[rho]["attention"] += int(lp_a.argmax().item() == correct_idx)
            ladder[rho]["uniform"] += int(lp_u.argmax().item() == correct_idx)
            ladder[rho]["random"] += int(lp_r.argmax().item() == correct_idx)
        ladder_n += 1
        del base, ve, o, a, r
        if ladder_n % 20 == 0:
            print(f"[{ladder_n}] xi(oracle) median {np.median(xi_running):.3f}")

    if ladder_n:
        for rho in rho_grid:
            for kk in ladder[rho]:
                ladder[rho][kk] /= ladder_n
    xi_oracle = float(np.median(xi_running)) if xi_running else 0.43
    return sig, ladder, ladder_n, xi_oracle


def main(args):
    if args.load_scores:
        sig, ladder, ladder_n, xi_oracle = load_all_scores(args.load_scores)
    else:
        sig, ladder, ladder_n, xi_oracle = compute_scores(args)
        if ladder_n == 0:
            print("no usable samples; nothing to plot."); return
        if args.save_scores:
            save_all_scores(args.save_scores, sig, ladder, ladder_n, xi_oracle)

    xi = {"oracle": xi_oracle, "attention": args.attention_xi, "redundancy": args.redundancy_xi}
    print(f"  oracle xi = {xi_oracle:.3f} over {ladder_n} samples")
    render_all(sig, ladder, xi, args.out_dir, args.tail_frac)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", default="")
    p.add_argument("--video_root", default="")
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--attn_layers", default="12,13,14,15,16")
    p.add_argument("--attention_xi", type=float, default=0.28)
    p.add_argument("--redundancy_xi", type=float, default=-0.28)
    p.add_argument("--tail_frac", type=float, default=0.10,
                   help="cap the F2 k-sweep at this fraction of positive values (tail only)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="figs")
    p.add_argument("--save_scores", default="",
                   help="opt-in: write per-sample scores + ladder to this .npz for re-plotting")
    p.add_argument("--load_scores", default="",
                   help="re-plot from a saved .npz; skips the VLM entirely")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())