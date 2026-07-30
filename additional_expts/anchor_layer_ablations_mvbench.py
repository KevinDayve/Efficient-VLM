"""
anchor_layer_ablations_mvbench.py -- Section 9 ("Design Choices to Validate") of
Efficient_VLMs_Method.pdf, benchmarked on MVBench.
=============================================================================================
The core method (Sections 1-8: value-norm debiased importance, moment-estimator anchor
selection, cross-shaped spatio-temporal diversity, real in-LLM prune) is already fully
implemented in anchor_layer_prune.py and benchmarked end to end by
anchor_layer_prune_mvbench.py (anchor_prune vs random_prune/uniform_prune/full). This file
does NOT re-implement that method -- it imports it -- and adds exactly the extra knobs
Section 9's table asks for, one CLI flag (--ablation) selecting which row to run:

  anchor_rule          tail-argmax L* vs. every fixed layer in B vs. entropy vs. Gini vs.
                        random layer.
  sink_debiasing        raw attention vs. value-norm s_t (default) vs. contrastive s_con_t
                        (free single-pass approximation, always computed; optional full
                        2-pass version via --include_contrastive_full) vs. two free
                        POSITIONAL-debias arms (pos_debias_seq: smooth recency detrend along
                        the sequence; pos_debias_frame: per-frame temporal-tilt subtraction --
                        both single-pass, no baseline prompt, targeting the position component
                        that value-norm cannot see); effect on BOTH L* (per-variant histogram)
                        and accuracy.
  estimator_stability    single-m tail-argmax vs. a Hill-plot-plateau average over
                        --m_fracs; the full gamma-hat(m) table per band layer is dumped to
                        the per-sample JSON for post-hoc plotting.
  redundancy_metric      both arms (Section 6 default) vs. spatial-arm-only vs.
                        temporal-arm-only vs. naive feature cosine (Section 6.1's rejected
                        default, kept here ONLY as the ablation foil).
  stability_gate         with vs. without g_t (does removing it wrongly delete fast
                        motion?), plus a --proj_dims sweep of the gate's projection width.
  arm_combination        max(Rs,Rt) (default) vs. weighted alpha*Rs+(1-alpha)*Rt over
                        --alphas.
  lambda_schedule        constant lambda (beta=0) vs. rho-indexed lambda0*rho^-beta over
                        --lambda_betas, reported across the full --rhos sweep (this row's
                        whole point is behaviour AS rho changes).
  all                    every row above, sharing the SAME one-dense-forward-per-clip
                        analysis pass (see `analyze_clip`) wherever a row doesn't need its
                        own extra forward (only `sink_debiasing`'s optional
                        --include_contrastive_full needs a second one).

Every row holds everything else fixed at the pipeline's own defaults (Section 9's header:
"experiments should settle [these] at matched compute... holding everything else fixed") --
see `build_variant_specs` for exactly what is isolated per row.

Run (one row):
    python anchor_layer_ablations_mvbench.py --ablation redundancy_metric \
        --data_root ~/Experiments/MVBench --tasks "Action Sequence" "Scene Transition" \
        --official_sampling --num_segments 16 --max_pixels 200704 \
        --max_samples 40 --rhos 0.05,0.10,0.25 --band 2,3,4,5,6,7,8 \
        --out results_ablation_redundancy_metric_mvbench.json

Run everything:
    python anchor_layer_ablations_mvbench.py --ablation all --data_root ~/Experiments/MVBench \
        --tasks "Action Sequence" --max_samples 20 --rhos 0.10 \
        --out results_ablation_all_mvbench.json

Fast, CPU-only, no-model correctness checks for the new machinery in this file (the base
method's own correctness is anchor_layer_prune.py --self_test's job):
    python anchor_layer_ablations_mvbench.py --self_test
"""

import os
import sys
import time
import types
import json
import warnings
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from efficient_vlm.utils import einmahlHaan, hill_tail_index
from oracle_check import build_full_positions  # noqa: F401 (re-exported for parity w/ sibling scripts)
from inference import DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt, official_frames  # noqa: F401
from inference_only_mvbench import letter_first_ids  # noqa: F401
from lcds_mvbench import select_uniform_exact  # noqa: F401 (not used as a strategy here; kept for parity)
from anchor_layer_prune import (
    token_grid, token_grid_video, stability_gates, greedy_select, select_anchor_layer,
    debiased_scores_by_layer, install_anchor_prune, set_anchor_prune, uninstall_anchor_prune,
    resolve_backbone, default_model_id, build_llava_video_prompt, full_sequence_keep_idx,
    _text_model, _anchor_prune_text_forward, _anchor_prune_plain_lm_forward,
    QWEN_MODEL_ID, LLAVA_VIDEO_MODEL_ID,
)
from anchor_layer_prune_mvbench import load_backbone, prepare_clip, score_letters

ABLATIONS = (
    "anchor_rule", "sink_debiasing", "estimator_stability", "redundancy_metric",
    "stability_gate", "arm_combination", "lambda_schedule",
)


# --------------------------------------------------------------------------- #
# 1. One dense analysis forward -> raw / value-norm / free-contrastive scores
#    for every band layer, plus per-layer hidden states (Section 4-5 inputs,
#    generalized beyond anchor_layer_prune.plan()'s single-L* return so every
#    ablation row below can share the SAME forward pass).
# --------------------------------------------------------------------------- #
def _dense_forward(model, base_embeds, position_ids, attn_mask):
    """Identical bypass trick to anchor_layer_prune.plan(): if install_anchor_prune
    already patched the text model's forward, swap back to the original for this
    one call (the patched forward has none of the output_attentions/
    output_hidden_states plumbing), then restore the patch."""
    text_model = _text_model(model)
    orig_forward = getattr(text_model, "_anchor_orig_forward", None)
    if orig_forward is not None:
        mrope = type(text_model).__name__ == "Qwen2_5_VLTextModel"
        patched = _anchor_prune_text_forward if mrope else _anchor_prune_plain_lm_forward
        text_model.forward = orig_forward
    try:
        with torch.no_grad():
            out = model(inputs_embeds=base_embeds, position_ids=position_ids,
                        attention_mask=attn_mask, use_cache=False,
                        output_attentions=True, output_hidden_states=True)
    finally:
        if orig_forward is not None:
            text_model.forward = types.MethodType(patched, text_model)
    return text_model, out


def raw_scores_by_layer(attentions, video_idx, query_mask, layers):
    """s_raw,(l)_t (Section 4's undebiased prior): mean attention received from
    `query_mask` positions, mean over heads -- no value-norm weighting. The
    sink-inflated baseline that `debiased_scores_by_layer` (value-norm) corrects."""
    out = {}
    for l in layers:
        A = attentions[l][0].float()                      # (H,Sq,Sk)
        recv = A[:, query_mask, :].mean(dim=1)             # (H,Sk)
        out[l] = recv.mean(dim=0)[video_idx]
    return out


def prefix_text_mask(seq_len, video_idx, device):
    """Positions of the system/instruction block S -- everything before the
    first visual token. Section 4's FREE single-pass contrastive approximation:
    attention this block receives stands in for the query-invariant baseline,
    with no second forward (sinks attract it regardless of the real query, so it
    cancels the same way a genuine neutral-query pass would)."""
    first_vid = int(video_idx.min())
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[:first_vid] = True
    return mask


def analyze_clip(model, base_embeds, position_ids, attn_mask, video_idx, band):
    """One dense forward -> {raw, valnorm, contrastive_free} debiased score
    dicts (layer -> (M,) tensor) for every layer in `band`, plus {layer -> (M,d)}
    input hidden states (feeds the Section 6.4 stability gate / cosine features
    for whichever layer a given ablation row ends up anchoring at)."""
    text_model, out = _dense_forward(model, base_embeds, position_ids, attn_mask)
    S = base_embeds.shape[1]
    query_mask = torch.ones(S, dtype=torch.bool, device=base_embeds.device)
    query_mask[video_idx] = False
    prefix_mask = prefix_text_mask(S, video_idx, base_embeds.device)
    if not bool(prefix_mask.any()):
        raise RuntimeError("no text positions before the video block -- contrastive_free needs a system/prefix block")

    raw = raw_scores_by_layer(out.attentions, video_idx, query_mask, band)
    valnorm = debiased_scores_by_layer(text_model, out.hidden_states, out.attentions,
                                       video_idx, query_mask, band)
    baseline = debiased_scores_by_layer(text_model, out.hidden_states, out.attentions,
                                        video_idx, prefix_mask, band)
    hidden = {l: out.hidden_states[l][0, video_idx] for l in band}
    return {
        "raw": raw, "valnorm": valnorm,
        "contrastive_free": {l: valnorm[l] - baseline[l] for l in band},
        "hidden": hidden,
    }


def contrastive_full_scores(model, backbone, processor, video_token_id, info, path, data_type,
                            has_bound, rec, args, device, dtype, min_pixels, band, M_expected):
    """Optional 2-pass upgrade (Section 4): a full second forward through the
    SAME video with the query text swapped for a neutral prompt (`args.neutral_query`),
    s_con = s_t(query) - s_t(baseline). Caveat: `prepare_clip` builds the baseline
    prompt via the same `build_prompt` as the real question, so the options block
    (which is answer-specific, not neutral) is still present -- an approximation,
    not a fully content-free baseline; refine the prompt construction before
    treating this as the paper's exact contrastive upgrade."""
    baseline_rec = dict(rec)
    baseline_rec["question"] = args.neutral_query
    base_b, pos_b, attn_b, vidx_b, _, _, _, _, _ = prepare_clip(
        backbone, model, processor, video_token_id, info, path, data_type,
        has_bound, baseline_rec, args, device, dtype, min_pixels)
    if vidx_b.numel() != M_expected:
        raise RuntimeError(f"contrastive_full baseline pass produced {vidx_b.numel()} video "
                          f"tokens, expected {M_expected} (query pass) -- cannot align.")
    text_model, out_b = _dense_forward(model, base_b, pos_b, attn_b)
    Sb = base_b.shape[1]
    qmask_b = torch.ones(Sb, dtype=torch.bool, device=device)
    qmask_b[vidx_b] = False
    return debiased_scores_by_layer(text_model, out_b.hidden_states, out_b.attentions,
                                    vidx_b, qmask_b, band)


def positional_debias(scores, coord, method="poly", degree=2, min_per_bin=2):
    """In-pass, prompt-free POSITIONAL debias -- the alternative to contrastive
    scoring for the *positional* component specifically (Section 4's contrastive
    upgrade cancels ALL query-invariant saliency and needs a second forward /
    a baseline prompt; this removes only the part of the importance score that
    position alone explains, from the SAME query pass, no baseline prompt).

    It is complementary to the value-norm debias: value-norm removes content-free
    sinks (high A, tiny ||v||), but a genuine full-||v|| token that is merely
    well-placed -- late in the sequence (recency) or in a favoured frame -- keeps
    an inflated score that value-norm cannot see. `positional_debias` subtracts
    that smooth score-vs-position trend, leaving the content-driven residual.

      scores : (M,) value-norm debiased importance per video token.
      coord  : (M,) positional coordinate driving the bias -- sequence position
               for a recency tilt (use `method='poly'`), or frame index f_t for a
               per-frame temporal tilt (use `method='bin'`).
      method : 'poly' least-squares polynomial detrend of degree `degree` in the
               normalized coordinate (smooth recency); 'bin' subtracts the mean
               within each distinct coordinate value (discrete frames), skipping
               bins with < `min_per_bin` tokens (too few to estimate an offset).

    Residuals may go negative; that is fine and matches contrastive scoring --
    top-k still ranks correctly, and the tail-index anchor selection drops the
    non-positive part exactly as it already does for `contrastive_free`."""
    s = scores.float()
    x = coord.float()
    if method == "bin":
        out = s.clone()
        for v in torch.unique(x):
            m = x == v
            if int(m.sum()) >= min_per_bin:
                out[m] = s[m] - s[m].mean()
        return out
    if method != "poly":
        raise ValueError(f"unknown positional_debias method {method!r}")
    n = x.numel()
    if n <= degree + 1 or float(x.max() - x.min()) < 1e-8:
        return s - s.mean()                                   # degenerate: constant-position
    xn = (x - x.min()) / (x.max() - x.min())                  # -> [0,1], conditions the fit
    V = torch.stack([xn ** k for k in range(degree + 1)], dim=1)   # (M, degree+1) Vandermonde
    # Ridge-regularized normal equations (tiny, degree+1 square) -- robust on CPU
    # and CUDA alike, no lstsq-driver dependence.
    eye = torch.eye(V.shape[1], device=V.device, dtype=V.dtype)
    coef = torch.linalg.solve(V.T @ V + 1e-6 * eye, V.T @ s)
    return s - V @ coef


# --------------------------------------------------------------------------- #
# 2. Anchor-rule alternatives (Section 9, row 1): tail-argmax already exists
#    (anchor_layer_prune.select_anchor_layer); entropy / Gini / random are new.
# --------------------------------------------------------------------------- #
def entropy_tail(scores):
    """Shannon entropy of the score distribution (normalized to sum 1). LOWER
    entropy = more concentrated = more discriminative -- the opposite sense of
    the tail index, so callers pick argmin, not argmax."""
    x = scores.clamp_min(0).float()
    total = x.sum()
    if total <= 0:
        return float("nan")
    p = x / total
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def gini_tail(scores):
    """Gini coefficient of the (nonnegative) score distribution. HIGHER Gini =
    more unequal = more discriminative, same sense as the tail index."""
    x = scores.clamp_min(0).float().sort().values
    n = x.numel()
    total = x.sum()
    if n == 0 or total <= 0:
        return float("nan")
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return float(((2 * idx - n - 1) * x).sum() / (n * total))


def select_anchor_by_rule(scores_by_layer, band, rule, k_frac=0.10, estimator="moment", rng=None):
    if rule == "tail_argmax":
        return select_anchor_layer(scores_by_layer, k_frac, estimator)[0]
    if rule == "entropy":
        vals = {l: entropy_tail(scores_by_layer[l]) for l in band}
        valid = {l: v for l, v in vals.items() if v == v}
        if not valid:
            raise RuntimeError("entropy rule: no layer produced a usable estimate")
        return min(valid, key=valid.get)
    if rule == "gini":
        vals = {l: gini_tail(scores_by_layer[l]) for l in band}
        valid = {l: v for l, v in vals.items() if v == v}
        if not valid:
            raise RuntimeError("gini rule: no layer produced a usable estimate")
        return max(valid, key=valid.get)
    if rule == "random":
        i = int(torch.randint(len(band), (1,), generator=rng))
        return band[i]
    raise ValueError(f"unknown anchor rule {rule!r}")


# --------------------------------------------------------------------------- #
# 3. Estimator stability (Section 9, row 3): Hill/moment plot gamma-hat(m) per
#    band layer, and a plateau-averaged anchor rule as the robustness variant
#    Section 5's "Robustness notes" (i) suggests.
# --------------------------------------------------------------------------- #
def hill_plot(scores, m_fracs, estimator="moment"):
    fn = einmahlHaan if estimator == "moment" else hill_tail_index
    return {m: fn(scores, m) for m in m_fracs}


def hill_plot_by_layer(scores_by_layer, band, m_fracs, estimator="moment"):
    return {l: hill_plot(scores_by_layer[l], m_fracs, estimator) for l in band}


def anchor_layer_plateau(scores_by_layer, band, m_fracs, estimator="moment"):
    """Layer with the heaviest AVERAGE tail index over `m_fracs` (the Hill-plot
    plateau), plus the per-layer averages -- Section 5's mitigation (i) against a
    hard arg max over one noisy m cutoff."""
    fn = einmahlHaan if estimator == "moment" else hill_tail_index
    avg = {}
    for l in band:
        vals = [v for v in (fn(scores_by_layer[l], m) for m in m_fracs) if v == v]
        avg[l] = float(np.mean(vals)) if vals else float("nan")
    valid = {l: v for l, v in avg.items() if v == v}
    if not valid:
        raise RuntimeError("no layer produced a usable plateau estimate")
    return max(valid, key=valid.get), avg


def anchor_stability_across_m(plot_by_layer, band, m_fracs):
    """{m: argmax layer at that m}, and whether the argmax is the SAME layer for
    every m in the plateau -- Section 5's "is the arg max layer stable?" check."""
    picks = {}
    for m in m_fracs:
        vals = {l: plot_by_layer[l][m] for l in band if plot_by_layer[l][m] == plot_by_layer[l][m]}
        picks[m] = max(vals, key=vals.get) if vals else None
    distinct = {v for v in picks.values() if v is not None}
    return picks, len(distinct) <= 1


# --------------------------------------------------------------------------- #
# 4. Generalized greedy selection (Section 9, rows 4-6): redundancy metric,
#    stability gate, and arm-combination all live in ONE parameterized version
#    of anchor_layer_prune.greedy_select / Algorithm 1, so isolating any one of
#    them holds the other two at their paper defaults automatically.
# --------------------------------------------------------------------------- #
def greedy_select_variant(scores, f, r, c, gate, feats, K, redundancy="both",
                          combine="max", alpha=0.5, use_gate=True,
                          sigma_s=1.5, sigma_tau=1.0, window=2, lam=1.0):
    """Behaviourally identical to anchor_layer_prune.greedy_select at its only
    supported point (redundancy='both', combine='max', use_gate=True) -- see
    `self_test` below for the regression check. Extra knobs:
      redundancy: 'both' (Section 6 default, max(Rs,Rt)) | 'spatial' only |
        'temporal' only | 'cosine' (naive feature cosine -- Section 6.1's
        rejected default; O(KN) here, kept ONLY as the ablation foil).
      combine: 'max' (default) | 'weighted' alpha*Rs+(1-alpha)*Rt -- relevant
        only when redundancy='both'.
      use_gate: apply the Section 6.4 stability gate g_t to the temporal arm.
        False = pure geometry, the failure mode the row is designed to catch
        (deletes fast-moving foreground that happens to revisit a grid cell).
    """
    device = scores.device
    N = scores.numel()
    K = min(K, N)
    s = scores.float()
    f, r, c = f.long(), r.long(), c.long()
    gate_eff = gate if use_gate else torch.ones_like(gate)
    feats_f = feats.float() if feats is not None else None

    r_spatial = torch.zeros(N, device=device, dtype=torch.float32)
    r_temporal = torch.zeros(N, device=device, dtype=torch.float32)
    r_cosine = torch.zeros(N, device=device, dtype=torch.float32)
    sel_mask = torch.zeros(N, dtype=torch.bool, device=device)
    order = []

    for _ in range(K):
        if redundancy == "spatial":
            r_run = r_spatial
        elif redundancy == "temporal":
            r_run = r_temporal
        elif redundancy == "cosine":
            r_run = r_cosine
        elif combine == "weighted":
            r_run = alpha * r_spatial + (1.0 - alpha) * r_temporal
        else:                                              # 'both' + 'max' (paper default)
            r_run = torch.maximum(r_spatial, r_temporal)

        obj = s - lam * r_run
        obj = obj.masked_fill(sel_mask, float("-inf"))
        t = int(obj.argmax())
        order.append(t)
        sel_mask[t] = True

        if redundancy in ("both", "spatial"):
            same_frame = f == f[t]
            d2 = (r[same_frame] - r[t]).float() ** 2 + (c[same_frame] - c[t]).float() ** 2
            contrib = torch.exp(-d2 / (2.0 * sigma_s ** 2))
            r_spatial[same_frame] = torch.maximum(r_spatial[same_frame], contrib)

        if redundancy in ("both", "temporal"):
            same_cell = (r == r[t]) & (c == c[t]) & ((f - f[t]).abs() <= window)
            dtau = (f[same_cell] - f[t]).float().abs()
            contrib_t = gate_eff[same_cell] * torch.exp(-dtau ** 2 / (2.0 * sigma_tau ** 2))
            r_temporal[same_cell] = torch.maximum(r_temporal[same_cell], contrib_t)

        if redundancy == "cosine":
            cos = F.cosine_similarity(feats_f, feats_f[t:t + 1].expand_as(feats_f), dim=-1)
            r_cosine = torch.maximum(r_cosine, cos.clamp_min(0.0))

    return torch.tensor(sorted(order), device=device, dtype=torch.long)


# --------------------------------------------------------------------------- #
# 5. Per-clip variant specs: one dict per ablation row, holding everything else
#    fixed at the pipeline's own defaults. Every entry is
#    {Lstar, scores, redundancy, combine, alpha, use_gate, proj_dim, lam_fn},
#    consumed identically by `main`'s inner rho loop regardless of ablation.
# --------------------------------------------------------------------------- #
def build_variant_specs(ablation, ctx, band, args, rng):
    default_lam_fn = lambda rho: args.lambda0 * (rho ** (-args.beta))         # noqa: E731
    default_L, _ = select_anchor_layer(ctx["valnorm"], args.k_frac, args.estimator)
    base_kwargs = dict(redundancy="both", combine="max", alpha=0.5, use_gate=True,
                       proj_dim=args.proj_dim, lam_fn=default_lam_fn)
    specs = {}

    if ablation == "anchor_rule":
        for l in band:
            specs[f"fixed_L{l}"] = dict(Lstar=l, scores=ctx["valnorm"][l], **base_kwargs)
        for rule in ("tail_argmax", "entropy", "gini", "random"):
            L = select_anchor_by_rule(ctx["valnorm"], band, rule, args.k_frac, args.estimator, rng)
            specs[rule] = dict(Lstar=L, scores=ctx["valnorm"][L], **base_kwargs)

    elif ablation == "sink_debiasing":
        score_sets = {"raw": ctx["raw"], "value_norm": ctx["valnorm"],
                     "contrastive_free": ctx["contrastive_free"]}
        # positional-debias arms (free, single-pass; see `positional_debias`) and
        # the optional 2-pass contrastive upgrade, whichever `main` populated.
        for extra in ("contrastive_full", "pos_debias_seq", "pos_debias_frame"):
            if extra in ctx:
                score_sets[extra] = ctx[extra]
        for name, sbl in score_sets.items():
            L = select_anchor_layer(sbl, args.k_frac, args.estimator)[0]
            specs[name] = dict(Lstar=L, scores=sbl[L], **base_kwargs)

    elif ablation == "estimator_stability":
        specs["single_m"] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **base_kwargs)
        m_fracs = [float(x) for x in args.m_fracs.split(",")]
        L_plateau, _ = anchor_layer_plateau(ctx["valnorm"], band, m_fracs, args.estimator)
        specs["plateau_avg"] = dict(Lstar=L_plateau, scores=ctx["valnorm"][L_plateau], **base_kwargs)

    elif ablation == "redundancy_metric":
        for mode in ("both", "spatial", "temporal", "cosine"):
            kw = dict(base_kwargs); kw["redundancy"] = mode
            specs[mode] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kw)

    elif ablation == "stability_gate":
        kw = dict(base_kwargs); kw["use_gate"] = True
        specs["with_gate"] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kw)
        kw2 = dict(base_kwargs); kw2["use_gate"] = False
        specs["without_gate"] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kw2)
        for d in [int(x) for x in args.proj_dims.split(",")]:
            kwd = dict(base_kwargs); kwd["proj_dim"] = d
            specs[f"proj_dim_{d}"] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kwd)

    elif ablation == "arm_combination":
        kw = dict(base_kwargs); kw["combine"] = "max"
        specs["max"] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kw)
        for a in [float(x) for x in args.alphas.split(",")]:
            kwa = dict(base_kwargs); kwa["combine"] = "weighted"; kwa["alpha"] = a
            specs[f"weighted_alpha{a}"] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kwa)

    elif ablation == "lambda_schedule":
        for beta in [float(x) for x in args.lambda_betas.split(",")]:
            kwb = dict(base_kwargs)
            kwb["lam_fn"] = (lambda rho, b=beta: args.lambda0 * (rho ** (-b)))
            tag = "constant" if beta == 0 else f"rho_indexed_beta{beta}"
            specs[tag] = dict(Lstar=default_L, scores=ctx["valnorm"][default_L], **kwb)

    else:
        raise ValueError(f"unknown ablation {ablation!r}")

    return specs


# --------------------------------------------------------------------------- #
# 6. Main: one dense analysis pass per clip, dispatched to every requested
#    ablation row. Mirrors anchor_layer_prune_mvbench.py's loop/skip/report
#    shape, generalized from a fixed STRATEGIES tuple to per-ablation variants.
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resolve_backbone(args.backbone, args.model_name)
    if not args.model_name:
        args.model_name = default_model_id(backbone)
    if args.dtype == "auto":
        args.dtype = "bf16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[backbone] {backbone}  model={args.model_name}  dtype={args.dtype}")
    model, processor, video_token_id, info = load_backbone(backbone, args.model_name, dtype)
    tm = install_anchor_prune(model)

    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    ablations = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    rhos = [float(x) for x in args.rhos.split(",")]
    band = [int(x) for x in args.band.split(",")]
    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    correct = {ab: {} for ab in ablations}                 # ab -> variant -> task -> rho -> hits
    lstar_hist = {ab: {} for ab in ablations}               # ab -> variant -> [L*, ...]
    full_correct = {t: 0 for t in tasks}
    skipped = []
    per_sample = []
    seen = {t: 0 for t in tasks}
    n = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        task_video_dir = os.path.join(video_dir, subdir)
        if not os.path.isdir(task_video_dir):
            print(f"[skip task] {task!r}: video dir not found ({task_video_dir}) -- skipping entirely.")
            continue
        try:                                    # a task json absent under this data_root
            with open(os.path.join(json_dir, fname)) as fh:   # (e.g. EgoSchema under MVBench/)
                records = json.load(fh)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip task] {task!r}: {e} -- skipping this task entirely.")
            continue
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task, unit="clip"):
            video_name = rec.get("video")
            path = os.path.join(video_dir, subdir, video_name)
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": video_name, "reason": "missing file"})
                tqdm.write(f"skip [{task}] {video_name}: missing file ({path})")
                continue

            try:
                base, pos, attn, vpos, grid, n_frames, seq_len, letter_ids, gt_idx = prepare_clip(
                    backbone, model, processor, video_token_id, info, path, data_type,
                    has_bound, rec, args, device, dtype, min_pixels)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue
            M = vpos.numel()
            f, r, c, T, Sframe = grid

            try:
                ctx = analyze_clip(model, base, pos, attn, vpos, band)
                if "sink_debiasing" in ablations:
                    # Free positional-debias arms off the SAME dense forward: strip
                    # the smooth score-vs-position trend (recency along the sequence,
                    # and a per-frame temporal tilt) that value-norm cannot see.
                    seq_coord = vpos.float()                 # absolute sequence position -> recency
                    frame_coord = f.float()                  # frame index f_t -> temporal tilt
                    ctx["pos_debias_seq"] = {l: positional_debias(ctx["valnorm"][l], seq_coord,
                                                                 method="poly", degree=args.pos_degree)
                                             for l in band}
                    ctx["pos_debias_frame"] = {l: positional_debias(ctx["valnorm"][l], frame_coord,
                                                                   method="bin")
                                               for l in band}
                    if args.include_contrastive_full:
                        ctx["contrastive_full"] = contrastive_full_scores(
                            model, backbone, processor, video_token_id, info, path, data_type,
                            has_bound, rec, args, device, dtype, min_pixels, band, M)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue

            # full-model reference (no pruning), shared across every ablation
            set_anchor_prune(tm, None, None)
            lp_full = score_letters(model, base, pos, attn, letter_ids)
            hit_full = int(lp_full.argmax().item() == gt_idx)
            full_correct[task] += hit_full

            rec_out = {"task": task, "video": rec.get("video"), "gt": gt_idx, "M": M,
                      "n_frames": n_frames, "full": hit_full, "ablations": {}}

            if "estimator_stability" in ablations:
                m_fracs = [float(x) for x in args.m_fracs.split(",")]
                plot = hill_plot_by_layer(ctx["valnorm"], band, m_fracs, args.estimator)
                picks, stable = anchor_stability_across_m(plot, band, m_fracs)
                rec_out["hill_plot"] = {str(l): {str(m): g for m, g in plot[l].items()} for l in band}
                rec_out["hill_plot_stable_across_m"] = stable

            for ab in ablations:
                specs = build_variant_specs(ab, ctx, band, args, rng)
                if not correct[ab]:
                    correct[ab] = {v: {t: {rho: 0 for rho in rhos} for t in tasks} for v in specs}
                    lstar_hist[ab] = {v: [] for v in specs}
                rec_out["ablations"][ab] = {}

                for variant_name, spec in specs.items():
                    lstar_hist[ab][variant_name].append(spec["Lstar"])
                    gate = stability_gates(ctx["hidden"][spec["Lstar"]], f, Sframe, proj_dim=spec["proj_dim"])
                    feats = ctx["hidden"][spec["Lstar"]] if spec["redundancy"] == "cosine" else None
                    rec_out["ablations"][ab][variant_name] = {"L_star": spec["Lstar"], "correct": {}}

                    for rho in rhos:
                        K = max(1, int(round(rho * M)))
                        lam = spec["lam_fn"](rho)
                        keep_local = greedy_select_variant(
                            spec["scores"], f, r, c, gate, feats, K,
                            redundancy=spec["redundancy"], combine=spec["combine"], alpha=spec["alpha"],
                            use_gate=spec["use_gate"], sigma_s=args.sigma_s, sigma_tau=args.sigma_tau,
                            window=args.window, lam=lam)
                        keep_abs = full_sequence_keep_idx(seq_len, vpos, keep_local)
                        set_anchor_prune(tm, spec["Lstar"], keep_abs)
                        lp = score_letters(model, base, pos, attn, letter_ids)
                        hit = int(lp.argmax().item() == gt_idx)
                        correct[ab][variant_name][task][rho] += hit
                        rec_out["ablations"][ab][variant_name]["correct"][str(rho)] = hit
                set_anchor_prune(tm, None, None)

            per_sample.append(rec_out)
            seen[task] += 1
            n += 1
            del base, ctx
            torch.cuda.empty_cache()
            if n % 25 == 0:
                print(f"[{n}] full acc so far: {sum(full_correct[t] for t in tasks) / n:.3f}")

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    valid = [t for t in tasks if seen[t]]

    def per_task_mean(counts):
        return float(np.mean([counts[t] / seen[t] for t in valid]))

    def micro(counts):
        return sum(counts[t] for t in valid) / n

    full_mean, full_micro = per_task_mean(full_correct), micro(full_correct)

    out_json = {"experiment": "mvbench_anchor_layer_ablations", "backbone": backbone,
               "model_name": args.model_name, "data_root": args.data_root, "rhos": rhos,
               "band": band, "k_frac": args.k_frac, "estimator": args.estimator,
               "sigma_s": args.sigma_s, "sigma_tau": args.sigma_tau, "window": args.window,
               "lambda0": args.lambda0, "beta": args.beta, "proj_dim": args.proj_dim,
               "ablations_run": ablations, "tasks": valid, "n": n,
               "full_accuracy_mean": full_mean, "full_accuracy_micro": full_micro,
               "per_task_seen": {t: seen[t] for t in valid},
               "skipped": skipped, "results": {}}

    print(f"\n==== Anchor-layer Section-9 ablations on MVBench ({n} samples, {len(valid)} task(s)) ====")
    if skipped:
        def _category(reason):
            if reason == "missing file":
                return "missing file"
            return "decode/processing error"
        by_cat = Counter(_category(s["reason"]) for s in skipped)
        print(f"\nskipped {len(skipped)} clip(s): " + ", ".join(f"{c} ({n_})" for c, n_ in by_cat.most_common()))
    print(f"\nfull-model accuracy: mean(per-task) {full_mean:.4f}  micro {full_micro:.4f}")

    for ab in ablations:
        print(f"\n---- {ab} ----")
        out_json["results"][ab] = {}
        hdr = f"{'variant':<24}{'rho':>6}{'acc_mean':>10}{'acc_micro':>11}{'L*_mean':>9}"
        print(hdr); print("-" * len(hdr))
        for variant_name in correct[ab]:
            all_l = lstar_hist[ab][variant_name]
            l_mean = float(np.mean(all_l)) if all_l else float("nan")
            l_hist = {int(l): all_l.count(l) for l in sorted(set(all_l))}
            out_json["results"][ab][variant_name] = {
                "L_star_mean": l_mean, "L_star_histogram": l_hist, "per_rho": {}}
            for rho in rhos:
                counts = {t: correct[ab][variant_name][t][rho] for t in tasks}
                acc_mean, acc_micro = per_task_mean(counts), micro(counts)
                out_json["results"][ab][variant_name]["per_rho"][str(rho)] = {
                    "accuracy_mean": acc_mean, "accuracy_micro": acc_micro,
                    "per_task": {t: correct[ab][variant_name][t][rho] / seen[t] for t in valid}}
                print(f"{variant_name:<24}{rho:>6.2f}{acc_mean:>10.4f}{acc_micro:>11.4f}{l_mean:>9.2f}")

    with open(args.out, "w") as fh:
        json.dump(out_json, fh, indent=2)
    ps_path = args.per_sample_out or os.path.splitext(args.out)[0] + "_per_sample.json"
    with open(ps_path, "w") as fh:
        json.dump({"experiment": "mvbench_anchor_layer_ablations_per_sample", "rhos": rhos,
                  "ablations_run": ablations, "n": n, "samples": per_sample}, fh, indent=2)
    print(f"\nsaved -> {args.out}\nsaved -> {ps_path}  (per-clip hits + Hill-plot dump, for paired tests / plotting)")
    print("\nread: within each ablation row, every variant shares the SAME rho and the same")
    print("      pipeline defaults everywhere except the one design choice under test (Section 9's")
    print("      'holding everything else fixed'), so any accuracy gap is attributable to that")
    print("      choice alone. Counts are unpaired -- resolve a claimed gap with McNemar on the")
    print("      per-clip dump, not with these margins.")


# --------------------------------------------------------------------------- #
# 7. Self-tests (fast, CPU-only, no model) -- correctness of the NEW machinery
#    in this file. The base method's own correctness is
#    anchor_layer_prune.py --self_test's job; this only covers the Section 9
#    additions layered on top of it.
# --------------------------------------------------------------------------- #
def self_test():
    torch.manual_seed(0)

    # --- greedy_select_variant, at its default point, must reproduce
    # anchor_layer_prune.greedy_select exactly (same algorithm, same result). ---
    print("[1] greedy_select_variant defaults == anchor_layer_prune.greedy_select")
    N = 64
    scores = torch.rand(N)
    f = torch.randint(0, 4, (N,))
    r = torch.randint(0, 4, (N,))
    c = torch.randint(0, 4, (N,))
    gate = torch.rand(N)
    keep_a = greedy_select(scores, f, r, c, gate, K=10, sigma_s=1.5, sigma_tau=1.0, window=2, lam=1.0)
    keep_b = greedy_select_variant(scores, f, r, c, gate, feats=None, K=10, redundancy="both",
                                   combine="max", use_gate=True, sigma_s=1.5, sigma_tau=1.0,
                                   window=2, lam=1.0)
    ok1 = torch.equal(keep_a, keep_b)
    print(f"  {'PASS' if ok1 else 'FAIL'}  identical selection ({keep_a.tolist()} vs {keep_b.tolist()})")

    # --- redundancy='cosine' suppresses a feature-duplicate even across frames,
    # something spatial/temporal-only (grid-based) modes cannot see. ---
    print("\n[2] cosine redundancy catches a cross-frame feature duplicate")
    T2, S2 = 4, 4
    N2 = T2 * S2
    idx = torch.arange(N2)
    f2, s_within = idx // S2, idx % S2
    r2, c2 = s_within // 2, s_within % 2
    feats = torch.randn(N2, 8)
    feats[8] = feats[0] + 0.001 * torch.randn(8)     # frame-2 token 8 duplicates frame-0 token 0's feature
    scores2 = torch.rand(N2) * 0.1
    scores2[0] = 1.0
    scores2[8] = 0.99                                 # nearly as important, but a feature-duplicate
    gate2 = torch.zeros(N2)
    keep_cos = greedy_select_variant(scores2, f2, r2, c2, gate2, feats, K=2, redundancy="cosine", lam=5.0)
    keep_grid = greedy_select_variant(scores2, f2, r2, c2, gate2, feats, K=2, redundancy="temporal", lam=5.0)
    dup_dropped_by_cosine = 8 not in keep_cos.tolist()
    dup_kept_by_grid = 8 in keep_grid.tolist()          # different (r,c) cell -> temporal arm misses it
    print(f"  {'PASS' if dup_dropped_by_cosine else 'FAIL'}  cosine mode drops the duplicate {keep_cos.tolist()}")
    print(f"  {'PASS' if dup_kept_by_grid else 'FAIL'}  temporal-only mode misses it (different cell) {keep_grid.tolist()}")

    # --- use_gate=False removes motion preservation: a moving cell is
    # suppressed just as much as a static one once geometry alone decides. ---
    print("\n[3] use_gate=False: motion preservation is lost")
    T3, S3 = 4, 4
    N3 = T3 * S3
    idx3 = torch.arange(N3)
    f3, within3 = idx3 // S3, idx3 % S3
    r3, c3 = within3 // 2, within3 % 2
    scores3 = torch.rand(N3) * 0.05
    static_cell = (r3 == 0) & (c3 == 0)
    moving_cell = (r3 == 1) & (c3 == 1)
    scores3[static_cell] = 1.0
    scores3[moving_cell] = 1.0
    gate3 = torch.zeros(N3)
    gate3[static_cell] = 1.0
    gate3[moving_cell] = 0.0
    keep_gated = greedy_select_variant(scores3, f3, r3, c3, gate3, None, K=6, use_gate=True,
                                       sigma_s=1.0, sigma_tau=1.0, window=3, lam=5.0)
    keep_ungated = greedy_select_variant(scores3, f3, r3, c3, gate3, None, K=6, use_gate=False,
                                         sigma_s=1.0, sigma_tau=1.0, window=3, lam=5.0)
    kept_moving_gated = int(moving_cell[keep_gated].sum())
    kept_moving_ungated = int(moving_cell[keep_ungated].sum())
    ok3 = kept_moving_ungated < kept_moving_gated
    print(f"  moving-cell kept: gated={kept_moving_gated}/4  ungated={kept_moving_ungated}/4")
    print(f"  {'PASS' if ok3 else 'FAIL'}  disabling the gate hurts (or at best ties) motion preservation")

    # --- entropy/gini: a peaked distribution must read as more discriminative
    # than a near-uniform one, in the correct direction for each metric. ---
    print("\n[4] entropy/Gini pick the peaked layer over the uniform one")
    band = [2, 3, 4]
    scores_by_layer = {
        2: torch.rand(2000) * 0.01,
        3: torch.distributions.Pareto(1.0, 1.5).sample((2000,)),
        4: torch.rand(2000) * 0.01,
    }
    L_ent = select_anchor_by_rule(scores_by_layer, band, "entropy")
    L_gini = select_anchor_by_rule(scores_by_layer, band, "gini")
    print(f"  entropy picks L={L_ent}  gini picks L={L_gini}")
    ok4 = L_ent == 3 and L_gini == 3
    print(f"  {'PASS' if ok4 else 'FAIL'}  both land on the heavy-tailed layer")

    # --- random rule: must return a layer that is actually in the band. ---
    print("\n[5] random anchor rule stays within the band")
    rng = torch.Generator().manual_seed(0)
    picks = [select_anchor_by_rule(scores_by_layer, band, "random", rng=rng) for _ in range(20)]
    ok5 = all(p in band for p in picks)
    print(f"  {'PASS' if ok5 else 'FAIL'}  all {len(picks)} picks in {band}")

    # --- hill_plot / anchor_layer_plateau: shape and plateau pick sanity. ---
    print("\n[6] Hill-plot plateau picks the heavy-tailed layer too")
    m_fracs = [0.05, 0.10, 0.20]
    L_plateau, avg = anchor_layer_plateau(scores_by_layer, band, m_fracs)
    ok6 = L_plateau == 3 and set(avg) == set(band)
    print(f"  plateau averages: { {l: round(v, 3) for l, v in avg.items()} }  L*={L_plateau}")
    print(f"  {'PASS' if ok6 else 'FAIL'}  plateau picks the heavy-tailed layer")

    # --- positional_debias (poly): a smooth recency ramp must stop dominating,
    # so a mid-sequence content spike wins after detrending but not before. ---
    print("\n[7] positional_debias (poly) removes a recency ramp, keeps a content spike")
    coord = torch.arange(100).float()                 # sequence position
    s_pos = 0.02 * coord                              # pure positional recency tilt
    s_pos[20] += 1.0                                  # one content-driven spike, mid-sequence
    raw_arg = int(s_pos.argmax())                     # 99: the ramp endpoint wins (artifact)
    deb = positional_debias(s_pos, coord, method="poly", degree=2)
    ok7 = raw_arg == 99 and int(deb.argmax()) == 20
    print(f"  raw argmax={raw_arg} (positional endpoint)  debiased argmax={int(deb.argmax())} (spike)")
    print(f"  {'PASS' if ok7 else 'FAIL'}  detrending surfaces the content spike over the recency tilt")

    # --- positional_debias (bin): a per-frame offset is removed, so a spike in a
    # low-scoring frame wins over uniformly-high tokens in a favoured frame. ---
    print("\n[8] positional_debias (bin) removes a per-frame temporal tilt")
    f_bin = torch.repeat_interleave(torch.arange(5), 4).float()   # 5 frames x 4 tokens
    s_bin = f_bin.clone()                             # frame index == pure per-frame tilt
    s_bin[2] += 5.0                                   # spike in frame 0 (the lowest-scoring frame)
    raw_arg_b = int(s_bin.argmax())                   # 2 already here; check the offset is gone
    debb = positional_debias(s_bin, f_bin, method="bin")
    frame4 = f_bin == 4
    ok8 = int(debb.argmax()) == 2 and float(debb[frame4].abs().max()) < 1e-5
    print(f"  debiased argmax={int(debb.argmax())} (spike)  favoured-frame residual max={float(debb[frame4].abs().max()):.2e}")
    print(f"  {'PASS' if ok8 else 'FAIL'}  the per-frame offset cancels, only the spike survives")

    ok = (ok1 and dup_dropped_by_cosine and dup_kept_by_grid and ok3 and ok4 and ok5
          and ok6 and ok7 and ok8)
    print("\n" + ("ALL SELF-TESTS PASSED" if ok else "SOME CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Section 9 ablations for the anchor-layer method, on MVBench.")
    p.add_argument("--self_test", action="store_true", help="run fast CPU-only correctness checks and exit.")
    p.add_argument("--ablation", choices=list(ABLATIONS) + ["all"], default=None,
                   help="which Section 9 row to run (or 'all'). Required unless --self_test.")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_video"], default="auto")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, llava_video={LLAVA_VIDEO_MODEL_ID}.")
    p.add_argument("--data_root", help="Dir holding json/ and video/. Required unless --self_test.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="qwen only: fps-sampling frame cap.")
    p.add_argument("--fps", type=float, default=2.0, help="qwen only: frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true", help="qwen only: mvbench.ipynb sampler.")
    p.add_argument("--num_segments", type=int, default=16, help="qwen only: frames for --official_sampling.")
    p.add_argument("--num_frames", type=int, default=8, help="llava_video only: uniform frame count.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--rhos", default="0.10", help="keep-ratios K/N to evaluate.")
    p.add_argument("--band", default="2,3,4,5,6,7,8", help="anchor-candidate layer band B.")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--sigma_s", type=float, default=1.5)
    p.add_argument("--sigma_tau", type=float, default=1.0)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--lambda0", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5, help="default lambda(rho)=lambda0*rho^-beta for every row except lambda_schedule.")
    p.add_argument("--proj_dim", type=int, default=64, help="default stability-gate projection dim.")
    # row-specific sweeps
    p.add_argument("--m_fracs", default="0.02,0.05,0.10,0.15,0.20,0.30",
                   help="estimator_stability: top-order-statistic fractions for the Hill plot.")
    p.add_argument("--alphas", default="0.25,0.5,0.75",
                   help="arm_combination: alpha values for the weighted alpha*Rs+(1-alpha)*Rt combine.")
    p.add_argument("--proj_dims", default="16,32,64,128,256",
                   help="stability_gate: projection-dim sweep for the with_gate variant.")
    p.add_argument("--lambda_betas", default="0.0,0.5",
                   help="lambda_schedule: beta values (0.0 = constant lambda) to compare across --rhos.")
    p.add_argument("--pos_degree", type=int, default=2,
                   help="sink_debiasing: polynomial degree for the pos_debias_seq recency detrend "
                        "(free single-pass positional debias; pos_debias_frame is always per-frame bins).")
    p.add_argument("--neutral_query", default="Describe what is happening in this video.",
                   help="sink_debiasing: neutral question for the optional contrastive_full baseline pass.")
    p.add_argument("--include_contrastive_full", action="store_true",
                   help="sink_debiasing: also run the 2-pass contrastive upgrade (one extra dense "
                        "forward per clip -- off by default, the free single-pass version always runs).")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_ablation_mvbench.json")
    p.add_argument("--per_sample_out", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.ablation:
            raise SystemExit("pass --ablation <row> (or 'all'), or --self_test.")
        if not args.data_root:
            raise SystemExit("pass --data_root <dir with json/ and video/>.")
        main(args)
