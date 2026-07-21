"""
lcds_ai2d.py -- Layer-Collected Diverse Selection, inference-only, on AI2D.
================================================================================
A training-free visual-token selection method, tested as a SELECTION strategy
(no in-LLM pruning yet): compute the selection from one dense forward, then score
the multiple-choice answer through the frozen model on the kept token subset.

The method:

  dense image tokens
        |
        v   at every k-th LLM layer  (independent, NO dropping)
  [text->vision attention top-C]   record the top-C image tokens by the attention
        |                          they RECEIVE from the text queries. No cascade,
        |                          no monotone shrink -- each selected layer votes
        |                          independently on its top candidates.
        v   union across layers
  [candidate pool]                 tokens that ANY selected layer ranked highly.
        |
        v   at the final layer
  [diversity selection]            on the FINAL-layer features of the pool, pick k
        |                          diverse tokens -- either farthest-point (Max-Min)
        v                          OR a conditional DPP-MAP (CDPruner, arXiv:2506.10967)
                                   whose relevance weight is the attention salience.
  kept subset -> score answer

Strategies reported (baselines for context; the method is the star):
  * uniform         : evenly spaced tokens over the merged grid              (floor)
  * attention_topk  : select-layers-averaged raw attention, global top-K     (foil)
  * lcds            : per-layer top-attention pool + farthest-point diversity (ours)
  * lcds_dpp        : same pool + conditional DPP-MAP diversity (CDPruner)    (ours)
  * lcds_mmr        : same pool + Maximal Marginal Relevance at --mmr_lambda  (ours)
                      lam=1 is attention_topk, lam=0 is pure repulsion, so a lam sweep
                      spans both endpoints at 1/rho the cost of the DPP Gram.

AI2D is the interesting stress case for the DIVERSITY half of the method: on a science
diagram the evidence is labels and arrows scattered over mostly-blank canvas, so a raw
attention top-K can pile its whole budget onto one label while uniform wastes most of
its budget on background. That is exactly the regime where the pool+diversity step
should separate from both.

Two backbones (--backbone, auto-detected from --model_name).  The METHOD is identical
for both -- collect_candidates / farthest_point / dpp_map / lcds_select and the whole
layer-overlap diagnostic only ever consume attentions, hidden states, and image-token
positions.  Only the plumbing differs:

  * qwen  : Qwen2.5-VL-3B.  Dynamic resolution -> M (image tokens) varies per image, so
            the budget k = rho*M varies too and numbers are only INTERNALLY comparable.
            mRoPE (3,1,S) positions via get_rope_index; 2x2 spatial merge; scoring via
            oracle_check.score_answer; --max_pixels/--min_pixels honoured.
  * llava : LLaVA-1.5-7B.  Fixed 336x336 -> a flat 24x24 = 576-token grid, so k is a
            CONSTANT 29 / 58 / 144 at rho = 0.05 / 0.10 / 0.25 -- the standard axis for
            the token-pruning literature (FastV, CDPruner both report there).  Plain 1D
            RoPE (position_ids = arange); scoring via score_answer_1d (its 1D twin);
            Llama-SentencePiece letter ids resolved against the real prompt (prefix-
            space sensitive); --max_pixels/--min_pixels ignored (hard-resized).

Run:
    # Qwen2.5-VL (default)
    python lcds_ai2d.py \
        --data_root ~/Experiments/AI2D \
        --max_samples 300 --rhos 0.05,0.10,0.25 --layer_stride 4 \
        --out results_lcds_ai2d.json

    # LLaVA-1.5
    python lcds_ai2d.py --backbone llava \
        --data_root ~/Experiments/AI2D \
        --max_samples 300 --rhos 0.05,0.10,0.25 --layer_stride 4 \
        --out results_lcds_ai2d_llava.json
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (Qwen2_5_VLForConditionalGeneration,
                          LlavaForConditionalGeneration, AutoProcessor)
from qwen_vl_utils import process_vision_info

# shared, validated helpers ------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# score_answer is visual-modality-agnostic: it masks all visual positions off, turns
# the kept ones back on, slices the (absolute) position ids, and reads the first
# answer-token log-probs. Works for image tokens exactly as for video tokens. Used by
# the qwen backbone; llava uses score_answer_1d below (its 1D-RoPE twin).
from oracle_check import score_answer
# AI2D data layer / prompt (single diagram, positional options, parquet PNG bytes).
# build_ai2d_prompt emits Qwen-style message dicts; the llava backbone builds its own
# flat prompt string (build_llava_prompt) instead.
from overlap_subset_ai2d import build_ai2d_prompt, decode_image, load_items

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

STRATEGIES = ("uniform", "attention_topk", "lcds", "lcds_dpp", "lcds_mmr")


# --------------------------------------------------------------------------- #
# scoring (llava: 1D-RoPE twin of oracle_check.score_answer)
# --------------------------------------------------------------------------- #
def score_answer_1d(model, inputs_embeds, position_ids, attention_mask,
                    image_positions, keep_image, letter_token_ids):
    """First-token log-prob of each candidate letter, on the kept-token seq.

    Identical to oracle_check.score_answer except for the position slice: that one
    does position_ids[:, :, keep_mask] for Qwen's (3,1,S) mRoPE ids, this one does
    position_ids[:, keep_mask] for LLaVA's (1,S) 1D ids. The semantics are preserved
    exactly: the kept tokens keep their ORIGINAL ABSOLUTE positions (so the id stream
    is gappy, and a surviving token still knows where in the image it came from)
    rather than being renumbered contiguously."""
    kept_abs = image_positions[keep_image]
    keep_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
    keep_mask[image_positions] = False
    keep_mask[kept_abs] = True
    e = inputs_embeds[:, keep_mask, :]
    p = position_ids[:, keep_mask]
    a = attention_mask[:, keep_mask]
    # inputs_embeds + no pixel_values -> the vision tower is skipped and this runs as
    # a pure language-model forward, which is the point: the image is already merged
    # into `base` by the single dense forward in main().
    out = model(inputs_embeds=e, position_ids=p, attention_mask=a, use_cache=False)
    lp = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return torch.tensor([lp[t].item() for t in letter_token_ids])


# --------------------------------------------------------------------------- #
# positions (qwen: mirror oracle_check.build_full_positions, image variant)
# --------------------------------------------------------------------------- #
def build_image_positions(model, input_ids, image_positions, image_grid_thw, attention_mask):
    """mRoPE position ids for a single-image sequence. Mirrors build_full_positions
    but marks image tokens (mm type 1; video used 2) and passes image_grid_thw.
    Falls back to the mm_token_type_ids-free signature if the fork doesn't take it."""
    try:
        mm = torch.zeros_like(input_ids)
        mm[0, image_positions] = 1
        pos, _ = model.model.get_rope_index(
            input_ids, mm_token_type_ids=mm,
            image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    except TypeError:
        pos, _ = model.model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    return pos  # (3,1,S)


# --------------------------------------------------------------------------- #
# per-layer text->vision attention (attention each image token RECEIVES)
# --------------------------------------------------------------------------- #
def received_attention(attentions, image_positions, layers):
    """{layer -> (M,)}: attention each image token RECEIVES from text queries,
    summed over text rows, head-averaged. Computed only for `layers` (a set)."""
    S = attentions[0].shape[-1]
    text_mask = torch.ones(S, dtype=torch.bool, device=attentions[0].device)
    text_mask[image_positions] = False
    out = {}
    for L in layers:
        a = attentions[L][0].mean(0)                     # (S,S) head-mean
        out[L] = a[text_mask][:, image_positions].sum(dim=0).float()  # (M,)
    return out


# --------------------------------------------------------------------------- #
# selection primitives
# --------------------------------------------------------------------------- #
def select_uniform_image(M, k, device):
    """Evenly spaced tokens over the flattened merged grid (content-free floor)."""
    if k >= M:
        return torch.arange(M, device=device)
    step = max(1, M // k)
    idx = list(range(0, M, step))[:k]
    i = 0
    while len(idx) < k and i < M:
        if i not in idx:
            idx.append(i)
        i += 1
    return torch.tensor(sorted(idx[:k]), device=device, dtype=torch.long)


def select_topk(scores, k):
    """Global top-k indices, sorted."""
    k = min(k, scores.numel())
    return torch.topk(scores, k).indices.sort().values


def farthest_point(feats, seed_scores, k):
    """Greedy maximal-distance selection in cosine space. Seeded at the highest-
    salience point, then repeatedly add the point farthest (max-min cosine
    distance) from the chosen set. Returns LOCAL indices into `feats`."""
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    X = F.normalize(feats.float(), dim=1)
    first = int(seed_scores.argmax())
    sel = [first]
    mind = 1.0 - (X @ X[first])                           # cosine distance to seed
    mind[first] = -1.0
    for _ in range(k - 1):
        nxt = int(mind.argmax())
        sel.append(nxt)
        mind = torch.minimum(mind, 1.0 - (X @ X[nxt]))
        mind[torch.tensor(sel, device=feats.device)] = -1.0
    return torch.tensor(sorted(sel), device=feats.device)


def mmr(feats, relevance, k, lam=0.5, eps=1e-6):
    """Maximal Marginal Relevance (Carbonell & Goldstein 1998): greedily add the token
    maximizing  lam*rel_i - (1-lam)*max_{j in S} sim(i,j)  -- the most salient token that
    is least like everything already chosen.

    Interpolates the two strategies already reported here, so ONE knob sweeps the whole
    relevance/diversity axis instead of testing two isolated points on it:
        lam=1 -> pure relevance argmax   (== attention_topk, restricted to the pool)
        lam=0 -> pure Max-Min repulsion  (== farthest_point's update rule)
    A lam sweep therefore subsumes both endpoints; read it as a curve, not a third point.

    Costs O(n k d): the max-similarity vector is carried incrementally, so unlike dpp_map
    the n x n Gram is never materialized -- cheaper by a factor 1/rho (10x at rho=0.1).
    Irrelevant while the picker runs once per forward, decisive if it is ever moved onto
    a per-layer path where it would run at every routed layer.

    `relevance` is min-max normalized to [0,1] so lam trades it against a cosine
    similarity on a comparable scale. Same (feats, relevance, k) -> LOCAL indices
    contract as farthest_point / dpp_map."""
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    X = F.normalize(feats.float(), dim=1)
    r = relevance.float()
    r = (r - r.min()) / (r.max() - r.min() + eps)         # min-max -> [0,1]
    sel = [int(r.argmax())]                               # seed at the most salient token
    maxsim = X @ X[sel[0]]                                # (n,) running max sim to S
    for _ in range(k - 1):
        score = lam * r - (1.0 - lam) * maxsim
        score[torch.tensor(sel, device=feats.device)] = -float("inf")
        j = int(score.argmax())
        sel.append(j)
        maxsim = torch.maximum(maxsim, X @ X[j])          # incremental: no n x n Gram
    return torch.tensor(sorted(sel), device=feats.device)


def dpp_map(feats, relevance, k, eps=1e-6):
    """Greedy MAP inference for a CONDITIONAL DPP (Chen et al. 2018 fast greedy;
    CDPruner, arXiv:2506.10967). The kernel is L = diag(r) K diag(r) with K the
    cosine-similarity Gram of `feats` (Eq.3) and r a min-max-normalized relevance
    weight (Eq.5-7), so the greedy maximizes
        log det(L_S) = sum_{i in S} log r_i^2 + log det(K_S)          (Eq.8)
    -- i.e. jointly HIGH relevance and LOW mutual similarity. Unlike farthest_point
    (Max-Min, nearest-neighbour), DPP scores the whole Gram VOLUME -> global, more
    balanced diversity, and it USES the relevance weight in the objective (not just
    as a seed). Here r = the text->vision attention salience. Returns LOCAL indices."""
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    X = F.normalize(feats.float(), dim=1)
    K = X @ X.t()                                        # cosine Gram (n,n), PSD
    r = relevance.float()
    r = (r - r.min()) / (r.max() - r.min() + eps)        # min-max -> [0,1]   (Eq.6)
    r = r.clamp_min(eps)                                 # keep every self-quality > 0
    L = r.unsqueeze(1) * K * r.unsqueeze(0)              # conditional kernel (Eq.7)
    d2 = torch.diagonal(L).clone()                       # d_i^2 = L_ii (Schur diag)
    c = torch.zeros(n, k, device=feats.device)           # incremental Cholesky rows
    sel = [int(torch.argmax(d2))]
    for t in range(k - 1):                               # add one item per step
        j = sel[-1]
        e = (L[j] - c[:, :t] @ c[j, :t]) / torch.sqrt(d2[j].clamp_min(eps))  # (n,)
        c[:, t] = e
        d2 = d2 - e * e                                  # Schur-complement update
        d2[torch.tensor(sel, device=feats.device)] = -float("inf")
        sel.append(int(torch.argmax(d2)))
    return torch.tensor(sorted(sel), device=feats.device)


# --------------------------------------------------------------------------- #
# the method: collect per-layer top-attention candidates (no dropping), then
# diversity-select k of them in the final-layer feature space.
# --------------------------------------------------------------------------- #
def collect_candidates(received, select_layers, M, cand, device):
    """Union of the top-`cand` text->vision-attention tokens at EACH selected layer
    (no dropping / no monotone cascade -- each layer votes independently). Returns
    (pool_idx over 0..M-1, aggregated salience over ALL M). Aggregated salience is
    the per-layer received attention summed over the selected layers -- used as the
    FPS seed."""
    pool = torch.zeros(M, dtype=torch.bool, device=device)
    sal = torch.zeros(M, device=device)
    for L in select_layers:
        a = received[L]
        sal += a
        top = torch.topk(a, min(cand, M)).indices
        pool[top] = True
    return pool.nonzero(as_tuple=False).flatten().sort().values, sal / max(1, len(select_layers))


def lcds_select(received, feats, select_layers, M, k, cand_mult, device, diversity="fps",
                mmr_lambda=0.5):
    """Take every k-th layer's top-attention tokens (no drop), union into a candidate
    pool, then diversity-select k of them in the final-layer feature space -> kept
    image-token indices (0..M-1). diversity: 'fps' = farthest-point (Max-Min),
    'dpp' = conditional DPP-MAP (CDPruner) seeded/weighted by attention salience,
    'mmr' = Maximal Marginal Relevance at `mmr_lambda` (spans attention_topk at lam=1
    and pure repulsion at lam=0, at 1/rho the cost of 'dpp')."""
    if k >= M:
        return torch.arange(M, device=device)
    cand = min(M, max(k, int(round(cand_mult * k))))       # per-layer candidate count
    pool, sal = collect_candidates(received, select_layers, M, cand, device)
    if diversity == "dpp":
        local = dpp_map(feats[pool], sal[pool], k)
    elif diversity == "mmr":
        local = mmr(feats[pool], sal[pool], k, lam=mmr_lambda)
    else:
        local = farthest_point(feats[pool], sal[pool], k)
    return pool[local].sort().values


# --------------------------------------------------------------------------- #
# layer-overlap diagnostic: how much do the per-layer candidate sets agree?
# --------------------------------------------------------------------------- #
#
# collect_candidates unions each selected layer's top-`cand` attention set, so the pool
# lcds_select picks from has size in [cand, min(M, cand*n_layers)]. Which end it lands
# on decides what LCDS *is*, and the accuracy table alone cannot tell you:
#
#   pool ~ cand   -> every layer ranks the same tokens; the union is a no-op and LCDS
#                    reduces to attention top-`cand` + diversity down to k.
#   pool ~ M      -> the layers disagree enough to cover the grid; the union is a no-op
#                    the other way and LCDS reduces to diversity over ALL tokens, the
#                    layer collection contributing nothing.
#
# Neither degenerate case is fatal, but both mean the "layer-collected" framing is not
# what earns the number, so measure it rather than assume it. With cand_mult=2 each
# per-layer set is 2*rho*M tokens, so cand*n_layers exceeds M whenever
# 2*rho*n_layers > 1 -- under the stride-4 plan (7 layers) that is every rho >= 0.08,
# and the pool CAN saturate. Not a hypothetical.
#
# Nothing here is on the method path: the per-layer sets are recomputed from `recv`
# (already in hand), so the method block above is backbone-agnostic. check_layer_overlap.py
# imports layer_overlap_stats / gap_profile from here.


def _layer_topk_masks(received, select_layers, M, cand, universe):
    """[bool (M,)] per layer: its top-`cand` received-attention set within `universe`."""
    masks = []
    for L in select_layers:
        a = received[L].clone()
        a[~universe] = -float("inf")                      # excluded -> never selectable
        m = torch.zeros(M, dtype=torch.bool, device=a.device)
        m[torch.topk(a, cand).indices] = True
        masks.append(m)
    return masks


def layer_overlap_stats(received, select_layers, M, cand, sink_exclude=0):
    """Pairwise Jaccard between the per-layer top-`cand` attention sets + pool size.

    Jaccard rather than the |A n B|/k overlap coefficient used in overlap_subset_*.py:
    every set here has the same size `cand`, so J = I/(2*cand - I) is a monotone
    function of that coefficient and the two carry identical information -- Jaccard is
    just the form a reader expects for a set-similarity matrix.

    Read it ONLY against `chance`, never against 0. Two random size-c subsets of an
    M-token grid already share c^2/M tokens, so E[J] ~ c/(2M - c): with cand_mult=2
    that is 0.05 at rho=0.05 but 0.33 at rho=0.25, where each layer's candidate set is
    half the image. `norm_jaccard` = (J - chance)/(1 - chance) is the version comparable
    across rho -- and, under Qwen's dynamic resolution, across SAMPLES: M varies per
    image, so chance varies per image, and only the per-sample normalized value may be
    averaged. (chance is the ratio of expectations rather than E[J] -- the standard
    approximation, exact enough at these set sizes to read a matrix by.)

    sink_exclude > 0 drops that many globally-hottest tokens (salience summed over the
    selected layers) from the universe BEFORE ranking. It controls the confounder that
    layers can agree merely by dumping attention on a few shared positional sinks: if
    mean_jaccard survives the exclusion, the layers agree about the IMAGE.

    Returns None when the question is vacuous for this sample -- fewer than 2 selected
    layers, or a candidate set as large as the surviving universe (every set is then
    the whole universe and J == 1 by construction). Reachable under dynamic resolution
    on a small image, so the caller counts these rather than averaging in a
    meaningless 1.0.
    """
    if len(select_layers) < 2:
        return None
    M_eff = M - sink_exclude
    if M_eff < 2 or cand >= M_eff:
        return None

    device = received[select_layers[0]].device
    universe = torch.ones(M, dtype=torch.bool, device=device)
    if sink_exclude > 0:
        sal = torch.stack([received[L] for L in select_layers]).sum(0)
        universe[torch.topk(sal, sink_exclude).indices] = False

    masks = _layer_topk_masks(received, select_layers, M, cand, universe)
    n = len(masks)
    J = np.eye(n)
    off = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = float((masks[i] & masks[j]).sum().item())
            union = float((masks[i] | masks[j]).sum().item())
            J[i, j] = J[j, i] = inter / union
            off.append(J[i, j])

    pool = torch.zeros(M, dtype=torch.bool, device=device)
    for m in masks:
        pool |= m
    pool_size = int(pool.sum().item())

    mean_j = float(np.mean(off))
    chance = cand / (2 * M_eff - cand)                     # cand < M_eff => chance < 1
    return {
        "jaccard": J,
        "mean_jaccard": mean_j,
        "chance_jaccard": chance,
        "norm_jaccard": (mean_j - chance) / (1 - chance),
        "pool_size": pool_size,
        # 1.0 = every layer agreed exactly; n_layers = they were disjoint.
        "pool_inflation": pool_size / cand,
        # the headroom pool_inflation had: it saturates once cand*n_layers passes M_eff.
        "max_pool_inflation": min(M_eff, cand * n) / cand,
        # fraction of the grid the diversity step actually chooses from; -> 1.0 means
        # LCDS has degenerated into diversity-over-everything.
        "pool_fraction": pool_size / M_eff,
        "cand": cand,
        "M_eff": M_eff,
    }


def gap_profile(J, select_layers):
    """{layer distance -> mean Jaccard} over the pairs at that distance. Separates
    'adjacent layers are redundant' from 'all layers agree': a profile that decays with
    distance says the stride is sampling correlated neighbours and a sparser --layers
    plan would buy a more diverse pool for the same compute. A flat one says it would not."""
    by_gap = defaultdict(list)
    for i in range(len(select_layers)):
        for j in range(i + 1, len(select_layers)):
            by_gap[select_layers[j] - select_layers[i]].append(J[i, j])
    return {g: float(np.mean(v)) for g, v in sorted(by_gap.items())}


# --------------------------------------------------------------------------- #
# spatial dispersion (single merged grid): 0 = clustered, 1 = spread
# --------------------------------------------------------------------------- #
def grid_dims(M, image_grid_thw, merge):
    """Merged-grid (Hm, Wm) from the patch grid; fall back to a square if it
    doesn't factor cleanly. (qwen: dynamic resolution; llava uses a fixed 24x24.)"""
    try:
        _, h, w = [int(x) for x in image_grid_thw[0].tolist()]
        Hm, Wm = h // merge, w // merge
        if Hm * Wm == M:
            return Hm, Wm
    except Exception:
        pass
    Wm = int(round(M ** 0.5))
    return max(1, M // max(1, Wm)), max(1, Wm)


def mean_dispersion(keep_idx, Hm, Wm):
    idx = keep_idx.detach().cpu().numpy()
    if idx.size < 2:
        return float("nan")
    ys, xs = idx // Wm, idx % Wm
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    d = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
    iu = np.triu_indices(coords.shape[0], k=1)
    max_d = max(1, max(Hm - 1, Wm - 1))
    return float(d[iu].mean() / max_d) if iu[0].size else float("nan")


# --------------------------------------------------------------------------- #
# LLaVA prompt + letter ids
# --------------------------------------------------------------------------- #
def build_llava_prompt(processor, text):
    """LLaVA-1.5's "USER: <image>\\n{text} ASSISTANT:", ending exactly where the
    model's next token is the answer letter. Uses the processor's shipped chat
    template when there is one, and falls back to the literal v1.5 format for
    processor configs that predate chat_template."""
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    try:
        return processor.apply_chat_template(conv, add_generation_prompt=True)
    except (AttributeError, ValueError, TypeError):
        return f"USER: <image>\n{text} ASSISTANT:"


def resolve_letter_ids(tokenizer, letters, prompt):
    """The id of the token the model actually emits FIRST for each option letter.

    Not the same as encode(letter): LLaVA uses Llama SentencePiece, which is prefix-
    space sensitive, so bare encode("A") yields "A" or "_A" depending on tokenizer
    version and the legacy flag -- and only one of those is what follows "ASSISTANT:".
    Picking the wrong one silently scores the wrong token and still produces a
    plausible-looking accuracy table, so resolve it against the REAL prompt instead:
    tokenize prompt vs prompt + " " + letter and take the first id that differs."""
    base = tokenizer.encode(prompt, add_special_tokens=False)
    ids = []
    for L in letters:
        full = tokenizer.encode(f"{prompt} {L}", add_special_tokens=False)
        if full[:len(base)] != base:
            raise RuntimeError(
                f"tokenizer re-segmented the prompt when appending '{L}'; the letter "
                f"id cannot be read off by diffing. Resolve the answer-token ids by "
                f"hand for this tokenizer before trusting any accuracy from this run.")
        ids.append(full[len(base)])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"option letters {letters} do not map to distinct tokens: {ids}")
    print(f"[letters] {list(zip(letters, ids, tokenizer.convert_ids_to_tokens(ids)))}")
    return ids


# --------------------------------------------------------------------------- #
# backbone plumbing -- the ONLY backbone-specific code besides the helpers above.
# Everything from received_attention down through the method + diagnostic is shared.
# --------------------------------------------------------------------------- #
def resolve_backbone(args):
    """'auto' -> infer from the model name ('llava' substring => llava, else qwen).
    With no --model_name given, 'auto' falls back to qwen (the default backbone)."""
    if args.backbone != "auto":
        return args.backbone
    return "llava" if "llava" in (args.model_name or "").lower() else "qwen"


def default_model_id(backbone):
    return LLAVA_MODEL_ID if backbone == "llava" else QWEN_MODEL_ID


def load_backbone(backbone, model_name, dtype):
    """Load model + processor + image-token id + a small `info` dict of backbone
    constants (qwen: spatial merge; llava: fixed grid side / token count)."""
    cls = LlavaForConditionalGeneration if backbone == "llava" else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(model_name, torch_dtype=dtype, device_map="auto",
                                attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_name)

    if backbone == "llava":
        # LLaVA names the placeholder image_token_index; newer transformers also expose
        # image_token_id. Take whichever exists rather than betting on the version.
        image_token_id = getattr(model.config, "image_token_index",
                                 getattr(model.config, "image_token_id", None))
        if image_token_id is None:
            raise RuntimeError("could not find the image token id on the LLaVA config "
                               "(looked for image_token_index / image_token_id).")
        # Fixed grid: 336/14 = 24 -> 24x24 = 576 tokens for every image, always.
        vcfg = model.config.vision_config
        side = vcfg.image_size // vcfg.patch_size
        expected_M = side * side
        if getattr(model.config, "vision_feature_select_strategy", "default") == "full":
            # "full" keeps CLS -> 577 tokens, which is not a square grid and would break
            # the dispersion metric's row/col decode. LLaVA-1.5 ships "default".
            raise RuntimeError("vision_feature_select_strategy='full' keeps the CLS token "
                               f"({expected_M + 1} tokens), which is not a {side}x{side} grid. "
                               "This script assumes the 'default' strategy.")
        print(f"[plan] LLaVA grid {side}x{side} -> {expected_M} image tokens per image (fixed)")
        info = {"side": side, "expected_M": expected_M}
    else:
        image_token_id = model.config.image_token_id
        info = {"merge": getattr(model.config.vision_config, "spatial_merge_size", 2)}
    return model, processor, image_token_id, info


def prepare_sample(backbone, model, processor, image_token_id, info, it, args,
                   device, dtype, letter_cache):
    """Build one sample's model inputs and everything the shared loop needs.

    Returns (base_forward_kwargs, pos, attn, ipos, M, letter_ids, Hm, Wm) or None to
    SKIP the sample (qwen: too few image tokens). base_forward_kwargs is fed straight
    to model(**kwargs, output_attentions=True, output_hidden_states=True); pos/attn are
    also handed to the scorer. The M mismatch on llava is a hard RuntimeError (a config
    problem to fix, not a sample to skip)."""
    image = decode_image(it["image_cell"])
    if backbone == "llava":
        prompt = build_llava_prompt(processor, it["text"])
        key = tuple(it["letters"])
        if key not in letter_cache:
            letter_cache[key] = resolve_letter_ids(processor.tokenizer, it["letters"], prompt)
        letter_ids = letter_cache[key]

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        # CLIP's patch conv is in `dtype`; the processor hands back fp32 pixels.
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        ipos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
        M = ipos.numel()
        if M != info["expected_M"]:
            raise RuntimeError(
                f"got {M} image tokens in input_ids, expected {info['expected_M']}. If M == 1, "
                f"the processor did not expand <image> into per-patch tokens -- that needs "
                f"`patch_size` in the processor config (transformers >= 4.44 with an "
                f"llava-hf checkpoint). Every index below is a per-patch index, so this "
                f"has to be fixed rather than skipped.")
        # plain 1D RoPE: no mRoPE, no get_rope_index, no image_grid_thw.
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        fwd = {"input_ids": input_ids, "attention_mask": attn, "position_ids": pos,
               "pixel_values": pixel_values}
        Hm = Wm = info["side"]
        return fwd, pos, attn, ipos, M, letter_ids, Hm, Wm

    # ---- qwen ----
    letter_ids = [processor.tokenizer.encode(L, add_special_tokens=False)[0]
                  for L in it["letters"]]
    messages = build_ai2d_prompt(image, it["text"], args.max_pixels, args.min_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    ipos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if ipos.numel() < args.min_tokens:                     # too few image tokens -> skip
        return None
    M = ipos.numel()
    pos = build_image_positions(model, input_ids, ipos, image_grid_thw, attn)
    fwd = {"input_ids": input_ids, "attention_mask": attn, "position_ids": pos,
           "pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
    Hm, Wm = grid_dims(M, image_grid_thw, info["merge"])
    return fwd, pos, attn, ipos, M, letter_ids, Hm, Wm


def score(backbone, model, base, pos, attn, ipos, keep, letter_ids):
    """Answer scoring dispatched by backbone: mRoPE (3,1,S) slice for qwen, 1D (1,S)
    slice for llava. Same semantics -- kept tokens keep their absolute positions."""
    if backbone == "llava":
        return score_answer_1d(model, base, pos, attn, ipos, keep, letter_ids)
    return score_answer(model, base, pos, attn, ipos, keep, letter_ids)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resolve_backbone(args)
    # --dtype auto: LLaVA-1.5 ships fp16, Qwen2.5-VL bf16.
    dtype_name = args.dtype if args.dtype != "auto" else ("fp16" if backbone == "llava" else "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model_name = args.model_name or default_model_id(backbone)
    out_path = args.out or ("results_lcds_ai2d_llava.json" if backbone == "llava"
                            else "results_lcds_ai2d.json")
    print(f"[backbone] {backbone}  model={model_name}  dtype={dtype_name}")
    model, processor, image_token_id, info = load_backbone(backbone, model_name, dtype)

    rhos = [float(x) for x in args.rhos.split(",")]
    items = load_items(args.data_root, args.split, args.max_samples)

    # layer plan resolved lazily once we know n_layers; llava letter ids cached per option set.
    select_layers = feature_hs_idx = None
    letter_cache = {}

    correct = {s: {r: 0 for r in rhos} for s in STRATEGIES}
    disp = {s: {r: [] for r in rhos} for s in STRATEGIES}
    kacc = {r: [] for r in rhos}     # per-sample budget k = max(1, round(rho*M)), for mean k
    full_correct = 0
    n = 0

    # layer-overlap accumulators: ov[rho][sink_exclude] -> running J matrix + pool stats.
    # 0 always runs; a second variant only if --sink_exclude asks for one (it is cheap
    # enough to ride along, and having both in ONE run makes the sink control a column
    # comparison rather than a second pass over the model).
    sink_variants = [0] + ([args.sink_exclude] if args.sink_exclude > 0 else [])
    ov = {r: {e: None for e in sink_variants} for r in rhos}
    ov_skipped = {r: {e: 0 for e in sink_variants} for r in rhos}

    from tqdm import tqdm
    for it in tqdm(items, desc="eval"):
        prepared = prepare_sample(backbone, model, processor, image_token_id, info,
                                  it, args, device, dtype, letter_cache)
        if prepared is None:                               # qwen: too few image tokens
            continue
        fwd, pos, attn, ipos, M, letter_ids, Hm, Wm = prepared

        # single dense forward: attentions (all layers) + merged embeds + features.
        with torch.no_grad():
            out = model(**fwd, use_cache=False, output_attentions=True, output_hidden_states=True)
        attentions = out.attentions
        base = out.hidden_states[0].detach()              # merged inputs_embeds (1,S,d)

        if select_layers is None:                         # resolve the layer plan once
            L = len(attentions)
            if args.layers:                               # explicit override
                select_layers = [int(v) for v in args.layers.split(",")]
            else:                                         # every k-th LLM layer
                select_layers = list(range(args.layer_stride, L, args.layer_stride))
            select_layers = sorted({min(x, L - 1) for x in select_layers})
            if args.min_select_layer > 0:                  # drop the early attention-sink /
                kept = [x for x in select_layers          # positional-bias band from the vote
                        if x >= args.min_select_layer]
                select_layers = kept or [L - 1]           # never empty -> keep the final layer
            if not select_layers:
                select_layers = [L - 1]
            # diversity features come from the FINAL layer by default (hidden_states[-1]);
            # --feature_layer l overrides -> hidden_states[l+1] (layer l output).
            feature_hs_idx = (len(out.hidden_states) - 1 if args.feature_layer < 0
                              else min(args.feature_layer + 1, len(out.hidden_states) - 1))
            print(f"[plan] n_layers={L} layer_stride={args.layer_stride} "
                  f"min_select_layer={args.min_select_layer} "
                  f"select_layers={select_layers} feature_hs_idx={feature_hs_idx}")

        feats = out.hidden_states[feature_hs_idx][0].detach()        # (S,d)
        feats_img = feats[ipos]                                      # (M,d)
        # attention_topk foil: raw received attention averaged over the SAME select
        # layers, then global top-K (single pool, no per-layer union, no diversity).
        recv = received_attention(attentions, ipos, set(select_layers))
        band = torch.stack([recv[L] for L in select_layers]).mean(0)
        del out, attentions

        # full-model reference (all image tokens kept)
        keep_all = torch.arange(M, device=device)
        lp_full = score(backbone, model, base, pos, attn, ipos, keep_all, letter_ids)
        full_correct += int(lp_full.argmax().item() == it["gt_idx"])

        for rho in rhos:
            k = max(1, int(round(rho * M)))
            kacc[rho].append(k)

            # cand mirrors lcds_select's line rather than being returned from it, to keep
            # the method block backbone-agnostic. Under qwen's dynamic resolution M varies
            # per image, so cand -- and the chance level it implies -- is per-sample; only
            # the normalized value is averaged at the end. Under llava M is fixed at 576.
            cand = min(M, max(k, int(round(args.cand_mult * k))))
            for ex in sink_variants:
                st = layer_overlap_stats(recv, select_layers, M, cand, ex)
                if st is None:
                    ov_skipped[rho][ex] += 1
                    continue
                if ov[rho][ex] is None:
                    ov[rho][ex] = {"J_sum": np.zeros_like(st["jaccard"]), "n": 0,
                                   "mean_j": [], "norm_j": [], "chance": [], "pool_infl": [],
                                   "max_infl": [], "pool_frac": [], "cand": [], "M_eff": []}
                a = ov[rho][ex]
                a["J_sum"] += st["jaccard"]
                a["n"] += 1
                for key, sk in (("mean_j", "mean_jaccard"), ("norm_j", "norm_jaccard"),
                                ("chance", "chance_jaccard"), ("pool_infl", "pool_inflation"),
                                ("max_infl", "max_pool_inflation"), ("pool_frac", "pool_fraction"),
                                ("cand", "cand"), ("M_eff", "M_eff")):
                    a[key].append(st[sk])

            for s in STRATEGIES:
                if s == "uniform":
                    keep = select_uniform_image(M, k, device)
                elif s == "attention_topk":
                    keep = select_topk(band, k)
                else:
                    div = {"lcds_dpp": "dpp", "lcds_mmr": "mmr"}.get(s, "fps")
                    keep = lcds_select(recv, feats_img, select_layers, M, k,
                                       args.cand_mult, device, div, args.mmr_lambda)
                lp = score(backbone, model, base, pos, attn, ipos, keep, letter_ids)
                correct[s][rho] += int(lp.argmax().item() == it["gt_idx"])
                d = mean_dispersion(keep, Hm, Wm)
                if d == d:
                    disp[s][rho].append(d)
        n += 1
        del base, feats, feats_img, recv
        if n % 25 == 0:
            r = rhos[-1]
            msg = " | ".join(f"{s[:8]} {correct[s][r]/n:.3f}" for s in STRATEGIES)
            print(f"[{n}] rho={r}: {msg}  (full {full_correct/n:.3f})")

    if n == 0:
        print("no usable samples."); return

    mean_k = {r: int(round(float(np.mean(kacc[r])))) if kacc[r] else 0 for r in rhos}
    out_json = {"n": n, "backbone": backbone, "model": model_name,
                "image_tokens": (info["expected_M"] if backbone == "llava" else "dynamic"),
                "full_accuracy": full_correct / n, "rhos": rhos,
                "budgets": {str(r): mean_k[r] for r in rhos},
                "layer_stride": args.layer_stride, "min_select_layer": args.min_select_layer,
                "select_layers": select_layers, "sink_exclude": args.sink_exclude,
                "feature_hs_idx": feature_hs_idx, "cand_mult": args.cand_mult, "table": {}}
    tok_note = f"   (all {info['expected_M']} image tokens)" if backbone == "llava" else ""
    print(f"\n==== LCDS on AI2D / {backbone}:{model_name.split('/')[-1]} ({n} samples) ====")
    print(f"full-model accuracy: {full_correct/n:.4f}{tok_note}\n")
    hdr = f"{'strategy':<16}{'rho':>6}{'k':>6}{'acc':>9}{'dispersion':>12}"
    print(hdr); print("-" * len(hdr))
    for s in STRATEGIES:
        out_json["table"][s] = {}
        for rho in rhos:
            acc = correct[s][rho] / n
            dsp = float(np.mean(disp[s][rho])) if disp[s][rho] else float("nan")
            out_json["table"][s][str(rho)] = {"accuracy": acc, "dispersion": dsp, "k": mean_k[rho]}
            print(f"{s:<16}{rho:>6.2f}{mean_k[rho]:>6d}{acc:>9.4f}{dsp:>12.3f}")

    # ---- layer-overlap diagnostic ------------------------------------------------
    # M is fixed (576) under llava, per-sample under qwen; the M / cand / chance columns
    # are per-sample means either way (constant for llava).
    out_json["layer_overlap"] = {}
    print(f"\n==== layer overlap: do the per-layer candidate sets agree? ====")
    print(f"select_layers = {select_layers}   (sink = # hottest tokens excluded; "
          f"M, cand, chance are per-sample means)\n")
    hdr = (f"{'rho':>6}{'sink':>6}{'M':>7}{'cand':>7}{'meanJ':>9}{'chance':>9}{'normJ':>9}"
           f"{'pool/cand':>11}{'ceil':>7}{'pool/M':>9}{'n':>6}")
    print(hdr); print("-" * len(hdr))
    for rho in rhos:
        out_json["layer_overlap"][str(rho)] = {}
        for ex in sink_variants:
            a = ov[rho][ex]
            if a is None or a["n"] == 0:
                print(f"{rho:>6.2f}{ex:>6d}   -- vacuous for all {ov_skipped[rho][ex]} sample(s)")
                continue
            J = a["J_sum"] / a["n"]
            rec = {"n": a["n"], "skipped": ov_skipped[rho][ex],
                   "mean_M_eff": float(np.mean(a["M_eff"])),
                   "mean_cand": float(np.mean(a["cand"])),
                   "mean_jaccard": float(np.mean(a["mean_j"])),
                   "chance_jaccard": float(np.mean(a["chance"])),
                   "norm_jaccard": float(np.mean(a["norm_j"])),
                   "pool_inflation": float(np.mean(a["pool_infl"])),
                   "max_pool_inflation": float(np.mean(a["max_infl"])),
                   "pool_fraction": float(np.mean(a["pool_frac"])),
                   "jaccard_matrix": J.tolist(),
                   "gap_profile": {str(g): v for g, v in gap_profile(J, select_layers).items()}}
            out_json["layer_overlap"][str(rho)][str(ex)] = rec
            print(f"{rho:>6.2f}{ex:>6d}{rec['mean_M_eff']:>7.0f}{rec['mean_cand']:>7.0f}"
                  f"{rec['mean_jaccard']:>9.3f}{rec['chance_jaccard']:>9.3f}"
                  f"{rec['norm_jaccard']:>9.3f}{rec['pool_inflation']:>11.2f}"
                  f"{rec['max_pool_inflation']:>7.2f}{rec['pool_fraction']:>9.3f}{a['n']:>6d}")
        if any(ov_skipped[rho][e] for e in sink_variants):
            print(f"       (rho={rho}: skipped "
                  f"{ {e: ov_skipped[rho][e] for e in sink_variants} } vacuous sample(s))")

    # The matrix and its distance profile only at sink=0 and the smallest rho, where the
    # pool has the most headroom and the layers can actually disagree.
    r0 = min(rhos)
    if ov[r0][0] is not None and ov[r0][0]["n"]:
        J = ov[r0][0]["J_sum"] / ov[r0][0]["n"]
        print(f"\nmean pairwise Jaccard, rho={r0} sink=0 (chance "
              f"{np.mean(ov[r0][0]['chance']):.3f}):")
        print("       " + "".join(f"{L:>7d}" for L in select_layers))
        for i, L in enumerate(select_layers):
            print(f"{L:>7d}" + "".join(f"{J[i, j]:>7.3f}" for j in range(len(select_layers))))
        prof = gap_profile(J, select_layers)
        print("\nby layer distance:  " + "  ".join(f"{g}:{v:.3f}" for g, v in prof.items()))

    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nsaved -> {out_path}")
    print("\nread: lcds > uniform         => the method beats the content-free floor.")
    print("      lcds > attention_topk  => the per-layer top-attention pool + final-layer")
    print("                                diversity beats a single-pool raw attention top-K.")
    print("      lcds_dpp vs lcds       => does conditional DPP-MAP (volume/global diversity,")
    print("                                relevance-weighted) beat farthest-point (Max-Min)?")
    print("      lcds_mmr (sweep lam)   => WHERE on the relevance<->diversity axis the optimum")
    print("                                sits. lam=1 reproduces attention_topk, lam=0 is pure")
    print("                                repulsion, so one sweep subsumes both endpoints -- at")
    print("                                1/rho the cost of dpp. Read it as a curve, not a point.")
    if backbone == "llava":
        print("      k is fixed here (576-token grid), so these sit on the same axis as the")
        print("      published LLaVA-1.5 pruning numbers.")
    print("\n      layer overlap -- read normJ (chance-corrected) and pool/M, never meanJ:")
    print("      normJ -> 1, pool/cand -> 1  => the layers rank the SAME tokens; the union is")
    print("                                     a no-op and lcds == attention top-cand +")
    print("                                     diversity. The layer axis earns nothing.")
    print("      pool/M -> 1                 => the union covers the grid; lcds == diversity")
    print("                                     over ALL tokens. The layer axis earns nothing")
    print("                                     the other way -- check this first at rho=0.25.")
    print("      pool/cand near ceil         => the pool is saturating against min(M, cand*L),")
    print("                                     so pool/cand is capped by the budget, not by")
    print("                                     how much the layers agree. Read normJ instead.")
    print("      sink>0 collapses normJ      => the layers only agreed on shared attention")
    print("                                     sinks, not on image content.")
    print("      gap profile decays          => the stride samples correlated neighbours; try")
    print("                                     a sparser --layers for a wider pool.")


def parse_args():
    p = argparse.ArgumentParser(description="Layer-Collected Diverse Selection on AI2D (inference-only).")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava"], default="auto",
                   help="VLM backbone. 'auto' infers from --model_name ('llava' substring "
                        "=> LLaVA-1.5, else Qwen2.5-VL). qwen: dynamic resolution + mRoPE; "
                        "llava: fixed 336x336 -> 576-token grid, 1D RoPE.")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, llava={LLAVA_MODEL_ID}.")
    p.add_argument("--data_root", required=True, help="AI2D dir with data/<split>-*.parquet.")
    p.add_argument("--split", default="test", choices=["test"],
                   help="AI2D ships answers with `test`; there is no other split to use.")
    p.add_argument("--rhos", default="0.05,0.10,0.25")
    p.add_argument("--layer_stride", type=int, default=4,
                   help="take the top text->vision attention at every k-th LLM layer (k = this).")
    p.add_argument("--layers", default="",
                   help="explicit select-layer list (overrides --layer_stride), e.g. 4,8,12,16.")
    p.add_argument("--min_select_layer", type=int, default=0,
                   help="drop select-layers below this index from the candidate vote, skipping "
                        "the early attention-sink / positional-bias band (0 = keep all, default). "
                        "Applies after --layer_stride/--layers; if it empties the set, the final "
                        "layer is kept.")
    p.add_argument("--feature_layer", type=int, default=-1,
                   help="decoder layer whose hidden states drive the final maximal-distance "
                        "selection (default = final layer). Values >= depth clamp to the final layer.")
    p.add_argument("--cand_mult", type=float, default=2.0,
                   help="per-layer candidate count = cand_mult*k (union across layers is the pool "
                        "the final diversity step selects k from). cand_mult=1 => literal top-k/layer.")
    p.add_argument("--mmr_lambda", type=float, default=0.5,
                   help="MMR relevance/diversity trade-off used by the lcds_mmr strategy: 1.0 = "
                        "pure attention salience (reproduces attention_topk on the pool), 0.0 = "
                        "pure repulsion. Sweep it -- lcds_mmr is a curve, not a point.")
    p.add_argument("--sink_exclude", type=int, default=0,
                   help="also report the layer-overlap diagnostic with the N globally-hottest "
                        "image tokens excluded, to test whether the per-layer sets agree only "
                        "via shared attention sinks. Rides along in the same run; 0 = off.")
    p.add_argument("--min_tokens", type=int, default=16,
                   help="qwen only: skip images with fewer than this many image tokens. "
                        "(llava is a fixed 576-token grid and hard-errors on a mismatch instead.)")
    p.add_argument("--max_pixels", type=int, default=None, help="qwen only (llava hard-resizes).")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="qwen only: lower bound on image area, e.g. 200704 (=448^2 -> >=256 merged "
                        "tokens). llava hard-resizes to 336x336, so this is ignored there.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 for qwen, fp16 for llava (its shipped weights).")
    p.add_argument("--out", default=None,
                   help="output JSON. Default per backbone: results_lcds_ai2d[_llava].json.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
