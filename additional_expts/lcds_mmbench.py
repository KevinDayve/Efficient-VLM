"""
lcds_mmbench.py -- Layer-Collected Diverse Selection, inference-only, on MMBench.
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

Run:
    python lcds_mmbench.py \
        --data_root ~/datasets/MMBench --split dev \
        --max_samples 300 --rhos 0.05,0.10,0.25 --layer_stride 4 \
        --out results_lcds_mmbench.json
"""

import os
import sys
import glob
import io
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# shared, validated helpers ------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# score_answer is visual-modality-agnostic: it masks all visual positions off, turns
# the kept ones back on, slices the (absolute) position ids, and reads the first
# answer-token log-probs. Works for image tokens exactly as for video tokens.
from oracle_check import score_answer
# MMBench data layer / prompt (single image, l2-category grouping, parquet PNG bytes).
from overlap_subset_mmbench import build_mmbench_prompt, build_prompt_text

STRATEGIES = ("uniform", "attention_topk", "lcds", "lcds_dpp", "lcds_mmr")


# --------------------------------------------------------------------------- #
# positions (mirror oracle_check.build_full_positions, image variant)
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
# spatial dispersion (single merged grid): 0 = clustered, 1 = spread
# --------------------------------------------------------------------------- #
def grid_dims(M, image_grid_thw, merge):
    """Merged-grid (Hm, Wm) from the patch grid; fall back to a square if it
    doesn't factor cleanly."""
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
# data
# --------------------------------------------------------------------------- #
def load_items(data_root, split, l2_categories, max_samples):
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    data_dir = os.path.join(os.path.expanduser(data_root), "data")
    matches = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not matches:
        raise FileNotFoundError(f"no {split}-*.parquet under {data_dir}")
    df = pd.read_parquet(matches[0])
    if l2_categories != ["all"]:
        df = df[df["l2-category"].isin(set(l2_categories))]
    if max_samples:
        df = df.iloc[:max_samples]

    def _decode(cell):
        b = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(b)).convert("RGB")

    items = []
    for _, rec in tqdm(df.iterrows(), total=len(df), desc=f"MMBench/{split}"):
        text, letters, gt_idx = build_prompt_text(rec)
        if len(letters) < 2 or gt_idx < 0:
            continue
        items.append({"task": rec.get("l2-category", "unknown"), "image": _decode(rec["image"]),
                      "text": text, "letters": letters, "gt_idx": gt_idx})
    return items


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    image_token_id = model.config.image_token_id
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)

    rhos = [float(x) for x in args.rhos.split(",")]
    items = load_items(args.data_root, args.split, args.l2_categories, args.max_samples)

    # layer plan resolved lazily once we know n_layers.
    select_layers = feature_hs_idx = None

    correct = {s: {r: 0 for r in rhos} for s in STRATEGIES}
    disp = {s: {r: [] for r in rhos} for s in STRATEGIES}
    full_correct = 0
    n = 0

    from tqdm import tqdm
    for it in tqdm(items, desc="eval"):
        letter_ids = [processor.tokenizer.encode(L, add_special_tokens=False)[0] for L in it["letters"]]
        messages = build_mmbench_prompt(it["image"], it["text"], args.max_pixels, args.min_pixels)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        ipos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
        if ipos.numel() < 16:
            continue
        M = ipos.numel()
        pos = build_image_positions(model, input_ids, ipos, image_grid_thw, attn)

        # single dense forward: attentions (all layers) + merged embeds + features.
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                        use_cache=False, output_attentions=True, output_hidden_states=True)
        attentions = out.attentions
        base = out.hidden_states[0].detach()              # merged inputs_embeds (1,S,d)

        if select_layers is None:                         # resolve the layer plan once
            L = len(attentions)
            if args.layers:                               # explicit override
                select_layers = [int(v) for v in args.layers.split(",")]
            else:                                         # every k-th LLM layer
                select_layers = list(range(args.layer_stride, L, args.layer_stride))
            select_layers = sorted({min(x, L - 1) for x in select_layers})
            if not select_layers:
                select_layers = [L - 1]
            # diversity features come from the FINAL layer by default (hidden_states[-1]);
            # --feature_layer l overrides -> hidden_states[l+1] (layer l output).
            feature_hs_idx = (len(out.hidden_states) - 1 if args.feature_layer < 0
                              else min(args.feature_layer + 1, len(out.hidden_states) - 1))
            print(f"[plan] n_layers={L} layer_stride={args.layer_stride} "
                  f"select_layers={select_layers} feature_hs_idx={feature_hs_idx}")

        feats = out.hidden_states[feature_hs_idx][0].detach()        # (S,d)
        feats_img = feats[ipos]                                      # (M,d)
        # attention_topk foil: raw received attention averaged over the SAME select
        # layers, then global top-K (single pool, no per-layer union, no diversity).
        recv = received_attention(attentions, ipos, set(select_layers))
        band = torch.stack([recv[L] for L in select_layers]).mean(0)
        del out, attentions

        Hm, Wm = grid_dims(M, image_grid_thw, merge)

        # full-model reference (all image tokens kept)
        keep_all = torch.arange(M, device=device)
        lp_full = score_answer(model, base, pos, attn, ipos, keep_all, letter_ids)
        full_correct += int(lp_full.argmax().item() == it["gt_idx"])

        for rho in rhos:
            k = max(1, int(round(rho * M)))
            for s in STRATEGIES:
                if s == "uniform":
                    keep = select_uniform_image(M, k, device)
                elif s == "attention_topk":
                    keep = select_topk(band, k)
                else:
                    div = {"lcds_dpp": "dpp", "lcds_mmr": "mmr"}.get(s, "fps")
                    keep = lcds_select(recv, feats_img, select_layers, M, k,
                                       args.cand_mult, device, div, args.mmr_lambda)
                lp = score_answer(model, base, pos, attn, ipos, keep, letter_ids)
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

    out_json = {"n": n, "full_accuracy": full_correct / n, "rhos": rhos,
                "layer_stride": args.layer_stride, "select_layers": select_layers,
                "feature_hs_idx": feature_hs_idx, "cand_mult": args.cand_mult, "table": {}}
    print(f"\n==== LCDS on MMBench ({n} samples) ====")
    print(f"full-model accuracy: {full_correct/n:.4f}\n")
    hdr = f"{'strategy':<16}{'rho':>6}{'acc':>9}{'dispersion':>12}"
    print(hdr); print("-" * len(hdr))
    for s in STRATEGIES:
        out_json["table"][s] = {}
        for rho in rhos:
            acc = correct[s][rho] / n
            dsp = float(np.mean(disp[s][rho])) if disp[s][rho] else float("nan")
            out_json["table"][s][str(rho)] = {"accuracy": acc, "dispersion": dsp}
            print(f"{s:<16}{rho:>6.2f}{acc:>9.4f}{dsp:>12.3f}")
    with open(args.out, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nread: lcds > uniform         => the method beats the content-free floor.")
    print("      lcds > attention_topk  => the per-layer top-attention pool + final-layer")
    print("                                diversity beats a single-pool raw attention top-K.")
    print("      lcds_dpp vs lcds       => does conditional DPP-MAP (volume/global diversity,")
    print("                                relevance-weighted) beat farthest-point (Max-Min)?")
    print("      lcds_mmr (sweep lam)   => WHERE on the relevance<->diversity axis the optimum")
    print("                                sits. lam=1 reproduces attention_topk, lam=0 is pure")
    print("                                repulsion, so one sweep subsumes both endpoints -- at")
    print("                                1/rho the cost of dpp. Read it as a curve, not a point.")


def parse_args():
    p = argparse.ArgumentParser(description="Layer-Collected Diverse Selection on MMBench (inference-only).")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_root", required=True, help="MMBench dir with data/<split>-*.parquet.")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--l2_categories", nargs="+", default=["all"])
    p.add_argument("--rhos", default="0.05,0.10,0.25")
    p.add_argument("--layer_stride", type=int, default=4,
                   help="take the top text->vision attention at every k-th LLM layer (k = this).")
    p.add_argument("--layers", default="",
                   help="explicit select-layer list (overrides --layer_stride), e.g. 4,8,12,16.")
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
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None,
                   help="lower bound on image area, e.g. 200704 (=448^2 -> >=256 merged tokens).")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="results_lcds_mmbench.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())