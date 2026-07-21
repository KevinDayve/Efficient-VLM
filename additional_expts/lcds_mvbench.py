"""
lcds_mvbench.py -- Layer-Collected Diverse Selection, inference-only, on MVBench.
================================================================================
MVBench sibling of ``lcds_mmbench.py``.  A training-free visual-token selection
method, tested as a SELECTION strategy (no in-LLM pruning yet): compute the
selection from one dense forward, then score the multiple-choice answer through
the frozen model on the kept video-token subset.

The method (identical to the MMBench version, video tokens instead of image):

  dense video tokens
        |
        v   at every k-th LLM layer  (independent, NO dropping)
  [text->vision attention top-C]   record the top-C video tokens by the attention
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
  * random               : uniform K-subset of the video tokens, no prior       (floor)
  * uniform              : stratified evenly-spaced tokens, per frame           (floor)
  * attention_topk       : select-layers-averaged raw attention, global top-K   (foil)
  * lcds                 : per-layer top-attention pool + farthest-point diversity (ours)
  * lcds_dpp             : same pool + conditional DPP-MAP diversity (CDPruner)   (ours)
  * lcds_mmr             : same pool + Maximal Marginal Relevance at --mmr_lambda (ours)
                           lam=1 is attention_topk, lam=0 is pure repulsion, so a lam
                           sweep spans both endpoints at 1/rho the DPP Gram's cost.

EVERY strategy spends EXACTLY k tokens, which is why the `uniform` floor uses the local
select_uniform_exact rather than the suite's oracle_check.select_uniform: that one takes
k//n_frames from each frame and so keeps only n_frames*(k//n_frames) <= k -- as much as
20% short of k at rho=0.01 (T=8, k=20 -> 16). A row on a quietly smaller budget than its
rivals can lose on budget alone, and at small k that is enough to invert a floor-vs-method
ranking (oracle_check.select_by_scores short-changes its callers the same way).

NOT reported here: a per-frame stratified version of attention_topk. Without it, a win for
lcds_* over the global attention foil cannot be attributed to feature diversity rather than
to temporal spread, since per-frame stratified top-K would buy that spread with none of the
pool/Gram machinery. Read temp_H below with that gap in mind.

"FRAME" throughout means a frame PAIR: Qwen2.5-VL pairs adjacent frames
(temporal_patch_size=2), so T = video_grid_thw[0] = num_segments/2 temporal groups --
the finest temporal unit the model exposes (see framewise_attn_tail_mvbench.py).

Uses the OFFICIAL MVBench data/prompt/sampling protocol (same imports as the
sibling MVBench experiments), so numbers are comparable across the suite, and
reports per-task accuracy (MVBench's task labels expose where diversity helps).

Reported per (strategy, rho): answer accuracy, mean WITHIN-frame dispersion, and the
TEMPORAL profile (fraction of frames touched + entropy of the per-frame token
histogram) -- the axis dispersion cannot see, since it saturates once k is large and
scores only the frames a selection actually landed on. Per-clip hits are dumped
separately so strategies can be compared PAIRED (McNemar) rather than by counts.

Run:
    python lcds_mvbench.py \
        --data_root ~/Experiments/MVBench --tasks "Action Sequence" "Scene Transition" \
        --official_sampling --num_segments 16 --max_pixels 200704 \
        --max_samples 40 --rhos 0.05,0.10,0.25 --layer_stride 4 \
        --out results_lcds_mvbench.json
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# shared, validated helpers ------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# score_answer masks all video positions off, turns the kept ones back on, slices the
# (absolute) position ids and reads the first answer-token log-probs; build_full_positions
# builds the video mRoPE ids. The suite's select_uniform / select_by_scores are deliberately
# NOT used here -- they return n_frames*(k//n_frames) <= k tokens; see select_uniform_exact.
from oracle_check import score_answer, build_full_positions
# OFFICIAL MVBench data / prompt / sampling protocol.
from inference import DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt
# identical video dispersion + option-letter readout as the sibling experiment.
from inference_only_mvbench import mean_dispersion, letter_first_ids

STRATEGIES = ("random", "uniform", "attention_topk", "lcds", "lcds_dpp", "lcds_mmr")


# --------------------------------------------------------------------------- #
# per-layer text->vision attention (attention each video token RECEIVES)
# --------------------------------------------------------------------------- #
def received_attention(attentions, video_positions, layers):
    """{layer -> (M,)}: attention each video token RECEIVES from text queries,
    summed over text rows, head-averaged. Computed only for `layers` (a set)."""
    S = attentions[0].shape[-1]
    text_mask = torch.ones(S, dtype=torch.bool, device=attentions[0].device)
    text_mask[video_positions] = False
    out = {}
    for L in layers:
        a = attentions[L][0].mean(0)                     # (S,S) head-mean
        out[L] = a[text_mask][:, video_positions].sum(dim=0).float()  # (M,)
    return out


# --------------------------------------------------------------------------- #
# selection primitives (modality-agnostic: indices into 0..M-1 of video tokens)
# --------------------------------------------------------------------------- #
def select_topk(scores, k):
    """Global top-k indices, sorted."""
    k = min(k, scores.numel())
    return torch.topk(scores, k).indices.sort().values


def select_uniform_exact(M, n_frames, k, device):
    """Content-free stratified floor that spends EXACTLY k tokens: evenly spaced within
    each frame, with the k - n_frames*(k//n_frames) remainder apportioned to evenly spaced
    FRAMES (largest-remainder). The remainder goes by position, never by score, so the
    floor stays content-free.

    Local rather than oracle_check.select_uniform: that one keeps n_frames*(k//n_frames)
    <= k, so at rho=0.01 (T=8, k=20) the floor would run on 16 tokens against the 20 that
    every attention/lcds row gets -- a 20% handicap in the regime where the rows are
    closest together."""
    per = M // n_frames
    kp, rem = divmod(k, n_frames)
    # frames granted the extra token: spread over time, not the first `rem` of them.
    bonus = set(np.linspace(0, n_frames - 1, rem, dtype=int).tolist()) if rem else set()
    idx = []
    for t in range(n_frames):
        m = min(per, kp + (1 if t in bonus else 0))
        if m <= 0:
            continue
        step = max(1, per // m)
        idx.extend(list(range(t * per, t * per + per, step))[:m])
    return torch.tensor(sorted(idx[:k]), device=device, dtype=torch.long)


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
    the per-layer received attention averaged over the selected layers -- the FPS seed."""
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
    video-token indices (0..M-1). diversity: 'fps' = farthest-point (Max-Min),
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
# temporal profile of a selection -- the axis mean_dispersion cannot see
# --------------------------------------------------------------------------- #
def temporal_profile(keep_idx, M, n_frames):
    """(fraction of frames touched, normalised entropy of the per-frame token histogram).
    Both are 1.0 when the budget is spread evenly over every frame and fall as the
    selection collapses onto a few frames -- entropy first, since it reacts to an uneven
    split long before a frame empties out entirely.

    Why this exists: mean_dispersion measures spread WITHIN a frame, saturates toward the
    grid's own mean once k is large, and skips frames holding <2 tokens -- so a temporally
    collapsed selection is scored only on the frames it piled onto. Nothing in the table
    otherwise reports where the budget went in TIME, which is the axis that separates
    global attention from the stratified rows."""
    per = max(1, M // n_frames)
    idx = keep_idx.detach().cpu().numpy()
    counts = np.bincount(np.minimum(idx // per, n_frames - 1),
                         minlength=n_frames).astype(np.float64)
    touched = float((counts > 0).sum()) / n_frames
    p = counts / max(1.0, counts.sum())
    nz = p[p > 0]
    H = (float(-(nz * np.log(nz)).sum() / np.log(n_frames)) if n_frames > 1
         else float("nan"))
    return touched, H


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    # min_pixels: default mirrors max_pixels (fixed per-clip token budget across clips);
    # <=0 disables the floor (variable budget, capped only by max_pixels).
    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    rhos = [float(x) for x in args.rhos.split(",")]
    # CPU generator for the `random` row, built exactly like the sibling script's, so the
    # two scripts' random floors are the same construction at the same seed.
    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    # layer plan resolved lazily once we know n_layers.
    select_layers = feature_hs_idx = None

    correct = {s: {t: {r: 0 for r in rhos} for t in tasks} for s in STRATEGIES}
    disp = {s: {r: [] for r in rhos} for s in STRATEGIES}
    tprof = {s: {r: {"frames": [], "H": []} for r in rhos} for s in STRATEGIES}
    per_sample = []                      # one record per clip -> paired tests downstream
    full_correct = {t: 0 for t in tasks}
    seen = {t: 0 for t in tasks}
    n = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task, unit="clip"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = build_prompt(rec)
                letter_ids = letter_first_ids(processor, letters)
                prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                             args.max_frames, args.max_pixels, args.fps,
                                             official=args.official_sampling,
                                             num_segments=args.num_segments,
                                             min_pixels=min_pixels)
                chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                chat += ANSWER_PREFIX  # force the answer; option letter is the next token after '('
                img_in, vid_in = process_vision_info(prompt)
                inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            input_ids = inputs["input_ids"].to(device)
            attn = inputs["attention_mask"].to(device)
            vpos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
            if vpos.numel() < 50:
                continue
            M = vpos.numel()
            grid_thw = inputs["video_grid_thw"].to(device)
            n_frames = int(grid_thw[0][0].item())
            pix = inputs["pixel_values_videos"].to(device)

            try:
                # merged embeds (video features spliced in) + video mRoPE positions.
                with torch.no_grad():
                    ve = model.get_video_features(pix, grid_thw).pooler_output
                    ve = torch.cat(ve, dim=0).to(device)
                    base = model.get_input_embeddings()(input_ids).clone()
                    base[0, vpos] = ve.to(base.dtype)
                pos = build_full_positions(model, input_ids, vpos, grid_thw, attn)
                # single dense forward: attentions (all layers) + hidden states (features).
                with torch.no_grad():
                    out = model(inputs_embeds=base, position_ids=pos, attention_mask=attn,
                                use_cache=False, output_attentions=True, output_hidden_states=True)
            except Exception as e:
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            attentions = out.attentions

            if select_layers is None:                     # resolve the layer plan once
                L = len(attentions)
                if args.layers:                           # explicit override
                    select_layers = [int(v) for v in args.layers.split(",")]
                else:                                     # every k-th LLM layer
                    select_layers = list(range(args.layer_stride, L, args.layer_stride))
                select_layers = sorted({min(x, L - 1) for x in select_layers})
                if args.min_select_layer > 0:             # drop the early attention-sink /
                    kept = [x for x in select_layers      # positional-bias band from the vote
                            if x >= args.min_select_layer]
                    select_layers = kept or [L - 1]       # never empty -> keep the final layer
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
            feats_vid = feats[vpos]                                      # (M,d)
            # attention_topk foil: raw received attention averaged over the SAME select
            # layers, then global top-K (single pool, no per-layer union, no diversity).
            recv = received_attention(attentions, vpos, set(select_layers))
            band = torch.stack([recv[L] for L in select_layers]).mean(0)
            del out, attentions

            # full-model reference (all video tokens kept)
            keep_all = torch.arange(M, device=device)
            lp_full = score_answer(model, base, pos, attn, vpos, keep_all, letter_ids)
            hit_full = int(lp_full.argmax().item() == gt_idx)
            full_correct[task] += hit_full

            # every strategy answers THIS clip, so their hits are paired; the aggregate
            # counts throw that pairing away, and the gaps here are small enough that the
            # paired test is the only one with the power to resolve them.
            rec_out = {"task": task, "video": rec.get("video"), "gt": gt_idx, "M": M,
                       "n_frames": n_frames, "full": hit_full,
                       "correct": {s: {} for s in STRATEGIES}}

            for rho in rhos:
                k = max(n_frames, int(round(rho * M)))     # >= 1 token/frame, like the suite
                for s in STRATEGIES:
                    if s == "random":
                        keep = torch.randperm(M, generator=rng)[:k].sort().values.to(device)
                    elif s == "uniform":
                        keep = select_uniform_exact(M, n_frames, k, device)
                    elif s == "attention_topk":
                        keep = select_topk(band, k)
                    else:
                        div = {"lcds_dpp": "dpp", "lcds_mmr": "mmr"}.get(s, "fps")
                        keep = lcds_select(recv, feats_vid, select_layers, M, k,
                                           args.cand_mult, device, div, args.mmr_lambda)
                    lp = score_answer(model, base, pos, attn, vpos, keep, letter_ids)
                    hit = int(lp.argmax().item() == gt_idx)
                    correct[s][task][rho] += hit
                    rec_out["correct"][s][str(rho)] = hit
                    d = mean_dispersion(keep, M, n_frames)
                    if d == d:
                        disp[s][rho].append(d)
                    frac, H = temporal_profile(keep, M, n_frames)
                    tprof[s][rho]["frames"].append(frac)
                    if H == H:
                        tprof[s][rho]["H"].append(H)
            per_sample.append(rec_out)
            seen[task] += 1
            n += 1
            del base, feats, feats_vid, recv
            torch.cuda.empty_cache()
            if n % 25 == 0:
                r = rhos[-1]
                msg = " | ".join(f"{s} {sum(correct[s][t][r] for t in tasks)/n:.3f}"
                                 for s in STRATEGIES)
                full_micro = sum(full_correct[t] for t in tasks) / n
                print(f"[{n}] rho={r}: {msg}  (full {full_micro:.3f})")

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    valid = [t for t in tasks if seen[t]]

    def per_task_mean(counts):     # counts: {task: int}
        return float(np.mean([counts[t] / seen[t] for t in valid]))

    def micro(counts):
        return sum(counts[t] for t in valid) / n

    full_mean, full_micro = per_task_mean(full_correct), micro(full_correct)

    out_json = {"experiment": "mvbench_lcds", "model_name": args.model_name,
                "data_root": args.data_root,
                "sampling": ("official" if args.official_sampling else "fps"),
                "num_frames": (args.num_segments if args.official_sampling else None),
                "fps": (None if args.official_sampling else args.fps),
                "max_frames": args.max_frames, "max_pixels": args.max_pixels,
                "min_pixels": min_pixels, "rhos": rhos, "layer_stride": args.layer_stride,
                "min_select_layer": args.min_select_layer,
                "select_layers": select_layers, "feature_hs_idx": feature_hs_idx,
                "cand_mult": args.cand_mult, "strategies": list(STRATEGIES),
                "tasks": valid, "n": n,
                "full_accuracy_mean": full_mean, "full_accuracy_micro": full_micro,
                "per_task_seen": {t: seen[t] for t in valid},
                "full_per_task": {t: full_correct[t] / seen[t] for t in valid},
                "table": {}}

    print(f"\n==== LCDS on MVBench ({n} samples, {len(valid)} task(s)) ====")
    print(f"full-model accuracy: mean(per-task) {full_mean:.4f}  micro {full_micro:.4f}\n")
    hdr = (f"{'strategy':<16}{'rho':>6}{'acc_mean':>10}{'acc_micro':>11}"
           f"{'dispersion':>12}{'frames':>9}{'temp_H':>9}")
    print(hdr); print("-" * len(hdr))
    for s in STRATEGIES:
        out_json["table"][s] = {}
        for rho in rhos:
            counts = {t: correct[s][t][rho] for t in tasks}
            acc_mean = per_task_mean(counts)
            acc_micro = micro(counts)
            dsp = float(np.mean(disp[s][rho])) if disp[s][rho] else float("nan")
            frac = (float(np.mean(tprof[s][rho]["frames"]))
                    if tprof[s][rho]["frames"] else float("nan"))
            H = float(np.mean(tprof[s][rho]["H"])) if tprof[s][rho]["H"] else float("nan")
            out_json["table"][s][str(rho)] = {
                "accuracy_mean": acc_mean, "accuracy_micro": acc_micro, "dispersion": dsp,
                "frames_touched": frac, "temporal_entropy": H,
                "per_task": {t: correct[s][t][rho] / seen[t] for t in valid}}
            print(f"{s:<16}{rho:>6.2f}{acc_mean:>10.4f}{acc_micro:>11.4f}"
                  f"{dsp:>12.3f}{frac:>9.3f}{H:>9.3f}")

    with open(args.out, "w") as f:
        json.dump(out_json, f, indent=2)
    ps_path = args.per_sample_out or os.path.splitext(args.out)[0] + "_per_sample.json"
    with open(ps_path, "w") as f:
        json.dump({"experiment": "mvbench_lcds_per_sample", "rhos": rhos,
                   "strategies": list(STRATEGIES), "n": n, "samples": per_sample}, f, indent=2)
    print(f"\nsaved -> {args.out}\nsaved -> {ps_path}  (per-clip hits, for paired tests)")
    print("\nread: every row spends exactly k tokens, so no comparison here turns on budget.")
    print("      lcds_* vs random / uniform => does a content signal beat no signal at all?")
    print("      lcds_* vs attention_topk   => does the per-layer pool + final-layer diversity")
    print("                                    beat a single-pool raw attention top-K?")
    print("      lcds_dpp vs lcds           => does conditional DPP-MAP (volume/global diversity,")
    print("                                    relevance-weighted) beat farthest-point (Max-Min)?")
    print("      lcds_mmr (sweep lam)       => WHERE on the relevance<->diversity axis the optimum")
    print("                                    sits. lam=1 reproduces attention_topk, lam=0 is pure")
    print("                                    repulsion, so one sweep subsumes both endpoints --")
    print("                                    at 1/rho the cost of dpp. A curve, not a point.")
    print("      frames / temp_H            => where the budget went in TIME: 1.0 = spread over")
    print("                                    every frame, low = collapsed onto a few.")
    print("      CAVEAT: an lcds_* win that arrives with high temp_H may be temporal spread, not")
    print("      feature diversity -- per-frame stratified attention would buy that spread with no")
    print("      pool and no Gram. There is no such row here, so the attribution stays open.")
    print("      The counts above are also unpaired: resolve any gap you mean to claim with")
    print("      McNemar on the per-clip dump, not with these margins.")


def parse_args():
    p = argparse.ArgumentParser(description="Layer-Collected Diverse Selection on MVBench (inference-only).")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames per clip (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="Use the reference mvbench.ipynb sampler (fixed --num_segments frames at "
                        "segment midpoints) for leaderboard-comparable numbers, instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="per-frame pixel ceiling (downscales large frames).")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="per-frame pixel floor. Default: mirror --max_pixels for a FIXED "
                        "per-clip token budget across clips; pass <=0 to disable the floor.")
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--rhos", default="0.01,0.05,0.10,0.25")
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
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42, help="seed for the `random` row.")
    p.add_argument("--out", default="results_lcds_mvbench.json")
    p.add_argument("--per_sample_out", default="",
                   help="per-clip correctness dump, for paired (McNemar) tests between "
                        "strategies. Default: <--out stem>_per_sample.json.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
