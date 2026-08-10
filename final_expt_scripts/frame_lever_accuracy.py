"""
frame_lever_accuracy.py -- the FRAME lever: is a single well-chosen frame worth more
than the whole clip, and does any cheap signal pick it?

This is Claim 4 of the narrative. The token-level thread (stage_topk_accuracy.py) ends
with "ranking tokens by importance does not beat a random keep". The remaining lever is
coarser: choose the frame. This script measures both halves of that claim in one pass --
how much headroom frame selection has (the oracle), and whether any per-frame score
captures it (every scorer, one table). It merges what were separate probes: frame
sufficiency, the per-frame scorer sweep, and the temporal change-point.

The two questions
-----------------
1. HEADROOM. Per clip we run the model on all F frames, on each frame ALONE, and on no
   frames at all, and report

       all           every frame, the normal inference
       single_mean   the expected accuracy of a frame picked at random (mean over F)
       single_best   correct if ANY single frame is correct -- the selection ORACLE
       single_worst  correct only if EVERY frame is correct
       blind         no visual tokens at all -- the language prior

   single_best > all says the model UNDER-EXPLOITS the full clip: a frame it already
   answers from is in there, and handing it the other F-1 frames loses accuracy. That
   gap, single_best - all, is the entire prize frame selection is playing for, and every
   scorer below is graded as the fraction of it recovered.

2. EXPLOITABILITY. Each scorer assigns one number per frame; the frame it argmaxes is
   the frame the model then answers from, so its accuracy is read straight off the
   per-frame correctness vector -- no extra forwards. A scorer that carries real signal
   lands above single_mean and moves toward single_best.

The scorers  (--scorers)
------------------------
Cheap, pre-LLM (computed from the frame's own post-projector features -- no LM forward
beyond the one that produced them):

    xi_encq    tail index of the frame's encoder x query token scores   [query-aware]
    xi_sal     tail index of the frame's token saliency ||v||           [query-free]
    mass_encq  mean encoder x query over the frame's tokens             [the plain-mass
               control for the two tail indices: if xi_* ~ mass_encq, concentration adds
               nothing over total query mass]
    motion     mean token novelty (1 - cosine) vs the neighbouring frame [query-free]

In-LLM, from the frame's own isolated forward (already run for single_best, so free):

    dec_xi     tail index of the frame's decoder importance over --dec_band
    dec_mass   the frame's total decoder attention mass
    conf       the answer-confidence margin (top option logit - second) -- the model's
               own sense of whether this frame answers the question

In-context, from the ONE full-clip forward (the trajectory a[t] of query-guided
attention over native temporal positions):

    ctx_level  a[t] itself -- the level baseline, and the thing the change-points below
               are derived from, so it is the number they have to beat
    ctx_cp_post |a[t] - a[t-1]| credited to the LATER position: the "an event just
               happened here" reading
    ctx_cp_pre  the same difference credited to the EARLIER position

External contrastive (--clip_model, off by default -- it loads a second model):

    clip_qmatch  SigLIP cosine(frame, question)                       [deployable]
    clip_qopt    top-1 minus top-2 cosine over "question + option_j"  [deployable]
    clip_qgt     cosine(frame, "question + the CORRECT option")       [uses the label:
                 a ceiling, not a method -- it answers "is the answerable frame even
                 VISIBLE in an aligned space?"]

Floors, carried as scorers so nothing downstream special-cases them:

    middle     the middle frame -- the positional prior
    random     one seeded draw per clip. This is the paired partner for McNemar;
               single_mean is its expectation over all F draws.

Position granularity (important for ctx_*)
------------------------------------------
Qwen merges frames in PAIRS temporally, so the finest unit its in-context attention has
is a POSITION covering two frames (T = F/2); LLaVA-OneVision and LLaVA-1.5 keep one
position per frame. The ctx_* scores are therefore computed per position and expanded
piecewise-constant back to frames, so every scorer is graded against ONE set of
frame-level anchors. On Qwen that expansion makes the ctx_* scores tie within each pair
-- an honest handicap of the merge, not of the statistic, and `frames_per_position` in
the output records it.

How each scorer is judged
-------------------------
    accuracy         of the frame it picks
    vs single_mean   points over the random-frame floor
    headroom         (acc - single_mean) / (single_best - single_mean): the share of the
                     oracle's prize recovered. Negative = worse than picking at random.
    pooled AUC       P(score of a correct frame > score of an incorrect one), all frames
    within-clip AUC  the same, computed INSIDE each clip and averaged -- restricted to
                     "mixed" clips (both a correct and an incorrect frame), which is what
                     controls for per-clip difficulty. This is the metric to trust: a
                     scorer can look good pooled purely by ranking easy clips above hard
                     ones, which is useless for choosing a frame WITHIN a clip.
    top-1-on-mixed   how often its argmax lands on a correct frame, on those clips
    McNemar          exact, two-sided, against the `random` draw and against `middle`,
                     on paired per-clip correctness -- so "no cheap score beats the
                     floor" is a test result and not a small gap.

Costs
-----
F + 2 forwards per clip: one full-clip (eager -- the in-context attention has to be
materialised), F short isolated ones, and the blind one. The isolated forwards dominate
the count but are cheap (one merged position on Qwen). Use --max_samples for a first look.

Single frames are passed as a --single_repeat (default 2) frame zero-motion clip [f,f]:
Qwen's temporal merge requires an even frame count, and repeating keeps the two backbones
on one protocol. The score of such a clip is content-at-that-time answerability, not an
in-context position effect.

The blind forward drops every visual token from the full-clip sequence at the input,
keeping the surviving text tokens' original position ids -- the identical operation
stage_topk_accuracy.py uses for its text-only floor, so the two experiments' floors are
the same number.

Frame sampling, prompts and input builders come from tail_vs_layer.py, and the letter
scorer and McNemar from lv_knockout_accuracy.py / stage_topk_accuracy.py, so these
numbers sit on the same scale as the tail-index, knockout and top-K runs, clip for clip.

Run
---
    # Qwen, EgoSchema
    python frame_lever_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --num_segments 16 --max_pixels 200704 \
        --out frame_ego_qwen.json --plot frame_ego_qwen.png

    # LLaVA-OneVision, MVBench, 50 clips/task, with the contrastive selector
    python frame_lever_accuracy.py --data_root ~/Experiments/MVBench --tasks mvbench \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --num_segments 16 \
        --max_samples 50 --clip_model google/siglip-base-patch16-224 \
        --out frame_mvb_llavaov.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from lv_knockout_accuracy import gold_index, option_token_ids
from stage_topk_accuracy import attach_capture, keep_abs_idx, mcnemar, parse_band, predict_pruned
from tail_vs_layer import (ANSWER_PREFIX, DATA_LIST, DEFAULT_SEGMENTS, LLAVA_FAMILY,
                           MVBENCH_TASKS, build_inputs, context_limit, infer_backbone,
                           iter_clips, load_model, moment_tail_index, sample_frames,
                           text_model_of, visual_query_masks)

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

CHEAP_SCORERS = ["xi_encq", "xi_sal", "mass_encq", "motion"]
LLM_SCORERS = ["dec_xi", "dec_mass", "conf"]
CTX_SCORERS = ["ctx_level", "ctx_cp_post", "ctx_cp_pre"]
CLIP_SCORERS = ["clip_qmatch", "clip_qopt", "clip_qgt"]
FLOOR_SCORERS = ["middle", "random"]
DEFAULT_SCORERS = CHEAP_SCORERS + LLM_SCORERS + CTX_SCORERS + FLOOR_SCORERS
# Anchors are not scorers: they are rho-independent references carried through the same
# per-clip bookkeeping so the report and the per-sample dump need no special cases.
ANCHORS = ["all", "blind", "single_best", "single_worst"]


# --------------------------------------------------------------------------- #
# 1. Frames <-> visual-token spans
# --------------------------------------------------------------------------- #
def frame_spans(inputs, M: int, n_frames: int, backbone: str):
    """(spans, frames_per_position): the visual-block slice of each native temporal
    position, and how many source frames one position covers.

    Qwen's video path merges frames in pairs (temporal_patch_size=2) and then pools 2x2
    spatially, so the decoder sees t = F/2 positions of (h*w)/4 tokens; video_grid_thw
    carries (t, h, w) in pre-merge patch units, which is the only place those numbers are
    stated for the clip actually built. Both LLaVA towers keep one position per frame at a
    fixed size, so the count divides out -- OneVision's trailing newline is the remainder
    and is deliberately left out of every span (it belongs to the clip, not to a frame)."""
    if backbone == "qwen":
        t, h, w = (int(x) for x in inputs["video_grid_thw"][0].tolist())
        per_pos, n_pos = (h * w) // 4, t
    else:
        per_pos, n_pos = M // n_frames, n_frames

    trailing = M - per_pos * n_pos
    if per_pos <= 0 or trailing < 0:
        raise RuntimeError(f"cannot split {M} visual tokens into {n_frames} frames "
                           f"(per_pos={per_pos}, trailing={trailing})")
    if n_frames % n_pos:
        raise RuntimeError(f"{n_frames} frames do not divide into {n_pos} temporal positions")
    spans = [(i * per_pos, (i + 1) * per_pos) for i in range(n_pos)]
    return spans, n_frames // n_pos


def expand_to_frames(per_pos: np.ndarray, frames_per_position: int, n_frames: int) -> np.ndarray:
    """A position-granular score, repeated onto the frames each position covers, so every
    scorer is graded against the same frame-level anchors. On Qwen this ties the two
    frames of a pair; on the LLaVA backbones it is the identity."""
    return np.repeat(np.asarray(per_pos, dtype=float), frames_per_position)[:n_frames]


# --------------------------------------------------------------------------- #
# 2. Per-frame scores
# --------------------------------------------------------------------------- #
def band_score(store: dict, band: list[int]) -> torch.Tensor:
    """Mean over the band of each layer's L1-normalised visual-token attention.

    Normalising per layer first is the same convention stage_topk_accuracy.py uses for
    attn_mid: attention rows sum to one over ALL keys, so a layer's visual mass sits at
    whatever scale its text/sink split leaves it, and an unnormalised mean is just
    whichever layer is loudest."""
    normed = torch.stack([store[l] / store[l].sum().clamp_min(1e-12) for l in band])
    return normed.mean(dim=0)


def cheap_frame_scores(V: np.ndarray, q_hat: np.ndarray, k_frac: float) -> dict:
    """The three per-frame pre-LLM scores, from the frame's post-projector token features.

    V is (P, D) -- the frame's own visual tokens as the LM receives them, which is the
    earliest point at which a query-aware score exists at all (before the projector there
    is no shared visual/text space). encq is the input-space visual.query projection.
    (motion is the fourth cheap score but needs a neighbour, so it is computed across
    frames in motion_scores.)"""
    encq = V @ q_hat
    sal = np.linalg.norm(V, axis=1)
    return {"xi_encq": moment_tail_index(torch.from_numpy(encq), k_frac),
            "xi_sal": moment_tail_index(torch.from_numpy(sal), k_frac),
            "mass_encq": float(encq.mean())}


def motion_scores(V_list: list[np.ndarray]) -> np.ndarray:
    """Novelty of each frame against its neighbour: 1 - mean cell-aligned cosine.

    Cell-aligned because every frame carries the same token grid (a fixed pixel budget on
    Qwen, a fixed frame size on both LLaVAs), so token p of frame f and token p of frame
    f-1 are the same spatial position. The ends reuse their only neighbour."""
    n = len(V_list)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    for f in range(n):
        g = f - 1 if f > 0 else 1
        a, b = V_list[f], V_list[g]
        if a.shape != b.shape:
            continue
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        out[f] = float(1.0 - cos.mean())
    return out


def ctx_scores(a: np.ndarray) -> dict:
    """The trajectory scorers. a[t] is query-guided attention at temporal position t.

    A change-point is only interesting if it beats the level it is built from, so
    ctx_level is reported beside them rather than as a separate control. The position
    that has no neighbour on the relevant side scores -inf, i.e. it is never picked,
    which is the honest reading of "this scorer has nothing to say there"."""
    d = np.abs(np.diff(a)) if a.size > 1 else np.zeros(0)
    post = np.concatenate([[-np.inf], d]) if a.size > 1 else np.full(a.size, -np.inf)
    pre = np.concatenate([d, [-np.inf]]) if a.size > 1 else np.full(a.size, -np.inf)
    return {"ctx_level": a, "ctx_cp_post": post, "ctx_cp_pre": pre}


# --------------------------------------------------------------------------- #
# 3. The contrastive selector (optional second model)
# --------------------------------------------------------------------------- #
class ContrastiveScorer:
    """SigLIP/CLIP frame x text cosine, on the RAW PIL frames.

    Raw frames, never the VLM's merged video tensor, so this is also the clean control
    for "is the [f,f] merge doing the damage?". The point of the probe is that this is
    the one signal Qwen's own space cannot give: its projector output is not
    contrastively text-aligned, so an input-space dot product with the question is close
    to meaningless there (see the encq scorers, which is exactly what they test)."""

    def __init__(self, model_name: str, device, dtype):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def _unit(self, frames, texts):
        px = self.processor(images=frames, return_tensors="pt").to(self.device)
        px["pixel_values"] = px["pixel_values"].to(self.dtype)
        tx = self.processor(text=texts, padding="max_length", truncation=True,
                            return_tensors="pt").to(self.device)
        f = self.model.get_image_features(**px).float()
        t = self.model.get_text_features(**tx).float()
        return (f / f.norm(dim=-1, keepdim=True)), (t / t.norm(dim=-1, keepdim=True))

    def scores(self, frames, record, gold: int) -> dict:
        """(F,) per scorer. qopt is the decisiveness of the best option at that frame --
        the deployable one, since it never touches the label; qgt is handed the correct
        option and so is a ceiling on what an aligned space could possibly mark."""
        q = record["question"]
        texts = [q] + [f"{q} {c}" for c in record["candidates"]]
        img, txt = self._unit(frames, texts)
        sim = (img @ txt.T).cpu().numpy()                    # (F, 1 + n_options)
        opts = sim[:, 1:]
        top2 = np.sort(opts, axis=1)[:, ::-1][:, :2]
        margin = top2[:, 0] - top2[:, 1] if opts.shape[1] > 1 else top2[:, 0]
        return {"clip_qmatch": sim[:, 0], "clip_qopt": margin, "clip_qgt": opts[:, gold]}


# --------------------------------------------------------------------------- #
# 4. Discrimination metrics
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, so tied scores neither win nor lose the comparison. Ties are the
    normal case for ctx_* on Qwen (a position covers two frames)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    s = a[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a correct frame > score of an incorrect one), ties counted as half --
    the Mann-Whitney form, so 0.5 is exactly "no signal"."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = _rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def pick(score: np.ndarray) -> int:
    """argmax, NaN-safe. An all-NaN scorer picks frame 0 rather than crashing the clip;
    it will sit at the floor in the table, which is the correct thing to report."""
    s = np.asarray(score, dtype=float)
    return int(np.nanargmax(s)) if np.isfinite(s).any() else 0


# --------------------------------------------------------------------------- #
# 5. One clip
# --------------------------------------------------------------------------- #
@torch.no_grad()
def isolated_frame(model, processor, frame, record, args, device, dtype, letter_ids,
                   visual_token_id, store, ctx, band):
    """One forward on the [f,f] zero-motion clip. Returns the frame's prediction, its
    answer-confidence margin, its own post-projector token features and its in-LLM
    decoder scores -- everything the per-frame scorers need, from a forward that has to
    happen anyway to know whether this frame alone answers the question."""
    inputs = build_inputs(processor, [frame] * args.single_repeat, record, args, device, dtype)
    ids = inputs["input_ids"][0]
    visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
    M = int(visual_idx.numel())
    if M < 8:
        raise RuntimeError(f"too few visual tokens on the isolated frame ({M})")

    store.clear()
    ctx.update({"capture": True, "visual_idx": visual_idx, "text_q": text_q,
                "embeds": None, "position_ids": None})
    logits = model(**inputs, use_cache=False).logits[0, -1]
    ctx["capture"] = False

    opt = logits[letter_ids].float()
    top = torch.sort(opt, descending=True).values
    conf = float(top[0] - top[1]) if top.numel() > 1 else float("nan")

    spans, _ = frame_spans(inputs, M, args.single_repeat, args.backbone)
    lo, hi = spans[0]                       # the frame's own tokens (the whole clip on Qwen)
    # Under device_map="auto" the embeddings can sit on a different GPU than the input_ids
    # the indices were built from, so the index tensor follows the tensor it indexes.
    embeds = ctx["embeds"]
    V = embeds[0][visual_idx[lo:hi].to(embeds.device)].float().cpu().numpy()

    dec = band_score(store, band)
    out = {"pred": int(torch.argmax(opt).item()), "conf": conf, "V": V,
           "dec_xi": moment_tail_index(dec, args.k_frac),
           "dec_mass": float(dec.sum())}
    ctx.update({"embeds": None, "position_ids": None})
    store.clear()
    del inputs, logits
    return out


@torch.no_grad()
def full_clip(model, processor, frames, record, args, device, dtype, letter_ids,
              visual_token_id, store, ctx, band, max_positions):
    """The one full-clip forward: the `all` prediction, the blind floor read off the same
    sequence, the in-context attention trajectory, and the question embedding the cheap
    scorers project onto."""
    inputs = build_inputs(processor, frames, record, args, device, dtype)
    ids = inputs["input_ids"][0]
    S = ids.numel()
    if max_positions is not None and S > max_positions:
        raise RuntimeError(f"sequence is {S} tokens but the LM holds {max_positions} "
                           f"-- lower --num_segments")
    visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
    M = int(visual_idx.numel())
    if M < 50:
        raise RuntimeError(f"too few visual tokens ({M} < 50)")

    store.clear()
    ctx.update({"capture": True, "visual_idx": visual_idx, "text_q": text_q,
                "embeds": None, "position_ids": None})
    dense = model(**inputs, use_cache=False)
    ctx["capture"] = False
    pred_all = int(torch.argmax(dense.logits[0, -1][letter_ids]).item())
    del dense

    embeds, pos = ctx["embeds"], ctx["position_ids"]
    if embeds is None or embeds.shape[1] != S:
        raise RuntimeError("did not capture the merged input embeddings")
    if pos is None:                          # the LM built them internally: plain 0..S-1
        pos = torch.arange(S, device=embeds.device)[None]

    # The blind floor: the same prune at K = 0. The text keeps the position ids it had
    # when the visual block was there, so this is the sweep's rho -> 0 limit and not a
    # differently-built prompt.
    pred_blind = predict_pruned(model, embeds, pos,
                                keep_abs_idx(S, visual_idx, np.empty(0, dtype=np.int64)),
                                letter_ids)

    # The question direction, from the text positions of this very sequence: a unit vector
    # in the LM's input space, which is where the visual tokens also live post-projector.
    q = embeds[0][text_q.to(embeds.device)].float().mean(0)
    q_hat = (q / q.norm().clamp_min(1e-12)).cpu().numpy()

    spans, fpp = frame_spans(inputs, M, args.num_segments, args.backbone)
    dec = band_score(store, band).numpy()
    a = np.array([dec[lo:hi].mean() for lo, hi in spans], dtype=float)

    ctx.update({"embeds": None, "position_ids": None})
    store.clear()
    del inputs, embeds
    return {"pred": pred_all, "blind": pred_blind, "q_hat": q_hat,
            "traj": a, "frames_per_position": fpp, "n_visual_tokens": M}


# --------------------------------------------------------------------------- #
# 6. Dataset sweep
# --------------------------------------------------------------------------- #
def resolve_args(args):
    """Backbone, frames, tasks, pixel budget and scorer list -- the same handling as the
    rest of the family, so the clips match clip for clip."""
    if args.backbone == "auto":
        args.backbone = infer_backbone(args.model_name)
    if args.num_segments is None:
        args.num_segments = DEFAULT_SEGMENTS[args.backbone]

    if args.tasks == ["all"]:
        args.tasks = list(DATA_LIST)
    elif args.tasks == ["mvbench"]:
        args.tasks = MVBENCH_TASKS
    unknown = [t for t in args.tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    if args.backbone in LLAVA_FAMILY:
        if args.max_pixels is not None or args.min_pixels is not None:
            fixed = "576" if args.backbone == "llava" else "196"
            print(f"[warn] --max_pixels/--min_pixels are ignored on {args.backbone} "
                  f"(fixed {fixed} decoder tokens/frame).")
        args.max_pixels = args.min_pixels = None
    else:
        if args.min_pixels is None:
            args.min_pixels = args.max_pixels
        if args.min_pixels is not None and args.min_pixels <= 0:
            args.min_pixels = None

    scorers = list(args.scorers)
    if args.clip_model:
        scorers += [s for s in CLIP_SCORERS if s not in scorers]
    else:
        scorers = [s for s in scorers if s not in CLIP_SCORERS]
    known = set(CHEAP_SCORERS + LLM_SCORERS + CTX_SCORERS + CLIP_SCORERS + FLOOR_SCORERS)
    bad = [s for s in scorers if s not in known]
    if bad:
        raise ValueError(f"unknown scorers {bad}; choices: {sorted(known)}")
    return scorers


def main(args):
    scorers = resolve_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[model] {args.model_name}  backbone={args.backbone}  dtype={args.dtype}  device={device}")
    model, processor, visual_token_id = load_model(args, dtype)   # eager: the scores need the weights
    max_positions = context_limit(model)
    n_layers = len(text_model_of(model).layers)
    band = parse_band(args.dec_band, n_layers)

    contrastive = None
    if args.clip_model:
        print(f"[clip] {args.clip_model}")
        contrastive = ContrastiveScorer(args.clip_model, device, dtype)

    F = args.num_segments
    print(f"[model] {n_layers} decoder layers, {max_positions} LM positions")
    print(f"[frames] {F}/clip; a single frame is a {args.single_repeat}-frame zero-motion clip")
    print(f"[score] decoder band {band[0]}-{band[-1]} ({len(band)} layers), queries={args.queries}")
    print(f"[score] scorers: {scorers}")
    print(f"[cost] {F + 2} forwards/clip  (1 full + {F} isolated + 1 blind)")
    print(f"[data] tasks={args.tasks}")

    store, ctx = {}, {"capture": False}
    uninstall = attach_capture(model, store, ctx)

    configs = ANCHORS + scorers
    letter_cache = {}
    hits = {c: [] for c in configs}                  # config -> per-clip 0/1, clip-aligned
    single_mean_sum = 0.0                            # fractional: E[acc of a random frame]
    per_task_hits, per_task_single_mean = {}, {}
    pos_correct = np.zeros(F)                        # accuracy by frame POSITION
    best_pos_hist = np.zeros(F)                      # where the oracle's frame sits
    pooled = {s: {"pos": [], "neg": []} for s in scorers}
    within_auc = {s: [] for s in scorers}
    within_top1 = {s: [] for s in scorers}
    seen_by_task, per_sample, skipped = {}, [], []
    n_seen = n_mixed = 0

    try:
        for idx, (task, rec, path, data_type, bound) in enumerate(
                tqdm(list(iter_clips(args)), unit="clip")):
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": rec["video"], "reason": "missing file"})
                continue
            try:
                gold = gold_index(rec)
                n_opt = len(rec["candidates"])
                if n_opt not in letter_cache:
                    letter_cache[n_opt] = torch.tensor(
                        option_token_ids(processor.tokenizer, n_opt), device=device)
                letter_ids = letter_cache[n_opt]

                frames = sample_frames(path, data_type, bound, F)
                if len(frames) != F:
                    raise RuntimeError(f"sampler returned {len(frames)} frames, expected {F}")

                clip = full_clip(model, processor, frames, rec, args, device, dtype,
                                 letter_ids, visual_token_id, store, ctx, band, max_positions)

                singles = [isolated_frame(model, processor, f, rec, args, device, dtype,
                                          letter_ids, visual_token_id, store, ctx, band)
                           for f in frames]
                correct = np.array([int(s["pred"] == gold) for s in singles], dtype=bool)

                # ---- per-frame scores; every scorer is a full (F,) vector ----
                V_list = [s["V"] for s in singles]
                cheap = [cheap_frame_scores(V, clip["q_hat"], args.k_frac) for V in V_list]
                score = {k: np.array([c[k] for c in cheap], dtype=float)
                         for k in ("xi_encq", "xi_sal", "mass_encq")}
                score["motion"] = motion_scores(V_list)
                for k in ("dec_xi", "dec_mass", "conf"):
                    score[k] = np.array([s[k] for s in singles], dtype=float)
                for k, v in ctx_scores(clip["traj"]).items():
                    score[k] = expand_to_frames(v, clip["frames_per_position"], F)
                if contrastive is not None:
                    score.update(contrastive.scores(frames, rec, gold))
                score["middle"] = -np.abs(np.arange(F) - (F - 1) / 2.0)
                rng = np.random.default_rng([args.seed, idx])
                score["random"] = rng.random(F)
            except Exception as e:
                ctx.update({"capture": False, "embeds": None, "position_ids": None})
                store.clear()
                skipped.append({"task": task, "video": rec["video"],
                                "reason": f"{type(e).__name__}: {e}"})
                tqdm.write(f"skip [{task}] {rec['video']}: {type(e).__name__}: {e}")
                continue

            # ---- anchors ----
            preds = {"all": int(clip["pred"] == gold), "blind": int(clip["blind"] == gold),
                     "single_best": int(correct.any()), "single_worst": int(correct.all())}
            picks = {}
            for s in scorers:
                picks[s] = pick(score[s])
                preds[s] = int(correct[picks[s]])

            n_seen += 1
            seen_by_task[task] = seen_by_task.get(task, 0) + 1
            frac = float(correct.mean())
            single_mean_sum += frac
            per_task_single_mean[task] = per_task_single_mean.get(task, 0.0) + frac
            tc = per_task_hits.setdefault(task, {c: 0 for c in configs})
            for c in configs:
                hits[c].append(preds[c])
                tc[c] += preds[c]
            pos_correct += correct
            if correct.any():
                best_pos_hist[int(np.argmax(correct))] += 1

            # ---- discrimination: does a high score mark a CORRECT frame? ----
            mixed = bool(correct.any() and not correct.all())
            n_mixed += int(mixed)
            for s in scorers:
                v = np.asarray(score[s], dtype=float)
                pooled[s]["pos"].append(v[correct])
                pooled[s]["neg"].append(v[~correct])
                if mixed:
                    within_auc[s].append(auc(v[correct], v[~correct]))
                    within_top1[s].append(float(correct[picks[s]]))

            per_sample.append({"task": task, "question_idx": rec.get("question_idx"),
                               "video": rec["video"], "gold": gold, "n_options": n_opt,
                               "n_visual_tokens": clip["n_visual_tokens"],
                               "frames_per_position": clip["frames_per_position"],
                               "frame_correct": correct.astype(int).tolist(),
                               "mixed": mixed,
                               "picked_frame": picks,
                               "hit_by_config": preds,
                               # null, not inf/nan: JSON has no encoding for either, and a
                               # non-finite entry means "this scorer says nothing here".
                               "scores": {s: [float(v) if np.isfinite(v) else None
                                              for v in score[s]] for s in scorers}})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        uninstall()

    if not n_seen:
        print("no usable clips -- check --data_root layout (json/ and video/).")
        return

    # ------------------------------------------------------------------ report
    acc = {c: sum(hits[c]) / n_seen for c in configs}
    acc["single_mean"] = single_mean_sum / n_seen
    floor, ceiling = acc["single_mean"], acc["single_best"]
    span = ceiling - floor

    disc = {}
    for s in scorers:
        pos = np.concatenate(pooled[s]["pos"]) if pooled[s]["pos"] else np.zeros(0)
        neg = np.concatenate(pooled[s]["neg"]) if pooled[s]["neg"] else np.zeros(0)
        wa, wt = np.asarray(within_auc[s], dtype=float), np.asarray(within_top1[s], dtype=float)
        # ctx_cp_* score -inf where they have no neighbour, so the mean is taken over the
        # finite entries only -- otherwise one sentinel swallows the whole statistic.
        fpos, fneg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        disc[s] = {"pooled_auc": auc(pos, neg),
                   "within_clip_auc": float(np.nanmean(wa)) if wa.size else float("nan"),
                   "top1_on_mixed": float(np.nanmean(wt)) if wt.size else float("nan"),
                   "mean_score_correct_minus_incorrect":
                       float(fpos.mean() - fneg.mean()) if fpos.size and fneg.size
                       else float("nan"),
                   "n_mixed_scored": int(wa.size)}

    # Every scorer is on trial against BOTH unranked floors: the random frame (its paired
    # partner) and the positional prior.
    tests = [dict(scorer=a, baseline=b, delta_pts=100 * (acc[a] - acc[b]),
                  **mcnemar(hits[a], hits[b]))
             for a in scorers if a not in FLOOR_SCORERS
             for b in FLOOR_SCORERS if b in scorers]
    # The prize itself: is the oracle's edge over all-frames more than noise?
    headroom_test = dict(scorer="single_best", baseline="all",
                         delta_pts=100 * (acc["single_best"] - acc["all"]),
                         **mcnemar(hits["single_best"], hits["all"]))
    vision_test = dict(scorer="all", baseline="blind",
                       delta_pts=100 * (acc["all"] - acc["blind"]),
                       **mcnemar(hits["all"], hits["blind"]))

    out = {"experiment": "frame_lever_accuracy",
           "backbone": args.backbone, "model_name": args.model_name,
           "data_root": args.data_root, "tasks": args.tasks,
           "num_segments": F, "single_repeat": args.single_repeat,
           "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
           "n_layers": n_layers, "dec_band": band, "queries": args.queries,
           "estimator": "moment", "k_frac": args.k_frac, "seed": args.seed,
           "clip_model": args.clip_model, "scorers": scorers,
           "scoring": "argmax over option-letter tokens", "answer_prefix": ANSWER_PREFIX,
           "n_clips": n_seen, "n_mixed_clips": n_mixed,
           "chance_accuracy": sum(1.0 / s["n_options"] for s in per_sample) / n_seen,
           "anchors": {"all": acc["all"], "blind": acc["blind"],
                       "single_mean": acc["single_mean"],
                       "single_best": acc["single_best"],
                       "single_worst": acc["single_worst"]},
           "selection_headroom_pts": 100 * (acc["single_best"] - acc["all"]),
           "accuracy_by_scorer": {s: acc[s] for s in scorers},
           "delta_vs_single_mean_pts": {s: 100 * (acc[s] - floor) for s in scorers},
           # The share of the oracle's prize a scorer recovers. Undefined when the oracle
           # has no edge over a random frame to begin with.
           "headroom_captured": {s: ((acc[s] - floor) / span if span else float("nan"))
                                 for s in scorers},
           "discrimination": disc,
           "mcnemar_vs_floors": tests,
           "mcnemar_single_best_vs_all": headroom_test,
           "mcnemar_all_vs_blind": vision_test,
           "accuracy_by_position": (pos_correct / n_seen).tolist(),
           "best_position_histogram": best_pos_hist.tolist(),
           "per_task_seen": seen_by_task,
           "accuracy_by_task": {t: dict({c: h[c] / seen_by_task[t] for c in configs},
                                        single_mean=per_task_single_mean[t] / seen_by_task[t])
                                for t, h in sorted(per_task_hits.items())},
           "skipped": skipped}

    print(f"\n==== the frame lever ({n_seen} clips, {F} frames) ====")
    print(f"{args.backbone}: {args.model_name}, {len(seen_by_task)} task(s)")
    print(f"  all           {100 * acc['all']:6.2f}%   <- every frame")
    print(f"  single_best   {100 * acc['single_best']:6.2f}%   <- ORACLE frame "
          f"({headroom_test['delta_pts']:+.2f} pts vs all, p={headroom_test['p_value']:.3g})")
    print(f"  single_mean   {100 * floor:6.2f}%   <- a random frame (the floor)")
    print(f"  single_worst  {100 * acc['single_worst']:6.2f}%")
    print(f"  blind         {100 * acc['blind']:6.2f}%   <- no video "
          f"(vision is worth {vision_test['delta_pts']:+.2f} pts, p={vision_test['p_value']:.3g})")
    print(f"  chance        {100 * out['chance_accuracy']:6.2f}%")
    print(f"\n  selection headroom: single_best - single_mean = {100 * span:+.2f} pts\n")

    print(f"  {'scorer':<12} {'acc':>7} {'vs mean':>8} {'headroom':>9} "
          f"{'pooled':>7} {'within':>7} {'top-1':>7}")
    for s in scorers:
        d = disc[s]
        print(f"  {s:<12} {100 * acc[s]:6.2f}% {100 * (acc[s] - floor):+7.1f} "
              f"{100 * (acc[s] - floor) / span if span else float('nan'):8.0f}% "
              f"{d['pooled_auc']:7.3f} {d['within_clip_auc']:7.3f} "
              f"{100 * d['top1_on_mixed']:6.1f}%")
    print(f"\n  (within-clip AUC over {n_mixed} mixed clips is the metric to trust: "
          f"0.5 = no signal)")

    print("\nMcNemar (exact, two-sided) against the unranked floors:")
    for t in tests:
        flag = "significant" if t["p_value"] < 0.05 else "n.s."
        print(f"  {t['scorer']:>12s} - {t['baseline']:<8s} {t['delta_pts']:+6.2f} pts  "
              f"(b={t['b']:>4d} c={t['c']:>4d}, p={t['p_value']:.3g}, {flag})")

    print("\naccuracy by frame position (is the answer frame positionally predictable?):")
    print("  " + " ".join(f"{100 * v / n_seen:.1f}" for v in pos_correct))
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s)")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
        xs = np.arange(len(scorers))
        ax0.bar(xs, [100 * acc[s] for s in scorers], color="C0")
        for y, ls, lab in ((100 * ceiling, "-", "single_best (oracle)"),
                           (100 * acc["all"], "--", "all frames"),
                           (100 * floor, ":", "single_mean (floor)"),
                           (100 * acc["blind"], "-.", "blind")):
            ax0.axhline(y, color="gray", lw=0.9, ls=ls, label=lab)
        ax0.set_xticks(xs, scorers, rotation=45, ha="right")
        ax0.set_ylabel("accuracy (%)")
        ax0.set_title(f"which frame each scorer picks ({n_seen} clips, {F}f)")
        ax0.legend(frameon=False, fontsize=7)

        ax1.bar(xs, [disc[s]["within_clip_auc"] for s in scorers], color="C2")
        ax1.axhline(0.5, color="gray", lw=0.9, ls=":", label="no signal")
        ax1.set_xticks(xs, scorers, rotation=45, ha="right")
        ax1.set_ylabel("within-clip AUC")
        ax1.set_ylim(0.3, 0.8)
        ax1.set_title(f"does the score mark an answerable frame? ({n_mixed} mixed clips)")
        ax1.legend(frameon=False, fontsize=7)
        fig.suptitle(os.path.basename(args.model_name))
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(
        description="The frame lever: single-frame sufficiency, and whether any cheap "
                    "per-frame score picks the answerable frame, on MVBench / EgoSchema.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' (all 20 MVBench tasks) / 'all' (+ EgoSchema). "
                        "MVBench and EgoSchema live under different roots -- don't mix in one run.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_ov", "llava"], default="auto",
                   help="auto infers from --model_name (onevision/-ov- -> llava_ov, else "
                        "'llava' -> llava-1.5, else qwen).")
    p.add_argument("--scorers", nargs="*", default=DEFAULT_SCORERS,
                   help=f"any of {CHEAP_SCORERS + LLM_SCORERS + CTX_SCORERS + FLOOR_SCORERS}; "
                        f"{CLIP_SCORERS} are added automatically by --clip_model.")
    p.add_argument("--clip_model", default=None,
                   help="optional contrastive model for the aligned-space selector, e.g. "
                        "google/siglip-base-patch16-224. Loads a SECOND model.")
    p.add_argument("--single_repeat", type=int, default=2,
                   help="frames in the zero-motion clip that carries one frame (default 2: "
                        "Qwen's temporal merge needs an even count).")
    p.add_argument("--dec_band", default="11:15",
                   help="0-indexed decoder layers the in-LLM scores read (default 11:15, the "
                        "paper's 12-16 and stage_topk_accuracy.py's mid band). 'a:b' inclusive, "
                        "negatives count from the end, comma-separated; a leading minus needs "
                        "the equals form, --dec_band=-5:-1.")
    p.add_argument("--queries", choices=["post", "all", "last"], default="post",
                   help="text query rows the decoder score averages over. post = non-visual "
                        "positions after the visual block, all = every non-visual position, "
                        "last = the answer slot only.")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="upper-tail fraction for the xi_* / dec_xi tail indices.")
    p.add_argument("--num_segments", type=int, default=None,
                   help=f"frames sampled per clip. Default per backbone: {DEFAULT_SEGMENTS}.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="qwen only: per-frame pixel cap, e.g. 200704. Both LLaVA backbones "
                        "have a fixed frame size and ignore this.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="qwen only: floor on per-frame pixels. Defaults to --max_pixels; 0 opts out.")
    p.add_argument("--max_samples", type=int, default=None, help="cap on records PER TASK.")
    p.add_argument("--seed", type=int, default=0, help="seeds the random-frame floor, per clip.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="frame_lever_accuracy.json")
    p.add_argument("--per_sample_out", default="", help="default: <--out stem>_per_sample.json")
    p.add_argument("--plot", default=None, help="optional PNG: accuracy + within-clip AUC.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
