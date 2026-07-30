"""
frame_tailindex_select_mvbench.py -- can a TAIL INDEX pick the right frame?

The pivot (Exp 3 `frame-selection-is-the-lever`, Exp 4 `closer-decoder-not-causal`):
token-level importance ranking is dead (pruning by decoder@L* did not beat random),
but frame SELECTION is a real lever -- a well-chosen single frame beats all-frames by
+12.7 pts (single_best 74.9% vs all 62.2%). This asks the natural next question with
our own tool: does the EVT tail index gamma, computed PER FRAME, choose that frame?

Hypothesis. A frame that actually contains the answer should have a *concentrated*
within-frame importance profile -- a sharp salient region (high gamma), versus a flat,
uniform frame (gamma<=0, nothing stands out). If so, argmax_f gamma_f is a cheap,
training-free, PRE-LLM frame selector.

The Exp-4 discipline (do NOT trust gamma, measure accuracy). gamma>0 was necessary but
not sufficient for causal token importance; the same trap applies to frames. So every
scorer here is judged by the accuracy of the frame it PICKS, against two anchors:
  single_mean  -- a typical (random) frame            [floor: beat this or it is useless]
  single_best  -- the oracle "any single frame right" [ceiling: how much headroom taken]
A scorer is useful iff picked-frame acc > single_mean and approaches single_best.

Design (no token->frame mapping needed). Each frame is scored on ITS OWN encoder
tokens: one frame passed as the 2-frame zero-motion clip [f,f] (the frame_sufficiency
protocol) temporally merges to exactly one frame's worth of S post-merger tokens, so
the frame's score uses only the frame's content. All scorers are cheap and PRE-LLM
(vision encoder + question embedding only); the LLM is run only to read the answer
(shared with the accuracy anchors, one forward per frame -- no extra cost).

Per-frame scorers (each -> a length-N vector; pick argmax frame):
  xi_encq   tail index of the frame's encoder x query token scores  [tail-idx, query-aware]  <- headline
  xi_sal    tail index of the frame's token saliency ||v||          [tail-idx, query-free]
  mass_encq mean encoder x query over the frame's tokens            [mass baseline, query-aware]
  motion    mean token novelty 1-cos vs the neighbour frame         [query-free heuristic]
  middle    the middle frame index                                  [content-free positional floor]

With --with_decoder, three STRONG in-LLM references are added (the cheap scorers above all
failed; this asks whether ANY signal picks the frame, or only the oracle does):
  dec_xi    tail index of the frame's ISOLATED decoder@L* importance [tail-idx, in-LLM]
  dec_mass  total decoder@L* attention mass on the frame's tokens    [mass, in-LLM]
  conf      the LLM's single-frame answer-confidence margin          [LLM-in-the-loop]
If conf picks the frame (-> single_best) but the cheap scorers don't => "the LLM knows
which frame, cheap pre-LLM scores don't"; if even conf ~ single_mean => the frame headroom
needs the full oracle. (Isolated per-[f,f] forward: the contextual joint-clip decoder score
is per-PAIR, not per-frame, under Qwen's temporal_patch_size=2 pairing.)

Reads (selection):
  xi_* >> single_mean, -> single_best   => a tail index DOES pick the frame (build it).
  xi_* ~ single_mean                    => tail-index frame selection fails like tokens did.
  xi_* ~ mass_encq                      => concentration adds nothing over plain query-mass.

Reads (discrimination -- higher power than the argmax, uses ALL frames): does a frame's
score correlate with that frame being INDIVIDUALLY correct? Reported as an AUC =
P(score of a correct frame > score of an incorrect frame). Pooled AUC over all frames,
and a within-clip AUC restricted to mixed clips (both a correct and an incorrect frame
present) so per-clip difficulty is controlled. AUC ~ 0.5 => the tail index does not mark
answerable frames; AUC >> 0.5 => a high tail index really does mark the right frame.

Run:
    # cheap pre-LLM scorers only (sdpa, fast)
    python frame_tailindex_select_mvbench.py --data_root ../MVBench/ --num_frames 8 \
        --max_samples 30 --max_pixels 200704
    # + strong in-LLM references (isolated decoder@L* concentration + answer confidence; eager)
    python frame_tailindex_select_mvbench.py --data_root ../MVBench/ --num_frames 8 \
        --max_samples 30 --max_pixels 200704 --with_decoder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))

from inference import (
    DATA_LIST, TEMPORAL_PATCH_SIZE, build_prompt, letter_token_ids, official_frames, predict,
)
from frame_sufficiency_mvbench import frames_prompt, build_inputs
from prune_accuracy_mvbench import question_direction
from anchor_layer_prune import debiased_scores_by_layer, select_anchor_layer, _text_model
from efficient_vlm.utils import einmahlHaan

# scorers that rank frames (references all/single_*/middle are handled separately)
PICK_SCORERS = ["xi_encq", "xi_sal", "mass_encq", "motion"]


def frame_forward(model, processor, frame, text, letter_ids, args, band, text_model):
    """One forward on the [f,f] zero-motion clip (temporal_patch_size merges the pair to
    ONE frame's S post-merger tokens). Returns:
      pred  -- predicted option letter (argmax over option-letter logits),
      conf  -- answer-confidence margin (top - second option logit); the LLM's own
               single-frame confidence, a strong in-LLM frame selector,
      V     -- (S,d) post-merger encoder tokens (from hidden_states[0]; equals
               get_video_features), the input to the cheap xi_*/mass/motion scorers,
      dec   -- (S,) debiased decoder importance at THIS frame's own L* (with --with_decoder),
               an ISOLATED, frame-granularity decoder signal. The contextual joint-clip
               decoder score is per-PAIR (grid_t = N/2 under Qwen's temporal pairing), so
               it cannot be mapped back to single-frame accuracy -- hence the isolated pass.
    """
    inputs = build_inputs(processor, model, frames_prompt([frame, frame], text, args.max_pixels))
    vid_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vidx = (inputs["input_ids"][0] == vid_id).nonzero(as_tuple=False).flatten()
    out = model(**inputs, output_hidden_states=True,
                output_attentions=args.with_decoder, use_cache=False)
    last = out.logits[0, -1]
    V = out.hidden_states[0][0, vidx].float()                 # (S,d) merged visual tokens
    dec = None
    if args.with_decoder:
        tqm = torch.ones(inputs["input_ids"].shape[1], dtype=torch.bool, device=model.device)
        tqm[vidx] = False
        sbl = debiased_scores_by_layer(text_model, out.hidden_states, out.attentions, vidx, tqm, band)
        Lstar, _ = select_anchor_layer(sbl, args.k_frac)
        dec = sbl[Lstar].float().cpu().numpy()                 # (S,) frame's own decoder importance
    opt = [max(last[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    pred = int(np.argmax(opt))
    top = sorted(opt, reverse=True)
    conf = float(top[0] - top[1]) if len(top) > 1 else float("nan")
    del out
    return {"pred": pred, "conf": conf, "V": V, "dec": dec}


def per_frame_scores(V_list, q_hat, k_frac):
    """From the per-frame token matrices, the length-N score vectors. gamma is
    einmahlHaan on the frame's positive token scores; NaN when a frame has too few
    positive samples (<12) -- those frames simply cannot be argmax-picked."""
    N = len(V_list)
    xi_encq = np.full(N, np.nan)
    xi_sal = np.full(N, np.nan)
    mass_encq = np.full(N, np.nan)
    for f, V in enumerate(V_list):
        proj = V @ q_hat                                 # (S,) encoder . unit(question)
        sal = V.norm(dim=-1)                             # (S,) query-free saliency
        xi_encq[f] = einmahlHaan(proj, k_frac)           # tail filters to proj>0 internally
        xi_sal[f] = einmahlHaan(sal, k_frac)
        mass_encq[f] = proj.mean().item()
    # motion: novelty vs the neighbour frame (symmetric at the ends), cell-aligned cosine
    motion = np.full(N, np.nan)
    if N >= 2:
        for f in range(N):
            g = f - 1 if f > 0 else 1                     # neighbour (frame 0 uses frame 1)
            cos = torch.nn.functional.cosine_similarity(V_list[f], V_list[g], dim=-1)
            motion[f] = (1.0 - cos).mean().item()
    return {"xi_encq": xi_encq, "xi_sal": xi_sal, "mass_encq": mass_encq, "motion": motion}


def pick(score_vec):
    """argmax frame ignoring NaN; None if every frame is NaN (scorer abstains)."""
    if np.all(np.isnan(score_vec)):
        return None
    return int(np.nanargmax(score_vec))


def _rankdata(a):
    """Average ranks (ties shared), like scipy.stats.rankdata -- no scipy dep."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.size, dtype=float)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0      # mean of 1-based ranks i..j
        i = j + 1
    return ranks


def _auc(pos, neg):
    """P(score of a positive > score of a negative), ties=0.5 (Mann-Whitney U/n1n0).
    0.5 = no relationship; >0.5 = a higher score marks a positive. O(n log n)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n0 = pos.size, neg.size
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata(np.concatenate([pos, neg]))
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Does a per-frame tail index pick the right frame? (MVBench)")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=8, help="candidate frames (rounded to even).")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--max_samples", type=int, default=30, help="cap samples PER TASK.")
    p.add_argument("--max_pixels", type=int, default=200704)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation for the cheap scorers.")
    p.add_argument("--with_decoder", action="store_true",
                   help="also score each frame by its ISOLATED decoder@L* concentration "
                        "(dec_xi/dec_mass) and answer-confidence margin (conf) -- the strong "
                        "in-LLM references. Forces eager attention (needs attentions).")
    p.add_argument("--band", default="2,3,4,5,6,7,8",
                   help="decoder anchor-candidate band for L* (--with_decoder only).")
    p.add_argument("--out", default="frame_tailindex_select_mvbench.json")
    args = p.parse_args()
    band = [int(x) for x in args.band.split(",")]

    N = max(TEMPORAL_PATCH_SIZE, round(args.num_frames / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    attn_impl = "eager" if args.with_decoder else args.attn   # decoder scores need attentions
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=attn_impl).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    text_model = _text_model(model) if args.with_decoder else None

    # ranking scorers; the strong in-LLM references are added only with --with_decoder
    pick_scorers = list(PICK_SCORERS) + (["dec_xi", "dec_mass", "conf"] if args.with_decoder else [])
    # every reported column; xi_* may abstain, so counts are tracked per key
    cols = ["all", "single_best", "single_mean", "middle"] + pick_scorers
    correct = defaultdict(lambda: defaultdict(float))    # task -> col -> sum correct
    count = defaultdict(lambda: defaultdict(int))        # task -> col -> denominator
    # how often each scorer's pick matches an actually-correct frame's index set is
    # captured by its accuracy; we also log the raw picks for later analysis.
    picks_log = []
    # per-FRAME discrimination: does a frame's score correlate with that frame being
    # individually correct?  pooled score split by frame-correctness (pooled AUC), and
    # per-mixed-clip AUC / top-1 hit (controls for per-clip difficulty).
    disc_pos = {s: [] for s in pick_scorers}     # scores of individually-CORRECT frames
    disc_neg = {s: [] for s in pick_scorers}     # scores of individually-INCORRECT frames
    within_auc = {s: [] for s in pick_scorers}   # one AUC per mixed clip
    within_top1 = {s: [] for s in pick_scorers}  # is the argmax frame a correct one?
    n_mixed = 0
    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=f"{task} (N={N})"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt = build_prompt(rec)
                letter_ids = letter_token_ids(processor, letters)
                frames = official_frames(path, data_type, has_bound, rec, N)   # N PIL frames
                q_hat = question_direction(processor, model, rec["question"], model.device).float()

                singles, V_list, dec_list, conf_list = [], [], [], []
                for f in frames:
                    res = frame_forward(model, processor, f, text, letter_ids, args, band, text_model)
                    singles.append(int(res["pred"] == gt))
                    V_list.append(res["V"])
                    dec_list.append(res["dec"])
                    conf_list.append(res["conf"])
                singles = np.array(singles, dtype=float)

                all_inp = build_inputs(processor, model, frames_prompt(frames, text, args.max_pixels))
                all_ok = int(predict(model, all_inp, letter_ids) == gt)

                scores = per_frame_scores(V_list, q_hat, args.k_frac)
                if args.with_decoder:                      # strong in-LLM references
                    scores["dec_xi"] = np.array(
                        [einmahlHaan(torch.as_tensor(d, dtype=torch.float32), args.k_frac)
                         if d is not None else np.nan for d in dec_list])
                    scores["dec_mass"] = np.array(
                        [float(np.sum(d)) if d is not None else np.nan for d in dec_list])
                    scores["conf"] = np.array(conf_list, dtype=float)
                mid = len(frames) // 2

                # per-frame discrimination: does score track individual-frame correctness?
                cor = singles.astype(bool)
                mixed = bool(cor.any() and (~cor).any())
                n_mixed += int(mixed)
                for s in pick_scorers:
                    sc = scores[s]
                    valid = ~np.isnan(sc)
                    disc_pos[s].extend(sc[valid & cor].tolist())
                    disc_neg[s].extend(sc[valid & ~cor].tolist())
                    if mixed:
                        vs, vc = sc[valid], cor[valid]
                        if vc.any() and (~vc).any():       # both classes survive NaN drop
                            within_auc[s].append(_auc(vs[vc], vs[~vc]))
                            within_top1[s].append(float(vc[int(np.argmax(vs))]))

                # references
                correct[task]["all"] += all_ok;                 count[task]["all"] += 1
                correct[task]["single_best"] += int(singles.any()); count[task]["single_best"] += 1
                correct[task]["single_mean"] += float(singles.mean()); count[task]["single_mean"] += 1
                correct[task]["middle"] += singles[mid];        count[task]["middle"] += 1
                # tail-index / heuristic scorers: accuracy of the frame each one picks
                rec_picks = {}
                for s in pick_scorers:
                    fp = pick(scores[s])
                    if fp is None:                              # abstain: all-NaN
                        continue
                    correct[task][s] += singles[fp]
                    count[task][s] += 1
                    rec_picks[s] = fp
                picks_log.append({"task": task, "video": rec.get("video"), "gt_frames": [
                    i for i, c in enumerate(singles) if c > 0], "picks": rec_picks})
                del V_list
                torch.cuda.empty_cache()
            except Exception as e:
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

    # ---- report ----
    def acc(task, c):
        n = count[task][c]
        return 100.0 * correct[task][c] / n if n else float("nan")

    print(f"\n{'task':26s} {'n':>4} " + " ".join(f"{c:>12s}" for c in cols))
    ntot = 0
    tot_c = defaultdict(float); tot_n = defaultdict(int)
    for task in tasks:
        n = count[task]["all"]
        if not n:
            continue
        ntot += n
        for c in cols:
            tot_c[c] += correct[task][c]; tot_n[c] += count[task][c]
        print(f"{task:26s} {n:>4} " + " ".join(f"{acc(task, c):11.1f}%" for c in cols))
    if ntot:
        def oacc(c):
            return 100.0 * tot_c[c] / tot_n[c] if tot_n[c] else float("nan")
        print(f"{'OVERALL':26s} {ntot:>4} " + " ".join(f"{oacc(c):11.1f}%" for c in cols))
        sm, sb = oacc("single_mean"), oacc("single_best")
        print(f"\n  anchors:  single_mean {sm:.1f}% (random-frame floor)   "
              f"single_best {sb:.1f}% (oracle ceiling)   all {oacc('all'):.1f}%")
        span = sb - sm
        for s in pick_scorers:
            a = oacc(s)
            frac = 100.0 * (a - sm) / span if span else float("nan")
            print(f"  {s:10s} {a:5.1f}%   vs single_mean {a - sm:+.1f}pts (useful?)   "
                  f"vs single_best {a - sb:+.1f}pts   headroom captured {frac:+.0f}%")

    # ---- per-frame discrimination: does a high score mark a CORRECT frame? ----
    # AUC = P(score of a correct frame > score of an incorrect one). 0.5 = no signal;
    # >0.5 = high score -> correct. within-clip AUC restricts to mixed clips (both
    # classes present), so it isolates the frame signal from per-clip difficulty.
    disc = {}
    print(f"\n  --- does a high score mark a CORRECT frame? (n_mixed={n_mixed}) ---")
    for s in pick_scorers:
        pos, neg = np.asarray(disc_pos[s]), np.asarray(disc_neg[s])
        p_auc = _auc(pos, neg)
        dmean = float(pos.mean() - neg.mean()) if pos.size and neg.size else float("nan")
        wa, wt = np.asarray(within_auc[s]), np.asarray(within_top1[s])
        w_auc = float(np.nanmean(wa)) if wa.size else float("nan")
        w_top1 = float(np.nanmean(wt)) if wt.size else float("nan")
        disc[s] = {"pooled_auc": p_auc, "mean_score_correct_minus_incorrect": dmean,
                   "n_correct_frames": int(pos.size), "n_incorrect_frames": int(neg.size),
                   "within_clip_auc": w_auc, "top1_hit_on_mixed": w_top1,
                   "n_mixed_scored": int(wa.size)}
        print(f"  {s:10s} pooled AUC {p_auc:.3f}  (Δmean score corr-incorr {dmean:+.4f})   "
              f"within-clip AUC {w_auc:.3f}   top1-on-mixed {100 * w_top1:.1f}%  (n={wa.size})")

    with open(args.out, "w") as fh:
        json.dump({"num_frames": N, "cols": cols, "n_mixed": n_mixed,
                   "correct": {t: dict(correct[t]) for t in correct},
                   "count": {t: dict(count[t]) for t in count},
                   "discrimination": disc, "picks": picks_log}, fh, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
