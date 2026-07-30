"""
frame_attn_changepoint_mvbench.py -- Exp 6: does a TEMPORAL CHANGE-POINT in the
query-guided patch attention localize the answerable frame?

Motivation (where the arc left off). Frame SELECTION is the real lever (Exp 3), but no
cheap PRE-LLM signal picks the frame (Exp 5): every scorer tested was a per-frame LEVEL
statistic -- "which frame has the most concentrated / highest-mass / highest-motion
attention" -- and all sat at the random floor (within-clip AUC ~0.5). The ONE signal that
worked was `conf` (the LLM's own single-frame answer margin): functional, not visual, and
expensive. This experiment tries the one axis Exp 5 never touched: not the LEVEL of the
attention field at a frame, but its TEMPORAL DERIVATIVE across frames -- a change-point.

Hypothesis. An event referenced by the question ("what does he do AFTER he stands up")
produces a DISRUPTION in the aggregated query-guided patch attention: the attention mass
shifts (off the bench, onto the new action) at the moment of the event. The answerable
position is at/next to that disruption. If argmax_t |a[t]-a[t-1]| marks a correct
position, a temporal change-point is a cheap frame localizer that a level statistic misses.

Three prior results are the threats this is designed against, and must be read WITH it:
  * Query-invariance (Exp 2b): the decoder kept-set is largely question-INVARIANT (a
    visual-saliency prior), so most of the trajectory's motion is scene/motion change, and
    `motion` already failed in Exp 5. The question-dependent residual is thin and lives on
    CLEVRER multi-object tasks -- so a WIN here is most plausible exactly on temporal/action
    tasks, and a null is the query-invariance prior showing through.
  * Attention != causal (Exp 4, THE CLOSER): reading attention MAGNITUDE (even its change)
    is CORRELATIONAL. This script is the cheap correlational test; the causal version
    (per-position patch ABLATION -> Delta-logit trajectory -> change-point) is a follow-up,
    warranted only if this survives. Do not call the change-point "causal".
  * Level vs change is the actual novelty: `attn_level` (the in-context mass a[t]) is scored
    alongside the change-points so the comparison is on ONE trajectory. attn_level ~ dec_mass
    (Exp 5, failed); if cp_* also ~0.5 the temporal-derivative axis is closed too.

Native granularity. Qwen2.5-VL pairs adjacent frames (TEMPORAL_PATCH_SIZE=2), so a full-clip
forward exposes only T = N/2 temporal POSITIONS, not N frames -- there is no per-frame
attention. Candidates are therefore the T native positions; each position is scored for
accuracy by an ISOLATED forward on its real frame-pair (frames[2t], frames[2t+1]) -- the
model's own smallest temporal unit, cleaner than Exp 5's [f,f] duplicate. Because T is
coarse (N=8 -> 4 positions -> 3 edges), run this at HIGHER N (default 16) and/or on long-form
EgoSchema (--tasks EgoSchema), where a change-point has temporal resolution to resolve.

The trajectory. One full-clip forward (eager, attentions) -> the debiased query-guided
importance s_t = mean_q mean_e A[q,t]*||v_t|| at the anchor layer L* (anchor_layer_prune's
`debiased_scores_by_layer` -- attention FROM the prompt TO the patches, the exact quantity
the change-point idea is about) -> aggregate over each temporal position's tokens (token_grid
gives the (frame,row,col) layout) -> a[t], t=0..T-1.

Scorers (each -> length-T vector; pick argmax position):
  cp_post     |a[t]-a[t-1]| assigned to the LATER position t     [change-point, "after"]
  cp_pre      |a[t]-a[t-1]| assigned to the EARLIER position t-1 [change-point, "before"]
  attn_level  a[t] itself                                        [in-context mass; level baseline]
  conf        isolated per-position answer margin                [LLM-in-the-loop; the Exp-5 winner]

Anchors: all (full clip) / single_mean (typical position, random floor) / single_best (any
position correct, oracle) / middle position. A scorer is useful iff picked-position acc >
single_mean and approaches single_best. AUC = P(score of a correct position > score of an
incorrect one); within-clip AUC (mixed clips only) controls per-clip difficulty. The
load-bearing read is the PER-TASK within-clip AUC: cp_* should, if anything, beat `conf` on
the temporal tasks (Action Localization, Moving Direction) where conf lost in Exp 5.

Run:
    # 8 positions per clip (needs eager for the decoder attentions)
    python frame_attn_changepoint_mvbench.py --data_root ../MVBench/ --num_frames 16 \
        --max_samples 30 --max_pixels 200704
    # long-form (more temporal resolution -- where a change-point matters most)
    python frame_attn_changepoint_mvbench.py --data_root ~/Experiments/EgoSchema \
        --tasks EgoSchema --num_frames 16 --max_pixels 200704
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
    DATA_LIST, TEMPORAL_PATCH_SIZE, build_prompt, letter_token_ids, official_frames,
)
from frame_sufficiency_mvbench import frames_prompt, build_inputs
from anchor_layer_prune import (
    debiased_scores_by_layer, select_anchor_layer, _text_model, token_grid,
)
from frame_tailindex_select_mvbench import _auc, pick   # reuse the Exp-5 AUC + argmax helpers

PICK_SCORERS = ["cp_post", "cp_pre", "attn_level", "conf"]


def _read_answer(logits_last, letter_ids):
    """(pred option index, confidence margin) from the final-position logits, using the
    same option-letter readout as inference.predict / Exp 5's frame_forward."""
    opt = [max(logits_last[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    pred = int(np.argmax(opt))
    top = sorted(opt, reverse=True)
    conf = float(top[0] - top[1]) if len(top) > 1 else float("nan")
    return pred, conf


def position_forward(model, processor, pair, text, letter_ids, args):
    """Isolated forward on ONE native temporal position = a real adjacent frame-pair
    (frames[2t], frames[2t+1]); TEMPORAL_PATCH_SIZE merges the pair to that position's S
    tokens. Returns (pred, conf) -- the per-position accuracy unit and the `conf` scorer.
    No attentions needed here, but the model is loaded eager for the clip trajectory."""
    inputs = build_inputs(processor, model, frames_prompt(list(pair), text, args.max_pixels))
    out = model(**inputs, use_cache=False)
    pred, conf = _read_answer(out.logits[0, -1], letter_ids)
    del out
    return pred, conf


def clip_trajectory(model, processor, frames, text, letter_ids, args, band, text_model, merge_size):
    """One full-clip forward (eager, attentions) -> (all-frames pred, trajectory a[T]).
    a[t] = total query-guided debiased importance on temporal position t's tokens at the
    anchor layer L* -- the aggregated patch attention whose temporal CHANGE the scorers read."""
    inputs = build_inputs(processor, model, frames_prompt(frames, text, args.max_pixels))
    vid_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vidx = (inputs["input_ids"][0] == vid_id).nonzero(as_tuple=False).flatten()
    out = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=False)
    pred, _ = _read_answer(out.logits[0, -1], letter_ids)

    tqm = torch.ones(inputs["input_ids"].shape[1], dtype=torch.bool, device=model.device)
    tqm[vidx] = False                                       # every non-visual (prompt) position
    sbl = debiased_scores_by_layer(text_model, out.hidden_states, out.attentions, vidx, tqm, band)
    Lstar, _ = select_anchor_layer(sbl, args.k_frac)
    dec = sbl[Lstar].float()                                # (M,) importance per visual token

    f, _, _, T, _ = token_grid(inputs["video_grid_thw"].to(model.device), merge_size)
    a = np.array([float(dec[f == t].sum()) for t in range(T)])   # (T,) aggregated per position
    del out
    return pred, a, int(T)


def changepoint_scores(a):
    """From the trajectory a[T], the length-T score vectors. |a[t]-a[t-1]| is an EDGE
    between positions t-1 and t; cp_post credits the later node t (the "after" frame),
    cp_pre the earlier node t-1. Endpoints with no incoming/outgoing edge are NaN (they
    simply cannot be argmax-picked, handled by `pick`)."""
    T = len(a)
    cp_post = np.full(T, np.nan)
    cp_pre = np.full(T, np.nan)
    if T >= 2:
        d = np.abs(np.diff(a))                              # d[i] = |a[i+1]-a[i]|, edge i<->i+1
        cp_post[1:] = d                                     # edge -> later node i+1
        cp_pre[:-1] = d                                     # edge -> earlier node i
    return {"cp_post": cp_post, "cp_pre": cp_pre, "attn_level": np.asarray(a, dtype=float)}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Does a change-point in query-guided patch attention pick the frame?")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=16,
                   help="candidate frames (rounded to even); T=N/2 native positions -- use >=16 for resolution.")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--max_samples", type=int, default=30, help="cap samples PER TASK.")
    p.add_argument("--max_pixels", type=int, default=200704)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--band", default="2,3,4,5,6,7,8", help="decoder anchor-candidate band for L*.")
    p.add_argument("--out", default="frame_attn_changepoint_mvbench.json")
    args = p.parse_args()
    band = [int(x) for x in args.band.split(",")]

    N = max(TEMPORAL_PATCH_SIZE, round(args.num_frames / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(     # eager: decoder attentions
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    text_model = _text_model(model)
    merge_size = model.config.vision_config.spatial_merge_size

    pick_scorers = list(PICK_SCORERS)
    cols = ["all", "single_best", "single_mean", "middle"] + pick_scorers
    correct = defaultdict(lambda: defaultdict(float))
    count = defaultdict(lambda: defaultdict(int))
    picks_log = []
    disc_pos = {s: [] for s in pick_scorers}     # scores of individually-CORRECT positions
    disc_neg = {s: [] for s in pick_scorers}     # scores of individually-INCORRECT positions
    within_auc = {s: [] for s in pick_scorers}   # one AUC per mixed clip (pooled over tasks)
    within_auc_task = {s: defaultdict(list) for s in pick_scorers}   # per-task mixed-clip AUCs
    within_top1 = {s: [] for s in pick_scorers}  # is the argmax position a correct one?
    n_mixed = 0
    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        try:                                    # a task json absent under this data_root
            with open(os.path.join(json_dir, fname)) as fh:   # (e.g. EgoSchema under MVBench/)
                records = json.load(fh)
        except (FileNotFoundError, OSError) as e:
            tqdm.write(f"skip task [{task}]: {e}")
            continue
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=f"{task} (N={N}, T={N // TEMPORAL_PATCH_SIZE})"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt = build_prompt(rec)
                letter_ids = letter_token_ids(processor, letters)
                frames = official_frames(path, data_type, has_bound, rec, N)   # N PIL frames

                # per-position accuracy + conf: isolated forward on each native frame-pair
                pairs = [(frames[2 * t], frames[2 * t + 1]) for t in range(N // TEMPORAL_PATCH_SIZE)]
                singles, conf_list = [], []
                for pr in pairs:
                    pred, conf = position_forward(model, processor, pr, text, letter_ids, args)
                    singles.append(int(pred == gt))
                    conf_list.append(conf)
                singles = np.array(singles, dtype=float)

                # full-clip trajectory + all-frames answer (one forward)
                all_pred, a, T = clip_trajectory(
                    model, processor, frames, text, letter_ids, args, band, text_model, merge_size)
                if T != len(singles):        # temporal-pairing sanity: trajectory must match positions
                    raise RuntimeError(f"trajectory T={T} != {len(singles)} positions")
                all_ok = int(all_pred == gt)

                scores = changepoint_scores(a)
                scores["conf"] = np.array(conf_list, dtype=float)
                mid = T // 2

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
                        if vc.any() and (~vc).any():
                            w = _auc(vs[vc], vs[~vc])
                            within_auc[s].append(w)
                            within_auc_task[s][task].append(w)
                            within_top1[s].append(float(vc[int(np.argmax(vs))]))

                correct[task]["all"] += all_ok;                     count[task]["all"] += 1
                correct[task]["single_best"] += int(singles.any());  count[task]["single_best"] += 1
                correct[task]["single_mean"] += float(singles.mean()); count[task]["single_mean"] += 1
                correct[task]["middle"] += singles[mid];             count[task]["middle"] += 1
                rec_picks = {}
                for s in pick_scorers:
                    fp = pick(scores[s])
                    if fp is None:
                        continue
                    correct[task][s] += singles[fp]
                    count[task][s] += 1
                    rec_picks[s] = fp
                picks_log.append({"task": task, "video": rec.get("video"),
                                  "gt_positions": [i for i, c in enumerate(singles) if c > 0],
                                  "traj": a.tolist(), "picks": rec_picks})
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
        print(f"\n  anchors:  single_mean {sm:.1f}% (random-position floor)   "
              f"single_best {sb:.1f}% (oracle ceiling)   all {oacc('all'):.1f}%")
        span = sb - sm
        for s in pick_scorers:
            a_ = oacc(s)
            frac = 100.0 * (a_ - sm) / span if span else float("nan")
            print(f"  {s:10s} {a_:5.1f}%   vs single_mean {a_ - sm:+.1f}pts (useful?)   "
                  f"vs single_best {a_ - sb:+.1f}pts   headroom captured {frac:+.0f}%")

    # ---- discrimination: does a high change-point mark a CORRECT position? ----
    disc = {}
    print(f"\n  --- does a high score mark a CORRECT position? (n_mixed={n_mixed}) ---")
    for s in pick_scorers:
        pos, neg = np.asarray(disc_pos[s]), np.asarray(disc_neg[s])
        p_auc = _auc(pos, neg)
        dmean = float(pos.mean() - neg.mean()) if pos.size and neg.size else float("nan")
        wa, wt = np.asarray(within_auc[s]), np.asarray(within_top1[s])
        w_auc = float(np.nanmean(wa)) if wa.size else float("nan")
        w_top1 = float(np.nanmean(wt)) if wt.size else float("nan")
        disc[s] = {"pooled_auc": p_auc, "mean_score_correct_minus_incorrect": dmean,
                   "n_correct_pos": int(pos.size), "n_incorrect_pos": int(neg.size),
                   "within_clip_auc": w_auc, "top1_hit_on_mixed": w_top1,
                   "n_mixed_scored": int(wa.size)}
        print(f"  {s:10s} pooled AUC {p_auc:.3f}  (Δmean corr-incorr {dmean:+.4f})   "
              f"within-clip AUC {w_auc:.3f}   top1-on-mixed {100 * w_top1:.1f}%  (n={wa.size})")

    # per-task within-clip AUC -- the load-bearing read: does a change-point (cp_post/cp_pre)
    # beat conf on the TEMPORAL tasks (Action Localization, Moving Direction) where conf lost
    # in Exp 5? A per-task mean over that task's mixed-clip AUCs (n = mixed clips in the task).
    per_task_auc = {}
    seen_tasks = [t for t in tasks if count[t]["all"]]
    if seen_tasks:
        print("\n  --- per-task within-clip AUC (mixed clips; the temporal-task read) ---")
        print(f"    {'task':26s} {'nmix':>5} " + " ".join(f"{s:>10s}" for s in pick_scorers))
        for task in seen_tasks:
            row, nmix = {}, 0
            cells = []
            for s in pick_scorers:
                wa = np.asarray(within_auc_task[s].get(task, []))
                nmix = max(nmix, wa.size)
                v = float(np.nanmean(wa)) if wa.size else float("nan")
                row[s] = {"within_clip_auc": v, "n_mixed": int(wa.size)}
                cells.append(f"{v:10.3f}")
            per_task_auc[task] = row
            print(f"    {task:26s} {nmix:>5} " + " ".join(cells))

    with open(args.out, "w") as fh:
        json.dump({"num_frames": N, "cols": cols, "n_mixed": n_mixed,
                   "correct": {t: dict(correct[t]) for t in correct},
                   "count": {t: dict(count[t]) for t in count},
                   "discrimination": disc, "per_task_within_auc": per_task_auc,
                   "picks": picks_log}, fh, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
