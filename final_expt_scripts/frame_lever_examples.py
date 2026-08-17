"""
frame_lever_examples.py -- pull the SHOWABLE cases out of frame_lever_accuracy.py's
per-sample dump: clips the model gets WRONG with all F frames but RIGHT from one frame
alone, rendered as a contact sheet with the question, the options and the gold answer --
and, with --rerun, WHERE inside those frames the text attends, all frames vs that frame
alone, plus what the model answers with no vision at all.

The bar chart says single_best > all by ~8 points on EgoSchema/OneVision. This script
turns rows of that gap into pictures. Nothing here re-derives a metric: the correctness
vector comes straight out of <run>_per_sample.json ("frame_correct" / "hit_by_config"),
so the example is guaranteed to be the same clip the plotted number came from.

Selection (--select)
--------------------
The default `gap` is the clips with hit_by_config["all"] == 0 and ["single_best"] == 1 --
the whole clip fails, some single frame succeeds -- ranked by how FEW frames are correct,
because a clip where exactly one of sixteen frames carries the answer is the sharpest
statement of the claim. But every other slice is available: `all` (every clip in the
dump), `mixed`, `all_frames_wrong`, `all_frames_right`, `reverse_gap` (EVERY frame answers
alone and the full clip still fails -- the case that cannot be blamed on selection),
`no_frame_right`, `every_frame_right`, `blind_beats_video`.

--summary / --summary_only: all 500 clips, frame by frame
---------------------------------------------------------
Reads the dump only -- no model, no video decoding, a second for the whole sweep -- and
writes which of the F frames answers every clip alone, plus four things the headline
table cannot say:

    frame_correct_matrix.csv  one row per clip: the F 0/1s, the correct frame indices and
                              the clip's anchors. The sweep in spreadsheet form.
    summary_matrix.png        the same as a clip x frame grid, rows sorted by how many
                              frames answer -- the picture "single_best > all" is made of.
    summary_panels.png        (a) the DISTRIBUTION of how many frames answer per clip;
                              (b) accuracy by frame position, all clips vs the gap clips;
                              (c) do the answerable frames sit TOGETHER in time? Observed
                                  runs of consecutive correct frames against the same
                                  frames shuffled within the clip, permutation p-value --
                                  clustering means the evidence is a locatable moment,
                                  no clustering means the oracle is a lottery over F draws
                                  and there is nothing to select;
                              (d) each scorer's within-clip AUC on all mixed clips vs on
                                  the clips where selection would actually pay. A scorer
                                  can be informative on the former and useless on the
                                  latter, which is the difference between a selector and
                                  a decoration.

--rerun: the three things the dump cannot store
-----------------------------------------------
The dump stores CORRECTNESS, not letters and not attention, so without a model this
script can say "all frames was wrong" but not "all frames answered (B)". --rerun redoes
1 + F forwards for the chosen clips only -- the full clip, and each frame as the same
[f,f] zero-motion clip frame_lever_accuracy.py used -- and adds:

  1. LETTERS. What each configuration actually answered. The recovered correctness vector
     is re-checked against the stored one; a mismatch means the frames or the prompt have
     drifted and is printed loudly rather than silently rendered.

  2. WHICH VISION TOKENS THE TEXT READS. The same mid-band decoder score every other
     script in the family uses (--dec_band, --queries), sliced per frame and reshaped onto
     the frame's patch grid, so the top-K patches can be drawn on the pixels. Two maps per
     frame: one from the FULL-CLIP forward and one from that frame's ISOLATED forward.
     Comparing them is the point -- the isolated map is what the model looks at when it
     answers correctly, the full-clip map is what it looks at when the same evidence is
     sitting in a 16-frame sequence.

     Both maps are renormalised INSIDE the frame before they are compared. Attention rows
     sum to one over all keys, and the two forwards have wildly different sequence lengths
     (~3.1k visual tokens vs ~390), so raw masses are not comparable and any "attention
     dropped" read off them would be an artefact of the length. What IS comparable is the
     shape within the frame (Spearman rho, top-K overlap, tail index) and, separately, the
     frame's SHARE of the clip's visual mass against the uniform 1/F.

     `--attn_control` runs the identical comparison on a wrong frame from the same clip.
     Without it, a difference between the two maps cannot be attributed to answerability
     rather than to the sequence length -- so it is on by default.

  3. THE BLIND ANSWER. The same clip with every visual token dropped at the input, the
     surviving text keeping its original position ids -- byte-for-byte the operation
     behind stage_topk_accuracy.py's text-only floor and frame_lever_accuracy.py's
     `blind` anchor. If the blind answer equals the all-frames answer, that clip's failure
     is the language prior surviving 3k visual tokens, which is a different disease from
     "the model looked at the wrong frame" and wants a different fix.

Outputs, per example, under --out_dir:
    <video>_<qidx>_sheet.png         the F frames, green = alone correct, + question
    <video>_<qidx>_f<NN>.png         each correct frame at full resolution
    <video>_<qidx>_attn_all.png      full-clip forward: attention overlay on every frame
    <video>_<qidx>_attn_single.png   each frame's own isolated forward, same overlay
    <video>_<qidx>_attn_compare.png  raw | full-clip | isolated, for the correct frame(s)
    examples.json                    everything above, machine-readable

Run
---
    # the whole sweep, frame by frame -- needs nothing but the dump
    python frame_lever_examples.py \
        --per_sample ../final_expt_results/frame_level_accuracy/frame_ego_llavaov_per_sample.json \
        --summary_only --out_dir ../final_expt_results/frame_level_accuracy/examples

    # no model: frames + question + which frames are correct  (CPU, seconds)
    python frame_lever_examples.py \
        --per_sample ../final_expt_results/frame_level_accuracy/frame_ego_llavaov_per_sample.json \
        --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --out_dir ../final_expt_results/frame_level_accuracy/examples

    # letters + attention maps + the blind answer
    python frame_lever_examples.py \
        --per_sample ../final_expt_results/frame_level_accuracy/frame_ego_llavaov_per_sample.json \
        --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --rerun \
        --n_examples 3 --out_dir ../final_expt_results/frame_level_accuracy/examples

    # one specific clip
    python frame_lever_examples.py ... --video 080eb552-9103-4f3e-a7e9-3417c1c0bcec.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import textwrap

import numpy as np
import torch

from frame_lever_accuracy import band_score, frame_spans
from lv_knockout_accuracy import gold_index, option_token_ids
from stage_topk_accuracy import attach_capture, keep_abs_idx, parse_band, predict_pruned
from tail_vs_layer import (ANSWER_PREFIX, DATA_LIST, DEFAULT_SEGMENTS, LLAVA_FAMILY,
                           MVBENCH_TASKS, build_inputs, build_question, infer_backbone,
                           iter_clips, load_model, moment_tail_index, sample_frames,
                           text_model_of, visual_query_masks)

LETTERS = [chr(ord("A") + i) for i in range(26)]


# --------------------------------------------------------------------------- #
# 1. Which clips to show
# --------------------------------------------------------------------------- #
SELECTORS = {
    # the headline case: the whole clip fails, some single frame succeeds
    "gap": lambda r: r["hit_by_config"]["all"] == 0 and r["hit_by_config"]["single_best"] == 1,
    "all": lambda r: True,
    "mixed": lambda r: r["mixed"],
    "all_frames_wrong": lambda r: r["hit_by_config"]["all"] == 0,
    "all_frames_right": lambda r: r["hit_by_config"]["all"] == 1,
    # the reverse gap: every frame answers alone and the full clip still fails
    "reverse_gap": lambda r: r["hit_by_config"]["all"] == 0 and r["hit_by_config"]["single_worst"] == 1,
    "no_frame_right": lambda r: r["hit_by_config"]["single_best"] == 0,
    "every_frame_right": lambda r: r["hit_by_config"]["single_worst"] == 1,
    "blind_beats_video": lambda r: r["hit_by_config"]["blind"] == 1 and r["hit_by_config"]["all"] == 0,
}


def candidates(per_sample: list[dict], args) -> list[dict]:
    """The clips to render, ordered so the most striking one comes first.

    `hit_by_config` is the same dict the accuracy table sums over, so filtering on it here
    reproduces the plotted numbers exactly -- with --select gap the row count IS the
    "single_best - all" gap, in clips, and with --select all it is the whole sweep."""
    rows = [r for r in per_sample if SELECTORS[args.select](r)]
    if args.video:
        rows = [r for r in rows if r["video"] == args.video]
    if args.question_idx:
        rows = [r for r in rows if str(r.get("question_idx")) == str(args.question_idx)]
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    key = (lambda r: -sum(r["frame_correct"])) if args.order == "all_wrong_margin" \
        else (lambda r: sum(r["frame_correct"]))
    return sorted(rows, key=key)


def record_index(args) -> dict:
    """(task, video, question_idx) -> (record, path, data_type, bound), from the same
    json/ the run read, so the question text and the option order match the labels the
    per-sample dump was scored against."""
    out = {}
    for task, rec, path, data_type, bound in iter_clips(args):
        out[(task, rec["video"], str(rec.get("question_idx")))] = (rec, path, data_type, bound)
    return out


# --------------------------------------------------------------------------- #
# 2. Every clip, frame by frame (--summary; needs no model and no video)
# --------------------------------------------------------------------------- #
GREEN, RED, BLUE, ORANGE = "#1a9850", "#d73027", "#2c7fb8", "#e08214"


def frame_matrix(per_sample: list[dict]) -> np.ndarray:
    """(n_clips, F) of 0/1: does frame f, alone, answer clip i's question?

    This is the whole experiment in one object -- `all`, `single_best`, `single_mean` and
    every scorer's accuracy are functions of it plus the per-clip anchors, so the summary
    below re-derives them from here rather than trusting a second copy."""
    lens = {len(r["frame_correct"]) for r in per_sample}
    if len(lens) != 1:
        raise RuntimeError(f"clips carry different frame counts {sorted(lens)} -- one dump per run")
    return np.array([r["frame_correct"] for r in per_sample], dtype=int)


def _n_runs(mat: np.ndarray) -> np.ndarray:
    """Per row, the number of maximal RUNS of correct frames: a 0->1 transition starts one.
    One run means the answerable frames are one contiguous stretch of time."""
    padded = np.concatenate([np.zeros((mat.shape[0], 1), dtype=int), mat], axis=1)
    return (np.diff(padded, axis=1) == 1).sum(axis=1)


def clustering(mat: np.ndarray, seed: int, n_perm: int) -> dict:
    """Are the frames that answer alone a MOMENT, or scattered luck?

    Restricted to mixed clips (a clip that is all-correct or all-wrong has no arrangement
    to test). Under exchangeability -- the same number of correct frames, in a random
    order -- a clip with c correct of F expects c(F-c+1)/F runs. Fewer observed runs than
    that means the answerable frames sit together in time, which is what "the evidence is
    an event in the video" predicts; equal means single-frame correctness is a coin flip
    per frame and the oracle is a lottery over F draws, not a locatable thing to select.

    The p-value is a permutation test on the same clips (shuffle each clip's own vector),
    so it conditions on both the number of clips and each clip's difficulty."""
    F = mat.shape[1]
    c = mat.sum(1)
    sel = mat[(c > 0) & (c < F)]
    if not sel.size:
        return {"n_mixed": 0}
    obs = float(_n_runs(sel).mean())
    cs = sel.sum(1)
    exp = float(np.mean(cs * (F - cs + 1) / F))
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        shuf = np.take_along_axis(sel, np.argsort(rng.random(sel.shape), axis=1), axis=1)
        perm[i] = _n_runs(shuf).mean()
    # Lag-1: given a frame answers alone, does the NEXT frame answer too? Compared with the
    # unconditional rate on the same clips, this is the same clustering read as a probability.
    pair = float((sel[:, :-1] * sel[:, 1:]).sum() / max(sel[:, :-1].sum(), 1))
    return {"n_mixed": int(sel.shape[0]),
            "mean_runs_observed": obs, "mean_runs_expected": exp,
            "p_fewer_runs_than_chance": float((np.sum(perm <= obs) + 1) / (n_perm + 1)),
            "p_next_frame_correct_given_correct": pair,
            "base_rate_on_these_clips": float(sel.mean()),
            "n_permutations": int(n_perm)}


def scorer_auc_by_group(rows: list[dict], scorers: list[str]) -> dict:
    """Each scorer's within-clip AUC and top-1 hit rate on whatever subset `rows` is.

    Run on the mixed clips as a whole and again on the gap clips, this answers the
    question the headline table cannot: a scorer's AUC is earned over ALL mixed clips,
    but the only clips where picking a frame would actually buy accuracy are the ones the
    full clip gets wrong. A scorer can be informative on the former and useless on the
    latter, and that is the difference between a usable selector and a decoration."""
    out = {}
    for s in scorers:
        au, top1 = [], []
        for r in rows:
            v = np.array([np.nan if z is None else z for z in r["scores"][s]], dtype=float)
            corr = np.array(r["frame_correct"], dtype=bool)
            if not corr.any() or corr.all():
                continue
            au.append(auc(v[corr], v[~corr]))
            fin = np.where(np.isfinite(v), v, -np.inf)
            top1.append(float(corr[int(np.argmax(fin))]))
        out[s] = {"within_clip_auc": float(np.nanmean(au)) if au else float("nan"),
                  "top1": float(np.mean(top1)) if top1 else float("nan"),
                  "n_clips": len(au)}
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a correct frame > score of an incorrect one), ties half -- the same
    Mann-Whitney form frame_lever_accuracy.py reports, so the numbers are comparable."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return float("nan")
    r = _ranks(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def summarize(per_sample: list[dict], args) -> dict:
    """The whole sweep, frame by frame: which of the F frames answer each clip alone, what
    that distribution looks like, whether the answerable frames cluster in time, and how
    much of each scorer's signal survives on the clips where selection would actually pay.

    Everything here reads the dump only -- no model, no video decoding -- so it covers all
    500 clips in a second and can be re-run against any other backbone's dump for free."""
    mat = frame_matrix(per_sample)
    n, F = mat.shape
    c = mat.sum(1)
    hit = {k: np.array([r["hit_by_config"][k] for r in per_sample], dtype=int)
           for k in ("all", "blind", "single_best", "single_worst")}
    gap = (hit["all"] == 0) & (hit["single_best"] == 1)
    scorers = sorted(per_sample[0]["scores"].keys()) if per_sample[0].get("scores") else []

    summary = {
        "n_clips": n, "n_frames": F,
        "anchors": {"all": float(hit["all"].mean()), "blind": float(hit["blind"].mean()),
                    "single_mean": float(mat.mean()),
                    "single_best": float(hit["single_best"].mean()),
                    "single_worst": float(hit["single_worst"].mean())},
        "clips_by_n_correct_frames": np.bincount(c, minlength=F + 1).tolist(),
        "clips_by_n_correct_frames_all_right":
            np.bincount(c[hit["all"] == 1], minlength=F + 1).tolist(),
        "n_gap_clips": int(gap.sum()),
        "accuracy_by_position": (mat.mean(0)).tolist(),
        "accuracy_by_position_gap_clips": (mat[gap].mean(0)).tolist() if gap.any() else None,
        "temporal_clustering": clustering(mat, args.seed, args.n_perm),
        "scorers_on_mixed": scorer_auc_by_group([r for r in per_sample if r["mixed"]],
                                                scorers) if scorers else {},
        "scorers_on_gap_clips": scorer_auc_by_group([r for r, g in zip(per_sample, gap) if g],
                                                    scorers) if scorers else {},
    }

    print(f"\n==== every clip, frame by frame ({n} clips x {F} frames) ====")
    a = summary["anchors"]
    print(f"  all {100 * a['all']:.2f}%   single_best {100 * a['single_best']:.2f}%   "
          f"single_mean {100 * a['single_mean']:.2f}%   single_worst "
          f"{100 * a['single_worst']:.2f}%   blind {100 * a['blind']:.2f}%")
    print(f"\n  how many of the {F} frames answer alone:")
    for k, v in enumerate(summary["clips_by_n_correct_frames"]):
        if v:
            r = summary["clips_by_n_correct_frames_all_right"][k]
            print(f"    {k:>2} frame(s): {v:>4} clips   ({r} of them the full clip also gets right)")
    cl = summary["temporal_clustering"]
    if cl.get("n_mixed"):
        print(f"\n  temporal clustering on {cl['n_mixed']} mixed clips: "
              f"{cl['mean_runs_observed']:.2f} runs of correct frames vs "
              f"{cl['mean_runs_expected']:.2f} expected if the same frames were shuffled "
              f"(p={cl['p_fewer_runs_than_chance']:.3g})")
        print(f"  P(next frame correct | this one is) = "
              f"{cl['p_next_frame_correct_given_correct']:.3f} vs base rate "
              f"{cl['base_rate_on_these_clips']:.3f}")
    if scorers:
        print(f"\n  within-clip AUC: all mixed clips -> the {int(gap.sum())} clips where "
              f"selection would actually pay")
        for s in scorers:
            m, g = summary["scorers_on_mixed"][s], summary["scorers_on_gap_clips"][s]
            print(f"    {s:<12} {m['within_clip_auc']:.3f} -> {g['within_clip_auc']:.3f}"
                  f"   (top-1 {100 * m['top1']:.1f}% -> {100 * g['top1']:.1f}%)")

    os.makedirs(args.out_dir, exist_ok=True)
    _write_matrix_csv(per_sample, mat, os.path.join(args.out_dir, "frame_correct_matrix.csv"))
    summary_figures(mat, summary, scorers, args)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out_dir}/summary.json, frame_correct_matrix.csv, "
          f"summary_matrix.png, summary_panels.png")
    return summary


def _write_matrix_csv(per_sample, mat, path):
    """One row per clip: which frames answer it alone, beside the clip's anchors. The
    per-frame columns are the same 0/1 the plotted accuracies are computed from, so this
    is the sweep in the form a spreadsheet or a second script can read."""
    import csv

    F = mat.shape[1]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "video", "question_idx", "gold", "n_options", "all", "blind",
                    "single_best", "single_worst", "n_correct_frames", "correct_frames"]
                   + [f"f{i:02d}" for i in range(F)])
        for r, row in zip(per_sample, mat):
            h = r["hit_by_config"]
            w.writerow([r["task"], r["video"], r.get("question_idx"), r["gold"],
                        r.get("n_options"), h["all"], h["blind"], h["single_best"],
                        h["single_worst"], int(row.sum()),
                        " ".join(str(i) for i in np.flatnonzero(row))] + row.tolist())


def summary_figures(mat, summary, scorers, args):
    """Two figures: the whole sweep as a clip x frame grid, and the four aggregate reads.

    The grid is the honest picture of what "single_best > all" is made of -- one row per
    clip, one cell per frame, green where that frame alone answers. Rows are sorted by how
    many frames answer and then by where the first one is, so any temporal structure would
    show as a diagonal band rather than as noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n, F = mat.shape
    first = np.where(mat.any(1), mat.argmax(1), F)
    order = np.lexsort((first, mat.sum(1)))
    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.imshow(mat[order], aspect="auto", interpolation="nearest",
              cmap=ListedColormap([RED, GREEN]), vmin=0, vmax=1)
    ax.set_xlabel("frame position")
    ax.set_ylabel("clip (sorted by how many frames answer alone)")
    ax.set_xticks(range(F))
    ax.set_title(f"does frame f alone answer clip i?\n{n} clips x {F} frames")
    ax.legend(handles=[Patch(facecolor=GREEN, label="correct alone"),
                       Patch(facecolor=RED, label="wrong alone")],
              loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "summary_matrix.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a) how many frames answer each clip -- the shape of the oracle's raw material
    ax = axes[0][0]
    xs = np.arange(F + 1)
    right = np.array(summary["clips_by_n_correct_frames_all_right"])
    tot = np.array(summary["clips_by_n_correct_frames"])
    ax.bar(xs, right, color=BLUE, label="full clip also right")
    ax.bar(xs, tot - right, bottom=right, color=ORANGE, label="full clip WRONG")
    ax.set_xlabel(f"number of the {F} frames that answer alone")
    ax.set_ylabel("clips")
    ax.set_xticks(xs)
    ax.set_title("how much of the clip is answerable frame-by-frame")
    ax.legend(frameon=False, fontsize=8)

    # (b) is the answerable frame positionally predictable? (the `middle` scorer's premise)
    ax = axes[0][1]
    ax.plot(np.arange(F), 100 * np.array(summary["accuracy_by_position"]), "-o",
            color=BLUE, lw=2, ms=5, label="all clips")
    if summary["accuracy_by_position_gap_clips"]:
        ax.plot(np.arange(F), 100 * np.array(summary["accuracy_by_position_gap_clips"]),
                "-o", color=ORANGE, lw=2, ms=5,
                label=f"the {summary['n_gap_clips']} gap clips")
    ax.axhline(100 * summary["anchors"]["single_mean"], color="gray", lw=0.9, ls=":",
               label="single_mean")
    ax.set_xlabel("frame position")
    ax.set_ylabel("accuracy alone (%)")
    ax.set_xticks(range(F))
    ax.set_title("is the answerable frame where you would guess?")
    ax.legend(frameon=False, fontsize=8)

    # (c) do the answerable frames sit together in time, or are they scattered?
    ax = axes[1][0]
    cl = summary["temporal_clustering"]
    if cl.get("n_mixed"):
        ax.bar([0, 1], [cl["mean_runs_observed"], cl["mean_runs_expected"]],
               color=[BLUE, ORANGE], width=0.6)
        for x, v in enumerate([cl["mean_runs_observed"], cl["mean_runs_expected"]]):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], ["observed", "shuffled within clip"])
        ax.set_ylabel("runs of consecutive correct frames")
        ax.set_title(f"are the answerable frames a moment? ({cl['n_mixed']} mixed clips, "
                     f"p={cl['p_fewer_runs_than_chance']:.3g})")

    # (d) the scorers, where it counts
    ax = axes[1][1]
    if scorers:
        xs = np.arange(len(scorers))
        ax.bar(xs - 0.2, [summary["scorers_on_mixed"][s]["within_clip_auc"] for s in scorers],
               width=0.38, color=BLUE, label="all mixed clips")
        ax.bar(xs + 0.2, [summary["scorers_on_gap_clips"][s]["within_clip_auc"] for s in scorers],
               width=0.38, color=ORANGE,
               label=f"the {summary['n_gap_clips']} clips where it would pay")
        ax.axhline(0.5, color="gray", lw=0.9, ls=":", label="no signal")
        ax.set_xticks(xs, scorers, rotation=45, ha="right")
        ax.set_ylabel("within-clip AUC")
        ax.set_ylim(0.3, 0.8)
        ax.set_title("does the signal survive where selection matters?")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "summary_panels.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Attention: per-frame maps and how to compare two of them
# --------------------------------------------------------------------------- #
def patch_grid(inputs, per_pos: int, backbone: str) -> tuple[int, int]:
    """(rows, cols) of the patch grid ONE temporal position covers, so a (per_pos,)
    attention vector can be reshaped onto the frame.

    Qwen states it in video_grid_thw, in pre-merge patch units: the 2x2 spatial merge
    halves each axis. Both LLaVA towers keep a square grid per frame (OneVision pools to
    14x14 = 196, LLaVA-1.5 is 24x24 = 576), so the side is just the square root -- and it
    is checked, because a non-square count would silently reshape into a wrong picture."""
    if backbone == "qwen":
        _, h, w = (int(x) for x in inputs["video_grid_thw"][0].tolist())
        gh, gw = h // 2, w // 2
        if gh * gw != per_pos:
            raise RuntimeError(f"video_grid_thw says {gh}x{gw} but the span holds {per_pos}")
        return gh, gw
    side = int(round(math.sqrt(per_pos)))
    if side * side != per_pos:
        raise RuntimeError(f"{per_pos} tokens/frame is not a square grid -- cannot map to pixels")
    return side, side


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks; ties share their mean rank so a flat region neither wins nor loses."""
    order = np.argsort(a, kind="stable")
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(1, a.size + 1, dtype=float)
    s = a[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def map_compare(full: np.ndarray, single: np.ndarray, k: int) -> dict:
    """How alike are the two within-frame attention maps?

    Both are renormalised to sum to one over the frame's own tokens FIRST. That is the
    only honest way to put them side by side: the full-clip forward spreads its rows over
    ~3.1k visual tokens and the isolated forward over ~390, so an unnormalised difference
    measures the sequence length and nothing else. After renormalising, three readings:

        spearman     does the frame's internal ORDERING survive the other 15 frames?
        top{k}_overlap  do the patches the text actually reads stay the same patches?
        tail index   is the map as CONCENTRATED in context as it is alone? (the same
                     moment estimator the xi_* scorers use, so it is on the family scale)
        mass_ratio   how much flatter/peakier the peak got, as a plain max ratio."""
    f = full / full.sum() if full.sum() > 0 else full
    s = single / single.sum() if single.sum() > 0 else single
    kf, ks = set(np.argsort(f)[::-1][:k].tolist()), set(np.argsort(s)[::-1][:k].tolist())
    rf, rs = _ranks(f), _ranks(s)
    return {"spearman": float(np.corrcoef(rf, rs)[0, 1]),
            f"top{k}_overlap": len(kf & ks) / float(k),
            "tail_index_full": moment_tail_index(torch.from_numpy(f)),
            "tail_index_single": moment_tail_index(torch.from_numpy(s)),
            "peak_ratio_full_over_single": float(f.max() / s.max()) if s.max() > 0 else float("nan"),
            "js_divergence": _js(f, s)}


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats: 0 = identical maps, log 2 = disjoint support."""
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(np.clip(a, 1e-12, None) /
                                                             np.clip(b, 1e-12, None)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top_patches(m: np.ndarray, grid: tuple[int, int], k: int) -> list[dict]:
    """The k highest-attention patches as (row, col) on the frame's grid, with the share
    of the frame's mass each carries -- the machine-readable half of the overlay."""
    _, gw = grid
    tot = m.sum() if m.sum() > 0 else 1.0
    return [{"token": int(i), "row": int(i // gw), "col": int(i % gw),
             "share_of_frame_mass": float(m[i] / tot)}
            for i in np.argsort(m)[::-1][:k]]


# --------------------------------------------------------------------------- #
# 3. The re-run: letters, maps, blind
# --------------------------------------------------------------------------- #
def probe_full(model, processor, frames, rec, args, device, dtype, letter_ids,
               visual_token_id, store, ctx, band) -> dict:
    """The one full-clip forward: the all-frames answer, the blind answer read off the
    SAME sequence, and the mid-band text->vision map split per frame.

    Blind is computed here rather than from a separately built prompt so it is the rho->0
    limit of the pruning sweep: the visual tokens are dropped from the merged embeddings
    and the surviving text keeps the position ids it had while they were there."""
    inputs = build_inputs(processor, frames, rec, args, device, dtype)
    ids = inputs["input_ids"][0]
    S = ids.numel()
    visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
    M = int(visual_idx.numel())

    store.clear()
    ctx.update({"capture": True, "visual_idx": visual_idx, "text_q": text_q,
                "embeds": None, "position_ids": None})
    with torch.no_grad():
        dense = model(**inputs, use_cache=False)
    ctx["capture"] = False
    pred = int(torch.argmax(dense.logits[0, -1][letter_ids]).item())
    del dense

    embeds, pos = ctx["embeds"], ctx["position_ids"]
    if pos is None:
        pos = torch.arange(S, device=embeds.device)[None]
    with torch.no_grad():
        blind = predict_pruned(model, embeds, pos,
                               keep_abs_idx(S, visual_idx, np.empty(0, dtype=np.int64)),
                               letter_ids)

    spans, fpp = frame_spans(inputs, M, args.num_segments, args.backbone)
    grid = patch_grid(inputs, spans[0][1] - spans[0][0], args.backbone)
    dec = band_score(store, band).numpy()          # (M,), sums to 1 over the visual block
    # Per POSITION, then repeated onto the frames it covers: on Qwen a position is a frame
    # PAIR, so the two frames of a pair share one map -- the honest granularity of the
    # merge, and the same expansion the ctx_* scorers use.
    per_pos = [dec[lo:hi] for lo, hi in spans]
    maps = [per_pos[f // fpp] for f in range(args.num_segments)]
    share = [float(per_pos[f // fpp].sum()) / fpp for f in range(args.num_segments)]

    ctx.update({"embeds": None, "position_ids": None})
    store.clear()
    del inputs, embeds
    return {"pred": pred, "blind": blind, "maps": maps, "share": share,
            "grid": grid, "frames_per_position": fpp, "n_visual_tokens": M,
            "uniform_share": 1.0 / len(spans) / fpp}


def probe_frame(model, processor, frame, rec, args, device, dtype, letter_ids,
                visual_token_id, store, ctx, band) -> dict:
    """One [f]*single_repeat zero-motion clip: the answer this frame gives on its own, and
    where the text looks while giving it. Identical construction to the experiment's
    isolated forward, so the prediction is the one behind the stored frame_correct entry."""
    inputs = build_inputs(processor, [frame] * args.single_repeat, rec, args, device, dtype)
    ids = inputs["input_ids"][0]
    visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
    M = int(visual_idx.numel())

    store.clear()
    ctx.update({"capture": True, "visual_idx": visual_idx, "text_q": text_q,
                "embeds": None, "position_ids": None})
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1]
    ctx["capture"] = False

    spans, _ = frame_spans(inputs, M, args.single_repeat, args.backbone)
    lo, hi = spans[0]                       # the frame's own tokens (the whole clip on Qwen)
    dec = band_score(store, band).numpy()
    ctx.update({"embeds": None, "position_ids": None})
    store.clear()
    del inputs
    return {"pred": int(torch.argmax(logits[letter_ids]).item()), "map": dec[lo:hi]}


# --------------------------------------------------------------------------- #
# 4. Rendering
# --------------------------------------------------------------------------- #
def caption(row: dict, rec: dict, gold: int, probe: dict | None,
            preds: list[int] | None, width: int = 118) -> str:
    """The text block under the sheet: what was asked, what the options were, which one is
    right, and the verdict of every configuration being contrasted."""
    corr = row["frame_correct"]
    good = [i for i, c in enumerate(corr) if c]
    pred_all = None if probe is None else probe["pred"]
    blind = None if probe is None else probe["blind"]
    lines = [f"{row['task']}  |  {row['video']}  |  question_idx {row.get('question_idx')}", ""]
    lines += textwrap.wrap(f"Q: {rec['question']}", width)
    lines.append("")
    for i, c in enumerate(rec["candidates"]):
        tags = []
        if i == gold:
            tags.append("GOLD")
        if pred_all is not None and i == pred_all:
            tags.append("ALL-FRAMES SAID THIS")
        if blind is not None and i == blind:
            tags.append("BLIND SAID THIS")
        mark = ("   <- " + " / ".join(tags)) if tags else ""
        body = textwrap.wrap(f"({LETTERS[i]}) {c}", width - 4)
        lines.append(body[0] if len(body) > 1 else body[0] + mark)
        lines += [" " * 4 + b for b in body[1:-1]]
        if len(body) > 1:
            lines.append(" " * 4 + body[-1] + mark)
    lines.append("")

    # Read off hit_by_config, never assumed: with --select all these sheets cover clips
    # the full clip gets RIGHT and clips no frame answers at all, and a hardcoded verdict
    # would caption them backwards.
    h = row["hit_by_config"]
    verdict = [f"all {len(corr)} frames: {'RIGHT' if h['all'] else 'WRONG'}" +
               (f" -> ({LETTERS[pred_all]})" if pred_all is not None else "")]
    if good:
        one = f"frame(s) {good} alone: CORRECT"
        if preds is not None:
            one += f" -> ({LETTERS[preds[good[0]]]})"
    else:
        one = f"none of the {len(corr)} frames answers alone"
    verdict.append(one)
    if blind is not None:
        verdict.append(f"no vision tokens (blind): ({LETTERS[blind]}) "
                       f"{'RIGHT' if blind == gold else 'wrong'}"
                       + (", SAME as all-frames" if blind == pred_all else ""))
    verdict.append(f"gold ({LETTERS[gold]})")
    lines.append("      |      ".join(verdict))
    lines.append(f"{sum(corr)}/{len(corr)} frames answer this question on their own")
    return "\n".join(lines)


def frame_table(row: dict, probe: dict | None, preds: list[int] | None) -> list[dict]:
    """One line per frame: does it answer alone, what does it answer, does that agree with
    the full clip and with the blind forward, and how much of the clip's visual attention
    it won.

    The agreement columns are the ones the correctness vector alone cannot give. A frame
    that is wrong alone AND says what the full clip says is a frame the clip is listening
    to; a frame that is right alone while the full clip says something else is evidence
    that was present and overruled. Which of those dominates decides whether the fix is
    selection or aggregation."""
    corr = row["frame_correct"]
    picks = {}
    for s, f in (row.get("picked_frame") or {}).items():
        picks.setdefault(int(f), []).append(s)
    rank = (np.argsort(np.argsort(probe["share"])[::-1]) + 1) if probe else None
    out = []
    for i, c in enumerate(corr):
        e = {"frame": i, "correct_alone": bool(c), "picked_by": sorted(picks.get(i, []))}
        if preds is not None:
            e.update({"answer": LETTERS[preds[i]],
                      "agrees_with_all_frames": preds[i] == probe["pred"],
                      "agrees_with_blind": preds[i] == probe["blind"]})
        if probe is not None:
            e.update({"attention_share": probe["share"][i], "attention_rank": int(rank[i])})
        out.append(e)
    return out


def print_frame_table(table: list[dict]):
    """The same table as text. Columns that need the model are simply absent without it,
    rather than printed empty -- the header is built from what the first row carries."""
    rich = "answer" in table[0]
    att = "attention_share" in table[0]
    head = f"  {'frame':>5} {'alone':>8}"
    if rich:
        head += f" {'answer':>7} {'==all':>6} {'==blind':>8}"
    if att:
        head += f" {'share%':>7} {'rank':>5}"
    print(head + "  picked by")
    for e in table:
        line = f"  {e['frame']:>5} {'CORRECT' if e['correct_alone'] else 'wrong':>8}"
        if rich:
            line += (f" {'(' + e['answer'] + ')':>7}"
                     f" {'yes' if e['agrees_with_all_frames'] else 'no':>6}"
                     f" {'yes' if e['agrees_with_blind'] else 'no':>8}")
        if att:
            line += f" {100 * e['attention_share']:>7.2f} {e['attention_rank']:>5}"
        print(line + "  " + ",".join(e["picked_by"]))


def _style(ax, correct: bool, title: str):
    ax.set_xticks([])
    ax.set_yticks([])
    col = "#1a9850" if correct else "#d73027"
    for sp in ax.spines.values():
        sp.set_edgecolor(col)
        sp.set_linewidth(3.0)
    ax.set_title(title, fontsize=8, color=col, fontweight="bold" if correct else "normal")


def _overlay(ax, frame, m: np.ndarray, grid: tuple[int, int], k: int, alpha: float):
    """The frame with its attention map on top and the k hottest patches boxed.

    The map is normalised by its own max, not globally: the question this figure answers
    is WHICH PART of this frame the text reads, and a global scale would flatten every
    frame that happens to sit in a low-mass position into an unreadable dark square. The
    absolute level is reported in the title instead, where it cannot be misread as shape."""
    import matplotlib.patches as mpatches

    gh, gw = grid
    W, H = frame.size
    ext = (-0.5, W - 0.5, H - 0.5, -0.5)
    ax.imshow(frame, extent=ext)
    img = m.reshape(gh, gw)
    ax.imshow(img / img.max() if img.max() > 0 else img, cmap="inferno", alpha=alpha,
              extent=ext, interpolation="bilinear", vmin=0.0, vmax=1.0)
    for i in np.argsort(m)[::-1][:k]:
        r, c = int(i) // gw, int(i) % gw
        ax.add_patch(mpatches.Rectangle((-0.5 + c * W / gw, -0.5 + r * H / gh),
                                        W / gw, H / gh, fill=False,
                                        edgecolor="#00e5ff", linewidth=1.2))


def sheet(frames, row, rec, gold, probe, preds, out_path, n_cols, show_picks):
    """The contact sheet. A frame's border is its verdict -- green if that frame ALONE
    gets the question right, red if not -- so the picture and the metric are the same
    object: the green frames are `single_best`, the whole grid together is `all`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr = row["frame_correct"]
    n_rows = int(np.ceil(len(frames) / n_cols))
    picks = {}
    if show_picks:
        for s, f in row.get("picked_frame", {}).items():
            picks.setdefault(int(f), []).append(s)

    cap = caption(row, rec, gold, probe, preds)
    fig, gs, fig_h = _canvas(n_rows, n_cols, cap)
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.imshow(frame)
        tag = "correct alone" if corr[i] else "wrong alone"
        if preds is not None:
            tag += f"  ({LETTERS[preds[i]]})"
        sub = "\n" + ",".join(sorted(picks[i])) if i in picks else ""
        _style(ax, bool(corr[i]), f"frame {i}  {tag}{sub}")
    h = row["hit_by_config"]
    n_good = sum(corr)
    title = (f"all {len(frames)} frames {'RIGHT' if h['all'] else 'WRONG'}   |   "
             + (f"{n_good} frame(s) alone RIGHT" if n_good else "no frame answers alone"))
    fig.text(0.01, 0.01, cap, fontsize=9, family="monospace", va="bottom", ha="left")
    fig.suptitle(title, fontsize=13, y=1.0 - 0.12 / fig_h)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _canvas(n_rows, n_cols, cap, tile=2.2):
    """A grid of `tile`-inch cells with room reserved underneath for the caption."""
    import matplotlib.pyplot as plt

    cap_lines = cap.count("\n") + 1
    fig_h = n_rows * tile + 0.22 * cap_lines + 0.9
    fig = plt.figure(figsize=(n_cols * tile, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols, top=1.0 - 0.5 / fig_h,
                          bottom=(0.22 * cap_lines + 0.4) / fig_h,
                          left=0.01, right=0.99, hspace=0.30, wspace=0.04)
    return fig, gs, fig_h


def attn_sheet(frames, row, maps, grid, out_path, title, n_cols, args,
               share=None, uniform=None, subtitles=None):
    """Every frame with its text->vision attention on top. Used twice: once for the
    full-clip forward (where `share` says how much of the clip's visual mass the frame
    won, against the uniform 1/F it would get if attention ignored content), once for the
    isolated forwards (where every frame is the whole clip, so share is meaningless and
    only the shape inside the frame carries information)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr = row["frame_correct"]
    n_rows = int(np.ceil(len(frames) / n_cols))
    cap = (f"top-{args.attn_topk} patches boxed; heat is the mid-band text->vision "
           f"attention (layers {args.dec_band}, queries={args.queries}), scaled per frame.")
    if share is not None:
        order = np.argsort(np.argsort(share)[::-1])
        cap += (f"\n'share' is the frame's fraction of the clip's visual attention mass; "
                f"uniform would be {100 * uniform:.2f}%.")
    fig, gs, fig_h = _canvas(n_rows, n_cols, cap)
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        _overlay(ax, frame, maps[i], grid, args.attn_topk, args.attn_alpha)
        t = f"frame {i}"
        if share is not None:
            t += f"  share {100 * share[i]:.2f}%  rank {int(order[i]) + 1}/{len(frames)}"
        if subtitles is not None:
            t += "\n" + subtitles[i]
        _style(ax, bool(corr[i]), t)
    fig.text(0.01, 0.01, cap, fontsize=9, family="monospace", va="bottom", ha="left")
    fig.suptitle(title, fontsize=13, y=1.0 - 0.12 / fig_h)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_sheet(frames, rows_to_show, full_maps, single_maps, grid, metrics,
                  out_path, args, header):
    """raw | full-clip attention | isolated attention, one row per frame of interest.

    This is the figure the claim rests on, so it always carries a CONTROL row: a frame
    that is wrong alone, put through the identical comparison. If the two maps diverge as
    much on the control as on the answerable frame, the divergence is about sequence
    length and not about the evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows_to_show)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3.3 * n), squeeze=False)
    for r, (i, label, correct) in enumerate(rows_to_show):
        axes[r][0].imshow(frames[i])
        _style(axes[r][0], correct, f"frame {i} -- {label}")
        _overlay(axes[r][1], frames[i], full_maps[i], grid, args.attn_topk, args.attn_alpha)
        _style(axes[r][1], correct, "attention with ALL frames")
        _overlay(axes[r][2], frames[i], single_maps[i], grid, args.attn_topk, args.attn_alpha)
        _style(axes[r][2], correct, "attention with THIS FRAME ALONE")
        m = metrics[i]
        axes[r][2].set_xlabel(
            f"spearman {m['spearman']:.2f}   top{args.attn_topk} overlap "
            f"{m[f'top{args.attn_topk}_overlap']:.2f}   JS {m['js_divergence']:.3f}\n"
            f"tail index  all {m['tail_index_full']:.3f}  alone {m['tail_index_single']:.3f}",
            fontsize=8)
    fig.suptitle(header, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def resolve_args(args):
    """Backbone / frames / tasks resolved the way frame_lever_accuracy.py resolves them,
    because the frames have to be the SAME frames: sample_frames is deterministic given
    (clip, num_segments), so any drift in --num_segments silently shows different pixels
    beside a correctness vector computed from the originals."""
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
        args.max_pixels = args.min_pixels = None
    elif args.min_pixels is None:
        args.min_pixels = args.max_pixels
    return args


def main(args):
    args = resolve_args(args)
    with open(args.per_sample) as fh:
        per_sample = json.load(fh)

    n_all_wrong = sum(1 for r in per_sample if r["hit_by_config"]["all"] == 0)
    n_gap = sum(1 for r in per_sample if SELECTORS["gap"](r))
    print(f"[dump] {len(per_sample)} clips, {n_all_wrong} wrong with all frames, "
          f"{n_gap} of those answered by SOME single frame (the single_best - all gap)")

    if args.summary or args.summary_only:
        summarize(per_sample, args)
    if args.summary_only:
        return

    rows = candidates(per_sample, args)
    print(f"[select] {args.select}: {len(rows)} clips match; rendering "
          f"{min(len(rows), args.n_examples)}")
    if not rows:
        print("no example matches the filters.")
        return
    rows = rows[: args.n_examples]

    if not args.data_root:
        raise SystemExit("--data_root is required to render frames "
                         "(use --summary_only for the model-free dataset summary)")
    index = record_index(args)
    os.makedirs(args.out_dir, exist_ok=True)

    model = processor = device = dtype = band = None
    store, ctx, uninstall, visual_token_id = {}, {"capture": False}, (lambda: None), None
    if args.rerun:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        print(f"[rerun] loading {args.model_name} on {device} ({args.dtype})")
        model, processor, visual_token_id = load_model(args, dtype)
        band = parse_band(args.dec_band, len(text_model_of(model).layers))
        uninstall = attach_capture(model, store, ctx)
        print(f"[score] decoder band {band[0]}-{band[-1]} ({len(band)} layers), "
              f"queries={args.queries}")

    out = []
    try:
        for row in rows:
            key = (row["task"], row["video"], str(row.get("question_idx")))
            if key not in index:
                print(f"skip {key}: not in {args.data_root}/json -- wrong --data_root/--tasks?")
                continue
            rec, path, data_type, bound = index[key]
            if not (os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)):
                print(f"skip {key}: missing media at {path}")
                continue

            gold = gold_index(rec)
            if gold != row["gold"]:
                print(f"[warn] {key}: gold in json is {gold} but the dump recorded "
                      f"{row['gold']} -- the json changed since the run; showing the json's.")
            frames = sample_frames(path, data_type, bound, args.num_segments)
            corr = row["frame_correct"]
            good = [i for i, c in enumerate(corr) if c]
            stem = f"{os.path.splitext(row['video'])[0]}_{row.get('question_idx')}"
            entry = {"task": row["task"], "video": row["video"],
                     "question_idx": row.get("question_idx"),
                     "question": rec["question"], "candidates": rec["candidates"],
                     "gold_index": gold, "gold_letter": LETTERS[gold],
                     "gold_text": rec["candidates"][gold],
                     "num_segments": args.num_segments,
                     "frame_correct": corr, "correct_frames": good,
                     "n_correct_frames": len(good),
                     "picked_frame": row.get("picked_frame"),
                     "hit_by_config": row["hit_by_config"],
                     "prompt": build_question(rec) + ANSWER_PREFIX}

            probe = preds = None
            if args.rerun:
                letter_ids = torch.tensor(
                    option_token_ids(processor.tokenizer, len(rec["candidates"])), device=device)
                probe = probe_full(model, processor, frames, rec, args, device, dtype,
                                   letter_ids, visual_token_id, store, ctx, band)
                singles = [probe_frame(model, processor, f, rec, args, device, dtype,
                                       letter_ids, visual_token_id, store, ctx, band)
                           for f in frames]
                preds = [s["pred"] for s in singles]
                single_maps = [s["map"] for s in singles]

                recovered = [int(p == gold) for p in preds]
                if recovered != corr:
                    print(f"[warn] {key}: re-run correctness {recovered} != stored {corr} "
                          f"-- frames or prompt have drifted.")
                if (probe["pred"] == gold) != bool(row["hit_by_config"]["all"]):
                    print(f"[warn] {key}: re-run all-frames hit disagrees with the dump.")
                if (probe["blind"] == gold) != bool(row["hit_by_config"]["blind"]):
                    print(f"[warn] {key}: re-run blind hit disagrees with the dump.")

                shapes_ok = all(m.shape == probe["maps"][0].shape for m in single_maps)
                metrics = {}
                if shapes_ok:
                    metrics = {i: map_compare(probe["maps"][i], single_maps[i], args.attn_topk)
                               for i in range(len(frames))}
                else:
                    print(f"[warn] {key}: full-clip and isolated token counts differ "
                          f"-- skipping the map comparison (pixel budget mismatch?)")

                attn_sheet(frames, row, probe["maps"], probe["grid"],
                           os.path.join(args.out_dir, f"{stem}_attn_all.png"),
                           "where the text looks in the FULL 16-frame clip",
                           args.n_cols, args, share=probe["share"],
                           uniform=probe["uniform_share"])
                attn_sheet(frames, row, single_maps, probe["grid"],
                           os.path.join(args.out_dir, f"{stem}_attn_single.png"),
                           "where the text looks when each frame is ALONE",
                           args.n_cols, args,
                           subtitles=[f"answers ({LETTERS[p]})" for p in preds])

                show = []
                if metrics:
                    show = [(i, "correct alone", True) for i in good[: args.max_saved_frames]]
                    if args.attn_control:
                        # The paired control: the wrong frame whose full-clip attention share
                        # is closest to the answerable one, so the two rows differ in
                        # answerability rather than in how loud the position is.
                        bad = [i for i, c in enumerate(corr) if not c]
                        if bad and good:
                            ref = probe["share"][good[0]]
                            ctrl = min(bad, key=lambda i: abs(probe["share"][i] - ref))
                            show.append((ctrl, "control: wrong alone", False))
                # A clip no frame answers alone (161 of the 500 on EgoSchema) has nothing to
                # compare against, so the compare figure is skipped rather than drawn empty.
                if show:
                    compare_sheet(frames, show, probe["maps"], single_maps, probe["grid"],
                                  metrics, os.path.join(args.out_dir, f"{stem}_attn_compare.png"),
                                  args, f"{row['video']} q{row.get('question_idx')} -- "
                                        f"does the frame's attention map survive the other "
                                        f"{args.num_segments - 1} frames?")

                entry.update({
                    "all_frames_pred_index": probe["pred"],
                    "all_frames_pred_letter": LETTERS[probe["pred"]],
                    "blind_pred_index": probe["blind"],
                    "blind_pred_letter": LETTERS[probe["blind"]],
                    "blind_equals_all_frames": probe["blind"] == probe["pred"],
                    "per_frame_pred_index": preds,
                    "per_frame_pred_letter": [LETTERS[p] for p in preds],
                    "n_visual_tokens": probe["n_visual_tokens"],
                    "patch_grid": list(probe["grid"]),
                    "frames_per_position": probe["frames_per_position"],
                    "visual_mass_share": probe["share"],
                    "uniform_share": probe["uniform_share"],
                    "share_rank_of_correct_frames":
                        [int(np.argsort(np.argsort(probe["share"])[::-1])[i]) + 1 for i in good],
                    "top_patches_full_clip":
                        {str(i): top_patches(probe["maps"][i], probe["grid"], args.attn_topk)
                         for i in good[: args.max_saved_frames]},
                    "top_patches_single_frame":
                        {str(i): top_patches(single_maps[i], probe["grid"], args.attn_topk)
                         for i in good[: args.max_saved_frames]},
                    "map_comparison": {str(i): metrics[i] for i in metrics} if metrics else None,
                    "attn_all_png": os.path.join(args.out_dir, f"{stem}_attn_all.png"),
                    "attn_single_png": os.path.join(args.out_dir, f"{stem}_attn_single.png"),
                    "attn_compare_png": os.path.join(args.out_dir, f"{stem}_attn_compare.png")
                                        if show else None})

            sheet_path = os.path.join(args.out_dir, f"{stem}_sheet.png")
            sheet(frames, row, rec, gold, probe, preds, sheet_path, args.n_cols, args.show_picks)
            frame_paths = []
            for i in good[: args.max_saved_frames]:
                p = os.path.join(args.out_dir, f"{stem}_f{i:02d}.png")
                frames[i].save(p)
                frame_paths.append(p)
            table = frame_table(row, probe, preds)
            entry.update({"sheet": sheet_path, "frame_files": frame_paths,
                          "frame_table": table})

            print("\n" + "=" * 100)
            print(caption(row, rec, gold, probe, preds))
            print()
            print_frame_table(table)
            if probe is not None:
                sh, uni = probe["share"], probe["uniform_share"]
                print(f"\nvisual attention share, full clip (uniform = {100 * uni:.2f}%):")
                print("  " + " ".join(f"{100 * s:.2f}" for s in sh))
                for i in good[: args.max_saved_frames]:
                    m = entry.get("map_comparison", {}) or {}
                    line = (f"  frame {i}: share {100 * sh[i]:.2f}% "
                            f"({'above' if sh[i] > uni else 'below'} uniform)")
                    if str(i) in m:
                        d = m[str(i)]
                        line += (f"   map vs alone: spearman {d['spearman']:.2f}, "
                                 f"top{args.attn_topk} overlap "
                                 f"{d[f'top{args.attn_topk}_overlap']:.2f}, "
                                 f"tail {d['tail_index_full']:.3f} -> {d['tail_index_single']:.3f}")
                    print(line)
            print(f"sheet -> {sheet_path}")
            if frame_paths:
                print("correct frame(s) -> " + ", ".join(frame_paths))
            out.append(entry)
    finally:
        uninstall()

    with open(os.path.join(args.out_dir, "examples.json"), "w") as fh:
        json.dump({"per_sample": os.path.abspath(args.per_sample),
                   "model_name": args.model_name, "backbone": args.backbone,
                   "dec_band": args.dec_band, "queries": args.queries,
                   "n_all_wrong": n_all_wrong, "n_all_wrong_single_right": len(rows),
                   "rerun": bool(args.rerun), "examples": out}, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'examples.json')}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Show clips frame_lever_accuracy.py got wrong with all frames and "
                    "right from one frame: the frames, the question, the answer, where "
                    "the text attends, and what the model says with no vision at all.")
    p.add_argument("--per_sample", required=True,
                   help="the <run>_per_sample.json written by frame_lever_accuracy.py")
    p.add_argument("--data_root", default=None,
                   help="dir holding json/ and video/. Required to render frames; "
                        "--summary_only needs nothing but the dump.")
    p.add_argument("--select", choices=sorted(SELECTORS), default="gap",
                   help="which clips to render: gap = the full clip is wrong and some "
                        "single frame is right (default, the headline case); all = every "
                        "clip in the dump; also mixed, all_frames_wrong, all_frames_right, "
                        "reverse_gap (EVERY frame right alone, full clip still wrong), "
                        "no_frame_right, every_frame_right, blind_beats_video.")
    p.add_argument("--summary", action="store_true",
                   help="also write the dataset-wide frame-by-frame analysis over ALL "
                        "clips in the dump (needs no model and no video).")
    p.add_argument("--summary_only", action="store_true",
                   help="write that summary and stop -- no frames rendered, no model.")
    p.add_argument("--seed", type=int, default=0, help="seeds the clustering permutation test.")
    p.add_argument("--n_perm", type=int, default=2000,
                   help="permutations for the temporal-clustering test.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' / 'all'. Must cover the tasks in the dump.")
    p.add_argument("--model_name", default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                   help="only sets the backbone (frame count / prompt) unless --rerun.")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_ov", "llava"], default="auto")
    p.add_argument("--num_segments", type=int, default=None,
                   help="MUST match the run that produced the dump (default per backbone).")
    p.add_argument("--n_examples", type=int, default=5, help="how many clips to render.")
    p.add_argument("--order", choices=["fewest_correct_frames", "all_wrong_margin"],
                   default="fewest_correct_frames",
                   help="fewest_correct_frames: the sharpest cases (one frame carries it). "
                        "all_wrong_margin: clips most frames answer but the full clip does not.")
    p.add_argument("--video", default=None, help="restrict to one video filename.")
    p.add_argument("--question_idx", default=None, help="restrict to one question_idx.")
    p.add_argument("--task", default=None, help="restrict to one task (MVBench dumps).")
    p.add_argument("--rerun", action="store_true",
                   help="load the model and redo 1 + F forwards per example: the letters "
                        "each configuration answers, the per-frame attention maps, and the "
                        "blind (no visual tokens) prediction.")
    p.add_argument("--single_repeat", type=int, default=2,
                   help="frames in the zero-motion single-frame clip; match the run.")
    p.add_argument("--dec_band", default="11:15",
                   help="decoder layers the attention maps read; match the run (default "
                        "11:15, the same mid band as stage_topk_accuracy.py).")
    p.add_argument("--queries", choices=["post", "all", "last"], default="post",
                   help="text query rows the attention is averaged over; match the run.")
    p.add_argument("--attn_topk", type=int, default=10,
                   help="how many hottest patches to box per frame and dump to json.")
    p.add_argument("--attn_alpha", type=float, default=0.55, help="heatmap opacity.")
    p.add_argument("--attn_control", action="store_true", default=True,
                   help="add a wrong-alone frame of matched attention share to the compare "
                        "figure -- without it a map difference cannot be attributed to "
                        "answerability rather than to sequence length.")
    p.add_argument("--no_attn_control", dest="attn_control", action="store_false")
    p.add_argument("--max_pixels", type=int, default=None, help="qwen only; match the run.")
    p.add_argument("--min_pixels", type=int, default=None, help="qwen only; match the run.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max_samples", type=int, default=None,
                   help="cap on records per task when reading json/ (leave unset).")
    p.add_argument("--n_cols", type=int, default=8, help="columns in the contact sheets.")
    p.add_argument("--show_picks", action="store_true",
                   help="also print, under each frame, which scorers argmaxed it.")
    p.add_argument("--max_saved_frames", type=int, default=4,
                   help="how many correct frames to save at full resolution and analyse.")
    p.add_argument("--out_dir", default="frame_lever_examples")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
