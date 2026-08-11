"""
plot_knockout_window_panels.py -- the Fine-grained (window) half of the LV-K sweep.

plot_knockout_panels.py draws the CUMULATIVE setting: knock out text->vision
attention from layer i onward and watch the curve decay. That answers "how deep
does the text still need to look at the video". This script draws the WINDOW
setting written by

    lv_knockout_accuracy.py --setting window --window 4

where exactly one block of consecutive layers is knocked out and every other
layer is left alone, so the question becomes WHICH layers carry the transfer.
y is the accuracy change acc(window) - acc(full) in points, with 0 (the full
model) as the reference line.

Note the x axis is categorical -- the windows are disjoint blocks, so the line
segments join measurements, they do not interpolate between them.

Same layout and plain-matplotlib look as plot_tail_panels.py: one panel per
MODEL, one line per benchmark, colour fixed to the benchmark, y shared so the
panels can be read against each other.

The band is a 95% CI on the accuracy CHANGE, computed from the run's
<stem>_per_sample.json: the knockout and the baseline are scored on the same
clips, so the difference is paired and its standard error comes from the per-clip
differences d_i = hit(window)_i - hit(full)_i, not from the two accuracies
separately. A band that clears 0 is the same statement the exact McNemar test
makes. Runs whose per-sample file is missing are drawn as a bare line.

Run
---
    # the two EgoSchema runs in the repo root, one panel each
    python final_expt_scripts/plot_scripts/plot_knockout_window_panels.py --out lvk_win_panels.png lvk_win_panels.pdf

    # a model's two benchmarks in one panel, two models side by side
    python final_expt_scripts/plot_scripts/plot_knockout_window_panels.py \
        --runs lvk_win_ego_llavaov.json lvk_win_mvb_llavaov.json \
               lvk_win_ego_qwen.json lvk_win_mvb_qwen.json \
        --out lvk_win_panels.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour per BENCHMARK, fixed across panels, so a colour always means the
# dataset and never the model -- as in plot_tail_panels.py.
COLORS = {"EgoSchema": "C0", "MVBench": "C1"}

PRETTY_MODEL = {"llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
                "llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
                "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": "LLaVA-OneVision-0.5B",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}


def dataset_label(meta) -> str:
    """The benchmark a run covers. A run is either the single EgoSchema task or
    some set of MVBench tasks; only a partial MVBench selection keeps its own name."""
    tasks = meta["tasks"]
    if len(tasks) > 1:
        return "MVBench"
    return tasks[0]


def paired_ci(path: str, labels, z: float = 1.96):
    """Half-width, in points, of the CI on acc(window) - acc(full), per window.

    Read off <stem>_per_sample.json: every config is scored on the SAME clips, so
    the per-clip differences d_i in {-1,0,+1} are what carries the uncertainty --
    treating the two accuracies as independent would overstate it badly, since
    most clips answer identically under both. None if the file is absent."""
    per_sample = os.path.splitext(path)[0] + "_per_sample.json"
    if not os.path.isfile(per_sample):
        return None
    with open(per_sample) as fh:
        recs = json.load(fh)
    if len(recs) < 2:
        return None
    hits = {lab: np.array([int(r["pred_by_config"][lab] == r["gold"]) for r in recs])
            for lab in ["full"] + list(labels)}
    n = len(recs)
    return np.array([100 * z * (hits[lab] - hits["full"]).std(ddof=1) / np.sqrt(n)
                     for lab in labels])


def load_run(path: str):
    """(window labels, accuracy change in points, CI half-width or None, meta)."""
    with open(path) as fh:
        d = json.load(fh)
    if d.get("setting") != "window":
        raise SystemExit(f"{path}: setting={d.get('setting')!r}, expected 'window'. "
                         "Use plot_knockout_panels.py for the cumulative sweep.")
    labels = list(d["windows"])
    delta = np.array([100 * d["accuracy_change_by_window"][lab] for lab in labels])
    return labels, delta, paired_ci(path, labels), d


def draw_panel(ax, runs, args):
    """One model's panel: a change line (+ paired CI band) per benchmark."""
    for labels, delta, ci, meta in runs:
        x = np.arange(len(labels))
        name = dataset_label(meta)
        c = COLORS.get(name, "C2")
        ax.plot(x, delta, marker="o", ms=3, color=c,
                label=f"{name} (n={meta['n_clips']})")
        if ci is not None and not args.no_ci:
            ax.fill_between(x, delta - ci, delta + ci, color=c, alpha=0.15)

    labels, _, _, meta = runs[0]
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(f"decoder layers knocked out (window of {meta['window']})")
    ax.legend(frameon=False, loc=args.legend_loc, fontsize=9)

    name = PRETTY_MODEL.get(meta["model_name"], os.path.basename(meta["model_name"]))
    ax.set_title(f"{name} ({meta['n_layers']} decoder layers)")


def group_by_model(runs):
    """[[run, ...]] -- one list per model, in the order the models first appear on
    the command line, so --runs sets both the panels and the lines within them."""
    panels: dict[str, list] = {}
    for run in runs:
        panels.setdefault(run[3]["model_name"], []).append(run)
    return list(panels.values())


def main(args):
    panels = group_by_model([load_run(p) for p in args.runs])

    # 5.5in per panel, same as plot_tail_panels.py, but never so narrow that the
    # figure title has to wrap off the canvas.
    width = max(5.5 * len(panels), 7.5)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.0),
                             sharey=not args.free_y, squeeze=False)
    axes = axes[0]
    for ax, runs in zip(axes, panels):
        draw_panel(ax, runs, args)

    ylab = "accuracy change (points)"
    axes[0].set_ylabel(ylab)
    if args.free_y:
        for ax in axes[1:]:
            ax.set_ylabel(ylab)

    fig.suptitle(args.title or "Accuracy change vs. knocked-out layer window")
    fig.tight_layout()
    for path in args.out:
        fig.savefig(path, dpi=args.dpi)
        print(f"saved {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["lvk_win_ego_llavaov.json", "lvk_win_ego_qwen.json"],
                   help="window-setting LV-K JSONs. Runs sharing a --model_name go in "
                        "one panel, one line each; panels follow first appearance.")
    p.add_argument("--free_y", action="store_true",
                   help="give each panel its own y range.")
    p.add_argument("--no_ci", action="store_true",
                   help="drop the paired 95%% CI band (drawn when the per-sample file "
                        "sits next to the run).")
    p.add_argument("--legend_loc", default="lower left",
                   help="the dip lands mid-x, so the bottom corners are the free ones.")
    p.add_argument("--title", default=None, help="figure-level title.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out", nargs="+", default=["lvk_win_panels.png"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
