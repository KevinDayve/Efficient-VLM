"""
plot_knockout_panels.py -- one figure, two VLM panels, two benchmarks per panel,
for the LV-K sweep written by lv_knockout_accuracy.py.

    left panel   LLaVA-1.5-7B    EgoSchema + MVBench
    right panel  Qwen2.5-VL-7B   EgoSchema + MVBench

x is the layer ratio i/L in %, y is the performance ratio acc(i)/acc(L) in %
(--y accuracy plots raw accuracy instead). The 100% and 95% guides are the ones
the paper reads its two-stage conclusion off: where the curve reaches 95%, the
text has already extracted everything it needs from the video.

Same plain-matplotlib look as plot_tail_panels.py.

Run
---
    python final_expt_scripts/plot_scripts/plot_knockout_panels.py --out lvk_panels.png lvk_panels.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"EgoSchema": "C0", "MVBench": "C1"}

PRETTY_MODEL = {"llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}


def load_curve(path: str):
    """(layer_ratio %, performance ratio %, accuracy %, meta) for one LV-K JSON."""
    with open(path) as fh:
        d = json.load(fh)
    cutoffs = [int(c) for c in d["cutoffs"]]
    x = np.array([100 * d["layer_ratio_by_cutoff"][str(c)] for c in cutoffs])
    ratio = np.array([100 * d["performance_ratio_by_cutoff"][str(c)] for c in cutoffs])
    acc = np.array([100 * d["accuracy_by_cutoff"][str(c)] for c in cutoffs])
    return x, ratio, acc, d


def draw_panel(ax, curves, title, args):
    for label, (x, ratio, acc, meta) in curves.items():
        y = acc if args.y == "accuracy" else ratio
        ax.plot(x, y, marker="o", ms=3, color=COLORS[label],
                label=f"{label} (n={meta['n_clips']}, "
                      f"full={100 * meta['baseline_accuracy']:.1f}%)")

    if args.y == "ratio":
        ax.axhline(100, color="gray", lw=0.8, ls=":")
        ax.axhline(95, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("layer ratio (%)")
    ax.set_title(title)
    ax.legend(frameon=False, loc=args.legend_loc, fontsize=9)


def main(args):
    panels = [{"EgoSchema": load_curve(args.llava_ego),
               "MVBench": load_curve(args.llava_mvb)},
              {"EgoSchema": load_curve(args.qwen_ego),
               "MVBench": load_curve(args.qwen_mvb)}]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=not args.free_y)
    for ax, curves in zip(axes, panels):
        meta = next(iter(curves.values()))[3]
        name = PRETTY_MODEL.get(meta["model_name"], os.path.basename(meta["model_name"]))
        draw_panel(ax, curves, f"{name} ({meta['n_layers']} decoder layers)", args)

    ylab = "accuracy (%)" if args.y == "accuracy" else "performance ratio (%)"
    axes[0].set_ylabel(ylab)
    if args.free_y:
        axes[1].set_ylabel(ylab)

    fig.suptitle(args.title or "Accuracy vs. the depth beyond which text->vision "
                               "attention is knocked out (LV-K)")
    fig.tight_layout()
    for path in args.out:
        fig.savefig(path, dpi=args.dpi)
        print(f"saved {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llava_ego", default="lvk_ego_llava.json")
    p.add_argument("--llava_mvb", default="lvk_mvb_llava.json")
    p.add_argument("--qwen_ego", default="lvk_ego_qwen.json")
    p.add_argument("--qwen_mvb", default="lvk_mvb_qwen.json")
    p.add_argument("--y", choices=["ratio", "accuracy"], default="ratio",
                   help="ratio = acc(i)/acc(L) in %% (the paper's y axis).")
    p.add_argument("--free_y", action="store_true",
                   help="give each panel its own y range.")
    p.add_argument("--legend_loc", default="lower right")
    p.add_argument("--title", default=None, help="figure-level title.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out", nargs="+", default=["lvk_panels.png"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
