"""
plot_tail_panels.py -- one figure, two VLM panels, two benchmarks per panel.

Reads the summary JSONs written by tail_vs_layer.py (the `gamma_mean_by_layer` /
`gamma_std_by_layer` blocks) and draws a single side-by-side figure:

    left panel   LLaVA-OneVision-7B   EgoSchema + MVBench
    right panel  Qwen2.5-VL-7B        EgoSchema + MVBench

Plain matplotlib defaults, same look as the per-run plot in tail_vs_layer.py
(default colour cycle, mean line + /- 1 sd band, gamma = 0 reference).

The x axes are per-panel so backbones of different depth still line up index for
index; the y axis is SHARED by default, because the two models can sit at
different tail-index levels and rescaling each panel to its own range would hide
exactly that. --free_y opts out when the within-model shape is what matters.

Run
---
    # decoder (defaults point at the four tail_dec_*.json in the repo root)
    python final_expt_scripts/plot_scripts/plot_tail_panels.py --out tail_dec_panels.png

    # encoder, same layout
    python final_expt_scripts/plot_scripts/plot_tail_panels.py --stack encoder \
        --llava_ego tail_enc_ego_llavaov.json --llava_mvb tail_enc_mvb_llavaov.json \
        --qwen_ego  tail_enc_ego_qwen.json  --qwen_mvb  tail_enc_mvb_qwen.json \
        --out tail_enc_panels.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour per BENCHMARK, fixed across both panels, so a colour always means the
# dataset and never the model. Default matplotlib cycle entries.
COLORS = {"EgoSchema": "C0", "MVBench": "C1"}

PRETTY_MODEL = {"llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}


def load_curve(path: str):
    """(mean, std, meta) for one tail_vs_layer.py summary JSON."""
    with open(path) as fh:
        d = json.load(fh)
    n = d["n_layers"]
    mean = np.array([d["gamma_mean_by_layer"][str(l)] for l in range(n)], dtype=float)
    std = np.array([d["gamma_std_by_layer"][str(l)] for l in range(n)], dtype=float)
    return mean, std, d


def draw_panel(ax, curves, title, args):
    """One model's panel: a mean line + /- 1 sd band per benchmark."""
    for label, (mean, std, meta) in curves.items():
        xs = np.arange(len(mean))
        if args.x == "depth":
            xs = xs / (len(mean) - 1)
        c = COLORS[label]
        ax.plot(xs, mean, marker="o", ms=3, color=c, label=f"{label} (n={meta['n_clips']})")
        ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.15)

    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("relative depth" if args.x == "depth" else f"{args.stack} layer")
    ax.set_title(title)
    ax.legend(frameon=False, loc=args.legend_loc)


def load_panel(ego: str, mvb: str):
    """{benchmark: curve} for one model, dropping the benchmarks passed as 'none'
    -- a stack is not always swept on both, and a panel with one line beats
    borrowing another model's run for the second."""
    paths = {"EgoSchema": ego, "MVBench": mvb}
    curves = {lab: load_curve(p) for lab, p in paths.items() if p.lower() != "none"}
    if not curves:
        raise SystemExit("a panel needs at least one run; both were 'none'.")
    return curves


def main(args):
    panels = [load_panel(args.llava_ego, args.llava_mvb),
              load_panel(args.qwen_ego, args.qwen_mvb)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=not args.free_y)
    for ax, curves in zip(axes, panels):
        meta = next(iter(curves.values()))[2]
        name = PRETTY_MODEL.get(meta["model_name"], os.path.basename(meta["model_name"]))
        draw_panel(ax, curves, f"{name} ({meta['n_layers']} {args.stack} layers)", args)

    axes[0].set_ylabel("tail index gamma")
    if args.free_y:
        axes[1].set_ylabel("tail index gamma")
    fig.suptitle(args.title or
                 f"Tail index of visual-token importance vs. {args.stack} layer")

    fig.tight_layout()
    for path in args.out:
        fig.savefig(path, dpi=args.dpi)
        print(f"saved {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llava_ego", default="tail_dec_ego_llavaov.json",
                   help="'none' leaves this benchmark out of the panel.")
    p.add_argument("--llava_mvb", default="tail_dec_mvb_llavaov.json",
                   help="'none' leaves this benchmark out of the panel.")
    p.add_argument("--qwen_ego", default="tail_dec_ego_qwen.json",
                   help="'none' leaves this benchmark out of the panel.")
    p.add_argument("--qwen_mvb", default="tail_dec_mvb_qwen.json",
                   help="'none' leaves this benchmark out of the panel.")
    p.add_argument("--stack", choices=["decoder", "encoder"], default="decoder",
                   help="axis/label wording only; pick the matching JSONs yourself.")
    p.add_argument("--x", choices=["layer", "depth"], default="layer",
                   help="depth = layer index normalised to [0,1], for comparing the "
                        "28- and 32-layer stacks position-for-position.")
    p.add_argument("--free_y", action="store_true",
                   help="give each panel its own y range (hides the level difference).")
    p.add_argument("--legend_loc", default="best")
    p.add_argument("--title", default=None,
                   help="figure-level title; default names the stack.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out", nargs="+", default=["tail_dec_panels.png"],
                   help="one or more output paths (e.g. a .png and a .pdf).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
