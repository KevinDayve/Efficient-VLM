"""
plot_topk_panels.py -- one figure, four panels, for the top-K sweep written by
stage_topk_accuracy.py.

    columns = model       LLaVA-OneVision-7B | Qwen2.5-VL-7B
    rows    = benchmark   EgoSchema (top) | MVBench (bottom)

Same plain-matplotlib look as plot_tail_panels.py / plot_knockout_panels.py: one
colour per selector, fixed across all four panels, one shared legend under the
figure instead of four copies inside it.

Line style encodes the selector FAMILY, so the three groups separate at a glance
even in greyscale:

    solid    attention top-K          (the method)
    dashed   random / uniform floors  (unranked baselines)
    dash-dot attention bottom-K       (anti-oracle, the other end of the ranking)

The no-pruning ceiling is per panel -- it is a property of the model/benchmark
pair, not of a selector -- so it stays a grey dotted line inside each panel,
labelled with its own value rather than pushed into the shared legend. The
text-only floor (`text_only_accuracy`, every visual token dropped) is drawn the
same way, in black dash-dot: it is likewise per panel and likewise not a
selector, and together the two lines bracket the band inside which any of these
curves can say anything. Runs written before the floor existed simply omit the
key and lose the line.

y axes are shared per ROW by default: the same benchmark on two backbones is the
comparison worth making pixel-for-pixel, while EgoSchema and MVBench sit at
different accuracy levels and sharing across rows would flatten both. --share_y
all forces one range everywhere, --share_y none gives every panel its own.
--y retention divides by each panel's own full accuracy, which puts every ceiling
at 100% and makes all four panels directly comparable.

Run
---
    python final_expt_scripts/plot_scripts/plot_topk_panels.py --out final_expt_results/topk_accuracy/topk_panels.png
    python final_expt_scripts/plot_scripts/plot_topk_panels.py --y retention --out topk_panels_retention.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = "final_expt_results/topk_accuracy"

# One colour + line style per SELECTOR, fixed across every panel, so a colour
# always means the selector and never the model or the benchmark. Colours keep
# the C0..C6 order the per-run plots in stage_topk_accuracy.py already use, so
# these panels and the single-run PNGs stay readable side by side.
STYLE = {"attn_early":      ("C0", "-"),
         "attn_mid":        ("C1", "-"),
         "random":          ("C2", "--"),
         "uniform":         ("C3", "--"),
         "uniform_stagger": ("C4", "--"),
         "bot_early":       ("C5", "-."),
         "bot_mid":         ("C6", "-.")}

PRETTY_SELECTOR = {"attn_early":      "attn top-K (early)",
                   "attn_mid":        "attn top-K (mid)",
                   "random":          "random",
                   "uniform":         "uniform",
                   "uniform_stagger": "uniform (staggered)",
                   "bot_early":       "attn bottom-K (early)",
                   "bot_mid":         "attn bottom-K (mid)"}

PRETTY_MODEL = {"llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}


def load_run(path: str):
    """(rho %, {selector: y%}, meta) for one stage_topk_accuracy.py summary JSON."""
    with open(path) as fh:
        d = json.load(fh)
    rhos = [float(r) for r in d["rhos"]]
    by_rho = d["accuracy_by_rho"]
    curves = {s: np.array([100 * by_rho[f"{r:g}"][s] for r in rhos])
              for s in d["selectors"]}
    return 100 * np.array(rhos), curves, d


def blind_pct(meta):
    """The run's text-only accuracy in %, or None if it did not measure one.

    `text_only_accuracy` is null (not absent) when a run was given --no-text_only,
    so presence of the key is not enough."""
    v = meta.get("text_only_accuracy")
    return None if v is None else 100 * v


def draw_panel(ax, run, args):
    """One model x benchmark panel: every selector, plus its own ceiling and floor."""
    xs, curves, meta = run
    full = 100 * meta["full_accuracy"]
    scale = (100 / full) if args.y == "retention" else 1.0

    for s in meta["selectors"]:
        color, ls = STYLE.get(s, (None, "-"))
        ax.plot(xs, scale * curves[s], ls, marker="o", ms=4, lw=1.6, color=color)

    ceiling = 100.0 if args.y == "retention" else full
    ax.axhline(ceiling, color="gray", lw=0.8, ls=":")
    ax.annotate(f"no pruning ({full:.1f}%)", xy=(xs[0], ceiling), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color="gray", va="bottom")

    # The floor, when the run measured one. Annotated BELOW its line so it cannot
    # collide with the ceiling label on a panel where the two sit close together --
    # which is itself the finding worth seeing.
    blind = blind_pct(meta)
    if blind is not None and not args.no_floor:
        ax.axhline(scale * blind, color="black", lw=0.9, ls="-.")
        ax.annotate(f"text only ({blind:.1f}%)", xy=(xs[0], scale * blind), xytext=(0, -4),
                    textcoords="offset points", fontsize=8, color="0.15", va="top")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def panel_note(meta) -> str:
    """Bottom-right stamp: which run this panel actually is."""
    task = "EgoSchema" if meta["tasks"] == ["EgoSchema"] else "MVBench"
    return f"{task}  n={meta['n_clips']}  {meta['num_segments']}f"


def headroom(ax, frac=0.10, foot=0.06):
    """Pad the y range so the ceiling label above its line and the floor label below
    its line both have room."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - foot * span, hi + frac * span)


def main(args):
    paths = [[args.llava_ego, args.qwen_ego],
             [args.llava_mvb, args.qwen_mvb]]
    runs = [[None if p.lower() == "none" else load_run(p) for p in row]
            for row in paths]
    if all(r is None for row in runs for r in row):
        raise SystemExit("nothing to plot: all four runs were 'none'.")

    share = {"row": "row", "all": True, "none": False}[args.share_y]
    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize),
                             sharex=True, sharey=share)

    for r in range(2):
        for c in range(2):
            ax, run = axes[r][c], runs[r][c]
            if run is None:
                ax.set_visible(False)
                continue
            draw_panel(ax, run, args)
            meta = run[2]
            if r == 0:
                ax.set_title(PRETTY_MODEL.get(meta["model_name"],
                                              os.path.basename(meta["model_name"])))
            if r == 1:
                ax.set_xlabel("visual tokens kept (%)")
            if c == 0 or share is False:
                ax.set_ylabel("accuracy / no pruning (%)" if args.y == "retention"
                              else "accuracy (%)")
            ax.annotate(panel_note(meta), xy=(0.98, 0.03), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=8, color="0.35")

    # Shared axes propagate set_ylim, so lift once per shared group -- doing it
    # per panel would stack the margin two (or four) times over.
    visible = [ax for ax in axes.ravel() if ax.get_visible()]
    if args.share_y == "all":
        groups = [visible]
    elif args.share_y == "row":
        groups = [[ax for ax in row if ax.get_visible()] for row in axes]
    else:
        groups = [[ax] for ax in visible]
    for group in groups:
        if group:
            headroom(group[0])

    # One legend for the whole figure, built from the fixed style table rather
    # than from whichever panel happened to be drawn first.
    order = [s for s in STYLE if any(run and s in run[2]["selectors"]
                                     for row in runs for run in row)]
    handles = [Line2D([], [], color=STYLE[s][0], ls=STYLE[s][1], marker="o", ms=4,
                      lw=1.6, label=s if args.raw_labels else PRETTY_SELECTOR.get(s, s))
               for s in order]
    handles.append(Line2D([], [], color="gray", lw=0.8, ls=":",
                          label="no pruning (per panel)"))
    if not args.no_floor and any(run and blind_pct(run[2]) is not None
                                 for row in runs for run in row):
        handles.append(Line2D([], [], color="black", lw=0.9, ls="-.",
                              label="text only, 0 visual tokens (per panel)"))
    fig.legend(handles=handles, loc="lower center", ncol=args.legend_ncol,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(args.title or "Accuracy vs. visual-token budget, "
                               "attention top-K against unranked floors")
    fig.tight_layout(rect=(0, args.legend_space, 1, 1))
    for path in args.out:
        fig.savefig(path, dpi=args.dpi)
        print(f"saved {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llava_ego", default=f"{RESULTS}/topk_ego_llavaov.json",
                   help="'none' blanks this panel.")
    p.add_argument("--llava_mvb", default=f"{RESULTS}/topk_mvb_llavaov.json",
                   help="'none' blanks this panel.")
    p.add_argument("--qwen_ego", default=f"{RESULTS}/topk_ego_qwen.json",
                   help="'none' blanks this panel.")
    p.add_argument("--qwen_mvb", default=f"{RESULTS}/topk_mvb_qwen.json",
                   help="'none' blanks this panel.")
    p.add_argument("--y", choices=["accuracy", "retention"], default="accuracy",
                   help="retention = accuracy / that panel's no-pruning accuracy.")
    p.add_argument("--share_y", choices=["row", "all", "none"], default="row",
                   help="row = one y range per benchmark (default).")
    p.add_argument("--raw_labels", action="store_true",
                   help="legend uses the bare selector names from the JSON.")
    p.add_argument("--no_floor", action="store_true",
                   help="hide the text-only floor line even where a run measured one.")
    p.add_argument("--legend_ncol", type=int, default=4)
    p.add_argument("--legend_space", type=float, default=0.075,
                   help="figure fraction reserved at the bottom for the legend.")
    p.add_argument("--figsize", type=float, nargs=2, default=[11.0, 7.5])
    p.add_argument("--title", default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out", nargs="+", default=[f"{RESULTS}/topk_panels.png"],
                   help="one or more output paths (e.g. a .png and a .pdf).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
