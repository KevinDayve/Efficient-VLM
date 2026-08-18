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

The diversity selectors (`div_*`: the same band's attention score selected greedily
under an MMR redundancy penalty) are solid too -- they are the method, not a floor --
but carry a square marker instead of a circle, because the comparison they exist for
is div_mid against attn_mid on the same panel and that pair has to be separable
without reading the legend. A div sweep is usually its own JSON rather than extra
columns in the top-K one, so it is folded into a panel with --extra_llava_ego and
friends, under the same same-model/benchmark/clips check the floors get.

The no-pruning ceiling is per panel -- it is a property of the model/benchmark
pair, not of a selector -- so it stays a grey dotted line inside each panel,
labelled with its own value rather than pushed into the shared legend. The
text-only floor (`text_only_accuracy`, every visual token dropped) is drawn the
same way, in black dash-dot: it is likewise per panel and likewise not a
selector, and together the two lines bracket the band inside which any of these
curves can say anything. Runs written before the floor existed simply omit the
key and lose the line -- but the floor is a property of the model/benchmark pair,
not of the sweep, so it can also be measured on its own (a rho-less
stage_topk_accuracy.py run with --text_only) and attached to a panel with
--floor_llava_ego and friends. Such a sidecar must come from the same model,
benchmark, clip set and (on Qwen) per-frame pixel budget as the sweep it decorates,
which is checked, not assumed -- see SETTING_KEYS. A mismatch is fatal, because a
curve measured under a different setting is read off this figure as if it shared the
panel's; --allow_drift downgrades that to a warning and stamps the panel with what
differs, for a mismatch you know about and intend to show.

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
    python final_expt_scripts/plot_scripts/plot_topk_panels.py \
        --floor_llava_ego final_expt_results/topk_accuracy/floor_ego_llavaov.json \
        --extra_llava_ego final_expt_results/topk_accuracy/div_ego_llavaov.json \
        --out final_expt_results/topk_accuracy/topk_panels.png final_expt_results/topk_accuracy/topk_panels.pdf

The four sweeps, four floors and all four div overlays are the defaults, so the figure
as committed is just

    python final_expt_scripts/plot_scripts/plot_topk_panels.py --allow_drift --no_drift_note \
        --out final_expt_results/topk_accuracy/topk_panels.png final_expt_results/topk_accuracy/topk_panels.pdf

--no_drift_note keeps the two Qwen panels clean, so the caveats below are NOT on the
figure and have to be carried by the caption instead; drop the flag to have each panel
stamp its own. --allow_drift is currently carrying two known mismatches:

    div_ego_qwen.json was swept at native per-frame resolution while topk_ego_qwen.json
    used --max_pixels/--min_pixels 200704, so its div_mid curve sits on a different
    unpruned model (62.9% vs the panel's 61.0%) and can cross the panel's ceiling
    without that meaning anything.

    div_mvb_qwen.json saw all 3800 MVBench clips of the 19 tasks it ran, while
    topk_mvb_qwen.json was missing 14 of them (12 Action Sequence, 2 Object Existence)
    at the time it was swept. Same model, resolution and frame count, so the curves are
    comparable in a way the EgoSchema pair is not, but the clip sets are not identical
    and the div run's own no-pruning accuracy is 61.2% against the panel's 60.9%.

Re-running the drifted sweep on the panel's setting removes both the flag and the stamp.
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

# One colour + line style + marker per SELECTOR, fixed across every panel, so a
# colour always means the selector and never the model or the benchmark. Colours
# keep the C0..C6 order the per-run plots in stage_topk_accuracy.py already use, so
# these panels and the single-run PNGs stay readable side by side; the div_*
# entries continue past C6 and take a square marker (see the module docstring).
STYLE = {"attn_early":      ("C0", "-",  "o"),
         "attn_mid":        ("C1", "-",  "o"),
         "div_early":       ("C8", "-",  "s"),
         "div_mid":         ("C9", "-",  "s"),
         "random":          ("C2", "--", "o"),
         "uniform":         ("C3", "--", "o"),
         "uniform_stagger": ("C4", "--", "o"),
         "bot_early":       ("C5", "-.", "o"),
         "bot_mid":         ("C6", "-.", "o")}

DEFAULT_STYLE = (None, "-", "o")

PRETTY_SELECTOR = {"attn_early":      "attn top-K (early)",
                   "attn_mid":        "attn top-K (mid)",
                   "div_early":       "attn + MMR diversity (early)",
                   "div_mid":         "attn + MMR diversity (mid)",
                   "random":          "random",
                   "uniform":         "uniform",
                   "uniform_stagger": "uniform (staggered)",
                   "bot_early":       "attn bottom-K (early)",
                   "bot_mid":         "attn bottom-K (mid)"}

PRETTY_MODEL = {"llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}

# The two per-panel reference lines. They stay achromatic so a colour never means
# anything but a selector, but dark enough to read against the 0.9 grid -- pale
# grey on white lost the ceiling exactly where it matters, next to the curves that
# are about to cross it. Kept here rather than inline so the panels and the shared
# legend cannot drift apart.
CEILING = dict(color="0.25", lw=1.3, ls=":")
FLOOR = dict(color="black", lw=0.9, ls="-.")


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


# Everything that has to match for a sidecar's numbers to mean the same thing as the
# panel's. max_pixels/min_pixels are in here because on Qwen they set the per-frame
# resolution and therefore n_visual_tokens: a sweep run with the clamp and a sidecar
# run without it decode different inputs, which moves the unpruned accuracy by more
# than the token budget does and makes the curves incomparable even though the model,
# benchmark and clip list all match.
SETTING_KEYS = ("model_name", "tasks", "n_clips", "num_segments",
                "max_pixels", "min_pixels")


def check_same_setting(side, meta, path: str, strict: bool = True):
    """Check that `side` (a sidecar JSON) is the same run setting as the panel it would
    decorate; return the list of human-readable mismatches.

    Fatal by default -- a curve measured under a different setting drawn on this panel
    is read as if it were measured under the panel's, which is the one failure mode
    these panels cannot survive. `strict=False` (--allow_drift) downgrades it to a
    warning for the case where the mismatch is known and the panel is going to be
    stamped with it, so it is on the figure rather than only in the shell.

    full_accuracy is never fatal -- it is a re-decode of the same unpruned model and
    can move a clip or two -- so it warns on its own, but a large gap is the symptom
    that one of the SETTING_KEYS above should have caught and did not."""
    drift = [f"{key} {side.get(key)!r} vs {meta.get(key)!r}"
             for key in SETTING_KEYS if side.get(key) != meta.get(key)]
    if drift and strict:
        raise SystemExit(f"{path}: {'; '.join(drift)}. This is not the same setting as "
                         f"the sweep it would decorate -- re-run it to match, or pass "
                         f"--allow_drift to plot it anyway with the panel stamped.")
    if drift:
        print(f"warning: {path} differs from the sweep it decorates: {'; '.join(drift)}.")

    delta = 100 * (side["full_accuracy"] - meta["full_accuracy"])
    if abs(delta) > 0.05:
        print(f"warning: {path} full_accuracy differs from the sweep's by "
              f"{delta:+.1f} pts; keeping the sweep's ceiling.")
    return drift


def attach_extra(run, path: str, strict: bool = True):
    """Fold a second sweep's selector curves into `run`'s panel, in place.

    The diversity selectors were swept separately from the top-K ones, so div_mid
    lives in its own JSON. Overlaying it is only worth doing if it is the same
    comparison -- same model, benchmark, clips and rho grid -- so all of that is
    checked rather than assumed, and a selector the panel already has is refused
    instead of silently overwriting the curve it is meant to be compared against.

    The rho grid stays fatal even under --allow_drift: a curve plotted against x
    values it was not measured at is not a caveat, it is a wrong line."""
    if run is None:
        raise SystemExit(f"{path}: extra run given for a panel that has no run.")
    xs, curves, meta = run
    xs_x, curves_x, extra = load_run(path)
    drift = check_same_setting(extra, meta, path, strict=strict)
    if len(xs_x) != len(xs) or not np.allclose(xs_x, xs):
        raise SystemExit(f"{path}: rhos are {list(xs_x)} but the sweep it would "
                         f"decorate has {list(xs)}.")

    for s in extra["selectors"]:
        if s in curves:
            raise SystemExit(f"{path}: selector {s!r} is already in the panel's sweep.")
        curves[s] = curves_x[s]
        meta["selectors"].append(s)

    # A drifted overlay carries its own no-pruning accuracy, which is the number its
    # curve should be read against -- not the panel's ceiling line. Keep both the
    # mismatch and that accuracy so the panel can print them where the curve is.
    if drift:
        meta.setdefault("_drift", []).append(
            (os.path.basename(path), drift, 100 * extra["full_accuracy"]))


def attach_floor(run, path: str):
    """Fold a standalone text-only run's floor into `run`'s meta, in place.

    The sweeps predate the floor, so theirs is measured separately: same model,
    same benchmark, same clips, no rhos, just `--text_only`. Attaching it is only
    honest if it really is the same setting, so the identifying fields are
    compared and a mismatch is fatal rather than quietly plotted. full_accuracy is
    the one field allowed to drift -- it is a re-decode of the same unpruned model
    and can move a clip or two -- so it warns instead."""
    if run is None:
        raise SystemExit(f"{path}: floor given for a panel that has no run.")
    meta = run[2]
    with open(path) as fh:
        floor = json.load(fh)

    check_same_setting(floor, meta, path)
    blind = blind_pct(floor)
    if blind is None:
        raise SystemExit(f"{path}: no text_only_accuracy to attach.")
    meta["text_only_accuracy"] = floor["text_only_accuracy"]


def draw_panel(ax, run, args, stamped: bool = False):
    """One model x benchmark panel: every selector, plus its own ceiling and floor.

    `stamped` says a drift note is about to be boxed into the top-left corner, which
    is where the ceiling label would otherwise sit on a panel whose ceiling is high:
    the label moves right of the box rather than under it."""
    xs, curves, meta = run
    full = 100 * meta["full_accuracy"]
    scale = (100 / full) if args.y == "retention" else 1.0

    for s in meta["selectors"]:
        color, ls, marker = STYLE.get(s, DEFAULT_STYLE)
        ax.plot(xs, scale * curves[s], ls, marker=marker, ms=4, lw=1.6, color=color)

    ceiling = 100.0 if args.y == "retention" else full
    ax.axhline(ceiling, **CEILING)
    ax.annotate(f"no pruning ({full:.1f}%)",
                xy=(0.36 if stamped else xs[0], ceiling),
                xycoords=("axes fraction", "data") if stamped else "data",
                xytext=(0, 3), textcoords="offset points", fontsize=8,
                color=CEILING["color"], va="bottom")

    # The floor, when the run measured one. Annotated BELOW its line so it cannot
    # collide with the ceiling label on a panel where the two sit close together --
    # which is itself the finding worth seeing.
    blind = blind_pct(meta)
    if blind is not None and not args.no_floor:
        ax.axhline(scale * blind, **FLOOR)
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


def drift_note(meta) -> str | None:
    """Top-left stamp naming any overlay that is NOT the panel's setting.

    Only reachable under --allow_drift. It names the file, what differs and the
    overlay's own no-pruning accuracy, because that last number is what its curve
    has to be read against -- the panel's dotted ceiling belongs to the sweep, and
    an overlay from a different setting can sit above it without meaning anything."""
    entries = meta.get("_drift")
    if not entries:
        return None
    lines = []
    for name, diffs, full in entries:
        lines.append(f"! {name} is a different setting:")
        lines += [f"    {d}" for d in diffs]
        lines.append(f"    its own no pruning: {full:.1f}%")
    return "\n".join(lines)


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

    # Extra selector curves first, so a panel's ceiling/floor checks and its y range
    # both see the full set of curves that will be drawn on it.
    extras = [[args.extra_llava_ego, args.extra_qwen_ego],
              [args.extra_llava_mvb, args.extra_qwen_mvb]]
    floors = [[args.floor_llava_ego, args.floor_qwen_ego],
              [args.floor_llava_mvb, args.floor_qwen_mvb]]
    for r in range(2):
        for c in range(2):
            for path in extras[r][c]:
                if path.lower() != "none":
                    attach_extra(runs[r][c], path, strict=not args.allow_drift)
            path = floors[r][c]
            if path.lower() != "none":
                attach_floor(runs[r][c], path)

    share = {"row": "row", "all": True, "none": False}[args.share_y]
    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize),
                             sharex=True, sharey=share)

    for r in range(2):
        for c in range(2):
            ax, run = axes[r][c], runs[r][c]
            if run is None:
                ax.set_visible(False)
                continue
            meta = run[2]
            note = None if args.no_drift_note else drift_note(meta)
            draw_panel(ax, run, args, stamped=bool(note))
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
            # Top-left: the one corner no curve uses on these panels, and boxed so it
            # stays readable if a future run puts one there anyway. The ceiling label
            # does live there, which is why draw_panel was told about the stamp.
            if note:
                ax.annotate(note, xy=(0.02, 0.97), xycoords="axes fraction",
                            ha="left", va="top", fontsize=6.5, color="0.15",
                            linespacing=1.4,
                            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                      ec="0.6", lw=0.6, alpha=0.9))

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
    handles = [Line2D([], [], color=STYLE[s][0], ls=STYLE[s][1], marker=STYLE[s][2],
                      ms=4, lw=1.6,
                      label=s if args.raw_labels else PRETTY_SELECTOR.get(s, s))
               for s in order]
    handles.append(Line2D([], [], label="no pruning (per panel)", **CEILING))
    if not args.no_floor and any(run and blind_pct(run[2]) is not None
                                 for row in runs for run in row):
        handles.append(Line2D([], [], label="text only, 0 visual tokens (per panel)",
                              **FLOOR))
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
    # The sweeps themselves carry no text_only_accuracy, so each panel's floor
    # comes from its own standalone run; 'none' leaves that panel without one.
    p.add_argument("--floor_llava_ego", default=f"{RESULTS}/floor_ego_llavaov.json")
    p.add_argument("--floor_llava_mvb", default=f"{RESULTS}/floor_mvb_llavaov.json")
    p.add_argument("--floor_qwen_ego", default=f"{RESULTS}/floor_ego_qwen.json")
    p.add_argument("--floor_qwen_mvb", default=f"{RESULTS}/floor_mvb_qwen.json")
    # Selectors swept separately from the main top-K run (the div_* sweeps) are
    # overlaid on their panel from their own JSON; repeat the flag for more than one.
    p.add_argument("--extra_llava_ego", nargs="+", default=[f"{RESULTS}/div_ego_llavaov.json"],
                   help="extra sweep JSONs whose selectors join this panel; 'none' for none.")
    p.add_argument("--extra_llava_mvb", nargs="+", default=[f"{RESULTS}/div_mvb_llavaov.json"],
                   help="extra sweep JSONs whose selectors join this panel; 'none' for none.")
    p.add_argument("--extra_qwen_ego", nargs="+", default=[f"{RESULTS}/div_ego_qwen.json"],
                   help="extra sweep JSONs whose selectors join this panel; 'none' for none.")
    p.add_argument("--extra_qwen_mvb", nargs="+", default=[f"{RESULTS}/div_mvb_qwen.json"],
                   help="extra sweep JSONs whose selectors join this panel; 'none' for none.")
    p.add_argument("--y", choices=["accuracy", "retention"], default="accuracy",
                   help="retention = accuracy / that panel's no-pruning accuracy.")
    p.add_argument("--share_y", choices=["row", "all", "none"], default="row",
                   help="row = one y range per benchmark (default).")
    p.add_argument("--raw_labels", action="store_true",
                   help="legend uses the bare selector names from the JSON.")
    p.add_argument("--no_floor", action="store_true",
                   help="hide the text-only floor line even where a run measured one.")
    p.add_argument("--no_drift_note", action="store_true",
                   help="hide the in-panel --allow_drift stamp. The mismatch is still "
                        "printed to stderr on every run; the figure just stops "
                        "carrying it, so say it in the caption instead.")
    p.add_argument("--allow_drift", action="store_true",
                   help="plot an --extra_* overlay whose setting does not match the "
                        "panel's, stamping the panel with what differs, instead of "
                        "refusing it. For a known mismatch you intend to show; the "
                        "curve is not comparable to the panel's other curves.")
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
