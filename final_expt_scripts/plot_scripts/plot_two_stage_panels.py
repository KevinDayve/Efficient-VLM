"""
plot_two_stage_panels.py -- the stage-one story, one column per model, from the JSONs
written by two_stage_accuracy.py.

    columns = model    LLaVA-OneVision-7B | Qwen2.5-VL-7B
    row (a) = curve    accuracy vs average visual retention across decoder layers
    row (b) = control  the same token budget reached by pruning vs by dropping frames

Same shape as plot_topk_panels.py: one colour per selector fixed across every panel, one
shared legend under the figure instead of a copy in each, y shared across a row so the
two backbones are comparable pixel-for-pixel.

Row (a) encodes the 2x2 the experiment actually is, rather than four arbitrary colours:

    colour      SELECTION    blue = max-min diversity, orange = unranked
    line style  DISPOSAL     solid = merged into nearest survivor, dashed = dropped

so both findings are readable off the figure without the legend -- the colours lie on
top of each other at every budget (selection does nothing), while solid and dashed
separate at the 1% end and nowhere else (merging hurts, but only once a survivor is an
average of a hundred tokens). The pair is #1b6ca8 / #d95f02, which clears CVD separation
at deltaE 21.3 (protan) and 3:1 contrast on white; line style carries the same
information redundantly, so the panel survives greyscale printing and colourblind
readers both.

Row (b) is deliberately ACHROMATIC. A colour in this figure means the selector and
nothing else -- the same convention plot_topk_panels.py holds to -- and the two routes
to a token budget are not selectors, so giving them the selection hues would assert a
correspondence that does not exist. Position and direct labels carry it instead. It is a
dot plot rather than bars because the interesting differences are 2-6 points on a 55-65
range: bars would either start at zero and hide them or start elsewhere and overstate
them, and dots carry no area, so a non-zero axis is honest.

x is AVERAGE VISUAL RETENTION ACROSS DECODER LAYERS, not the final keep rate. For
stage-one-only arms the two coincide (nothing is pruned mid-stack, so every layer sees
the same tokens); for the two-stage arms they do not, and plotting the final rate would
credit a layer-14 prune with a saving it does not make in layers 0..13. Retention is
computed PER CONFIG, not per budget -- see config_retention.

Stage-one arms only. The decoder-side (stage-two) arms are measured and live in
two_stage_ego_*.json, but they are not drawn here: at 30% retention they land on a curve
point and need their own marker vocabulary to be read at all, and the panel is worth more
as one uncluttered claim about stage one. Plot them separately if they are wanted.

A model with no sweep yet still gets its column, drawn with whatever anchors exist and
stamped as pending, so the figure can be regenerated unchanged as runs land.

Run
---
    python final_expt_scripts/plot_scripts/plot_two_stage_panels.py \
        --out final_expt_results/two_stage/two_stage_panels.png \
              final_expt_results/two_stage/two_stage_panels.pdf
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

RESULTS = "final_expt_results/two_stage"

# colour = selection, line style = disposal. See the module docstring.
SELECT_COLOR = {"maxmin": "#1b6ca8", "random": "#d95f02"}
DISPOSAL_STYLE = {True: "-", False: "--"}          # True = merged

ARMS = [("maxmin+none", "maxmin", True),
        ("random+none", "random", True),
        ("maxmin_nomerge+none", "maxmin", False),
        ("random_nomerge+none", "random", False)]

PRETTY_ARM = {"maxmin+none": "max-min, merged",
              "random+none": "unranked, merged",
              "maxmin_nomerge+none": "max-min, dropped",
              "random_nomerge+none": "unranked, dropped"}

PRETTY_MODEL = {"llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
                "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B"}

# Achromatic, so a colour never means anything but the selector.
CEILING = dict(color="0.25", lw=1.3, ls=":")
FLOOR = dict(color="black", lw=0.9, ls="-.")
DOT_REF = "0.62"
DOT_INK = "0.20"

# Everything that has to match for two JSONs to describe the same measurement.
# num_segments is deliberately NOT here: the frame control differs from its column in
# exactly that field, which is the point of it, so it is checked separately.
SETTING_KEYS = ("model_name", "tasks", "n_clips", "max_pixels", "min_pixels")


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def bench_key(meta: dict) -> tuple:
    """The benchmark a JSON was measured on, as a hashable key.

    Taken from `tasks` rather than from the filename: `tasks` is already one of the
    fields check_same_setting refuses to mix, so grouping on it means a run can never
    land in a row it does not belong to however the file was named."""
    return tuple(meta["tasks"])


def bench_label(key: tuple, n_tasks: int | None = None) -> str:
    """Row label. The task count comes from what was SEEN, never from `tasks`.

    `tasks` records what was asked for; a task whose videos are missing is skipped
    wholesale and never appears in per_task_seen. MVBench here is 19 of the 20, and a
    caption claiming 20 would be the figure asserting something the run did not measure."""
    if len(key) == 1:
        return key[0]
    return f"MVBench ({n_tasks or len(key)} tasks)"


def n_visual_tokens(summary_path: str) -> int:
    """Median visual-token count from the run's per-sample sidecar.

    Median rather than a config field: LLaVA-OneVision is constant per frame count, but
    Qwen's block length depends on each clip's aspect ratio, and a figure that prints an
    exact token count should print what was measured."""
    side = os.path.splitext(summary_path)[0] + "_per_sample.json"
    with open(side) as fh:
        rows = json.load(fh)
    return int(np.median([r["n_visual_tokens"] for r in rows]))


def check_same_setting(side: dict, ref: dict, path: str, strict: bool = True) -> list[str]:
    """Check `side` describes the same measurement as `ref`; return the mismatches.

    Fatal by default. A number measured under a different setting drawn on this panel is
    read as if it were measured under the panel's, which is the one failure mode these
    panels cannot survive -- and the dense reference for a column can legitimately come
    from a sibling experiment's JSON, so this is not a hypothetical."""
    drift = [f"{k}: {side.get(k)!r} vs {ref.get(k)!r}"
             for k in SETTING_KEYS if side.get(k) != ref.get(k)]
    if drift and strict:
        raise SystemExit(f"{path}: {'; '.join(drift)} -- not the same setting as the "
                         f"column it would decorate. Re-run it to match, or pass "
                         f"--allow_drift to plot it anyway.")
    if drift:
        print(f"[warn] {path}: {'; '.join(drift)}")
    return drift


def config_retention(meta: dict, spec: str, r1: float, r2: float) -> float:
    """Average retention in % for ONE config at this budget.

    `average_retention_accounting` is keyed on the budget, so every config in a run
    shares its number -- but a config names which stages actually run. `none+attn` skips
    stage one and therefore retains rho1 = 1, not the budget's 0.5, and belongs at 60%
    rather than 30%; reading the budget's figure would plot it two-fold cheaper than it
    is and put it under a curve it actually sits above."""
    s1, s2 = spec.split("+")
    e1 = 1.0 if s1 == "none" else r1
    e2 = 1.0 if s2 == "none" else r2
    k, n = meta["prune_layer"], meta["n_layers"]
    return 100 * e1 * (k + (n - k) * e2) / n


def per_sample(summary_path: str) -> dict:
    """The run's per-clip predictions, keyed by question so two runs can be joined."""
    side = os.path.splitext(summary_path)[0] + "_per_sample.json"
    with open(side) as fh:
        return {(r["task"], r["question_idx"]): r for r in json.load(fh)}


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on b/c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_contrast(a_path: str, a_key: str, b_path: str, b_key: str):
    """(delta in points, p) for arm a against arm b, paired over the clips both scored.

    Cross-file on purpose: the frame control and the sweep are separate runs over the same
    500 questions, so the only comparison the control row exists to make is available only
    by joining their sidecars per question. Doing it unpaired would discard the pairing --
    and the pairing is most of the power here, since the two arms agree on ~90% of clips."""
    A, B = per_sample(a_path), per_sample(b_path)
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    for src, key, path in ((A, a_key, a_path), (B, b_key, b_path)):
        if key not in src[keys[0]]["pred_by_config"]:
            raise SystemExit(f"{path}: no config {key!r} in the per-sample sidecar")
    b = c = na = nb = 0
    for k in keys:
        ca = A[k]["pred_by_config"][a_key] == A[k]["gold"]
        cb = B[k]["pred_by_config"][b_key] == B[k]["gold"]
        na, nb = na + ca, nb + cb
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    return 100 * (na - nb) / len(keys), mcnemar_p(b, c)


def curve_points(metas: list[dict], arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(x%, y%) for one arm, pooled across however many sweeps supply its budgets."""
    pts = []
    for meta in metas:
        for r1, r2 in meta["budgets"]:
            lab = f"{arm}@{r1:g}/{r2:g}"
            if lab in meta["accuracy_by_config"]:
                pts.append((config_retention(meta, arm, r1, r2),
                            100 * meta["accuracy_by_config"][lab]))
    pts.sort()
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.array(x), np.array(y)


def find_arm_at(metas: list[dict], arm: str, retention: float):
    """(meta, label) for `arm` at the budget whose average retention is `retention`%."""
    for meta in metas:
        for r1, r2 in meta["budgets"]:
            lab = f"{arm}@{r1:g}/{r2:g}"
            if (lab in meta["accuracy_by_config"]
                    and abs(config_retention(meta, arm, r1, r2) - retention) < 1e-6):
                return meta, lab
    return None, None


# --------------------------------------------------------------------------- #
# A column: everything measured for one model
# --------------------------------------------------------------------------- #
def build_column(curve_paths, frames8_path, dense_ref_path, strict: bool) -> dict:
    metas = [load(p) for p in curve_paths]
    for m, p in zip(metas, curve_paths):
        m["_tokens"] = n_visual_tokens(p)
        m["_path"] = p
    for m in metas[1:]:
        check_same_setting(m, metas[0], m["_path"], strict)
        if m["num_segments"] != metas[0]["num_segments"]:
            raise SystemExit(f"{m['_path']}: num_segments {m['num_segments']} vs "
                             f"{metas[0]['num_segments']} -- budgets from different frame "
                             "counts cannot share this x axis")

    frames8 = dense_ref = None
    if frames8_path:
        frames8 = load(frames8_path)
        frames8["_tokens"] = n_visual_tokens(frames8_path)
        frames8["_path"] = frames8_path
    if dense_ref_path:
        dense_ref = load(dense_ref_path)
        dense_ref["_tokens"] = n_visual_tokens(dense_ref_path)
        dense_ref["_path"] = dense_ref_path

    # The column's identity and its anchors come from the sweep when there is one, and
    # from the sidecar reference when there is not.
    anchor = metas[0] if metas else dense_ref
    if anchor is None:
        raise SystemExit("a column needs at least a sweep or a dense reference")
    if dense_ref is not None:
        check_same_setting(dense_ref, anchor, dense_ref["_path"], strict)
    if frames8 is not None:
        check_same_setting(frames8, anchor, frames8["_path"], strict)

    return {"metas": metas, "frames8": frames8,
            "model": anchor["model_name"], "tasks": anchor["tasks"],
            "n_clips": anchor["n_clips"], "frames": anchor["num_segments"],
            "n_tasks": len(anchor.get("per_task_seen") or anchor["tasks"]),
            "dense": 100 * anchor["full_accuracy"],
            "dense_tokens": anchor["_tokens"],
            "floor": (None if anchor.get("text_only_accuracy") is None
                      else 100 * anchor["text_only_accuracy"]),
            "pending": not metas}


# --------------------------------------------------------------------------- #
# row (a) -- the curve
# --------------------------------------------------------------------------- #
def panel_curve(ax, col: dict, title: str, dense_at_100: bool):
    dense, floor = col["dense"], col["floor"]

    for arm, select, merged in ARMS:
        x, y = curve_points(col["metas"], arm)
        if not x.size:
            continue
        if dense_at_100:                      # rho=1 is the identity, i.e. the dense run
            x, y = np.append(x, 100.0), np.append(y, dense)
        ax.plot(x, y, color=SELECT_COLOR[select], ls=DISPOSAL_STYLE[merged],
                marker="o", ms=4.5, lw=1.8, zorder=3)

    # Both reference labels sit at the LEFT edge: the curves are lowest there, and the
    # right-hand side is where the dense point lives.
    ax.axhline(dense, **CEILING, zorder=1)
    ax.annotate(f"no pruning  {dense:.1f}%", xy=(0.015, dense),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="left", fontsize=8,
                color=CEILING["color"])
    if floor is not None:
        ax.axhline(floor, **FLOOR, zorder=1)
        ax.annotate(f"no visual tokens  {floor:.1f}%", xy=(0.015, floor),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="left", fontsize=8, color="0.15")

    if col["pending"]:
        ax.annotate("sweep not yet measured", xy=(0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=10, color="0.55", style="italic")

    ax.set_xscale("log")
    ax.set_xlim(0.8, 130)
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("average visual retention across decoder layers (%)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="both", color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# row (b) -- the iso-token control
# --------------------------------------------------------------------------- #
def panel_frames(ax, col: dict, title: str, iso_arm: str, iso_retention: float):
    """Ways to spend a token budget, two of them identical in tokens.

    The point is that the two matched rows cost the same and differ only in WHERE the
    tokens came from -- half of every frame, or all of half the frames.

    That contrast is stated with its paired p, because the two backbones split on it: the
    same picture is a solid effect on one and not resolvable on the other, and three dots
    laid out identically in both panels otherwise invite reading the second as confirming
    the first."""
    frames8 = col["frames8"]
    rows = [("%d frames, dense" % col["frames"], col["dense_tokens"], col["dense"], DOT_REF)]
    if frames8 is not None:
        rows.append(("%d frames, dense" % frames8["num_segments"],
                     frames8["_tokens"], 100 * frames8["full_accuracy"], DOT_INK))
    pruned, lab = find_arm_at(col["metas"], iso_arm, iso_retention)
    if pruned is not None:
        # The pruned row's token count is the run's MEASURED keep rate against its own
        # block, so the figure cannot claim a budget the run did not actually hit.
        n_tok = int(round(pruned["realised_visual_keep_rate"][lab] * pruned["_tokens"]))
        rows.append(("%d frames, %.0f%% pruned" % (col["frames"], iso_retention),
                     n_tok, 100 * pruned["accuracy_by_config"][lab], DOT_INK))

    ys = np.arange(len(rows))[::-1]
    base = col["dense"]
    ax.axvline(base, **CEILING, zorder=1)
    for y, (label, ntok, acc, color) in zip(ys, rows):
        ax.plot([base, acc], [y, y], color=color, lw=1.4, alpha=0.5, zorder=2,
                solid_capstyle="butt")
        ax.plot([acc], [y], marker="o", ms=9, color=color, zorder=3)
        ax.annotate(f"{acc:.1f}%", xy=(acc, y), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9, color="0.15")
        ax.annotate(f"{ntok} tokens", xy=(base, y), xytext=(6, -15),
                    textcoords="offset points", ha="left", fontsize=7.5, color="0.5")

    # The iso-token contrast, stated rather than left to the eye. Both rows carry the same
    # token count, so this is the only difference in the panel that is not confounded.
    if frames8 is not None and pruned is not None:
        got = paired_contrast(pruned["_path"], lab, frames8["_path"], "full")
        if got is not None:
            delta, p = got
            ax.annotate(f"matched budget: {delta:+.1f} pts for pruning · p = {p:.3f}"
                        + ("" if p < 0.05 else "  (n.s.)"),
                        xy=(0.5, 0.045), xycoords="axes fraction", ha="center",
                        fontsize=8, color="0.15" if p < 0.05 else "0.45",
                        bbox=dict(fc="white", ec="none", pad=1.5))

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    # Bottom margin leaves a clear lane for the contrast line; at -0.7 it ran through the
    # last row's token label.
    ax.set_ylim(-1.05, len(rows) - 0.3)
    ax.margins(x=0.18)
    ax.set_xlabel("accuracy (%)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if pruned is None:
        ax.annotate(f"no {iso_retention:.0f}%% arm measured yet".replace("%%", "%"),
                    xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
                    fontsize=8, color="0.55", style="italic")


def main(args):
    exists = lambda ps: [p for p in (ps or []) if p and os.path.exists(p)]

    # models x benchmarks. Each model supplies flat lists of JSONs; which benchmark row
    # a file belongs to is read off the file, so adding MVBench is adding paths.
    models, benches = [], []
    for curve, frames8, dense_ref in (
            (args.llava_curve, args.llava_frames8, args.llava_dense_ref),
            (args.qwen_curve, args.qwen_frames8, args.qwen_dense_ref)):
        by_bench = {}
        groups = {"curve": exists(curve), "frames8": exists(frames8),
                  "dense_ref": exists(dense_ref)}
        for kind, paths in groups.items():
            for p in paths:
                by_bench.setdefault(bench_key(load(p)), {}).setdefault(kind, []).append(p)
        if by_bench:
            models.append(by_bench)
            for k in by_bench:
                if k not in benches:
                    benches.append(k)
    if not models:
        raise SystemExit("nothing to plot -- no sweep or dense reference found")
    # EgoSchema (a single task) before MVBench (twenty), then by label.
    benches.sort(key=lambda k: (len(k) > 1, bench_label(k)))

    cells = {}                      # (bench, model index) -> column, or None
    for j, by_bench in enumerate(models):
        for b in benches:
            g = by_bench.get(b, {})
            if not g.get("curve") and not g.get("dense_ref"):
                continue
            cells[(b, j)] = build_column(
                g.get("curve", []), (g.get("frames8") or [None])[0],
                (g.get("dense_ref") or [None])[0], not args.allow_drift)

    # One curve row per benchmark, then one iso-token row per benchmark that has a frame
    # control. A benchmark with no frame control simply contributes no second row.
    rows = [("curve", b) for b in benches]
    rows += [("frames", b) for b in benches
             if any(cells.get((b, j), {}) and cells[(b, j)]["frames8"] is not None
                    for j in range(len(models)))]

    h = {"curve": 1.5, "frames": 1.0}
    fig, axes = plt.subplots(len(rows), len(models),
                             figsize=(args.figsize[0], sum(h[k] for k, _ in rows) * 2.6),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [h[k] for k, _ in rows]})
    tags, t = "abcdefghijkl", 0
    for i, (kind, b) in enumerate(rows):
        for j, by_bench in enumerate(models):
            ax = axes[i][j]
            col = cells.get((b, j))
            if col is None:
                ax.set_visible(False)
                continue
            name = PRETTY_MODEL.get(col["model"], col["model"])
            title = f"({tags[t]})  {name}"
            t += 1
            if kind == "curve":
                panel_curve(ax, col, title, not args.no_dense_point)
            else:
                panel_frames(ax, col, title, args.iso_arm, args.iso_retention)
        first = next((axes[i][j] for j in range(len(models))
                      if axes[i][j].get_visible()), None)
        if first is not None:
            row_cells = [cells[(b, j)] for j in range(len(models)) if (b, j) in cells]
            ns = sorted({c["n_clips"] for c in row_cells})
            # Cells in a row can rest on different clip counts while one of them is still
            # a sidecar reference; say so rather than picking one and implying agreement.
            clips = f"{ns[0]}" if len(ns) == 1 else f"{ns[0]}–{ns[-1]}"
            lab = bench_label(b, max(c["n_tasks"] for c in row_cells))
            first.set_ylabel(f"{lab}, {clips} clips\naccuracy (%)"
                             if kind == "curve" else lab)

    # Shared scales WITHIN a row only: two backbones on one benchmark is the comparison
    # worth making pixel-for-pixel, while EgoSchema and MVBench sit at different accuracy
    # levels and sharing across them would flatten both.
    for i, (kind, _) in enumerate(rows):
        vis = [ax for ax in axes[i] if ax.get_visible()]
        if len(vis) < 2:
            continue
        if kind == "curve":
            lo, hi = min(a.get_ylim()[0] for a in vis), max(a.get_ylim()[1] for a in vis)
            for ax in vis:
                ax.set_ylim(lo, hi)
            for ax in vis[1:]:
                ax.tick_params(labelleft=False)
        else:
            lo, hi = min(a.get_xlim()[0] for a in vis), max(a.get_xlim()[1] for a in vis)
            for ax in vis:
                ax.set_xlim(lo, hi)

    handles = [Line2D([], [], color=SELECT_COLOR[s], ls=DISPOSAL_STYLE[m],
                      marker="o", ms=4.5, lw=1.8, label=PRETTY_ARM[a])
               for a, s, m in ARMS]
    handles += [Line2D([], [], **CEILING, label="no pruning"),
                Line2D([], [], **FLOOR, label="no visual tokens")]
    # handlelength has to show a full dash cycle -- at the default the dropped arms
    # render as solid and the panel's second finding disappears.
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               handlelength=3.0, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))

    # Benchmark and clip count live on each row's label, so the suptitle carries only
    # what every panel shares.
    frames = sorted({c["frames"] for c in cells.values()})
    fig.suptitle(f"{'/'.join(map(str, frames))} frames per clip · "
                 f"stage one only, no decoder-side prune", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, args.legend_space, 1, 0.975))
    for path in args.out:
        fig.savefig(path, dpi=args.dpi)
        print(f"wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage-one retention curve and iso-token "
                                            "frame control, one column per model.")
    # Flat lists per model; the benchmark row each file belongs to is read off the file.
    p.add_argument("--llava_curve", nargs="+",
                   default=[f"{RESULTS}/stage1_curve_ego_llavaov.json",
                            f"{RESULTS}/stage1_only_ego_llavaov.json",
                            f"{RESULTS}/stage1_mid_ego_llavaov.json",
                            f"{RESULTS}/stage1_curve_mvb_llavaov.json"])
    p.add_argument("--llava_frames8", nargs="+",
                   default=[f"{RESULTS}/frames8_ego_llavaov.json",
                            f"{RESULTS}/frames8_mvb_llavaov.json"])
    p.add_argument("--llava_dense_ref", nargs="+", default=[],
                   help="fallback for a cell's dense/floor anchors when it has no sweep "
                        "yet; must be the same setting (see SETTING_KEYS).")
    p.add_argument("--qwen_curve", nargs="+",
                   default=[f"{RESULTS}/stage1_curve_ego_qwen.json",
                            f"{RESULTS}/stage1_hi_ego_qwen.json",
                            f"{RESULTS}/stage1_hi2_ego_qwen.json",
                            f"{RESULTS}/stage1_curve_mvb_qwen.json"])
    p.add_argument("--qwen_frames8", nargs="+",
                   default=[f"{RESULTS}/frames8_ego_qwen.json",
                            f"{RESULTS}/frames8_mvb_qwen.json"])
    # floor_* rather than topk_*: both report the same dense accuracy, but only these
    # also carry text_only_accuracy, so a pending cell gets both anchors.
    p.add_argument("--qwen_dense_ref", nargs="+",
                   default=["final_expt_results/topk_accuracy/floor_ego_qwen.json",
                            "final_expt_results/topk_accuracy/floor_mvb_qwen.json"])
    p.add_argument("--iso_arm", default="random+none",
                   help="which pruned arm row (b) shows against the frame control.")
    p.add_argument("--iso_retention", type=float, default=50.0,
                   help="average retention %% matching the frame control's token count.")
    p.add_argument("--no-dense_point", dest="no_dense_point", action="store_true",
                   help="do not close each curve at (100%%, dense).")
    p.add_argument("--allow_drift", action="store_true",
                   help="downgrade a setting mismatch from fatal to a warning.")
    p.add_argument("--figsize", type=float, nargs=2, default=[11.0, 7.0])
    p.add_argument("--legend_space", type=float, default=0.07)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", nargs="+", default=[f"{RESULTS}/two_stage_panels.png"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
