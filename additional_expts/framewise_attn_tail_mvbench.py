"""
framewise_attn_tail_mvbench.py -- per-FRAME EVT tail index (xi) of the language->video
attention.  Is every frame's tail equally heavy, or do frames differ?
======================================================================================
Frame-resolved sibling of ``layerwise_attn_tail_mvbench.py``.  Same forward, same
attention read-out, same Dekkers-Einmahl-de Haan moment estimator, same official MVBench
protocol -- only the distribution xi is estimated ON changes.

The layerwise script pools ALL M video tokens of a clip into one distribution per layer
and asks which extreme-value domain its upper tail is in.  That pooled tail mixes two
effects: some frames may be heavy-tailed internally, AND some frames may simply soak up
more mass than others.  Here we split the temporal axis out and ask the narrower
question, per frame:

  p_i^(t) = attention mass video-token i of FRAME t receives, summed over all text
            queries, renormalized WITHIN the frame so sum_i p_i^(t) = 1

  xi(layer, t) = tail index of that within-frame distribution
      xi > 0  Frechet   -> this frame's attention sits on a few spatial tokens
      xi = 0  Gumbel    -> the frame's tail IS exponential
      xi < 0  Weibull   -> bounded / light -> toward uniform over the frame (xi = -1)

So xi(t) flat across t => every frame concentrates the same way and there is nothing to
gain by treating frames differently.  xi(t) varying across t => the heavy tail is a
property of PARTICULAR frames, and a temporal budget could exploit it.

TOKEN -> FRAME MAP.  Qwen2.5-VL pairs adjacent frames (temporal_patch_size=2) and merges
2x2 spatially, so the M video tokens are laid out as (T, S) with T = video_grid_thw[0] =
num_segments/2 temporal GROUPS and S = (H/merge)*(W/merge) tokens per group.  "Frame" in
this script therefore means a frame PAIR -- the finest temporal unit the model exposes.

Reported, mirroring the sibling: mean xi +/- SEM per (layer, frame); the xi of the exact
band-averaged distribution the scorer ranks on (layers 12..16), per frame and per task;
and the across-frame SPREAD std_t(xi) per clip, which is the direct answer to "do the
frames differ?" -- compare it against its own SEM, not against zero.

Run:
    python framewise_attn_tail_mvbench.py \
        --data_root ~/Experiments/MVBench \
        --tasks "Action Sequence" "Object Existence" "Scene Transition" \
        --official_sampling --num_segments 16 --max_samples 40 \
        --out results_framewise_attn_tail_mvbench.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

# Reuse the shared position builder ...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oracle_check import build_full_positions
# ... the OFFICIAL MVBench data/prompt/sampling protocol ...
from inference import DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt
# ... and the estimator itself, so both scripts measure xi identically.
from layerwise_attn_tail_mvbench import evt_xi, per_layer_video_attention

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    band = [int(x) for x in args.baseline_band.split(",")] if args.baseline_band else []
    merge = model.config.vision_config.spatial_merge_size

    # min_pixels: default mirrors max_pixels (fixed per-clip token budget across clips);
    # <=0 disables the floor (variable budget, capped only by max_pixels).
    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    xi_lt = None          # (n_layers, T) -> list of per-clip xi
    band_xi_frame = None  # T -> list of per-clip band-averaged xi
    band_xi_task = {t: None for t in tasks}   # task -> same, per frame
    band_share = None     # T -> list of per-clip band-averaged attention SHARE of the frame
    spread = []           # per-clip std of the band xi ACROSS frames
    n_layers = None
    n_frames = None
    seen = {t: 0 for t in tasks}
    n = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task, unit="clip"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, _letters, _gt = build_prompt(rec)
                prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                             args.max_frames, args.max_pixels, args.fps,
                                             official=args.official_sampling,
                                             num_segments=args.num_segments,
                                             min_pixels=min_pixels)
                chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                chat += ANSWER_PREFIX  # match the scorer's queries (answer-forcing suffix)
                img_in, vid_in = process_vision_info(prompt)
                inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            input_ids = inputs["input_ids"].to(device)
            attn = inputs["attention_mask"].to(device)
            vpos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
            grid_thw = inputs["video_grid_thw"].to(device)
            pix = inputs["pixel_values_videos"].to(device)

            T = int(grid_thw[0, 0])
            S = (int(grid_thw[0, 1]) // merge) * (int(grid_thw[0, 2]) // merge)
            if T * S != vpos.numel():
                tqdm.write(f"skip [{task}] {rec.get('video')}: grid T*S={T*S} vs "
                           f"{vpos.numel()} video tokens")
                continue
            if S < 12:  # the moment estimator needs enough within-frame order statistics
                tqdm.write(f"skip [{task}] {rec.get('video')}: only {S} tokens per frame")
                continue
            # Frames are compared position-by-position across clips, so T must not vary.
            if n_frames is not None and T != n_frames:
                tqdm.write(f"skip [{task}] {rec.get('video')}: T={T} != {n_frames} "
                           f"(clip has a different frame count)")
                continue

            try:
                with torch.no_grad():
                    ve = model.get_video_features(pix, grid_thw).pooler_output
                    ve = torch.cat(ve, dim=0).to(device)
                    base = model.get_input_embeddings()(input_ids).clone()
                    base[0, vpos] = ve.to(base.dtype)
                pos = build_full_positions(model, input_ids, vpos, grid_thw, attn)
                dists, L = per_layer_video_attention(model, base, vpos, pos, attn)
            except Exception as e:
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            if xi_lt is None:
                n_layers, n_frames = L, T
                bad = [b for b in band if b >= L]
                if bad:
                    raise ValueError(f"--baseline_band {bad} >= n_layers {L}")
                xi_lt = [[[] for _ in range(T)] for _ in range(L)]
                band_xi_frame = [[] for _ in range(T)]
                band_share = [[] for _ in range(T)]
                band_xi_task = {t: [[] for _ in range(T)] for t in tasks}

            grid = dists.reshape(L, T, S)          # (layer, frame, within-frame token)
            for li in range(n_layers):
                for t in range(T):
                    x = evt_xi(grid[li, t], args.k_frac)   # xi is scale-invariant: no renorm needed
                    if x == x:                             # skip NaNs
                        xi_lt[li][t].append(x)

            # The distribution the scorer actually ranks on: the band average, per frame.
            if band:
                bg = grid[band].mean(0)                    # (T, S)
                share = bg.sum(1) / bg.sum()               # frame's share of the clip's mass
                xs = np.array([evt_xi(bg[t], args.k_frac) for t in range(T)])
                for t in range(T):
                    if xs[t] == xs[t]:
                        band_xi_frame[t].append(float(xs[t]))
                        band_xi_task[task][t].append(float(xs[t]))
                    band_share[t].append(float(share[t]))
                ok = xs[~np.isnan(xs)]
                if ok.size > 1:
                    spread.append(float(ok.std()))         # do the frames differ, in this clip?

            seen[task] += 1
            n += 1
            del base, ve, dists, grid
            torch.cuda.empty_cache()

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    def _stats(vals):
        a = np.array([v for v in vals if v == v], dtype=np.float64)
        if not a.size:
            return {"mean": float("nan"), "sem": float("nan"), "n": 0}
        return {"mean": float(a.mean()), "sem": float(a.std() / np.sqrt(a.size)), "n": int(a.size)}

    eps = args.domain_eps
    layers, frames = list(range(n_layers)), list(range(n_frames))
    xi_mean = np.array([[_stats(xi_lt[li][t])["mean"] for t in frames] for li in layers])
    xi_sem = np.array([[_stats(xi_lt[li][t])["sem"] for t in frames] for li in layers])

    valid = [t for t in tasks if seen[t]]
    band_summary = None
    if band:
        band_summary = {
            "per_frame": [_stats(band_xi_frame[t]) for t in frames],
            "share_per_frame": [_stats(band_share[t]) for t in frames],
            "spread_across_frames": _stats(spread),
            "per_task": {tk: [_stats(band_xi_task[tk][t]) for t in frames] for tk in valid}}

    out = {"experiment": "mvbench_framewise_attention_evt_xi",
           "model_name": args.model_name, "data_root": args.data_root,
           "sampling": ("official" if args.official_sampling else "fps"),
           "num_frames": (args.num_segments if args.official_sampling else None),
           "fps": (None if args.official_sampling else args.fps),
           "max_frames": args.max_frames, "max_pixels": args.max_pixels,
           "min_pixels": min_pixels, "tasks": valid, "n": n,
           "n_layers": n_layers, "n_frame_groups": n_frames,
           "per_task_seen": {t: seen[t] for t in valid},
           "baseline_band": band, "k_frac": args.k_frac, "domain_eps": eps,
           "estimator": "dekkers_einmahl_dehaan_moment",
           "layers": layers, "frames": frames,
           "xi_mean": xi_mean.tolist(), "xi_sem": xi_sem.tolist(),
           "band_xi": band_summary}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {args.out}")

    # ---- console summary ----
    print(f"\n==== per-FRAME EVT tail index xi on MVBench ({n} clips, {len(valid)} task(s), "
          f"{n_layers} layers x {n_frames} frame groups) ====")
    print("  xi is estimated WITHIN each frame, over its own spatial tokens")
    print("  xi > 0 Frechet (a few tokens hold the frame) | xi = 0 Gumbel (exponential) | "
          "xi < 0 Weibull (toward uniform over the frame)")
    if band:
        print(f"\nband{band} averaged (the tail the scorer actually ranks on), per frame:")
        hdr = f"{'frame':>6}{'xi':>9}{'sem':>8}{'share':>9}{'domain':>16}"
        print(hdr); print("-" * len(hdr))
        for t in frames:
            s = band_summary["per_frame"][t]
            sh = band_summary["share_per_frame"][t]["mean"]
            dom = ("heavy/Frechet" if s["mean"] > eps else
                   ("uniform/Weibull" if s["mean"] < -eps else "~exponential"))
            print(f"{t:>6}{s['mean']:>9.3f}{s['sem']:>8.3f}{sh:>9.3f}{dom:>16}")
        sp = band_summary["spread_across_frames"]
        print(f"\nacross-frame spread of xi, per clip: std_t(xi) = {sp['mean']:.3f} "
              f"+/- {sp['sem']:.3f} (n={sp['n']})")
        print(f"  {'task':>22}{'xi range over frames':>22}")
        for tk in valid:
            m = np.array([band_summary['per_task'][tk][t]['mean'] for t in frames])
            print(f"  {tk:>22}{np.nanmin(m):>11.3f} .. {np.nanmax(m):<9.3f}")

    print("\nread: xi(frame) flat within its SEM => every frame concentrates alike, the heavy")
    print("      tail is not a property of particular frames (nothing temporal to exploit);")
    print("      xi(frame) varying >> SEM => some frames are Frechet while others are not,")
    print("      i.e. the tail IS frame-specific and a temporal budget could act on it.")
    print("      Compare with the SHARE column: a frame can hold little mass yet still be heavy.")

    if not args.no_plot:
        plot(out, os.path.splitext(args.out)[0] + ".png")


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def plot(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE = "#2a78d6"; RED = "#e34948"
    INK = "#0b0b0b"; INK_SECOND = "#52514e"; MUTED = "#898781"
    GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECOND, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})

    layers = np.array(out["layers"]); frames = np.array(out["frames"])
    band = out["baseline_band"]
    xi_mean = np.array(out["xi_mean"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: xi vs FRAME for the scorer band, against the xi=0 / xi=-1 lines ----
    if out["band_xi"]:
        m = np.array([s["mean"] for s in out["band_xi"]["per_frame"]])
        e = np.array([s["sem"] for s in out["band_xi"]["per_frame"]])
        axA.axhline(0, color=INK, lw=1.4)
        axA.text(frames.max(), 0, "  xi = 0  (exponential / Gumbel)", color=INK,
                 va="bottom", ha="right", fontsize=9)
        axA.axhline(-1, color=MUTED, lw=1.2, ls="--")
        axA.text(frames.max(), -1, "  xi = -1  (uniform / Weibull)", color=MUTED,
                 va="bottom", ha="right", fontsize=9)
        axA.fill_between(frames, m - e, m + e, color=BLUE, alpha=0.18, zorder=1)
        axA.plot(frames, m, "-", color=INK_SECOND, lw=1.4, zorder=2,
                 label=f"xi of the band {min(band)}-{max(band)} average")
        pos = m > 0
        axA.scatter(frames[pos], m[pos], s=26, color=RED, zorder=3)
        axA.scatter(frames[~pos], m[~pos], s=26, color=BLUE, zorder=3)
        axA.set_ylim(min(-1.1, float(np.nanmin(m - e)) - 0.1), float(np.nanmax(m + e)) + 0.12)
        axA.legend(frameon=False, fontsize=9, loc="best")
    axA.set_xlabel("frame group  (2 frames each, in clip order)")
    axA.set_ylabel("EVT tail index  xi  (within frame)")
    axA.set_title("Does the tail differ across frames?", loc="left", weight="bold")

    # ---- Panel B: xi(layer, frame) heatmap, diverging about xi = 0 ----
    lim = float(np.nanmax(np.abs(xi_mean)))
    im = axB.imshow(xi_mean, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-lim, vmax=lim,
                    extent=[frames.min() - 0.5, frames.max() + 0.5,
                            layers.min() - 0.5, layers.max() + 0.5])
    if band:
        for y in (min(band) - 0.5, max(band) + 0.5):
            axB.axhline(y, color=INK, lw=1.0, ls="--")
        axB.text(frames.max() + 0.4, np.mean(band), f" band {min(band)}-{max(band)}",
                 color=INK, va="center", ha="left", fontsize=9, rotation=90)
    axB.set_xlabel("frame group  (2 frames each, in clip order)")
    axB.set_ylabel("LLM layer")
    axB.set_title("xi per (layer, frame)   red = heavy, blue = light", loc="left", weight="bold")
    cbar = fig.colorbar(im, ax=axB, pad=0.02)
    cbar.set_label("EVT tail index  xi", color=INK_SECOND)
    cbar.outline.set_edgecolor(MUTED)

    nf = out["num_frames"] if out["sampling"] == "official" else f"{out['fps']}fps"
    fig.suptitle(f"MVBench ({out['sampling']}, {nf} frames), n={out['n']}, "
                 f"{len(out['tasks'])} task(s), {out['model_name'].split('/')[-1]} — "
                 f"is the heavy tail of video attention frame-specific?",
                 x=0.01, ha="left", weight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    print(f"wrote {png_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-frame EVT tail index (xi) of language->video attention on MVBench.")
    p.add_argument("--model_name", default=MODEL_ID)
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames per clip (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="Use the reference mvbench.ipynb sampler (fixed --num_segments frames at "
                        "segment midpoints) for leaderboard-comparable numbers, instead of fps sampling. "
                        "Recommended here: it fixes the frame count, so frame t means the same thing "
                        "across clips.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="per-frame pixel ceiling (downscales large frames).")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="per-frame pixel floor. Default: mirror --max_pixels for a FIXED "
                        "per-clip token budget across clips; pass <=0 to disable the floor.")
    p.add_argument("--max_samples", type=int, default=40, help="cap samples PER TASK.")
    p.add_argument("--baseline_band", default="12,13,14,15,16",
                   help="scorer band: marked in the plot AND used for the band-averaged per-frame "
                        "xi ('' to disable).")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="top fraction of order statistics used by the moment estimator. Note it "
                        "now applies to the S tokens of ONE frame, not to all M video tokens.")
    p.add_argument("--domain_eps", type=float, default=0.05,
                   help="|xi|<=eps counts as Gumbel/exponential in the domain labels.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out", default="results_framewise_attn_tail_mvbench.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())