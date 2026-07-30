"""
decoder_tail_index_mvbench.py -- anchor_layer_prune.plan()'s decoder tail-index
diagnostic (Section 5), aggregated over MVBench. Decoder-side companion to
vision_encoder_tail_index_mvbench.py.
=============================================================================================
anchor_layer_prune_mvbench.py already tracks the tail index AT the chosen anchor
L* per clip (`lstar_tail`), but only as part of the full anchor-prune-vs-random-
vs-uniform accuracy benchmark, and only at L* -- not the full gamma-vs-layer curve
across the whole band B. This is a leaner harness that reuses the exact same clip
loading (`load_backbone` / `prepare_clip`) but skips the prune/accuracy machinery
entirely (no `install_anchor_prune`, no greedy_select, no STRATEGIES loop) and
just calls `plan()` per clip, pooling `out["tail_indices"]` -- the gamma for EVERY
band layer, not just L* -- across the dataset.

Backbone-agnostic (--backbone qwen|llava_video|auto), unlike the vision-encoder
MVBench script which is Qwen2.5-VL only -- see decoder_tail_index.py's docstring.

Run:
    python decoder_tail_index_mvbench.py --data_root ~/Experiments/MVBench \
        --tasks "Action Sequence" "Scene Transition" --official_sampling \
        --num_segments 16 --max_pixels 200704 --max_samples 40 \
        --out results_decoder_tail_index_mvbench.json
"""
import os
import sys
import json
import warnings
import argparse
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference import DATA_LIST
from anchor_layer_prune import plan, resolve_backbone, default_model_id, _text_model, QWEN_MODEL_ID, LLAVA_VIDEO_MODEL_ID
from anchor_layer_prune_mvbench import load_backbone, prepare_clip


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resolve_backbone(args.backbone, args.model_name)
    if not args.model_name:
        args.model_name = default_model_id(backbone)
    if args.dtype == "auto":
        args.dtype = "bf16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[backbone] {backbone}  model={args.model_name}  dtype={args.dtype}")
    model, processor, video_token_id, info = load_backbone(backbone, args.model_name, dtype)

    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    if args.band == "all":
        band = list(range(len(_text_model(model).layers)))
    else:
        band = [int(x) for x in args.band.split(",")]
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    gamma_sum = {t: {l: 0.0 for l in band} for t in tasks}
    gamma_n = {t: {l: 0 for l in band} for t in tasks}
    lstar_hist = {t: [] for t in tasks}
    lstar_tail = {t: [] for t in tasks}          # gamma AT the chosen L*, per clip
    skipped = []
    per_sample = []
    seen = {t: 0 for t in tasks}
    n = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        task_video_dir = os.path.join(video_dir, subdir)
        if not os.path.isdir(task_video_dir):
            print(f"[skip task] {task!r}: video dir not found ({task_video_dir}) -- skipping entirely.")
            continue
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task, unit="clip"):
            video_name = rec.get("video")
            path = os.path.join(video_dir, subdir, video_name)
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": video_name, "reason": "missing file"})
                tqdm.write(f"skip [{task}] {video_name}: missing file ({path})")
                continue

            try:
                base, pos, attn, vpos, grid, n_frames, seq_len, letter_ids, gt_idx = prepare_clip(
                    backbone, model, processor, video_token_id, info, path, data_type,
                    has_bound, rec, args, device, dtype, min_pixels)
                out = plan(model, base, pos, attn, vpos, band, k_frac=args.k_frac, estimator=args.estimator)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue

            gammas, Lstar = out["tail_indices"], out["L_star"]
            for l in band:
                g = gammas[l]
                if g == g:
                    gamma_sum[task][l] += g
                    gamma_n[task][l] += 1
            lstar_hist[task].append(Lstar)
            lstar_tail[task].append(gammas[Lstar])

            per_sample.append({"task": task, "video": video_name, "gammas": gammas, "L_star": Lstar})
            seen[task] += 1
            n += 1
            del base, out
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    valid_tasks = [t for t in tasks if seen[t]]

    def mean_curve(task_list):
        overall_sum = {l: sum(gamma_sum[t][l] for t in task_list) for l in band}
        overall_n = {l: sum(gamma_n[t][l] for t in task_list) for l in band}
        return {l: (overall_sum[l] / overall_n[l] if overall_n[l] else float("nan")) for l in band}

    per_task_curve = {t: mean_curve([t]) for t in valid_tasks}
    overall_curve = mean_curve(valid_tasks)
    overall_valid = {l: g for l, g in overall_curve.items() if g == g}
    L_star_overall = max(overall_valid, key=overall_valid.get) if overall_valid else None

    all_lstar = [l for t in valid_tasks for l in lstar_hist[t]]
    all_tail = [g for t in valid_tasks for g in lstar_tail[t] if g == g]
    lstar_counts = {int(l): all_lstar.count(l) for l in sorted(set(all_lstar))}

    out_json = {"experiment": "decoder_tail_index_mvbench", "backbone": backbone,
               "model_name": args.model_name, "data_root": args.data_root,
               "sampling": "official" if (backbone != "llava_video" and args.official_sampling)
                          else ("official_frames" if backbone == "llava_video" else "fps"),
               "max_pixels": args.max_pixels, "min_pixels": min_pixels,
               "band": band, "k_frac": args.k_frac, "estimator": args.estimator,
               "tasks": valid_tasks, "n": n,
               "per_task_seen": {t: seen[t] for t in valid_tasks},
               "gamma_curve_overall": overall_curve,
               "gamma_curve_by_task": per_task_curve,
               "L_star_overall": L_star_overall,
               "L_star_histogram": lstar_counts,
               "L_star_mean": float(np.mean(all_lstar)) if all_lstar else float("nan"),
               "tail_index_at_anchor_mean": float(np.mean(all_tail)) if all_tail else float("nan"),
               "skipped": skipped}

    print(f"\n==== Decoder tail index on MVBench ({n} samples, {len(valid_tasks)} task(s)) ====")
    if skipped:
        by_cat = Counter(s["reason"].split(":")[0] if ":" in s["reason"] else s["reason"] for s in skipped)
        by_task = Counter(s["task"] for s in skipped)
        print(f"\nskipped {len(skipped)} clip(s):")
        for cat, cnt in by_cat.most_common():
            print(f"  {cat:<24}{cnt:>5}")
        print("  by task: " + ", ".join(f"{t} ({c})" for t, c in by_task.most_common()))

    print(f"\nmean gamma per decoder layer (over {n} clips):")
    for l in band:
        g = overall_curve[l]
        mark = "  <- L*" if l == L_star_overall else ""
        print(f"  layer {l:>3}: gamma = {g:+.4f}{mark}" if g == g else f"  layer {l:>3}: gamma = NaN{mark}")
    print(f"\nheaviest-tailed decoder layer, pooled over all clips: L* = {L_star_overall}")
    print(f"per-clip L* histogram: {lstar_counts}  (mean {out_json['L_star_mean']:.2f})")
    print(f"tail index at the chosen anchor: mean gamma = {out_json['tail_index_at_anchor_mean']:.3f}")

    with open(args.out, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"\nwrote {args.out}")

    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ys = [overall_curve[l] for l in band]
        plt.figure(figsize=(6, 4))
        plt.plot(band, ys, marker="o", label="overall")
        if L_star_overall is not None:
            plt.axvline(L_star_overall, color="red", linestyle="--", label=f"L*={L_star_overall}")
        plt.xlabel("decoder layer")
        plt.ylabel("mean tail index (gamma)")
        plt.title(f"{os.path.basename(args.model_name)}: decoder tail index vs. layer, MVBench")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Decoder tail index gamma vs. layer, on MVBench.")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_video"], default="auto")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, "
                        f"llava_video={LLAVA_VIDEO_MODEL_ID}.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="qwen only: fps-sampling frame cap.")
    p.add_argument("--fps", type=float, default=2.0, help="qwen only.")
    p.add_argument("--official_sampling", action="store_true", help="qwen only.")
    p.add_argument("--num_segments", type=int, default=16, help="qwen only: frames for --official_sampling.")
    p.add_argument("--num_frames", type=int, default=8, help="llava_video only.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--band", default="2,3,4,5,6,7,8",
                   help="anchor-candidate layer band B: comma-separated layer indices, "
                        "or 'all' for every decoder layer.")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--out", default="results_decoder_tail_index_mvbench.json")
    p.add_argument("--per_sample_out", default="",
                   help="per-clip gamma-curve dump. Default: <--out stem>_per_sample.json.")
    p.add_argument("--plot", default=None, help="path to save an overall gamma-vs-layer PNG (optional).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
