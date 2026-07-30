"""
vision_encoder_tail_index_mvbench.py -- vision_encoder_tail_index.py's encoder
tail-index diagnostic (Section 11), aggregated over MVBench.
=============================================================================================
Section 11.3 predicts the tail index should pick out LATE vision-encoder layers (a
condensed, heavy-tailed set carrying global context), as opposed to the decoder's
shallow-band anchor (Section 5). This runs that check over full MVBench, reusing
anchor_layer_prune_mvbench.py's own official-protocol clip loading so the numbers
sit next to the rest of this repo's MVBench results.

No extra encoder forward: `prepare_clip`'s qwen path already calls
`model.get_video_features(pix, grid_thw)` -> the vision tower's own forward once
per clip to build the merged embeddings anchor_layer_prune needs. `install_capture`
is installed on that same module (`_visual_model(model)`, see
vision_encoder_tail_index.py) up front, so this just reads that SAME forward's
attention weights off afterward via `read_captured_gammas` -- no separate pass, no
separate pixel/grid_thw plumbing.

Qwen2.5-VL only (LLaVA-OneVision's SigLIP encoder is a different attention module,
out of scope -- see vision_encoder_tail_index.py's docstring).

Run:
    python vision_encoder_tail_index_mvbench.py --data_root ~/Experiments/MVBench \
        --tasks "Action Sequence" "Scene Transition" --official_sampling \
        --num_segments 16 --max_pixels 200704 --max_samples 40 \
        --out results_vision_encoder_tail_index_mvbench.json
"""
import os
import sys
import json
import warnings
import argparse
from collections import Counter

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
from inference import DATA_LIST
from anchor_layer_prune_mvbench import load_backbone, prepare_clip
from anchor_layer_prune import QWEN_MODEL_ID
from vision_encoder_tail_index import install_capture, uninstall_capture, read_captured_gammas, _visual_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        args.dtype = "bf16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[backbone] qwen  model={args.model_name}  dtype={args.dtype}")
    model, processor, video_token_id, info = load_backbone("qwen", args.model_name, dtype)
    visual = _visual_model(model)
    install_capture(visual)

    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    depth = len(visual.blocks)
    gamma_sum = {t: np.zeros(depth) for t in tasks}          # sum of per-clip gamma, NaN-safe
    gamma_n = {t: np.zeros(depth) for t in tasks}             # count of non-NaN gamma per layer
    estar_hist = {t: [] for t in tasks}                        # per-clip heaviest-tailed layer
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
                prepare_clip("qwen", model, processor, video_token_id, info, path, data_type,
                            has_bound, rec, args, device, dtype, min_pixels)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue

            gammas = read_captured_gammas(visual, k_frac=args.k_frac, estimator=args.estimator)
            valid = {l: g for l, g in gammas.items() if g == g}
            Estar = max(valid, key=valid.get) if valid else None
            for l, g in valid.items():
                gamma_sum[task][l] += g
                gamma_n[task][l] += 1
            estar_hist[task].append(Estar)

            per_sample.append({"task": task, "video": video_name, "gammas": gammas, "E_star": Estar})
            seen[task] += 1
            n += 1

    uninstall_capture(visual)

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    valid_tasks = [t for t in tasks if seen[t]]
    depth_range = list(range(depth))

    def mean_curve(sum_arr, n_arr):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_arr > 0, sum_arr / np.maximum(n_arr, 1), np.nan)

    per_task_curve = {t: mean_curve(gamma_sum[t], gamma_n[t]).tolist() for t in valid_tasks}
    overall_sum = sum(gamma_sum[t] for t in valid_tasks)
    overall_n = sum(gamma_n[t] for t in valid_tasks)
    overall_curve = mean_curve(overall_sum, overall_n)
    overall_valid = {l: overall_curve[l] for l in depth_range if overall_curve[l] == overall_curve[l]}
    E_star_overall = max(overall_valid, key=overall_valid.get) if overall_valid else None

    all_estar = [e for t in valid_tasks for e in estar_hist[t] if e is not None]
    estar_counts = {int(l): all_estar.count(l) for l in sorted(set(all_estar))}

    out_json = {"experiment": "vision_encoder_tail_index_mvbench", "backbone": "qwen",
               "model_name": args.model_name, "data_root": args.data_root,
               "sampling": "official" if args.official_sampling else "fps",
               "num_frames": args.num_segments if args.official_sampling else None,
               "fps": None if args.official_sampling else args.fps,
               "max_pixels": args.max_pixels, "min_pixels": min_pixels,
               "k_frac": args.k_frac, "estimator": args.estimator,
               "encoder_depth": depth, "tasks": valid_tasks, "n": n,
               "per_task_seen": {t: seen[t] for t in valid_tasks},
               "gamma_curve_overall": overall_curve.tolist(),
               "gamma_curve_by_task": per_task_curve,
               "E_star_overall": E_star_overall,
               "E_star_histogram": estar_counts,
               "E_star_mean": float(np.mean(all_estar)) if all_estar else float("nan"),
               "skipped": skipped}

    print(f"\n==== Vision-encoder tail index on MVBench ({n} samples, {len(valid_tasks)} task(s)) ====")
    if skipped:
        by_cat = Counter(s["reason"].split(":")[0] if ":" in s["reason"] else s["reason"] for s in skipped)
        by_task = Counter(s["task"] for s in skipped)
        print(f"\nskipped {len(skipped)} clip(s):")
        for cat, cnt in by_cat.most_common():
            print(f"  {cat:<24}{cnt:>5}")
        print("  by task: " + ", ".join(f"{t} ({c})" for t, c in by_task.most_common()))

    print(f"\nmean gamma per encoder layer (over {n} clips):")
    for l in depth_range:
        g = overall_curve[l]
        mark = "  <- E*" if l == E_star_overall else ""
        print(f"  layer {l:>2}: gamma = {g:+.4f}{mark}" if g == g else f"  layer {l:>2}: gamma = NaN{mark}")
    print(f"\nheaviest-tailed encoder layer, pooled over all clips: E* = {E_star_overall}")
    print(f"per-clip E* histogram: {estar_counts}  (mean {out_json['E_star_mean']:.2f})")

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
        plt.figure(figsize=(6, 4))
        plt.plot(depth_range, overall_curve, marker="o", label="overall")
        if E_star_overall is not None:
            plt.axvline(E_star_overall, color="red", linestyle="--", label=f"E*={E_star_overall}")
        plt.xlabel("vision encoder layer")
        plt.ylabel("mean tail index (gamma)")
        plt.title(f"{os.path.basename(args.model_name)}: encoder tail index vs. layer, MVBench")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Vision-encoder tail index gamma vs. layer, on MVBench (Qwen2.5-VL).")
    p.add_argument("--model_name", default=QWEN_MODEL_ID)
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames per clip (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="use the reference mvbench.ipynb sampler instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None, help="per-frame pixel ceiling.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="per-frame pixel floor. Default: mirror --max_pixels. <=0 disables.")
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--out", default="results_vision_encoder_tail_index_mvbench.json")
    p.add_argument("--per_sample_out", default="",
                   help="per-clip gamma-curve dump. Default: <--out stem>_per_sample.json.")
    p.add_argument("--plot", default=None, help="path to save an overall gamma-vs-layer PNG (optional).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
