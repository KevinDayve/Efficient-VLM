"""
encoder_information_horizon_mvbench.py -- encoder_information_horizon.py's
depth-truncation check ("does the vision encoder need its full depth?"),
aggregated over MVBench as letter-choice accuracy per exit layer.
=============================================================================================
anchor_layer_prune_mvbench.py's prepare_clip() is a black box that calls
model.get_video_features(...) internally ONCE (at full encoder depth) and only
returns the already-merged embeddings -- it can't be reused here, since this sweep
needs to re-run the vision encoder multiple times per clip, once per candidate
exit layer, with a DIFFERENT (truncated) result each time. So `prepare_clip_raw`
below duplicates prepare_clip's qwen branch up to (not including) the
get_video_features call, exposing the raw pixel_values/grid_thw/position_ids so
this script can drive get_video_features itself per exit layer -- see
encoder_information_horizon.py's own load_qwen_inputs/run_at_exit_layer split,
which this mirrors at MVBench scale. Qwen2.5-VL only (same scope as the rest of
the encoder-side scripts: the early-exit patch targets Qwen2_5_VLVisionBlock).

Run:
    python encoder_information_horizon_mvbench.py --data_root ~/Experiments/MVBench \
        --tasks "Action Sequence" "Scene Transition" --official_sampling \
        --num_segments 16 --max_pixels 200704 --max_samples 40 \
        --exit_layers 7,15,23,27,31 --out results_encoder_information_horizon_mvbench.json
"""
import os
import sys
import json
import warnings
import argparse
from collections import Counter

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference import DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt
from inference_only_mvbench import letter_first_ids
from oracle_check import build_full_positions
from anchor_layer_prune import QWEN_MODEL_ID
from anchor_layer_prune_mvbench import load_backbone, score_letters
from vision_encoder_tail_index import _visual_model
from encoder_information_horizon import install_early_exit, uninstall_early_exit, set_exit_layer

FULL = "full"          # sentinel strategy name: dense, uncut encoder


def prepare_clip_raw(model, processor, video_token_id, path, data_type, has_bound, rec,
                     args, device, min_pixels):
    """anchor_layer_prune_mvbench.prepare_clip's qwen branch, stopped BEFORE the
    get_video_features call so the caller can drive it itself per exit layer."""
    text, letters, gt_idx = build_prompt(rec)
    letter_ids = letter_first_ids(processor, letters)

    prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                 args.max_frames, args.max_pixels, args.fps,
                                 official=args.official_sampling,
                                 num_segments=args.num_segments,
                                 min_pixels=min_pixels)
    chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    chat += ANSWER_PREFIX
    img_in, vid_in = process_vision_info(prompt)
    inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    video_idx = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    if video_idx.numel() < 50:
        raise RuntimeError(f"too few video tokens ({video_idx.numel()} < 50)")
    grid_thw = inputs["video_grid_thw"].to(device)
    pix = inputs["pixel_values_videos"].to(device)
    position_ids = build_full_positions(model, input_ids, video_idx, grid_thw, attn_mask)
    return input_ids, attn_mask, video_idx, grid_thw, pix, position_ids, letter_ids, gt_idx


@torch.no_grad()
def score_at_exit_layer(model, input_ids, attn_mask, video_idx, grid_thw, pix, position_ids, letter_ids):
    ve = model.get_video_features(pix, grid_thw).pooler_output
    ve = torch.cat(ve, dim=0).to(input_ids.device)
    base_embeds = model.get_input_embeddings()(input_ids).clone()
    base_embeds[0, video_idx] = ve.to(base_embeds.dtype)
    return score_letters(model, base_embeds, position_ids, attn_mask, letter_ids)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        args.dtype = "bf16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[backbone] qwen  model={args.model_name}  dtype={args.dtype}")
    model, processor, video_token_id, info = load_backbone("qwen", args.model_name, dtype)
    visual = _visual_model(model)
    depth = len(visual.blocks)
    install_early_exit(visual)

    exit_layers = [int(x) for x in args.exit_layers.split(",")]
    for e in exit_layers:
        if not (0 <= e < depth):
            raise SystemExit(f"--exit_layers: {e} out of range [0, {depth - 1}]")
    strategies = [FULL] + exit_layers

    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    correct = {s: {t: 0 for t in tasks} for s in strategies}
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
                clip = prepare_clip_raw(model, processor, video_token_id, path, data_type,
                                        has_bound, rec, args, device, min_pixels)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue

            input_ids, attn_mask, video_idx, grid_thw, pix, position_ids, letter_ids, gt_idx = clip
            rec_out = {"task": task, "video": video_name, "gt": gt_idx, "correct": {}}
            for s in strategies:
                set_exit_layer(visual, None if s == FULL else s)
                lp = score_at_exit_layer(model, input_ids, attn_mask, video_idx, grid_thw,
                                         pix, position_ids, letter_ids)
                hit = int(lp.argmax().item() == gt_idx)
                correct[s][task] += hit
                rec_out["correct"][str(s)] = hit
            set_exit_layer(visual, None)                  # leave the model clean between clips

            per_sample.append(rec_out)
            seen[task] += 1
            n += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()

    uninstall_early_exit(visual)

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    valid_tasks = [t for t in tasks if seen[t]]

    def per_task_mean(counts):
        return float(np.mean([counts[t] / seen[t] for t in valid_tasks]))

    def micro(counts):
        return sum(counts[t] for t in valid_tasks) / n

    out_json = {"experiment": "encoder_information_horizon_mvbench", "backbone": "qwen",
               "model_name": args.model_name, "data_root": args.data_root,
               "encoder_depth": depth, "exit_layers": exit_layers,
               "max_pixels": args.max_pixels, "min_pixels": min_pixels,
               "tasks": valid_tasks, "n": n,
               "per_task_seen": {t: seen[t] for t in valid_tasks},
               "skipped": skipped, "table": {}}

    print(f"\n==== Encoder information horizon on MVBench ({n} samples, {len(valid_tasks)} task(s)) ====")
    if skipped:
        by_cat = Counter(s["reason"].split(":")[0] if ":" in s["reason"] else s["reason"] for s in skipped)
        by_task = Counter(s["task"] for s in skipped)
        print(f"\nskipped {len(skipped)} clip(s):")
        for cat, cnt in by_cat.most_common():
            print(f"  {cat:<24}{cnt:>5}")
        print("  by task: " + ", ".join(f"{t} ({c})" for t, c in by_task.most_common()))

    print()
    hdr = f"{'exit_layer':<12}{'acc_mean':>10}{'acc_micro':>11}"
    print(hdr); print("-" * len(hdr))
    for s in strategies:
        acc_mean = per_task_mean(correct[s])
        acc_micro = micro(correct[s])
        out_json["table"][str(s)] = {"acc_mean": acc_mean, "acc_micro": acc_micro,
                                     "per_task": {t: correct[s][t] / seen[t] for t in valid_tasks}}
        print(f"{str(s):<12}{acc_mean:>10.4f}{acc_micro:>11.4f}")

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
        full_acc = out_json["table"][FULL]["acc_mean"]
        xs = exit_layers
        ys = [out_json["table"][str(e)]["acc_mean"] for e in exit_layers]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o", label="truncated encoder")
        plt.axhline(full_acc, color="gray", linestyle="--", label="full depth")
        plt.xlabel("encoder exit layer c (blocks 0..c run, rest frozen)")
        plt.ylabel("accuracy (mean over tasks)")
        plt.title(f"{os.path.basename(args.model_name)}: encoder information horizon, MVBench")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Vision-encoder depth-truncation accuracy sweep, on MVBench.")
    p.add_argument("--model_name", default=QWEN_MODEL_ID)
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames per clip (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="use the reference mvbench.ipynb sampler instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--exit_layers", required=True,
                   help="comma-separated encoder cut layers to sweep (0-indexed, inclusive).")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--out", default="results_encoder_information_horizon_mvbench.json")
    p.add_argument("--per_sample_out", default="",
                   help="per-clip correctness dump. Default: <--out stem>_per_sample.json.")
    p.add_argument("--plot", default=None, help="path to save an accuracy-vs-exit-layer PNG (optional).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
