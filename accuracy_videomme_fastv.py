"""FastV accuracy evaluation on Video-MME.

Adapted from accuracy_mvbench_fastv.py and evaluate_videomme.py.

Runs the standard Video-MME multiple-choice evaluation while using FastV
(training-free visual-token pruning inside the LLM) instead of the learned
token scorer.

Example:
    python accuracy_videomme_fastv.py \
        --data_root ~/VideoMME \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --fastv_k 2 3 5 \
        --rhos 0.25 0.5 0.75
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluate_videomme import (
    DURATIONS,
    build_prompt,
    load_records,
    frame_timestamps,
    subtitles_for_frames,
)

from accuracy_mvbench import (
    build_inputs,
    letter_token_ids,
    make_mvbench_prompt,
    predict,
    video_text_counts,
)

from efficient_vlm.fastv import install_fastv, set_fastv, set_visual_mask


def fastv_flops_ratio(n_tokens, n_kept, k, n_layers, hidden, inter):
    d, m = hidden, inter

    def layer_cost(n):
        return 4 * n * d * d + 2 * n * n * d + 2 * n * d * m

    full = n_layers * layer_cost(n_tokens)
    fast = k * layer_cost(n_tokens) + (n_layers - k) * layer_cost(n_kept)
    return 1.0 - fast / full


def main():
    p = argparse.ArgumentParser(description="FastV accuracy on Video-MME.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--durations", nargs="+", default=["all"])
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--fastv_k", type=int, nargs="+", default=[2])
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--max_frames", type=int, default=16, help="upper cap on frames per clip; fps sampling clamps to the clip length below this.")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for video sampling (qwen_vl_utils default 2.0); short clips yield fewer frames instead of being skipped. Video-MME clips are long, so the cap binds and sampling matches the old fixed-count behaviour.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_subs", action="store_true")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--no_full", action="store_true")
    args = p.parse_args()

    durations = list(DURATIONS) if args.durations == ["all"] else args.durations

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn,
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_name)

    if hasattr(model, "disable_token_gating"):
        model.disable_token_gating()

    text_model = install_fastv(model)
    video_token_id = model.config.video_token_id

    tcfg = model.config.text_config
    n_layers = tcfg.num_hidden_layers
    hidden = tcfg.hidden_size
    inter = tcfg.intermediate_size

    settings = [] if args.no_full else [("full", None, None)]
    for k in args.fastv_k:
        for r in args.rhos:
            settings.append((f"K{k}@{r}", k, r))

    records = load_records(os.path.expanduser(args.data_root))

    by_duration = {}
    for d in durations:
        recs = [r for r in records if str(r["duration"]) == d]
        by_duration[d] = recs[: args.max_samples] if args.max_samples else recs

    correct = {d: {name: 0 for name, _, _ in settings} for d in durations}
    seen = {d: 0 for d in durations}

    kept_vid = {name: 0 for name, _, _ in settings}
    flops_red = {name: 0.0 for name, _, _ in settings}

    vid_total = 0
    text_total = 0
    n = 0

    video_dir = os.path.join(os.path.expanduser(args.data_root), "data")
    sub_dir = os.path.join(os.path.expanduser(args.data_root), "subtitle")

    for duration in durations:
        for rec in tqdm(by_duration[duration], desc=duration):
            try:
                video_path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
                timestamps = frame_timestamps(video_path, args.max_frames)

                subs = None
                if args.use_subs:
                    subs = subtitles_for_frames(
                        os.path.join(sub_dir, f"{rec['videoID']}.srt"),
                        timestamps,
                    )

                text, letters, gt_idx = build_prompt(rec, subs)

                prompt = make_mvbench_prompt(
                    video_path, "video", False, rec, text,
                    args.max_frames, args.max_pixels, args.fps,
                )
                inputs = build_inputs(processor, model, prompt)

                letter_ids = letter_token_ids(processor, letters)

            except Exception as e:
                tqdm.write(f"skip [{duration}] {rec.get('videoID')}: {e}")
                continue

            n_video, n_text = video_text_counts(model, inputs)
            seq_len = n_video + n_text

            visual = (inputs["input_ids"][0] == video_token_id)

            vid_total += n_video
            text_total += n_text

            for name, k, rho in settings:
                if k is None:
                    set_fastv(text_model, None)
                    kept = n_video
                else:
                    set_visual_mask(text_model, visual)
                    set_fastv(text_model, k, rho)

                    kept = min(n_video, max(1, round(rho * n_video)))

                    flops_red[name] += fastv_flops_ratio(
                        seq_len,
                        seq_len - (n_video - kept),
                        k,
                        n_layers,
                        hidden,
                        inter,
                    )

                pred = predict(model, inputs, letter_ids)
                correct[duration][name] += int(pred == gt_idx)
                kept_vid[name] += kept

            set_fastv(text_model, None)

            seen[duration] += 1
            n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated.")

    text_avg = text_total / n

    print(f"\\nEvaluated {n} Video-MME questions\\n")

    cols = [name for name, _, _ in settings]

    print(f"{'duration':<12}{'n':>6}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (18 + 12 * len(cols)))

    for d in durations:
        if not seen[d]:
            continue

        row = "".join(
            f"{correct[d][c] / seen[d]:>12.4f}"
            for c in cols
        )
        print(f"{d:<12}{seen[d]:>6}{row}")

    print("-" * (18 + 12 * len(cols)))

    overall = "".join(
        f"{sum(correct[d][c] for d in durations) / n:>12.4f}"
        for c in cols
    )
    print(f"{'overall':<12}{n:>6}{overall}")

    print(f"\\n{'setting':<12}{'vid%':>8}{'vid_tok':>10}{'text_tok':>10}{'FLOPs-':>10}")
    print("-" * 52)

    for name, k, _ in settings:
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        fr = (flops_red[name] / n * 100) if k is not None else 0.0

        print(
            f"{name:<12}{vid_pct:>7.1f}%"
            f"{kept_vid[name] / n:>10.0f}"
            f"{text_avg:>10.0f}"
            f"{fr:>9.1f}%"
        )


if __name__ == "__main__":
    main()
