"""MC accuracy on Video-MME with the native Qwen2.5-VL token gating.

The scorer (train.py) counterpart to ``accuracy_videomme_fastv.py``: same frames,
prompt and option-letter readout, but instead of FastV's in-LLM pruning this uses
the learned pre-LLM scorer attached to the model (we just flip ``token_keep_ratio``),
matching ``accuracy_mvbench.py`` so the two methods are directly comparable at the
same visual-token budget.

Video-MME is 2700 multiple-choice questions over 900 videos (3 questions each),
split by clip ``duration`` into short / medium / long. The data layer (parquet
annotations, uniform full-clip frame sampling, optional subtitles, prompt) is
reused verbatim from ``evaluate_videomme.py``; the model plumbing (input build,
letter-token readout, single gated forward) is reused from ``accuracy_mvbench.py``.

For every sample we run ONE forward (no autoregressive decoding) and read the
next-token logits at the final position, restricted to the option-letter tokens.
The full model is the accuracy reference; each retention ratio rho reuses the
attached scorer, and at the same budget any requested content-free baselines
(random / uniform). The Video-MME headline is overall accuracy (micro over all
questions); we also report the per-duration breakdown.

By default we run the standard *without subtitles* setting (frames only), the
meaningful one for a visual-token-pruning ablation -- subtitles let the LLM answer
many questions without the frames. Pass ``--use_subs`` for the w/ subs setting.

Example:
    python accuracy_videomme.py \
        --data_root ~/VideoMME \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --scorer_ckpt checkpoints/scorer_best.pt \
        --rhos 0.25 0.5 0.75
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Video-MME data layer (parquet load, frame sampling, subtitles, prompt).
from evaluate_videomme import (
    DURATIONS,
    build_prompt,
    load_records,
    read_video,
    subtitles_for_frames,
)

# Native-gating model plumbing, shared with the MVBench scorer eval so the two
# benchmarks (and the FastV counterparts) read out accuracy identically.
from accuracy_mvbench import (
    build_inputs,
    kept_video_count,
    letter_token_ids,
    predict,
    video_text_counts,
)


def main():
    p = argparse.ArgumentParser(description="Gated MC accuracy on Video-MME.")
    p.add_argument("--data_root", required=True, help="Dir holding videomme/, data/, subtitle/ (see evaluate_videomme.py).")
    p.add_argument("--durations", nargs="+", default=["all"], help="duration splits to run, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--scorer_ckpt", default="checkpoints/scorer_best.pt")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--baselines", nargs="*", default=["random", "uniform"],
                   choices=["random", "uniform"],
                   help="Content-free selection baselines to run at each rho (same token "
                        "budget as the scorer). Pass empty to skip, e.g. --baselines.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random-selection baseline.")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--k_min", type=int, default=1)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--max_frames", type=int, default=16, help="frames sampled per clip (uniform over the full video).")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap questions PER DURATION split (debug).")
    p.add_argument("--use_subs", action="store_true", help="Inject frame-aligned .srt subtitles (Video-MME w/ subs).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model baseline.")
    p.add_argument("--out", default="results_videomme_accuracy.json", help="Path to dump the metrics JSON.")
    args = p.parse_args()

    durations = list(DURATIONS) if args.durations == ["all"] else args.durations
    unknown = [d for d in durations if d not in DURATIONS]
    if unknown:
        raise ValueError(f"unknown durations {unknown}; choices: {list(DURATIONS)}")

    torch.manual_seed(args.seed)  # reproducible "random" selection baseline
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.load_token_scorer(args.scorer_ckpt, keep_ratio=args.rhos[0], hidden_dim=args.hidden_dim,
                            k_min=args.k_min, temp=args.temp, beta_max=args.beta_max)

    # (column name, rho, selection mode). The learned scorer plus, at the same
    # budget, any content-free baselines requested via --baselines.
    settings = [("full", None, None)] if not args.no_full else []
    for r in args.rhos:
        settings.append((f"rho={r}", r, "scorer"))
        for b in args.baselines:
            settings.append((f"{b[:4]}@{r}", r, b))   # e.g. rand@0.25, unif@0.25

    correct = {d: {name: 0 for name, _, _ in settings} for d in durations}
    seen = {d: 0 for d in durations}
    kept_vid = {name: 0 for name, _, _ in settings}
    vid_total = 0
    text_total = 0
    n = 0

    data_root = os.path.expanduser(args.data_root)
    video_dir = os.path.join(data_root, "data")
    sub_dir = os.path.join(data_root, "subtitle")
    records = load_records(data_root)

    by_duration = {}
    for d in durations:
        recs = [r for r in records if str(r["duration"]) == d]
        by_duration[d] = recs[: args.max_samples] if args.max_samples else recs

    for duration in durations:
        for rec in tqdm(by_duration[duration], desc=duration):
            try:
                video_path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
                frames, timestamps = read_video(video_path, args.max_frames)
                subs = None
                if args.use_subs:
                    subs = subtitles_for_frames(
                        os.path.join(sub_dir, f"{rec['videoID']}.srt"), timestamps)
                text, letters, gt_idx = build_prompt(rec, subs)
                inputs = build_inputs(processor, model, frames, text, args.max_pixels)
                letter_ids = letter_token_ids(processor, letters)
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{duration}] {rec.get('videoID')}: {e}")
                continue

            n_video, n_text = video_text_counts(model, inputs)
            vid_total += n_video
            text_total += n_text
            for name, rho, mode in settings:
                if rho is None:
                    model.disable_token_gating()
                else:
                    model.model.token_keep_ratio = rho
                    model.model.token_selection_mode = mode
                correct[duration][name] += int(predict(model, inputs, letter_ids) == gt_idx)
                kept_vid[name] += kept_video_count(model, inputs, rho)
            seen[duration] += 1
            n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (videomme/ and data/).")

    text_avg = text_total / n
    valid = [d for d in durations if seen[d]]
    col = [name for name, _, _ in settings]
    # Video-MME headline: overall (micro) accuracy over all questions. Retention is
    # each setting's overall accuracy vs the full model.
    setting_overall = {c: sum(correct[d][c] for d in valid) / n for c in col}
    full_overall = setting_overall.get("full", float("nan"))

    def retention(c):
        return setting_overall[c] / full_overall if ("full" in col and full_overall > 0) else float("nan")

    print(f"\nEvaluated {n} Video-MME questions across {len(durations)} duration split(s)  "
          f"(avg video tokens/clip = {vid_total / n:.0f}, avg text tokens = {text_avg:.0f})\n")

    # per-duration accuracy table (one column per setting)
    print(f"{'duration':<12}{'n':>6}" + "".join(f"{c:>11}" for c in col))
    print("-" * (18 + 11 * len(col)))
    for d in durations:
        if not seen[d]:
            continue
        accs = "".join(f"{correct[d][c] / seen[d]:>11.4f}" for c in col)
        print(f"{d:<12}{seen[d]:>6}{accs}")
    print("-" * (18 + 11 * len(col)))
    overall_row = "".join(f"{setting_overall[c]:>11.4f}" for c in col)
    print(f"{'overall':<12}{n:>6}{overall_row}")

    # per-setting accuracy / retention / token-budget summary
    print(f"\n{'setting':<10}{'acc':>9}{'ret%':>8}{'vid%':>8}{'vid_tok':>9}{'text_tok':>10}")
    print("-" * 54)
    for name in col:
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        ret = retention(name)
        ret_str = f"{ret * 100:>7.1f}%" if ret == ret else f"{'--':>8}"
        print(f"{name:<10}{setting_overall[name]:>9.4f}{ret_str}{vid_pct:>7.1f}%"
              f"{kept_vid[name] / n:>9.0f}{text_avg:>10.0f}")

    # JSON dump (headline + per-duration + per-setting table, mirrors evaluate_videomme).
    result = {
        "experiment": "videomme_scorer_accuracy",
        "model_name": args.model_name,
        "scorer_ckpt": args.scorer_ckpt,
        "data_root": args.data_root,
        "use_subs": args.use_subs,
        "durations": valid,
        "n_evaluated": n,
        "rhos": args.rhos,
        "baselines": args.baselines,
        "settings": col,
        "full_accuracy": full_overall if "full" in col else float("nan"),
        "per_duration": {d: {"n": seen[d], **{c: correct[d][c] / seen[d] for c in col}} for d in valid},
        "table": {
            name: {
                "accuracy": setting_overall[name],
                "accuracy_per_duration": {d: correct[d][name] / seen[d] for d in valid},
                "accuracy_retention": retention(name),
                "kept_vid_pct": (kept_vid[name] / vid_total * 100 if vid_total else 0.0),
                "vid_tok": kept_vid[name] / n,
                "text_tok": text_avg,
            }
            for name in col
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
