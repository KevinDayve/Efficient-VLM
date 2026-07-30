"""
frame_sufficiency_mvbench.py -- how much does MVBench accuracy need MORE THAN ONE frame?

Clean, self-contained accuracy probe (NOT a token-pruning experiment): for each
question it compares the official all-frames answer against single-frame answers,
using inference.py's exact MVBench protocol (official midpoint sampling, option-letter
logit readout, ANSWER_PREFIX forcing) so the numbers are leaderboard-comparable.

Per question we run, on the SAME official frames:
  all         -- the N-frame clip (the normal protocol)                  [reference]
  single_i    -- frame i alone, for each of the N frames
  blind       -- no video at all (question+options only)                 [language prior]

Qwen2.5-VL pairs adjacent frames (TEMPORAL_PATCH_SIZE=2), so a true T=1 video is
invalid; a single frame is passed as [f, f] -- a 2-frame zero-motion clip carrying
exactly one frame of CONTENT through the identical video pipeline.

Reported per task and overall:
  all           accuracy with all N frames
  single_mean   mean accuracy across the N single-frame runs  (a typical 1 frame)
  single_best   correct if ANY single frame is correct        ("any 1 frame kept")
  single_worst  correct only if EVERY single frame is correct (frame-robustness floor)
  blind         accuracy with no video                        (text-only prior)

READ:
  single_best ~ all (or above)  => the answer lives in SOME single frame; multi-frame
      integration adds little beyond frame SELECTION -> temporal is largely redundant
      for accuracy, the game is picking the right frame/tokens.
  single_best << all            => genuine multi-frame temporal integration is required.
  single_mean ~ blind           => an average lone frame barely beats the language prior.

Run:
    python frame_sufficiency_mvbench.py --data_root ~/MVBench --num_frames 8 \
        --tasks "Action Sequence" "Scene Transition" --max_samples 50
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from inference import (
    DATA_LIST, ANSWER_PREFIX, TEMPORAL_PATCH_SIZE,
    build_prompt, letter_token_ids, official_frames, predict,
)


def frames_prompt(frames, text, max_pixels=None):
    """MVBench user turn with an explicit PIL-frame list as the video item."""
    vid = {"type": "video", "video": frames}
    if max_pixels is not None:
        vid["max_pixels"] = max_pixels
    return [{"role": "user", "content": [vid, {"type": "text", "text": text}]}]


def build_inputs(processor, model, prompt):
    chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    chat += ANSWER_PREFIX
    imgs, vids = process_vision_info(prompt)
    return processor(text=[chat], images=imgs, videos=vids, return_tensors="pt").to(model.device)


def blind_inputs(processor, model, text):
    msg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
    return processor(text=[chat], return_tensors="pt").to(model.device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="1-frame vs all-frames MVBench accuracy probe.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=8, help="all-frames count (rounded to even).")
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--blind", action="store_true", help="also run the no-video language-prior baseline.")
    p.add_argument("--out", default="frame_sufficiency_mvbench.json")
    args = p.parse_args()

    N = max(TEMPORAL_PATCH_SIZE, round(args.num_frames / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    # per-task counters
    C = defaultdict(lambda: defaultdict(int))   # task -> metric -> count correct
    seen = defaultdict(int)
    Pos = defaultdict(lambda: np.zeros(N))       # task -> per-position correct-count (temporal order 0..N-1)
    Best = defaultdict(lambda: np.zeros(N))      # task -> best-position histogram (argmax single | any correct)
    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=f"{task} (N={N})"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt = build_prompt(rec)
                letter_ids = letter_token_ids(processor, letters)
                frames = official_frames(path, data_type, has_bound, rec, N)  # N PIL frames

                # all N frames
                all_ok = int(predict(model, build_inputs(processor, model,
                             frames_prompt(frames, text, args.max_pixels)), letter_ids) == gt)

                # each single frame, duplicated to satisfy the even-frame requirement
                singles = [int(predict(model, build_inputs(processor, model,
                            frames_prompt([f, f], text, args.max_pixels)), letter_ids) == gt)
                           for f in frames]

                seen[task] += 1
                C[task]["all"] += all_ok
                C[task]["single_mean"] += float(np.mean(singles))
                C[task]["single_best"] += int(any(singles))
                C[task]["single_worst"] += int(all(singles))
                Pos[task] += np.asarray(singles, dtype=float)      # per-position correctness
                if any(singles):                                    # best-position among solvable-by-some
                    Best[task][int(np.argmax(singles))] += 1
                if args.blind:
                    C[task]["blind"] += int(predict(model, blind_inputs(processor, model, text), letter_ids) == gt)
            except Exception as e:
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

    metrics = ["all", "single_mean", "single_best", "single_worst"] + (["blind"] if args.blind else [])
    print(f"\n{'task':26s} {'n':>5} " + " ".join(f"{m:>12s}" for m in metrics))
    tot = defaultdict(float); ntot = 0
    for task in tasks:
        n = seen[task]
        if not n:
            continue
        ntot += n
        row = []
        for m in metrics:
            acc = 100.0 * C[task][m] / n
            tot[m] += C[task][m]
            row.append(f"{acc:11.1f}%")
        print(f"{task:26s} {n:>5} " + " ".join(row))
    if ntot:
        print(f"{'OVERALL':26s} {ntot:>5} " +
              " ".join(f"{100.0 * tot[m] / ntot:11.1f}%" for m in metrics))
        gap = 100.0 * (tot['all'] - tot['single_best']) / ntot
        print(f"\n  all - single_best = {gap:+.1f} pts   "
              f"(<=0 => some single frame already suffices; temporal largely redundant for accuracy)")

        # per-position (temporal) single-frame accuracy, pooled over tasks. Position i = the
        # i-th midpoint-sampled frame (~temporal fraction (i+0.5)/N). Each frame is scored in
        # ISOLATION ([f,f]), so this is content-at-that-time answerability, not an in-context
        # position bias. best_pos = distribution of which position is the answerable one.
        pos_tot = np.sum([Pos[t] for t in seen if seen[t]], axis=0)   # (N,) correct-count per position
        best_tot = np.sum([Best[t] for t in seen if seen[t]], axis=0) # (N,) best-position histogram
        acc = 100.0 * pos_tot / ntot
        idx = np.arange(N)
        # monotone temporal trend: Pearson r between position index and per-position accuracy
        r = float(np.corrcoef(idx, acc)[0, 1]) if N > 1 and acc.std() > 0 else float("nan")
        print("\n  per-position single-frame accuracy (temporal order 0..N-1):")
        print("   " + "  ".join(f"{i}:{acc[i]:.1f}" for i in idx))
        print(f"   argmax position {int(acc.argmax())} ({acc.max():.1f}%)  min {int(acc.argmin())} "
              f"({acc.min():.1f}%)  spread {acc.max()-acc.min():.1f} pts  position-trend r={r:+.2f}")
        print("  best-frame position histogram (share of solvable Qs whose argmax is at each position):")
        print("   " + "  ".join(f"{i}:{100.0*best_tot[i]/max(best_tot.sum(),1):.1f}" for i in idx))

    with open(args.out, "w") as fh:
        json.dump({"num_frames": N, "seen": dict(seen),
                   "correct": {t: dict(C[t]) for t in C},
                   "pos_correct": {t: Pos[t].tolist() for t in Pos},   # per-position correct-count, task->[N]
                   "best_pos_hist": {t: Best[t].tolist() for t in Best}}, fh, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
