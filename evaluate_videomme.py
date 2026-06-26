"""Standalone gated evaluation on Video-MME: ours vs uniform vs random.

The Video-MME twin of ``evaluate_mvbench.py``. It reuses the same self-contained
gating path -- merged-token features (``model.get_video_features(...).pooler_output``,
exactly what the scorer was trained on), stratified Pareto selection, and the
*physically shortened* answer forward (dropped video tokens removed; survivors keep
their original M-RoPE ids) -- and only swaps the data layer for Video-MME's parquet
annotations and uniform full-clip frame sampling.

Video-MME is 2700 multiple-choice questions over 900 videos (3 questions each),
split by clip ``duration`` into short / medium / long (900 questions each). The
annotations live in ``<data_root>/videomme/test-00000-of-00001.parquet`` and clips
under ``<data_root>/data/<videoID>.mp4``; optional subtitles are
``<data_root>/subtitle/<videoID>.srt``. Each question has exactly 4 options that are
already letter-prefixed (``"A. ..."``) and a single-letter ``answer``.

For each retention ratio rho we keep K = round(rho * n_video) video tokens with
three strategies and read the option-letter logits at the answer position:

  * ours    -- learned scorer + stratified Pareto selection (paper section 2.4)
  * uniform -- evenly spaced video tokens (matched retention, no importance)
  * random  -- random video tokens (matched-retention chance baseline)

The full model (keep all tokens) is the accuracy reference and the LLM-forward
speedup denominator. The Video-MME headline is overall accuracy (micro over all
questions); we also report the per-duration breakdown and timing/speedup.

By default we run the standard *without subtitles* setting (frames only), which is
the meaningful one for a visual-token-pruning ablation -- subtitles let the LLM
answer many questions without the frames, masking the effect of dropping video
tokens. Pass ``--use_subs`` for the frame-aligned w/ subs setting.

Setup (run once on the remote machine):
    pip install -U "huggingface_hub[cli]" decord pillow pandas pyarrow
    hf download lmms-lab/Video-MME --repo-type dataset --local-dir ~/VideoMME

Example:
    python evaluate_videomme.py \
        --data_root ~/VideoMME \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --scorer_ckpt checkpoints_oracle/oracle_scorer_best.pt \
        --hidden_dim 512 \
        --rhos 0.25 0.50 0.75 \
        --strategies ours uniform random
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# Reuse evaluate_mvbench's generic data/scoring helpers (frame sampling, the
# no_grad gated-forward sample prep, MC scoring) so Video-MME eval matches MVBench
# and training exactly; those in turn reuse evaluate_oracle's selection helpers.
from evaluate_mvbench import frame_indices, prepare_sample, mc_correct
from evaluate_oracle import (
    STRATEGIES,
    select_ours,
    select_uniform,
    select_random,
    k_from_rho,
    gated_answer_logits,
    load_scorer,
)

DURATIONS = ("short", "medium", "long")

# Official Video-MME multiple-choice prompt (lmms-eval convention).
PRE_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the "
    "video. Respond with only the letter (A, B, C, or D) of the correct option."
)
POST_PROMPT = "The best answer is:"


# --------------------------------------------------------------------------- #
# Video-MME data loading
# --------------------------------------------------------------------------- #
def load_records(data_root):
    """Read the Video-MME parquet into a list of plain dicts (one per question)."""
    import pandas as pd

    path = os.path.join(data_root, "videomme", "test-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    return df.to_dict("records")


def frame_timestamps(path, num_segments):
    """Timestamps (sec) of the frames ``process_vision_info`` will sample.

    The clip itself is decoded and sampled inside ``prepare_sample`` (via
    ``process_vision_info``); for an unbounded video that samples at
    ``linspace(0, total-1, nframes)`` -- exactly what ``frame_indices`` returns --
    so these timestamps align with the frames the model actually sees, and let us
    attach the right subtitle line to each sampled frame."""
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    idxs = frame_indices(None, fps, len(vr) - 1, num_segments, first_idx=0)
    return [float(i) / fps for i in idxs]


def parse_srt(path):
    """Parse an .srt file into a list of (start_sec, end_sec, text) entries."""
    def to_sec(stamp):  # "00:01:02,500" -> seconds
        hms, ms = stamp.split(",")
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    entries, lines = [], open(path, encoding="utf-8", errors="ignore").read().splitlines()
    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            a, b = lines[i].split("-->")
            start, end = to_sec(a.strip()), to_sec(b.strip())
            i += 1
            text = []
            while i < len(lines) and lines[i].strip():
                text.append(lines[i].strip())
                i += 1
            entries.append((start, end, " ".join(text)))
        i += 1
    return entries


def subtitles_for_frames(srt_path, timestamps):
    """Subtitle lines active at any sampled-frame timestamp, deduped in order."""
    if not os.path.exists(srt_path):
        return ""
    entries = parse_srt(srt_path)
    chosen = []
    for ts in timestamps:
        for start, end, text in entries:
            if start <= ts <= end and text and text not in chosen:
                chosen.append(text)
    return "\n".join(chosen)


def build_prompt(rec, subs_text=None):
    """Video-MME option block + the letters present, and the ground-truth index."""
    options = [str(o) for o in rec["options"]]
    letters = [chr(ord("A") + i) for i in range(len(options))]
    opt_block = "\n".join(options)
    head = f"This video's subtitles are listed below:\n{subs_text}\n\n" if subs_text else ""
    text = f"{head}{PRE_PROMPT}\n{rec['question']}\n{opt_block}\n{POST_PROMPT}"
    gt_idx = ord(str(rec["answer"]).strip()) - ord("A")
    return text, letters, gt_idx


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
def run(args):
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)  # cpu generator for 'random' strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    durations = list(DURATIONS) if args.durations == ["all"] else args.durations
    unknown_d = [d for d in durations if d not in DURATIONS]
    if unknown_d:
        raise ValueError(f"unknown durations {unknown_d}; choices: {list(DURATIONS)}")

    strategies = list(args.strategies)
    unknown_s = [s for s in strategies if s not in STRATEGIES]
    if unknown_s:
        raise ValueError(f"unknown strategies {unknown_s}; choices: {list(STRATEGIES)}")
    if "ours" in strategies and not args.scorer_ckpt:
        raise ValueError("strategy 'ours' requires --scorer_ckpt (trained via train_oracle.py).")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn
    )
    model.eval()
    model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    data_root = os.path.expanduser(args.data_root)
    video_dir = os.path.join(data_root, "data")
    sub_dir = os.path.join(data_root, "subtitle")
    records = load_records(data_root)
    # group by duration, capping per-split for balanced debug runs
    by_dur = {}
    for d in durations:
        recs = [r for r in records if str(r["duration"]) == d]
        by_dur[d] = recs[: args.max_samples] if args.max_samples else recs
    print(f"Eval: durations={durations} (n={[len(by_dur[d]) for d in durations]}) | "
          f"strategies={strategies} | rhos={args.rhos} | subs={args.use_subs}")

    scorer = None
    correct = {d: {s: {r: 0 for r in args.rhos} for s in strategies} for d in durations}
    full_correct = {d: 0 for d in durations}
    seen = {d: 0 for d in durations}
    llm_ms = {s: {r: 0.0 for r in args.rhos} for s in strategies}
    n_timed = {s: {r: 0 for r in args.rhos} for s in strategies}
    full_llm_ms = 0.0
    full_n_timed = 0
    evaluated = 0

    for d in durations:
        for rec in tqdm(by_dur[d], desc=d):
            try:
                video_path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
                timestamps = frame_timestamps(video_path, args.max_frames)
                subs = (subtitles_for_frames(os.path.join(sub_dir, f"{rec['videoID']}.srt"),
                                             timestamps) if args.use_subs else None)
                text, letters, gt_idx = build_prompt(rec, subs)
                sample = prepare_sample(model, processor, video_path, "video", False, rec,
                                        text, letters, gt_idx, video_token_id, device, args)
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{d}] {rec.get('videoID')}: {e}")
                continue
            if sample is None:
                continue

            features = sample["features"]
            n_video = features.shape[0]
            n_frames = sample["n_frames"]
            warm = evaluated >= args.warmup

            # Lazily build the scorer once we know the feature width.
            scores = None
            if "ours" in strategies:
                if scorer is None:
                    scorer = load_scorer(args.scorer_ckpt, features.shape[-1], args.hidden_dim, device)
                scores = scorer(features.unsqueeze(0))[0]

            # Full-model reference (keep all tokens).
            if not args.no_full:
                keep_all = torch.arange(n_video, device=device)
                logits, ms = gated_answer_logits(model, sample, keep_all, time_llm=True)
                full_correct[d] += int(mc_correct(logits, sample["letter_ids"], sample["gt_idx"]))
                if warm:
                    full_llm_ms += ms
                    full_n_timed += 1

            for rho in args.rhos:
                k = k_from_rho(n_video, rho)
                for strat in strategies:
                    if strat == "ours":
                        kept = select_ours(scores, k, n_frames, args)
                    elif strat == "uniform":
                        kept = select_uniform(n_video, k, device)
                    else:  # random
                        kept = select_random(n_video, k, device, rng)
                    logits, ms = gated_answer_logits(model, sample, kept, time_llm=True)
                    correct[d][strat][rho] += int(
                        mc_correct(logits, sample["letter_ids"], sample["gt_idx"])
                    )
                    if warm:
                        llm_ms[strat][rho] += ms
                        n_timed[strat][rho] += 1
            seen[d] += 1
            evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (videomme/ and data/).")

    _report(args, durations, strategies, evaluated, seen, correct, full_correct,
            llm_ms, n_timed, full_llm_ms, full_n_timed)


def _report(args, durations, strategies, evaluated, seen, correct, full_correct,
            llm_ms, n_timed, full_llm_ms, full_n_timed):
    valid = [d for d in durations if seen[d]]
    full_llm = full_llm_ms / max(1, full_n_timed)

    # column specs: full first (if run), then strat-major x rho.
    col_specs = []
    if not args.no_full:
        col_specs.append(("full", None, None))
    for s in strategies:
        for r in args.rhos:
            col_specs.append((f"{s[:4]}@{r:g}", s, r))

    def split_acc(d, s, r):
        if s is None:
            return full_correct[d] / seen[d]
        return correct[d][s][r] / seen[d]

    # Video-MME headline: overall accuracy = total correct / total (micro).
    def overall(s, r):
        tot = sum((full_correct[d] if s is None else correct[d][s][r]) for d in valid)
        return tot / evaluated

    full_overall = overall(None, None) if not args.no_full else float("nan")

    table = {}
    for s in strategies:
        table[s] = {}
        for r in args.rhos:
            nt = max(1, n_timed[s][r])
            acc = overall(s, r)
            s_llm = llm_ms[s][r] / nt
            table[s][str(r)] = {
                "accuracy": acc,
                "accuracy_per_duration": {d: split_acc(d, s, r) for d in valid},
                "accuracy_retention": (acc / full_overall) if (not args.no_full and full_overall > 0) else float("nan"),
                "compute_saving": 1.0 - r,
                "speedup_llm": (full_llm / s_llm) if (not args.no_full and s_llm > 0) else float("nan"),
                "llm_ms": s_llm,
            }

    result = {
        "experiment": "videomme_oracle_eval",
        "model_name": args.model_name,
        "scorer_ckpt": args.scorer_ckpt,
        "data_root": args.data_root,
        "use_subs": args.use_subs,
        "durations": valid,
        "n_evaluated": evaluated,
        "rhos": args.rhos,
        "strategies": strategies,
        "k_min": args.k_min,
        "full_accuracy": full_overall,
        "full_accuracy_per_duration": {d: split_acc(d, None, None) for d in valid} if not args.no_full else {},
        "full_llm_ms": full_llm if not args.no_full else float("nan"),
        "per_duration": {
            d: {"n": seen[d], **{name: split_acc(d, s, r) for name, s, r in col_specs}}
            for d in valid
        },
        "table": table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # ---- per-duration accuracy table (one column per setting) ----
    cols = [name for name, _, _ in col_specs]
    print(f"\nEvaluated {evaluated} questions across {len(valid)} duration split(s)"
          + (f" | full(overall)={full_overall:.4f} | full_llm={full_llm:.2f}ms" if not args.no_full else "")
          + "\n")
    hdr = f"{'duration':<12}{'n':>5}" + "".join(f"{c:>11}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for d in valid:
        accs = "".join(f"{split_acc(d, s, r):>11.4f}" for _, s, r in col_specs)
        print(f"{d:<12}{seen[d]:>5}{accs}")
    print("-" * len(hdr))
    overall_row = "".join(f"{overall(s, r):>11.4f}" for _, s, r in col_specs)
    print(f"{'overall':<12}{evaluated:>5}{overall_row}")

    # ---- retention / speedup summary (mirrors evaluate_oracle) ----
    hdr2 = f"\n{'strat':<9}{'rho':>6}{'acc':>9}{'ret%':>8}{'save%':>8}{'sp_llm':>9}"
    print(hdr2)
    print("-" * len(hdr2.strip("\n")))
    for s in strategies:
        for r in args.rhos:
            tt = table[s][str(r)]
            print(f"{s:<9}{r:>6.2f}{tt['accuracy']:>9.4f}{tt['accuracy_retention'] * 100:>8.1f}"
                  f"{tt['compute_saving'] * 100:>8.1f}{tt['speedup_llm']:>9.2f}")
    print(f"\nSaved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Oracle-scorer gated eval on Video-MME: ours vs uniform vs random.")
    p.add_argument("--data_root", required=True, help="Dir holding videomme/, data/, subtitle/ (see module docstring).")
    p.add_argument("--durations", nargs="+", default=["all"], help="duration splits to run, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--scorer_ckpt", default="", help="Scorer checkpoint from train_oracle.py (oracle_scorer_*.pt).")
    p.add_argument("--hidden_dim", type=int, default=512, help="Must match the trained scorer (train_oracle default 512).")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.50, 0.75], help="retention ratios")
    p.add_argument("--strategies", nargs="+", default=["ours", "uniform", "random"],
                   help=f"gating strategies to sweep: {list(STRATEGIES)} (full is always the baseline)")
    p.add_argument("--max_samples", type=int, default=None, help="Cap questions evaluated PER DURATION split (debug).")
    p.add_argument("--max_frames", type=int, default=16, help="frames sampled per clip (uniform over the full video).")
    p.add_argument("--max_pixels", type=int, default=None, help="Cap per-frame resolution (e.g. 100352) on small GPUs.")
    p.add_argument("--use_subs", action="store_true", help="Inject frame-aligned .srt subtitles (Video-MME w/ subs).")
    # stratified-Pareto selection knobs (must match how you intend to deploy 'ours')
    p.add_argument("--k_min", type=int, default=1, help="per-frame coverage floor for stratified selection")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--warmup", type=int, default=3, help="samples excluded from timing stats (CUDA warm-up)")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model reference (no retention/speedup).")
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_videomme_eval.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
