"""Content-free video-token reduction baselines on Video-MME.

Companion to ``inference.py`` (which runs the same baselines on MVBench). Most
video tokens are unnecessary; this measures how Video-MME accuracy holds up when,
at a retention ratio rho, we keep only ``k = round(rho * tokens)`` video tokens
per clip chosen *without looking at content*:

  * random         -- k random tokens (chance baseline at the budget)
  * uniform        -- k evenly-spaced tokens over the flattened (frame-major)
                      sequence (a space-time diagonal sweep)
  * uniform_strat  -- even per-frame quota of evenly-spaced spatial tokens
                      (a uniform spatial grid replicated across frames)
  * regvar         -- content-free regularly-varying (Pareto) per-frame budget
                      drawn by inverse-transform sampling (the content-free null
                      for adaptive budgeting; --rv_alpha sets the tail index)
  * first / last   -- the first / last k tokens in order (prefix / suffix frames)

They reuse the model's native pre-LLM gating (``token_selection_mode``), so the
decoder runs on a genuinely shorter sequence -- no scorer checkpoint required.
The full (no-drop) model is the reference, and ``--blind`` adds a text-only
baseline (zero vision tokens, the model's language prior).

Prompt and answer readout follow the official Video-MME protocol (the lmms-eval
template): the multiple-choice instruction, the option block, and a closing
"The best answer is:" cue. ``--subtitles`` prepends the clip's .srt text (and
switches the instruction to the "...video and the subtitles..." wording). The
letter is read in a single forward pass as the argmax over the A/B/C/D option-
letter logits at that cue -- no autoregressive decoding or regex parsing.

Data layout (official Video-MME release):
    <data_root>/Video-MME.json          # grouped-by-video or flat list of questions
    <data_root>/data/<videoID>.mp4      # videos
    <data_root>/subtitle/<videoID>.srt  # subtitles (only needed for --subtitles)

Setup (questions json):
    hf download lmms-lab/Video-MME --repo-type dataset --local-dir ~/Video-MME
    pip install -U decord pillow

Example:
    python inference_videomme.py \
        --data_root ~/Video-MME \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --num_frames 32 \
        --rhos 0.25 0.5 0.75 --baselines random uniform
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Official Video-MME (lmms-eval) instruction; the "and the subtitles" clause is
# added only when --subtitles is on. The answer is forced by the closing cue, so
# the very next token is the option letter -- the logit readout needs no parsing.
INSTRUCTION = ("Select the best answer to the following multiple-choice question based on "
               "the video{subs}. Respond with only the letter (A, B, C, or D) of the correct option.")
ANSWER_CUE = "The best answer is:"
SUBTITLE_HEADER = "This video's subtitles are listed below: \n"

LETTERS = ["A", "B", "C", "D"]
DURATIONS = ["short", "medium", "long"]
TEMPORAL_PATCH_SIZE = 2  # Qwen2.5-VL pairs adjacent frames; the sampled count must be even.


def load_questions(path):
    """Flatten the Video-MME data to a list of question dicts (videoID/duration/
    question/options/answer). Accepts a .parquet file, the official grouped-by-video
    JSON format, or a flat per-question JSON list."""
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return [
            {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}
            for row in df.to_dict("records")
        ]
    with open(path) as fh:
        data = json.load(fh)
    out = []
    for rec in data:
        if "questions" in rec:  # grouped: one video -> several questions
            for q in rec["questions"]:
                out.append({"videoID": rec["videoID"], "duration": rec["duration"], **q})
        else:  # already flat
            out.append(rec)
    return out


def uniform_indices(num_frames, num_segments):
    """num_segments frame indices at the midpoints of equal temporal segments over
    the whole clip (Video-MME clips have no temporal bounds)."""
    seg = num_frames / num_segments
    return np.array([min(num_frames - 1, int(seg / 2 + round(seg * i))) for i in range(num_segments)])


def sampled_frames(path, num_segments):
    """num_segments midpoint-sampled PIL frames decoded with decord (uniform over
    the whole video). num_segments is rounded up to an even count for Qwen pairing."""
    from decord import VideoReader, cpu  # lazy import: only needed to read pixels
    n = max(TEMPORAL_PATCH_SIZE, round(num_segments / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idxs = uniform_indices(len(vr), n)
    return [Image.fromarray(f) for f in vr.get_batch(idxs).asnumpy()]


def load_subtitle(path):
    """Concatenated subtitle text from a .srt file (drops indices/timestamps and
    consecutive duplicate lines). Returns '' if the file is missing."""
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    lines = []
    for block in content.strip().split("\n\n"):
        rows = [r for r in block.strip().split("\n") if r.strip()]
        text = " ".join(rows[2:]).strip() if len(rows) >= 3 else ""
        text = re.sub(r"<[^>]+>", "", text)  # strip any inline markup
        if text and (not lines or lines[-1] != text):
            lines.append(text)
    return "\n".join(lines)


def make_video_prompt(path, text, num_frames, max_pixels):
    """Qwen chat prompt whose video item is an explicit list of num_frames
    midpoint-sampled PIL frames (the protocol Video-MME evals use: a fixed number
    of frames sampled uniformly over the clip)."""
    vid = {"type": "video", "video": sampled_frames(path, num_frames)}
    if max_pixels is not None:
        vid["max_pixels"] = max_pixels
    return [{"role": "user", "content": [vid, {"type": "text", "text": text}]}]


def build_prompt(record, subtitle=None):
    """Official Video-MME prompt + the ground-truth index. Options already carry
    their letter prefixes (``A. ...``); the closing cue forces the answer letter."""
    subs = " and the subtitles" if subtitle else ""
    header = f"{SUBTITLE_HEADER}{subtitle}\n" if subtitle else ""
    options = "\n".join(record["options"])
    text = (f"{header}{INSTRUCTION.format(subs=subs)}\n"
            f"{record['question']}\n{options}\n{ANSWER_CUE}")
    gt_idx = LETTERS.index(record["answer"].strip().upper())
    return text, gt_idx


def letter_token_ids(processor):
    """First-token id candidates per option letter (with/without leading space)."""
    tok = processor.tokenizer
    out = []
    for L in LETTERS:
        cands = set()
        for form in (L, f" {L}"):
            ids = tok.encode(form, add_special_tokens=False)
            if ids:
                cands.add(ids[0])
        out.append(sorted(cands))
    return out


@torch.no_grad()
def predict(model, inputs, letter_ids):
    last = model(**inputs).logits[0, -1]
    opt = [max(last[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    return int(np.argmax(opt))


def video_text_counts(model, inputs):
    ids = inputs["input_ids"][0]
    n_video = int((ids == model.config.video_token_id).sum())
    return n_video, int(ids.numel()) - n_video


def kept_video_count(model, inputs, keep_ratio):
    """Video tokens the gating keeps (deterministic: max(1, round(rho*tokens)) per clip)."""
    n_video, _ = video_text_counts(model, inputs)
    if keep_ratio is None:
        return n_video
    merge = model.config.vision_config.spatial_merge_size**2
    kept = sum(max(1, round(keep_ratio * (int(row.prod()) // merge))) for row in inputs["video_grid_thw"])
    return min(kept, n_video)


def main():
    p = argparse.ArgumentParser(
        description="Content-free video-token reduction baselines (random / uniform) on Video-MME.")
    p.add_argument("--data_root", required=True, help="Dir holding Video-MME.json, data/ and subtitle/.")
    p.add_argument("--json_name", default="Video-MME.json", help="Questions json under --data_root.")
    p.add_argument("--durations", nargs="+", default=["all"], choices=["all", *DURATIONS],
                   help="Video-MME duration buckets to evaluate, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="Retention ratios: fraction of video tokens kept per clip.")
    p.add_argument("--baselines", nargs="*", default=["random", "uniform", "uniform_strat"],
                   choices=["random", "uniform", "uniform_strat", "regvar", "first", "last"],
                   help="Content-free selection modes to run at each rho (same token budget).")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random / regvar baselines.")
    p.add_argument("--rv_alpha", type=float, default=2.0,
                   help="Pareto tail index for the 'regvar' baseline (smaller = heavier tail).")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model (no-drop) reference.")
    p.add_argument("--blind", action="store_true",
                   help="Also run a text-only (no video) baseline: the question+options with ZERO "
                        "vision tokens, measuring the model's language prior.")
    p.add_argument("--subtitles", action="store_true",
                   help="Prepend each clip's .srt subtitle text (official 'with subtitles' setting).")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Frames sampled uniformly over each clip (rounded up to even for Qwen pairing).")
    p.add_argument("--video_subdir", default="data", help="Video subdir under --data_root.")
    p.add_argument("--subtitle_subdir", default="subtitle", help="Subtitle subdir under --data_root.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap questions evaluated (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--out", default="results_videomme_baselines.json", help="Path to dump metrics JSON.")
    args = p.parse_args()

    durations = list(DURATIONS) if args.durations == ["all"] else args.durations

    torch.manual_seed(args.seed)  # reproducible "random" selection
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.model.token_gating_rv_alpha = args.rv_alpha  # Pareto tail index for the regvar baseline
    letter_ids = letter_token_ids(processor)  # fixed A/B/C/D, same for every question

    # (column name, rho, selection mode). full = no dropping; blind = no video tokens.
    label = {"random": "rand", "uniform": "unif", "uniform_strat": "ustr", "regvar": "rvar",
             "first": "frst", "last": "last"}
    settings = [] if args.no_full else [("full", None, None)]
    if args.blind:
        settings.append(("blind", None, "blind"))
    for r in args.rhos:
        for b in args.baselines:
            settings.append((f"{label[b]}@{r}", r, b))
    col = [name for name, _, _ in settings]
    need_video = any(m != "blind" for _, _, m in settings)

    questions = [q for q in load_questions(os.path.join(os.path.expanduser(args.data_root), args.json_name))
                 if q["duration"] in durations]
    if args.max_samples:
        questions = questions[:args.max_samples]

    correct = {d: {c: 0 for c in col} for d in durations}
    seen = {d: 0 for d in durations}
    kept_vid = {c: 0 for c in col}
    vid_total = 0
    n = 0

    video_dir = os.path.join(os.path.expanduser(args.data_root), args.video_subdir)
    sub_dir = os.path.join(os.path.expanduser(args.data_root), args.subtitle_subdir)

    for rec in tqdm(questions, desc="Video-MME"):
        dur = rec["duration"]
        try:
            path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
            subtitle = load_subtitle(os.path.join(sub_dir, f"{rec['videoID']}.srt")) if args.subtitles else None
            text, gt_idx = build_prompt(rec, subtitle=subtitle)
            inputs = None
            if need_video:
                prompt = make_video_prompt(path, text, args.num_frames, args.max_pixels)
                chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                imgs, vids = process_vision_info(prompt)
                inputs = processor(text=[chat], images=imgs, videos=vids, return_tensors="pt").to(model.device)
            blind_inputs = None
            if args.blind:  # text-only prompt: same question/options, no video item
                bmsg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                bchat = processor.apply_chat_template(bmsg, tokenize=False, add_generation_prompt=True)
                blind_inputs = processor(text=[bchat], return_tensors="pt").to(model.device)
        except Exception as e:  # missing/corrupt clip -> skip
            tqdm.write(f"skip [{dur}] {rec.get('videoID')}: {e}")
            continue

        if need_video:
            vid_total += video_text_counts(model, inputs)[0]
        for name, rho, mode in settings:
            if mode == "blind":  # no video tokens at all -- language prior
                correct[dur][name] += int(predict(model, blind_inputs, letter_ids) == gt_idx)
                continue
            if rho is None:
                model.model.disable_token_gating()
            else:
                model.model.token_keep_ratio = rho
                model.model.token_selection_mode = mode
            correct[dur][name] += int(predict(model, inputs, letter_ids) == gt_idx)
            kept_vid[name] += kept_video_count(model, inputs, rho)
        seen[dur] += 1
        n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (Video-MME.json and data/).")

    valid = [d for d in durations if seen[d]]
    setting_mean = {c: float(np.mean([correct[d][c] / seen[d] for d in valid])) for c in col}
    setting_micro = {c: sum(correct[d][c] for d in valid) / n for c in col}

    print(f"\nEvaluated {n} questions across {len(valid)} duration bucket(s)\n")
    print(f"{'duration':<12}{'n':>6}" + "".join(f"{c:>10}" for c in col))
    print("-" * (18 + 10 * len(col)))
    for d in valid:
        accs = "".join(f"{correct[d][c] / seen[d]:>10.4f}" for c in col)
        print(f"{d:<12}{seen[d]:>6}{accs}")
    print("-" * (18 + 10 * len(col)))
    print(f"{'mean (per-bucket)':<12}{'':>6}" + "".join(f"{setting_mean[c]:>10.4f}" for c in col))
    print(f"{'micro (overall)':<12}{n:>6}" + "".join(f"{setting_micro[c]:>10.4f}" for c in col))

    # per-setting accuracy / kept-token budget
    print(f"\n{'setting':<10}{'acc':>9}{'vid%':>8}{'vid_tok':>9}")
    print("-" * 36)
    for name in col:
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        print(f"{name:<10}{setting_micro[name]:>9.4f}{vid_pct:>7.1f}%{kept_vid[name] / n:>9.0f}")

    result = {
        "experiment": "videomme_content_free_baselines",
        "model_name": args.model_name,
        "data_root": args.data_root,
        "num_frames": args.num_frames,
        "subtitles": args.subtitles,
        "rhos": args.rhos,
        "baselines": args.baselines,
        "rv_alpha": args.rv_alpha,
        "seed": args.seed,
        "settings": col,
        "durations": valid,
        "n_evaluated": n,
        "per_duration": {d: {"n": seen[d], **{c: correct[d][c] / seen[d] for c in col}} for d in valid},
        "table": {
            name: {
                "accuracy_mean": setting_mean[name],
                "accuracy_micro": setting_micro[name],
                "kept_vid_pct": (kept_vid[name] / vid_total * 100 if vid_total else 0.0),
                "vid_tok": kept_vid[name] / n,
            }
            for name in col
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
