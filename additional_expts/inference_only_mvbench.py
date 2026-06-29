"""
inference_only_mvbench.py -- inference-only token-selection baselines on MVBench.
=================================================================================
Experiment 1, ported to the OFFICIAL MVBench protocol (the same data layout,
midpoint frame sampling, prompt, ``Best option:(`` answer forcing and
option-letter logit readout as ``inference.py``).

Compares four INFERENCE-ONLY selection strategies at matched retention, through
the frozen VLM. No backward pass, no training, no caching:

  * random            : uniform K-subset of video tokens (per sample)
  * stratified_uniform: K evenly spaced tokens, per-frame stratified
  * attention_topk    : top-K by language->video attention (the academic proxy)
  * inverse_transform : per-frame sampling prop. to softmax(attn / temp)
                        (between top-k and uniform; the diversity hypothesis)

Unlike ``inference.py`` -- which routes content-free baselines through the
model's native pre-LLM gating -- this script selects tokens explicitly by
editing the input embedding sequence, so it can run CONTENT-AWARE strategies
(attention_topk / inverse_transform) that gating cannot express. As a result
its absolute accuracies are not directly comparable to ``inference.py`` (tokens
are dropped from the embedding sequence, not gated pre-LLM); compare strategies
WITHIN this script.

For each strategy and retention rho we report downstream answer accuracy AND the
mean spatial dispersion of the selected set (so the result can be explained, not
just reported): clustered selections (low dispersion) vs space-filling ones.

Dispersion = mean pairwise Chebyshev distance between selected tokens' (x,y) grid
coordinates within a frame, averaged over frames and samples, normalised by grid
size. High = spread across the frame; low = clustered.

Setup (run once):
    pip install -U "huggingface_hub[cli]" decord pillow
    hf download OpenGVLab/MVBench --repo-type dataset --local-dir ~/MVBench
    cd ~/MVBench/video && for z in $(find . -name '*.zip'); do unzip -n -q "$z"; done

Run:
    python inference_only_mvbench.py \
        --data_root ~/MVBench \
        --tasks "Action Sequence" "Scene Transition" \
        --rhos 0.01,0.05,0.10,0.25 --max_frames 8 --out results_exp1.json

    python inference_only_mvbench.py \
    --data_root ~/MVBench \
    --tasks "Action Sequence" \
    --max_frames 16 \
    --max_pixels 200704 \
    --out results_exp1_action_sequence.json

"""

import os
import json
import argparse
import numpy as np
import sys
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Reuse the shared, validated selection/scoring helpers ...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oracle_check import (build_full_positions, select_uniform, select_by_scores,
                          score_answer)
# ... and the OFFICIAL MVBench data/prompt/sampling protocol from inference.py.
from inference import DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt


STRATEGIES = ("random", "stratified_uniform", "attention_topk", "inverse_transform")


# --------------------------------------------------------------------------- #
# attention signal (language -> video), the only signal this experiment needs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_scores(model, base_embeds, video_positions, position_ids, attn, layers):
    """Mean attention received by each video token from text queries, over `layers`."""
    out = model(inputs_embeds=base_embeds, position_ids=position_ids,
                attention_mask=attn, use_cache=False, output_attentions=True)
    text_mask = torch.ones(base_embeds.shape[1], dtype=torch.bool, device=base_embeds.device)
    text_mask[video_positions] = False
    acc = torch.zeros(video_positions.numel(), device=base_embeds.device)
    for L in layers:
        a = out.attentions[L][0].mean(0)              # (S,S) mean over heads
        acc += a[text_mask][:, video_positions].sum(dim=0)
    del out
    return (acc / max(1, len(layers))).float()


# --------------------------------------------------------------------------- #
# spatial dispersion of a selected set (per-frame, normalised)
# --------------------------------------------------------------------------- #
def frame_grid_dims(n_video, n_frames):
    """Infer (h, w) per-frame grid from token count. Qwen merged grid is square-ish;
    we recover w = round(sqrt(per)) and h = per // w as a robust default."""
    per = n_video // n_frames
    w = int(round(per ** 0.5))
    w = max(1, w)
    h = max(1, per // w)
    return h, w, per


def mean_dispersion(keep_idx, n_video, n_frames):
    """Mean pairwise Chebyshev distance among selected tokens within each frame,
    averaged over frames, normalised by max possible distance. 0=clustered, 1=spread.
    keep_idx: 1-D LongTensor of selected local video-token indices (0..n_video-1)."""
    h, w, per = frame_grid_dims(n_video, n_frames)
    if per < 2:
        return float("nan")
    idx = keep_idx.detach().cpu().numpy()
    max_d = max(1, max(h - 1, w - 1))
    vals = []
    for t in range(n_frames):
        lo, hi = t * per, (t + 1) * per
        sel = idx[(idx >= lo) & (idx < hi)] - lo
        if sel.size < 2:
            continue
        ys, xs = sel // w, sel % w
        coords = np.stack([ys, xs], axis=1).astype(np.float64)
        # mean pairwise Chebyshev distance
        d = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
        iu = np.triu_indices(coords.shape[0], k=1)
        if iu[0].size:
            vals.append(d[iu].mean() / max_d)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def select_inverse_transform(scores, n_frames, k, temp, rng):
    """Per-frame sampling WITHOUT replacement, with inclusion probability
    proportional to softmax(score / temp). temp->0 approaches top-k;
    temp->inf approaches uniform-random. Keeps high-attention tokens while
    preserving spread (the diversity/coverage hypothesis)."""
    n_video = scores.numel()
    per = n_video // n_frames
    kp = max(1, k // n_frames)
    s = scores.detach().float().cpu()
    idx_all = []
    for t in range(n_frames):
        seg = s[t * per:(t + 1) * per]
        m = min(kp, seg.numel())
        if m <= 0:
            continue
        logits = seg / max(temp, 1e-6)
        logits = logits - logits.max()                 # stabilise
        w = torch.softmax(logits, dim=0)
        # Gumbel top-m == sampling m without replacement prop. to w
        g = -torch.log(-torch.log(torch.rand(seg.numel(), generator=rng).clamp_min(1e-12)))
        keys = torch.log(w.clamp_min(1e-12)) + g
        top = torch.topk(keys, m).indices + t * per
        idx_all.append(top)
    keep = torch.cat(idx_all).sort().values if idx_all else torch.arange(min(k, n_video))
    return keep[:k]


def select(strategy, scores_attn, n_video, n_frames, k, device, rng, it_temp):
    if strategy == "random":
        return torch.sort(torch.randperm(n_video, generator=rng)[:k]).values.to(device)
    if strategy == "stratified_uniform":
        return select_uniform(n_video, n_frames, k, device)
    if strategy == "attention_topk":
        return select_by_scores(scores_attn, n_frames, k)
    if strategy == "inverse_transform":
        return select_inverse_transform(scores_attn, n_frames, k, it_temp, rng).to(device)
    raise ValueError(strategy)


def letter_first_ids(processor, letters):
    """First-token id of each bare option letter -- the token that follows the
    ``Best option:(`` prefix (no leading space), matching the official readout."""
    tok = processor.tokenizer
    return [tok.encode(L, add_special_tokens=False)[0] for L in letters]


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
    attn_layers = [int(x) for x in args.attn_layers.split(",")]
    rhos = [float(x) for x in args.rhos.split(",")]

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    # per (strategy, rho, task) correct counts and per-task seen counts.
    correct = {s: {r: {t: 0 for t in tasks} for r in rhos} for s in STRATEGIES}
    disp = {s: {r: [] for r in rhos} for s in STRATEGIES}
    full_correct = {t: 0 for t in tasks}
    seen = {t: 0 for t in tasks}
    n = 0
    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = build_prompt(rec)
                letter_ids = letter_first_ids(processor, letters)
                prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                             args.max_frames, args.max_pixels, args.fps,
                                             official=args.official_sampling,
                                             num_segments=args.num_segments)
                chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                chat += ANSWER_PREFIX  # force the answer; option letter is the next token after '('
                img_in, vid_in = process_vision_info(prompt)
                inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            input_ids = inputs["input_ids"].to(device)
            attn = inputs["attention_mask"].to(device)
            vpos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
            if vpos.numel() < 50:
                continue
            n_video = vpos.numel()
            grid_thw = inputs["video_grid_thw"].to(device)
            n_frames = int(grid_thw[0][0].item())
            pix = inputs["pixel_values_videos"].to(device)

            with torch.no_grad():
                ve = model.get_video_features(pix, grid_thw).pooler_output
                ve = torch.cat(ve, dim=0).to(device)
                base = model.get_input_embeddings()(input_ids).clone()
                base[0, vpos] = ve.to(base.dtype)
            pos = build_full_positions(model, input_ids, vpos, grid_thw, attn)

            scores_attn = attention_scores(model, base, vpos, pos, attn, attn_layers)

            # full-model reference (all video tokens kept)
            keep_all = torch.arange(n_video, device=device)
            lp_full = score_answer(model, base, pos, attn, vpos, keep_all, letter_ids)
            full_correct[task] += int(lp_full.argmax().item() == gt_idx)

            for rho in rhos:
                k = max(n_frames, int(round(rho * n_video)))
                for s in STRATEGIES:
                    keep = select(s, scores_attn, n_video, n_frames, k, device, rng, args.it_temp)
                    lp = score_answer(model, base, pos, attn, vpos, keep, letter_ids)
                    correct[s][rho][task] += int(lp.argmax().item() == gt_idx)
                    d = mean_dispersion(keep, n_video, n_frames)
                    if d == d:
                        disp[s][rho].append(d)
            seen[task] += 1
            n += 1
            del base, ve, scores_attn

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    # ---- aggregate: per-task mean (leaderboard metric) and micro (per-sample) ----
    valid = [t for t in tasks if seen[t]]

    def per_task_mean(counts):  # mean over valid tasks of correct/seen
        return float(np.mean([counts[t] / seen[t] for t in valid]))

    def micro(counts):          # total correct / total samples
        return sum(counts[t] for t in valid) / n

    full_mean, full_micro = per_task_mean(full_correct), micro(full_correct)

    out = {"experiment": "mvbench_inference_only_selection",
           "model_name": args.model_name, "data_root": args.data_root,
           "sampling": ("official" if args.official_sampling else "fps"),
           "num_frames": (args.num_segments if args.official_sampling else None),
           "fps": (None if args.official_sampling else args.fps),
           "max_frames": args.max_frames, "rhos": rhos, "attn_layers": attn_layers,
           "it_temp": args.it_temp, "tasks": valid, "n": n,
           "full_accuracy_mean": full_mean, "full_accuracy_micro": full_micro,
           "per_task_seen": {t: seen[t] for t in valid},
           "full_per_task": {t: full_correct[t] / seen[t] for t in valid},
           "table": {}}

    print(f"\n==== EXPERIMENT 1: inference-only selection on MVBench "
          f"({n} samples, {len(valid)} task(s)) ====")
    print(f"full-model accuracy: mean(per-task) {full_mean:.4f}  micro {full_micro:.4f}\n")
    hdr = f"{'strategy':<20}{'rho':>6}{'acc':>9}{'micro':>9}{'dispersion':>12}"
    print(hdr); print("-" * len(hdr))
    for s in STRATEGIES:
        out["table"][s] = {}
        for rho in rhos:
            acc = per_task_mean(correct[s][rho])
            acc_micro = micro(correct[s][rho])
            dsp = float(np.mean(disp[s][rho])) if disp[s][rho] else float("nan")
            out["table"][s][str(rho)] = {
                "accuracy_mean": acc, "accuracy_micro": acc_micro, "dispersion": dsp,
                "per_task": {t: correct[s][rho][t] / seen[t] for t in valid}}
            print(f"{s:<20}{rho:>6.2f}{acc:>9.4f}{acc_micro:>9.4f}{dsp:>12.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nread: random ~ stratified_uniform > attention_topk => attention selection fails.")
    print("      inverse_transform between top-k and uniform tests whether spreading an")
    print("      attention-anchored sample helps; pair accuracy with dispersion to explain.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference-only token-selection baselines on MVBench (official protocol).")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames per clip (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="Use the reference mvbench.ipynb sampler (fixed --num_segments frames at "
                        "segment midpoints) for leaderboard-comparable numbers, instead of fps sampling.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK (debug).")
    p.add_argument("--rhos", default="0.01,0.05,0.10,0.25")
    p.add_argument("--attn_layers", default="12,13,14,15,16")
    p.add_argument("--it_temp", type=float, default=1.0,
                   help="inverse-transform temperature: ->0 top-k, ->inf uniform")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp1.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
