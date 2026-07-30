"""
anchor_layer_prune_mvbench.py -- Efficient_VLMs_Method.pdf's anchor-layer prune, on MVBench.
=============================================================================================
Benchmark harness for ``anchor_layer_prune.py``'s method: per clip, ONE dense analysis
forward picks the anchor layer L* and the K kept video tokens (value-norm debiased
importance + cross-shaped spatio-temporal diversity, Sections 4-7), then a SECOND forward
through the FastV-style monkey-patched text model actually restricts layers L*+1..end to
those K tokens (Section 8) -- a real compute/KV reduction, not a selection-only score.

Two backbones (--backbone): qwen (Qwen2.5-VL, dynamic resolution, mRoPE) and llava_video
(LLaVA-OneVision / LLaVA-Video-7B-Qwen2, REAL multi-frame video via a fixed --num_frames
uniform sample, 1D RoPE, Qwen2Model text backbone). The method itself and everything
below the per-clip input construction (scoring, strategies, skip-tracking, tables) is
fully shared -- see anchor_layer_prune.py's own backbone plumbing for the same pattern.

Strategies reported, all via the SAME real prune mechanism at the SAME anchor layer L*
(so only WHICH video tokens are kept differs -- the isolate-one-variable approach used
throughout this repo, e.g. lcds_mvbench.py's mmr_lambda sweep):
  * random_prune  : K random video tokens kept at L*                      (floor)
  * uniform_prune : K stratified evenly-spaced video tokens kept at L*    (floor)
  * anchor_prune  : importance-diversity greedy selection at L*           (ours)
`full` (no pruning at all) is reported separately as the reference ceiling, exactly as
lcds_mvbench.py reports its `full_correct`.

Uses the OFFICIAL MVBench data/prompt/sampling protocol (same imports as the sibling
MVBench experiments), so numbers are comparable across the suite.

Run (qwen):
    python anchor_layer_prune_mvbench.py \
        --data_root ~/Experiments/MVBench --tasks "Action Sequence" "Scene Transition" \
        --official_sampling --num_segments 16 --max_pixels 200704 \
        --max_samples 40 --rhos 0.05,0.10,0.25 --band 2,3,4,5,6,7,8 \
        --out results_anchor_layer_prune_mvbench.json

Run (llava_video):
    python anchor_layer_prune_mvbench.py --backbone llava_video \
        --data_root ~/Experiments/MVBench --tasks "Action Sequence" "Scene Transition" \
        --num_frames 8 --max_samples 40 --rhos 0.05,0.10,0.25 \
        --out results_anchor_layer_prune_mvbench_llava.json
"""

import os
import sys
import time
import json
import warnings
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Noise-only: torchvision's deprecation notice fires on every decord fallback
# (missing/corrupt clip) and adds nothing a skip-reason line doesn't already say.
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oracle_check import build_full_positions
from inference import DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt, official_frames
from inference_only_mvbench import letter_first_ids
from lcds_mvbench import select_uniform_exact               # exact-budget stratified floor
from anchor_layer_prune import (
    plan, token_grid, token_grid_video, keep_indices_from_plan, full_sequence_keep_idx,
    install_anchor_prune, set_anchor_prune, resolve_backbone, default_model_id,
    build_llava_video_prompt, _text_model, QWEN_MODEL_ID, LLAVA_VIDEO_MODEL_ID,
)

STRATEGIES = ("random_prune", "uniform_prune", "anchor_prune")


def load_backbone(backbone, model_name, dtype):
    """Load model + processor + video-token id + a small `info` dict of
    backbone constants (qwen: spatial merge; llava_video: pooled grid side)."""
    if backbone == "llava_video":
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
        model.eval(); model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(model_name)
        video_token_id = model.config.video_token_id
        vcfg = model.config.vision_config
        patches_side = vcfg.image_size // vcfg.patch_size
        pooled_side = -(-patches_side // 2)          # ceil(side/2): apply_pooling's 2x downsample
        info = {"pooled_side": pooled_side}
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
        model.eval(); model.requires_grad_(False)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        info = {"merge_size": model.config.vision_config.spatial_merge_size}
    return model, processor, video_token_id, info


def prepare_clip(backbone, model, processor, video_token_id, info, path, data_type,
                 has_bound, rec, args, device, dtype, min_pixels):
    """Build one clip's merged embeds/positions/mask/video-idx/grid, dispatched by
    backbone. Returns (base_embeds, position_ids, attn_mask, video_idx, grid,
    n_frames, seq_len, letter_ids, gt_idx)."""
    text, letters, gt_idx = build_prompt(rec)
    letter_ids = letter_first_ids(processor, letters)

    if backbone == "llava_video":
        frames = official_frames(path, data_type, has_bound, rec, args.num_frames)
        prompt = build_llava_video_prompt(processor, text) + ANSWER_PREFIX
        inputs = processor(text=prompt, videos=[frames], return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        pix = inputs["pixel_values_videos"].to(device=device, dtype=dtype)

        T = len(frames)
        S = info["pooled_side"] ** 2
        video_idx_full = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if video_idx_full.numel() != T * S + 1:
            raise RuntimeError(f"got {video_idx_full.numel()} video-token positions, expected "
                              f"{T * S + 1} ({T} frames x {S} pooled tokens + 1 newline).")
        video_idx = video_idx_full[:-1]                  # drop the trailing newline slot (Section 4 guard)

        # If the harness already installed the anchor-prune patch (it does, once up
        # front), the text model's forward is a plain function with NONE of the
        # original's output_hidden_states capture machinery (newer transformers wires
        # that up via a decorator on the ORIGINAL method), so this merging forward would
        # return hidden_states=None. Temporarily restore the original forward for this
        # one pass -- the exact guard plan() uses for its own analysis forward.
        tm = _text_model(model)
        orig_forward = getattr(tm, "_anchor_orig_forward", None)
        if orig_forward is not None:
            patched_forward = tm.forward
            tm.forward = orig_forward
        try:
            with torch.no_grad():
                merged = model(input_ids=input_ids, attention_mask=attn,
                              pixel_values_videos=pix, use_cache=False,
                              output_hidden_states=True)
                base = merged.hidden_states[0].detach()
        finally:
            if orig_forward is not None:
                tm.forward = patched_forward
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        grid = token_grid_video(T, info["pooled_side"], device)
        n_frames = T
    else:
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
        attn = inputs["attention_mask"].to(device)
        video_idx = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if video_idx.numel() < 50:
            raise RuntimeError(f"too few video tokens ({video_idx.numel()} < 50)")
        grid_thw = inputs["video_grid_thw"].to(device)
        n_frames = int(grid_thw[0][0].item())
        pix = inputs["pixel_values_videos"].to(device)

        with torch.no_grad():
            ve = model.get_video_features(pix, grid_thw).pooler_output
            ve = torch.cat(ve, dim=0).to(device)
            base = model.get_input_embeddings()(input_ids).clone()
            base[0, video_idx] = ve.to(base.dtype)
        pos = build_full_positions(model, input_ids, video_idx, grid_thw, attn)
        grid = token_grid(grid_thw, info["merge_size"])

    return base, pos, attn, video_idx, grid, n_frames, input_ids.shape[1], letter_ids, gt_idx


@torch.no_grad()
def score_letters(model, base_embeds, position_ids, attn_mask, letter_ids):
    """First-token log-prob of each candidate letter, whatever the text model's
    current anchor-prune configuration is (dense if `set_anchor_prune(tm, None,
    None)` was last called, pruned otherwise)."""
    out = model(inputs_embeds=base_embeds, position_ids=position_ids,
               attention_mask=attn_mask, use_cache=False)
    lp = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return torch.tensor([lp[t].item() for t in letter_ids])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resolve_backbone(args.backbone, args.model_name)
    if not args.model_name:
        args.model_name = default_model_id(backbone)
    if args.dtype == "auto":
        args.dtype = "bf16"                          # both qwen and llava_video ship bf16
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[backbone] {backbone}  model={args.model_name}  dtype={args.dtype}")
    model, processor, video_token_id, info = load_backbone(backbone, args.model_name, dtype)
    tm = install_anchor_prune(model)

    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    rhos = [float(x) for x in args.rhos.split(",")]
    band = [int(x) for x in args.band.split(",")]
    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    correct = {s: {t: {r: 0 for r in rhos} for t in tasks} for s in STRATEGIES}
    full_correct = {t: 0 for t in tasks}
    lstar_hist = {t: [] for t in tasks}          # chosen anchor layer per clip, per task
    lstar_tail = {t: [] for t in tasks}          # tail index AT the chosen anchor, per task
    skipped = []                                 # [{"task", "video", "reason"}], for a clean report
    latency_full, latency_pruned = [], {r: [] for r in rhos}   # wall-clock ms, CUDA only
    per_sample = []
    seen = {t: 0 for t in tasks}
    n = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        task_video_dir = os.path.join(video_dir, subdir)
        if not os.path.isdir(task_video_dir):
            # Fail fast with ONE clear message instead of a per-clip "skip" line for
            # every record (e.g. "Fine-grained Pose" needs video/nturgbd/, which MVBench
            # does not ship -- NTU-RGB+D requires a separate license/download; the repo
            # only ships MVBench_videos_ntu.txt, a manifest, not the clips themselves).
            print(f"[skip task] {task!r}: video dir not found ({task_video_dir}) -- "
                 f"skipping this task entirely.")
            continue
        try:                                    # a task json absent under this data_root
            with open(os.path.join(json_dir, fname)) as fh:   # (e.g. EgoSchema under MVBench/)
                records = json.load(fh)
        except (FileNotFoundError, OSError) as e:
            print(f"[skip task] {task!r}: {e} -- skipping this task entirely.")
            continue
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task, unit="clip"):
            video_name = rec.get("video")
            path = os.path.join(video_dir, subdir, video_name)
            # Check existence BEFORE ever touching process_vision_info: a missing file still
            # gets decord to print a raw C++ error line and fall back to torchvision (another
            # warning) before finally raising -- an opaque exception like KeyError('video_fps')
            # that hides the real cause. Catching it here skips straight past all of that
            # with one clear, correctly-labeled line.
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": video_name, "reason": "missing file"})
                tqdm.write(f"skip [{task}] {video_name}: missing file ({path})")
                continue

            try:
                base, pos, attn, vpos, grid, n_frames, seq_len, letter_ids, gt_idx = prepare_clip(
                    backbone, model, processor, video_token_id, info, path, data_type,
                    has_bound, rec, args, device, dtype, min_pixels)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue
            M = vpos.numel()

            try:
                out = plan(model, base, pos, attn, vpos, band,
                          k_frac=args.k_frac, estimator=args.estimator)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append({"task": task, "video": video_name, "reason": reason})
                tqdm.write(f"skip [{task}] {video_name}: {reason}")
                continue

            Lstar = out["L_star"]
            lstar_hist[task].append(Lstar)
            lstar_tail[task].append(out["tail_indices"][Lstar])

            # full-model reference (no pruning)
            set_anchor_prune(tm, None, None)
            t0 = time.perf_counter()
            lp_full = score_letters(model, base, pos, attn, letter_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_full.append((time.perf_counter() - t0) * 1000.0)
            hit_full = int(lp_full.argmax().item() == gt_idx)
            full_correct[task] += hit_full

            rec_out = {"task": task, "video": rec.get("video"), "gt": gt_idx, "M": M,
                      "n_frames": n_frames, "L_star": Lstar, "full": hit_full,
                      "correct": {s: {} for s in STRATEGIES}}

            for rho in rhos:
                keep_abs_anchor, _, K = keep_indices_from_plan(
                    out, vpos, grid, rho, seq_len,
                    sigma_s=args.sigma_s, sigma_tau=args.sigma_tau, window=args.window,
                    lambda0=args.lambda0, beta=args.beta, proj_dim=args.proj_dim)
                keep_rand_local = torch.randperm(M, generator=rng)[:K].sort().values.to(device)
                keep_abs_rand = full_sequence_keep_idx(seq_len, vpos, keep_rand_local)
                keep_unif_local = select_uniform_exact(M, n_frames, K, device)
                keep_abs_unif = full_sequence_keep_idx(seq_len, vpos, keep_unif_local)

                plans = {"random_prune": keep_abs_rand, "uniform_prune": keep_abs_unif,
                        "anchor_prune": keep_abs_anchor}
                for s in STRATEGIES:
                    set_anchor_prune(tm, Lstar, plans[s])
                    t0 = time.perf_counter()
                    lp = score_letters(model, base, pos, attn, letter_ids)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if s == "anchor_prune":
                        latency_pruned[rho].append((time.perf_counter() - t0) * 1000.0)
                    hit = int(lp.argmax().item() == gt_idx)
                    correct[s][task][rho] += hit
                    rec_out["correct"][s][str(rho)] = hit
            set_anchor_prune(tm, None, None)                 # leave the model clean between clips

            per_sample.append(rec_out)
            seen[task] += 1
            n += 1
            del base, out
            torch.cuda.empty_cache()
            if n % 25 == 0:
                r = rhos[-1]
                msg = " | ".join(f"{s} {sum(correct[s][t][r] for t in tasks)/n:.3f}" for s in STRATEGIES)
                print(f"[{n}] rho={r}: {msg}  (full {sum(full_correct[t] for t in tasks)/n:.3f})")

    if n == 0:
        print("no usable samples -- check --data_root layout (json/ and video/).")
        return

    valid = [t for t in tasks if seen[t]]

    def per_task_mean(counts):
        return float(np.mean([counts[t] / seen[t] for t in valid]))

    def micro(counts):
        return sum(counts[t] for t in valid) / n

    full_mean, full_micro = per_task_mean(full_correct), micro(full_correct)
    all_lstar = [l for t in valid for l in lstar_hist[t]]
    all_tail = [g for t in valid for g in lstar_tail[t] if g == g]     # drop NaN
    lstar_counts = {int(l): all_lstar.count(l) for l in sorted(set(all_lstar))}
    tail_at_anchor_mean = float(np.mean(all_tail)) if all_tail else float("nan")

    if backbone == "llava_video":
        sampling, reported_num_frames, reported_fps = "official_frames", args.num_frames, None
    else:
        sampling = "official" if args.official_sampling else "fps"
        reported_num_frames = args.num_segments if args.official_sampling else None
        reported_fps = None if args.official_sampling else args.fps

    out_json = {"experiment": "mvbench_anchor_layer_prune", "backbone": backbone,
               "model_name": args.model_name, "data_root": args.data_root,
               "sampling": sampling, "num_frames": reported_num_frames, "fps": reported_fps,
               "max_frames": args.max_frames, "max_pixels": args.max_pixels,
               "min_pixels": min_pixels, "rhos": rhos, "band": band,
               "k_frac": args.k_frac, "estimator": args.estimator,
               "sigma_s": args.sigma_s, "sigma_tau": args.sigma_tau, "window": args.window,
               "lambda0": args.lambda0, "beta": args.beta, "proj_dim": args.proj_dim,
               "strategies": list(STRATEGIES), "tasks": valid, "n": n,
               "full_accuracy_mean": full_mean, "full_accuracy_micro": full_micro,
               "per_task_seen": {t: seen[t] for t in valid},
               "full_per_task": {t: full_correct[t] / seen[t] for t in valid},
               "anchor_layer_histogram": lstar_counts,
               "anchor_layer_mean": float(np.mean(all_lstar)) if all_lstar else float("nan"),
               "tail_index_at_anchor_mean": tail_at_anchor_mean,
               "latency_ms_full_mean": float(np.mean(latency_full)) if latency_full else float("nan"),
               "latency_ms_pruned_mean": {str(r): (float(np.mean(latency_pruned[r]))
                                                   if latency_pruned[r] else float("nan"))
                                          for r in rhos},
               "skipped": skipped,
               "table": {}}

    print(f"\n==== Anchor-layer prune on MVBench ({n} samples, {len(valid)} task(s)) ====")
    if skipped:
        def _category(reason):
            if reason == "missing file":
                return "missing file"
            if reason.startswith("too few video tokens"):
                return "too few video tokens"
            return "decode/processing error"
        by_cat = Counter(_category(s["reason"]) for s in skipped)
        by_task = Counter(s["task"] for s in skipped)
        print(f"\nskipped {len(skipped)} clip(s):")
        for cat, cnt in by_cat.most_common():
            print(f"  {cat:<24}{cnt:>5}")
        print("  by task: " + ", ".join(f"{t} ({c})" for t, c in by_task.most_common()))
        print(f"  {'task':<22}{'video':<18}reason")
        for s in skipped:
            print(f"  {s['task']:<22}{str(s['video']):<18}{s['reason']}")
    print(f"\nfull-model accuracy: mean(per-task) {full_mean:.4f}  micro {full_micro:.4f}")
    print(f"anchor layer L* histogram: {lstar_counts}  (mean {out_json['anchor_layer_mean']:.2f})")
    print(f"tail index at the chosen anchor: mean gamma = {tail_at_anchor_mean:.3f}")
    if device.type == "cuda" and latency_full:
        print(f"mean forward latency: full {np.mean(latency_full):.1f} ms  |  "
             + "  ".join(f"anchor_prune@{r} {np.mean(latency_pruned[r]):.1f} ms" for r in rhos
                         if latency_pruned[r]))
    print()
    hdr = f"{'strategy':<16}{'rho':>6}{'acc_mean':>10}{'acc_micro':>11}"
    print(hdr); print("-" * len(hdr))
    for s in STRATEGIES:
        out_json["table"][s] = {}
        for rho in rhos:
            counts = {t: correct[s][t][rho] for t in tasks}
            acc_mean = per_task_mean(counts)
            acc_micro = micro(counts)
            out_json["table"][s][str(rho)] = {
                "accuracy_mean": acc_mean, "accuracy_micro": acc_micro,
                "per_task": {t: correct[s][t][rho] / seen[t] for t in valid}}
            print(f"{s:<16}{rho:>6.2f}{acc_mean:>10.4f}{acc_micro:>11.4f}")

    with open(args.out, "w") as f:
        json.dump(out_json, f, indent=2)
    ps_path = args.per_sample_out or os.path.splitext(args.out)[0] + "_per_sample.json"
    with open(ps_path, "w") as f:
        json.dump({"experiment": "mvbench_anchor_layer_prune_per_sample", "rhos": rhos,
                  "strategies": list(STRATEGIES), "n": n, "samples": per_sample}, f, indent=2)
    print(f"\nsaved -> {args.out}\nsaved -> {ps_path}  (per-clip hits, for paired tests)")
    print("\nread: random_prune / uniform_prune / anchor_prune all prune at the SAME anchor L*,")
    print("      so any gap between them is attributable to WHICH tokens are kept, not to a")
    print("      different amount of realized compute reduction.")
    print("      anchor_prune vs full         => cost of pruning at all, at this rho.")
    print("      anchor_prune vs random/uniform => value of the importance-diversity criterion.")
    print("      The counts above are unpaired: resolve a claimed gap with McNemar on the")
    print("      per-clip dump, not with these margins.")


def parse_args():
    p = argparse.ArgumentParser(description="Anchor-layer importance-diversity prune, benchmarked on MVBench.")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_video"], default="auto",
                   help="VLM backbone. 'auto' infers from --model_name ('onevision'/'video' "
                        "substring => llava_video, else qwen). qwen: dynamic resolution, mRoPE, "
                        "fps or --official_sampling frame count. llava_video: LLaVA-OneVision / "
                        "LLaVA-Video-7B-Qwen2, REAL multi-frame video via a fixed --num_frames "
                        "uniform sample (always the mvbench.ipynb midpoint convention via "
                        "inference.official_frames), 1D RoPE, Qwen2Model.")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, "
                        f"llava_video={LLAVA_VIDEO_MODEL_ID}.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--max_frames", type=int, default=8,
                   help="qwen only: upper cap on frames per clip (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="qwen only: frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="qwen only: use the reference mvbench.ipynb sampler instead of fps sampling "
                        "(llava_video always uses this convention, via --num_frames).")
    p.add_argument("--num_segments", type=int, default=16, help="qwen only: frames for --official_sampling.")
    p.add_argument("--num_frames", type=int, default=8,
                   help="llava_video only: uniform frame count sampled per clip (each frame costs "
                        "pooled_side^2 tokens, so this drives the token budget directly).")
    p.add_argument("--max_pixels", type=int, default=None, help="per-frame pixel ceiling.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="per-frame pixel floor. Default: mirror --max_pixels. <=0 disables.")
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK.")
    p.add_argument("--rhos", default="0.05,0.10,0.25", help="keep-ratios K/N to evaluate.")
    p.add_argument("--band", default="2,3,4,5,6,7,8", help="anchor-candidate layer band B.")
    p.add_argument("--k_frac", type=float, default=0.10, help="top-order-statistic fraction for the tail estimator.")
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--sigma_s", type=float, default=1.5, help="spatial-arm radius (patch units).")
    p.add_argument("--sigma_tau", type=float, default=1.0, help="temporal-arm decay (frame-group units).")
    p.add_argument("--window", type=int, default=2, help="temporal-arm window W (frame groups).")
    p.add_argument("--lambda0", type=float, default=1.0, help="lambda(rho) = lambda0 * rho^-beta.")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--proj_dim", type=int, default=64, help="stability-gate projection dim d'.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 for both backbones (llava_video ships bf16, unlike LLaVA-1.5's fp16).")
    p.add_argument("--seed", type=int, default=42, help="seed for the random_prune baseline.")
    p.add_argument("--out", default="results_anchor_layer_prune_mvbench.json")
    p.add_argument("--per_sample_out", default="",
                   help="per-clip correctness dump, for paired (McNemar) tests. "
                        "Default: <--out stem>_per_sample.json.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
