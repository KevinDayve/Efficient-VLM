#!/usr/bin/env python3
"""
video_token_horizon_probe.py
================================================================================
Visualize per-layer visual-token geometry across the LM decoder of a video VLLM,
decomposed into SPATIAL (within-frame) and TEMPORAL (between-frame) axes, for
MVBench and/or VideoMME clips using the OFFICIAL benchmark protocols from
inference.py and inference_videomme.py in this directory.

Signals computed per decoder layer (all NORM-CORRECTED: tokens are L2-normalized
before geometry, because transformer residual norms grow with depth and a few
"massive-activation" tokens otherwise dominate every statistic):

  * trace(Sigma)  -- mean pairwise squared distance == total spread (the "MSE")
  * participation ratio PR = tr(Sigma)^2 / tr(Sigma^2)  -- effective dimensionality
                            (eigendecomposition-free; PR -> 1 means collapse)
  * residual velocity 1 - cos(H^i, H^{i-1})  -- how much the model is still
                            updating the visual stream ("still reading the image")

Each of spread/PR is reported THREE ways:  total | within-frame (spatial) |
between-frame (temporal).  Velocity is reported as overall + frame-mean.

Optional (heavier, --drop_all): label-free "ground-truth horizon" via
D_i = JS(p_full || p_drop-all-visual-from-layer-i).

--------------------------------------------------------------------------------
Data:
    MVBench  --  same layout as inference.py: <data_root>/json/ + <data_root>/video/
    VideoMME --  same layout as inference_videomme.py: <data_root>/Video-MME.json +
                 <data_root>/data/<videoID>.mp4

Run:
    python video_token_horizon_probe.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --mvbench ~/MVBench \
        --videomme ~/Video-MME \
        --n_frames 16 \
        --max_clips 20 \
        --out horizon_signals.png
================================================================================
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

# Same-directory imports are deferred to run() so --help works without a GPU env.
_HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- #
# Geometry helpers  (all operate on a token matrix X of shape (N, d), float32)
# ----------------------------------------------------------------------------- #
def _spread_and_pr(X):
    """Eigendecomposition-free spread and participation ratio.

    tr(Sigma)   = (1/N) * sum_k ||x_k - xbar||^2          (the 'MSE' / total spread)
    tr(Sigma^2) = (1/N^2) * ||Xc Xc^T||_F^2               (via the Gram trick)
    PR          = tr(Sigma)^2 / tr(Sigma^2)               (effective dimensionality)
    """
    N = X.shape[0]
    if N < 2:
        return 0.0, 1.0
    Xc = X - X.mean(dim=0, keepdim=True)
    t1 = (Xc.pow(2).sum() / N)
    G = Xc @ Xc.t()
    t2 = (G.pow(2).sum() / (N * N))
    pr = (t1 * t1 / t2).item() if t2 > 0 else 1.0
    return t1.item(), pr


def layer_signals(hv_TSD):
    """hv_TSD: (T, S, d) visual hidden states for ONE layer, ONE clip.
    Returns a dict of scalar signals, norm-corrected (unit-L2 rows)."""
    T, S, d = hv_TSD.shape
    flat = hv_TSD.reshape(T * S, d).float()
    mean_norm = flat.norm(dim=-1).mean().item()
    flat = torch.nn.functional.normalize(flat, dim=-1)
    hv = flat.reshape(T, S, d)

    tr_tot, pr_tot = _spread_and_pr(flat)

    tr_sp, pr_sp = [], []
    for f in range(T):
        t, p = _spread_and_pr(hv[f])
        tr_sp.append(t); pr_sp.append(p)
    tr_spatial = float(np.mean(tr_sp))
    pr_spatial = float(np.mean(pr_sp))

    frame_means = torch.nn.functional.normalize(hv.mean(dim=1), dim=-1)  # (T, d)
    tr_temporal, pr_temporal = _spread_and_pr(frame_means)

    return dict(
        mean_norm=mean_norm,
        trace_total=tr_tot, pr_total=pr_tot,
        trace_spatial=tr_spatial, pr_spatial=pr_spatial,
        trace_temporal=tr_temporal, pr_temporal=pr_temporal,
        _flat_unit=flat, _frame_means_unit=frame_means,
    )


def velocity(curr, prev):
    """Residual velocity between consecutive layers (1 - cosine), on unit vectors."""
    overall = (1.0 - (curr["_flat_unit"] * prev["_flat_unit"]).sum(-1)).mean().item()
    fm_now, fm_prev = curr["_frame_means_unit"], prev["_frame_means_unit"]
    temporal = (1.0 - (fm_now * fm_prev).sum(-1)).mean().item()
    return overall, temporal


# ----------------------------------------------------------------------------- #
# Locate the visual tokens in the LM sequence and reshape to (T, S, d)
# ----------------------------------------------------------------------------- #
def locate_visual_grid(model, input_ids, video_grid_thw):
    video_token_id = model.config.video_token_id
    visual_mask = (input_ids[0] == video_token_id)
    visual_idx = visual_mask.nonzero(as_tuple=False).squeeze(-1)
    n_visual = visual_idx.numel()

    merge = model.config.vision_config.spatial_merge_size  # usually 2
    T  = int(video_grid_thw[0, 0])
    Hm = int(video_grid_thw[0, 1]) // merge
    Wm = int(video_grid_thw[0, 2]) // merge
    S  = Hm * Wm
    # >>> VERIFY: token layout. If this fires, print T/S/n_visual and adjust.
    assert T * S == n_visual, (
        f"grid mismatch: T={T} S={S} (T*S={T*S}) vs n_visual={n_visual}")
    return visual_idx, T, S


# ----------------------------------------------------------------------------- #
# Optional: label-free ground-truth horizon via drop-all JS divergence
# ----------------------------------------------------------------------------- #
def _js(p, q, eps=1e-8):
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum()
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).item()


def drop_all_js_curve(model, inputs, visual_idx, n_layers, clean_logits):
    """For each candidate layer i, zero the visual positions at the INPUT to
    decoder layer i and measure JS between the next-token distribution and the
    clean one.  Returns D_i over layers."""
    p_full = torch.softmax(clean_logits[0, -1].float(), dim=-1)
    layers = model.model.layers  # >>> VERIFY: decoder layer list path
    D = []
    for i in range(n_layers):
        def pre_hook(module, args, kwargs):
            hs = args[0].clone()
            hs[:, visual_idx, :] = 0.0
            return (hs,) + args[1:], kwargs

        handle = layers[i].register_forward_pre_hook(pre_hook, with_kwargs=True)
        with torch.no_grad():
            out = model(**inputs)
        handle.remove()
        p_drop = torch.softmax(out.logits[0, -1].float(), dim=-1)
        D.append(_js(p_full, p_drop))
    return np.array(D)


# ----------------------------------------------------------------------------- #
# Benchmark-specific input iterators (using the official protocols)
# ----------------------------------------------------------------------------- #
def iter_mvbench(args, processor, device):
    """Yields (inputs, label) for MVBench clips using the official protocol from inference.py."""
    sys.path.insert(0, _HERE)
    from qwen_vl_utils import process_vision_info
    from inference import DATA_LIST, build_prompt, make_mvbench_prompt

    json_dir = os.path.join(os.path.expanduser(args.mvbench), "json")
    video_dir = os.path.join(os.path.expanduser(args.mvbench), "video")

    tasks = list(DATA_LIST) if args.mvbench_tasks == ["all"] else args.mvbench_tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown MVBench tasks {unknown}; choices: {list(DATA_LIST)}")

    count = 0
    for task in tasks:
        if args.max_clips and count >= args.max_clips:
            return
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        for rec in records:
            if args.max_clips and count >= args.max_clips:
                return
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = build_prompt(rec)
                prompt = make_mvbench_prompt(
                    path, data_type, has_bound, rec, text,
                    args.n_frames, args.max_pixels, args.fps,
                    official=True, num_segments=args.n_frames,
                )
                chat = processor.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(prompt)
                inputs = processor(
                    text=[chat], images=img_in, videos=vid_in, return_tensors="pt"
                ).to(device)
                yield inputs, f"MVBench/{task}"
                count += 1
            except Exception as e:
                print(f"[MVBench/{task}] skip {rec.get('video')}: {e}")


def iter_videomme(args, processor, device):
    """Yields (inputs, label) for VideoMME clips using the official protocol from inference_videomme.py."""
    sys.path.insert(0, _HERE)
    from qwen_vl_utils import process_vision_info
    from inference_videomme import DURATIONS, load_questions, build_prompt, make_video_prompt

    json_path = os.path.join(os.path.expanduser(args.videomme), "Video-MME.json")
    video_dir = os.path.join(os.path.expanduser(args.videomme), "data")

    durations = list(DURATIONS) if args.vmme_durations == ["all"] else args.vmme_durations
    unknown = [d for d in durations if d not in DURATIONS]
    if unknown:
        raise ValueError(f"unknown VideoMME durations {unknown}; choices: {list(DURATIONS)}")

    questions = [q for q in load_questions(json_path) if q["duration"] in durations]

    count = 0
    for rec in questions:
        if args.max_clips and count >= args.max_clips:
            return
        try:
            path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
            text, gt_idx = build_prompt(rec)
            prompt = make_video_prompt(path, text, args.n_frames, args.max_pixels)
            chat = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(prompt)
            inputs = processor(
                text=[chat], images=img_in, videos=vid_in, return_tensors="pt"
            ).to(device)
            yield inputs, f"VideoMME/{rec['duration']}"
            count += 1
        except Exception as e:
            print(f"[VideoMME/{rec.get('duration')}] skip {rec.get('videoID')}: {e}")


# ----------------------------------------------------------------------------- #
# Main geometry loop
# ----------------------------------------------------------------------------- #
def run(args, model, processor, clips_iter, bench_name):
    """Iterate clips from clips_iter, compute geometry signals, return (per_clip, drop_curves)."""
    device = next(model.parameters()).device
    per_clip = []
    drop_curves = []

    for ci, (inputs, label) in enumerate(clips_iter):
        try:
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # tuple len = n_layers + 1 (embeds first)
            clean_logits = out.logits

            visual_idx, T, S = locate_visual_grid(
                model, inputs["input_ids"], inputs["video_grid_thw"])

            sig = defaultdict(list)
            prev = None
            for li in range(1, len(hs)):
                hv = hs[li][0, visual_idx, :].reshape(T, S, -1)
                s = layer_signals(hv)
                for k, v in s.items():
                    if not k.startswith("_"):
                        sig[k].append(v)
                vo, vt = velocity(s, prev) if prev is not None else (0.0, 0.0)
                sig["velocity"].append(vo)
                sig["velocity_temporal"].append(vt)
                prev = s
            per_clip.append({k: np.array(v) for k, v in sig.items()})

            if args.drop_all:
                D = drop_all_js_curve(model, inputs, visual_idx,
                                      len(model.model.layers), clean_logits)
                drop_curves.append(D)

            print(f"[{bench_name} {ci+1}] {label}  T={T} S={S} "
                  f"visual_tokens={T*S} layers={len(hs)-1}")
        except Exception as e:
            print(f"[{bench_name} {ci+1}] skipped: {e}")

    return per_clip, (np.stack(drop_curves) if drop_curves else None)


def stack_signal(per_clip, name):
    """Pad/truncate to a common length and stack -> (n_clips, n_layers)."""
    arrs = [c[name] for c in per_clip if name in c]
    L = min(len(a) for a in arrs)
    return np.stack([a[:L] for a in arrs])


def plot_results(results_by_bench, out_path, drop_by_bench=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("PR (effective dim): spatial", "pr_spatial"),
        ("PR (effective dim): temporal", "pr_temporal"),
        ("Spread tr(Σ): spatial", "trace_spatial"),
        ("Spread tr(Σ): temporal", "trace_temporal"),
        ("Residual velocity (overall)", "velocity"),
        ("Residual velocity (temporal / frame-means)", "velocity_temporal"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    for ax, (title, key) in zip(axes, panels):
        for bench, per_clip in results_by_bench.items():
            M = stack_signal(per_clip, key)
            x = np.arange(1, M.shape[1] + 1)
            mu, sd = M.mean(0), M.std(0)
            ax.plot(x, mu, label=bench, linewidth=2)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        ax.set_title(title); ax.set_xlabel("decoder layer"); ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")

    if drop_by_bench:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        for bench, D in drop_by_bench.items():
            x = np.arange(1, D.shape[1] + 1)
            mu, sd = D.mean(0), D.std(0)
            ax2.plot(x, mu, label=bench, linewidth=2)
            ax2.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        ax2.set_title("Ground-truth horizon: D_i = JS(full ‖ drop-all-visual@i)")
        ax2.set_xlabel("decoder layer"); ax2.set_ylabel("JS divergence")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig2.tight_layout()
        dp = out_path.replace(".png", "_dropall.png")
        fig2.savefig(dp, dpi=130); print(f"saved {dp}")


def main():
    ap = argparse.ArgumentParser(
        description="Per-layer visual-token geometry probe on MVBench / VideoMME clips.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    # MVBench
    ap.add_argument("--mvbench", default="~/MVBench",
                    help="MVBench data root holding json/ and video/ (same as inference.py --data_root).")
    ap.add_argument("--mvbench_tasks", nargs="+", default=["all"],
                    help="MVBench task names, or 'all'.")
    # VideoMME
    ap.add_argument("--videomme", default="~/Video-MME",
                    help="VideoMME data root holding Video-MME.json and data/ "
                         "(same as inference_videomme.py --data_root).")
    ap.add_argument("--vmme_durations", nargs="+", default=["all"],
                    choices=["all", "short", "medium", "long"],
                    help="VideoMME duration buckets, or 'all'.")
    # Shared sampling
    ap.add_argument("--n_frames", type=int, default=16,
                    help="Frames per clip: MVBench uses official midpoint sampling at this count; "
                         "VideoMME uses uniform midpoint sampling at this count.")
    ap.add_argument("--max_pixels", type=int, default=None,
                    help="Per-frame resolution cap (e.g. 200704). Passed to both benchmarks.")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="FPS for MVBench fps-mode fallback (only used when official_sampling=True "
                         "does not apply, i.e. frame-folder tasks already handled by official_frames).")
    ap.add_argument("--max_clips", type=int, default=0,
                    help="Max clips PER BENCHMARK (for quick runs; 0 = no cap, the default).")
    ap.add_argument("--drop_all", action="store_true",
                    help="Also compute the (heavier) label-free horizon D_i.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--out", default="horizon_signals.png")
    args = ap.parse_args()

    if not args.mvbench and not args.videomme:
        ap.error("Provide at least one of --mvbench or --videomme.")

    from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        attn_implementation="sdpa",
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model)

    results, drops = {}, {}

    if args.mvbench:
        per_clip, D = run(args, model, processor,
                          iter_mvbench(args, processor, device), "MVBench")
        if per_clip:
            results["MVBench"] = per_clip
        if D is not None:
            drops["MVBench"] = D

    if args.videomme:
        per_clip, D = run(args, model, processor,
                          iter_videomme(args, processor, device), "VideoMME")
        if per_clip:
            results["VideoMME"] = per_clip
        if D is not None:
            drops["VideoMME"] = D

    if not results:
        print("No clips processed -- check data roots and paths.")
        return

    plot_results(results, args.out, drops or None)


if __name__ == "__main__":
    main()
