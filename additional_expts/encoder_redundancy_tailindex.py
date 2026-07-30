"""
encoder_redundancy_tailindex.py -- tail-index companion to encoder_redundancy_probe.py.

Same probe, same clips, same temporal/spatial redundancy maps. The ONLY thing added
is the question Gini can't answer: is the concentration a genuine HEAVY TAIL (a few
tokens carry disproportionate importance -> a small keep-set that transfers across
clip sizes), or just moderate global inequality?

Score fed to the estimator is importance = 1 - redundancy, exactly as in the probe's
`summarise`. The UPPER tail of importance = the rare tokens that CHANGED vs the previous
frame (temporal) / vs neighbours (spatial) -- the ones a pruner must keep. Heavier that
tail (larger gamma), the smaller and more separable the keep-set.

Why the moment estimator and not Hill (see efficient_vlm.utils.einmahlHaan): cosine
redundancy is bounded above by 1, so importance is bounded below -- but importance's own
upper tail is what we estimate, and the sign-aware moment estimator correctly returns
gamma <= 0 when that tail is light/bounded (nothing concentrated to rank on). Hill would
always report a positive, heavy tail. This is the SAME estimator pareto_budget deploys,
now pointed at the encoder temporal-cosine signal.

READ:
  gamma > 0 and rising with the Gini  => importance is heavy-tailed => small, separable
                                          keep-set => aggressive temporal prune/skip is safe,
                                          and the budget follows from gamma (pareto_budget).
  gamma ~ 0 / negative                 => despite a high mean/Gini the tail is bounded/flat;
                                          you'd be pruning the body, not a copy population.

NOT executed where written (no GPU/model). The two # VERIFY spots from the probe still
govern whether every number is real -- check them on one clip first.
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from efficient_vlm.utils import einmahlHaan            # sign-aware moment estimator (gamma)
from inference import DATA_LIST                         # task -> (json, subdir, data_type, has_bound)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
K_FRAC = 0.10                                          # top-k fraction for the tail fit

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
model.eval(); model.requires_grad_(False)
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID)
MERGE = model.config.vision_config.spatial_merge_size   # 2 for Qwen2.5-VL


# --- verbatim from encoder_redundancy_probe.py: same feature extraction + maps --- #
def grid_dims(grid_thw, merge=MERGE):
    """Post-merge (T, H, W). # VERIFY: T*H*W == number of visual tokens the LLM gets."""
    t, h, w = [int(x) for x in grid_thw]
    return t, h // merge, w // merge


def encoder_features(video_path, query, layer=-1, max_frames=32, fps=2.0, max_pixels=None):
    """Post-merge visual token features (M, d) at one encoder layer, plus (T,H,W).
    # VERIFY: confirm POST-merge tokens (M == T*H*W), not pre-merge (merge^2 larger)."""
    vid_item = {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames}
    if max_pixels is not None:
        vid_item["max_pixels"] = max_pixels
    messages = [{"role": "user", "content": [
        vid_item,
        {"type": "text", "text": query}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, vid = process_vision_info(messages)
    inputs = processor(text=[chat], videos=vid, return_tensors="pt").to(model.device)
    T, H, W = grid_dims(inputs["video_grid_thw"][0])

    pix = inputs["pixel_values_videos"]
    if layer == -1:
        out = model.model.visual(pix, grid_thw=inputs["video_grid_thw"])
        feats = out.pooler_output   # post-merge, raster order
    else:
        cap = {}
        blocks = model.model.visual.blocks
        h = blocks[layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("x", o[0] if isinstance(o, tuple) else o))
        _ = model.model.visual(pix, grid_thw=inputs["video_grid_thw"])
        h.remove()
        feats = cap["x"]            # NOTE: pre-merge granularity at an intermediate block
    return feats.float(), (T, H, W)


def redundancy_maps(feats, dims):
    """Per-token temporal (same cell, prev frame) and spatial (4-neighbour) cosine
    redundancy. Row-major (T,H,W): token index = f*(H*W) + r*W + c."""
    T, H, W = dims
    M = T * H * W
    assert feats.shape[0] == M, f"feature count {feats.shape[0]} != T*H*W {M} -- grid bookkeeping wrong"
    x = F.normalize(feats, dim=-1)
    x3 = x.view(T, H, W, -1)

    temp = torch.full((T, H, W), float("nan"), device=x.device)
    if T > 1:
        temp[1:] = (x3[1:] * x3[:-1]).sum(-1)

    sims, cnt = torch.zeros(T, H, W, device=x.device), torch.zeros(T, H, W, device=x.device)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rs, re = max(0, dr), H + min(0, dr)
        cs, ce = max(0, dc), W + min(0, dc)
        s = (x3[:, rs:re, cs:ce] * x3[:, rs - dr:re - dr, cs - dc:ce - dc]).sum(-1)
        sims[:, rs:re, cs:ce] += s
        cnt[:, rs:re, cs:ce] += 1
    spat = sims / cnt.clamp_min(1)
    return temp.flatten(), spat.flatten()


# --------- multi-frame: the FULL time axis of one spatial position ----------- #
def temporal_track(feats, dims, max_lag=None):
    """Beyond lag-1: how redundant is each spatial cell ACROSS all T frames.
    Row-major (T,H,W); per cell s the 'track' is its (T, d) time series.

    Returns:
      lags, autocos   -- mean same-cell cosine at temporal lag k=1..L, averaged
                         over cells + frame pairs (the redundancy-decay curve:
                         does redundancy persist past adjacent frames or not?)
      static_cell (S,) -- per-cell mean cosine of each frame to the cell's own
                         temporal centroid (=1 => that position never changes)
      eff_frames  (S,) -- participation ratio of the (T,d) track in [1, T]:
                         ->1 the whole time series is one mode (keep 1 frame),
                         ->T every frame is a distinct mode (no compressibility)."""
    T, H, W = dims
    S = H * W
    x = F.normalize(feats, dim=-1).view(T, S, -1)      # (T, S, d), unit rows

    L = (T - 1) if max_lag is None else min(max_lag, T - 1)
    autocos = [(x[k:] * x[:-k]).sum(-1).mean().item() for k in range(1, L + 1)]

    centroid = F.normalize(x.mean(0), dim=-1)          # (S, d) per-cell time mean
    static_cell = (x * centroid.unsqueeze(0)).sum(-1).mean(0)          # (S,)

    xc = x - x.mean(0, keepdim=True)                   # center each track over time
    t1 = xc.pow(2).sum(-1).sum(0)                       # (S,)  == tr(Gram_s)
    G = torch.einsum("tsd,usd->stu", xc, xc)           # (S, T, T) per-cell Gram
    t2 = G.pow(2).sum(dim=(1, 2))                       # (S,)  == ||Gram_s||_F^2
    eff_frames = t1 * t1 / t2.clamp_min(1e-12)         # PR: 1/T factors cancel
    return list(range(1, L + 1)), autocos, static_cell, eff_frames


def summarise_track(feats, dims):
    lags, autocos, static_cell, eff = temporal_track(feats, dims)
    T = dims[0]
    sc = static_cell.cpu().numpy(); ef = eff.cpu().numpy()
    curve = "  ".join(f"k{k}:{c:.3f}" for k, c in zip(lags, autocos))
    print(f"  lag-decay cosine   {curve}")
    print(f"  static-to-centroid mean {sc.mean():.3f}  Gini {gini(1 - sc):.3f}  "
          f"gamma[copy] {tail_gamma(sc):+.3f}   (upper tail near 1 = globally-static cells)")
    print(f"  eff frames / T     mean {ef.mean():.2f} / {T}  "
          f"=> ~{T / max(ef.mean(), 1e-6):.1f}x temporal compression headroom  "
          f"gamma[keep_t] {tail_gamma(ef):+.3f}   (upper tail = few dynamic positions)")
    return sc, ef / T                                   # normalized eff for pooling


# --------------------------- tail-index reporting ---------------------------- #
def gini(v):
    v = np.sort(v[~np.isnan(v)])
    n = len(v)
    if n == 0 or v.min() < 0:
        v = v - min(0, v.min())
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum() + 1e-12)) if n else float("nan")


def tail_gamma(imp_np):
    """gamma of the importance UPPER tail via the moment estimator. NaN if too few samples."""
    return einmahlHaan(torch.from_numpy(imp_np.astype(np.float32)), K_FRAC)


def summarise(temp, spat, dims):
    t, s = temp.cpu().numpy(), spat.cpu().numpy()
    tv, sv = t[~np.isnan(t)], s[~np.isnan(s)]

    def block(name, v):
        imp = 1 - v                                    # importance = 1 - redundancy
        g_keep = tail_gamma(imp)                        # upper tail of IMPORTANCE = keeper elite?
        g_copy = tail_gamma(v)                          # upper tail of REDUNDANCY (cos->1) = copy elite?
        print(f"  {name:9s} redundancy mean {v.mean():.3f}  "
              f"importance Gini {gini(imp):.3f}  "
              f"gamma[keep] {g_keep:+.3f}  gamma[copy] {g_copy:+.3f}")

    print(f"  dims T,H,W = {dims}   tokens = {dims[0]*dims[1]*dims[2]}   k_frac = {K_FRAC}")
    block("temporal", tv)
    block("spatial", sv)
    print("  READ: gamma[keep]>0 => separable must-keep elite (top-k selection ok); "
          "gamma[copy]>0 => separable near-copy population (drop/merge-the-redundant ok); "
          "both <=0 => bounded/flat, only soft recoverable-skip is safe.")


def run(clips, query="Describe what happens in the video.", layer=-1,
        max_frames=32, fps=2.0, max_pixels=None):
    all_t, all_s = [], []
    all_static, all_effn = [], []
    for i, path in enumerate(clips):
        feats, dims = encoder_features(path, query, layer=layer,
                                       max_frames=max_frames, fps=fps, max_pixels=max_pixels)
        temp, spat = redundancy_maps(feats, dims)
        print(f"\n[{i}] {path}")
        summarise(temp, spat, dims)
        if dims[0] > 1:
            sc, effn = summarise_track(feats, dims)     # multi-frame, per spatial cell
            all_static.append(sc); all_effn.append(effn)
        all_t.append(temp[~torch.isnan(temp)]); all_s.append(spat[~torch.isnan(spat)])
        torch.cuda.empty_cache()

    print("\n=== POOLED ===")
    T = torch.cat(all_t).cpu().numpy(); S = torch.cat(all_s).cpu().numpy()
    for name, v in (("temporal", T), ("spatial", S)):
        imp = 1 - v
        print(f"  {name:9s} redundancy mean {v.mean():.3f}  "
              f"importance Gini {gini(imp):.3f}  "
              f"gamma[keep] {tail_gamma(imp):+.3f}  gamma[copy] {tail_gamma(v):+.3f}")
    if all_static:
        sc = np.concatenate(all_static); effn = np.concatenate(all_effn)
        print(f"  multiframe static-to-centroid mean {sc.mean():.3f}  Gini {gini(1 - sc):.3f}  "
              f"gamma[copy] {tail_gamma(sc):+.3f}")
        print(f"  multiframe eff-frames/T       mean {effn.mean():.3f}  "
              f"gamma[keep_t] {tail_gamma(effn):+.3f}   "
              f"(low mean => big temporal-collapse headroom; gamma>0 => it's concentrated)")


def load_clips_from_task(data_root, tasks, max_clips, seed):
    """Build clip paths the same way inference.py does: <data_root>/video/<subdir>/<video>,
    reading each task's json. Samples up to max_clips clips (evenly across tasks)."""
    root = os.path.expanduser(data_root)
    json_dir, video_dir = os.path.join(root, "json"), os.path.join(root, "video")
    paths = []
    for task in tasks:
        if task not in DATA_LIST:
            raise ValueError(f"unknown task {task!r}; choices: {list(DATA_LIST)}")
        fname, subdir, _dtype, _bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            recs = json.load(fh)
        task_paths = [os.path.join(video_dir, subdir, r["video"]) for r in recs]
        # de-dup videos (EgoSchema has 1 clip per question; many Qs share a clip)
        seen, uniq = set(), []
        for p in task_paths:
            if p not in seen:
                seen.add(p); uniq.append(p)
        paths.extend(uniq)
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:max_clips] if max_clips else paths


DEFAULT_MVBENCH_CLIPS = [
        "/home/ubuntu/Experiments/MVBench/video/FunQA_test/test/test_humor/H_T_1227_0000_0191.mp4",
        "/home/ubuntu/Experiments/MVBench/video/FunQA_test/test/test_humor/H_T_568_0000_0500.mp4",
        "/home/ubuntu/Experiments/MVBench/video/FunQA_test/test/test_creative/C_KT_14_7911_8000.mp4",
        "/home/ubuntu/Experiments/MVBench/video/Moments_in_Time_Raw/videos/validation/giving/getty-pantages-theater-marquee-for-the-30th-annual-academy-awards-honoring-video-id136343623_28.mp4",
        "/home/ubuntu/Experiments/MVBench/video/Moments_in_Time_Raw/videos/validation/raining/getty-close-up-woman-looking-out-window-at-rain-with-worried-expression-video-id587-33_6.mp4",
        "/home/ubuntu/Experiments/MVBench/video/Moments_in_Time_Raw/videos/validation/punting/yt-NQy2ohW9OaI_94.mp4",
        "/home/ubuntu/Experiments/MVBench/video/clevrer/video_validation/video_12178.mp4",
        "/home/ubuntu/Experiments/MVBench/video/clevrer/video_validation/video_14309.mp4",
        "/home/ubuntu/Experiments/MVBench/video/clevrer/video_validation/video_14661.mp4",
        "/home/ubuntu/Experiments/MVBench/video/perception/videos/video_3740.mp4",
        "/home/ubuntu/Experiments/MVBench/video/perception/videos/video_6303.mp4",
        "/home/ubuntu/Experiments/MVBench/video/perception/videos/video_4415.mp4",
        "/home/ubuntu/Experiments/MVBench/video/scene_qa/video/Top079_02905.mp4",
        "/home/ubuntu/Experiments/MVBench/video/scene_qa/video/Top019_04990.mp4",
        "/home/ubuntu/Experiments/MVBench/video/scene_qa/video/Top069_05960.mp4",
        "/home/ubuntu/Experiments/MVBench/video/ssv2_video/131887.webm",
        "/home/ubuntu/Experiments/MVBench/video/ssv2_video/175245.webm",
        "/home/ubuntu/Experiments/MVBench/video/ssv2_video/92623.webm",
        "/home/ubuntu/Experiments/MVBench/video/sta/sta_video/4BIMI.mp4",
        "/home/ubuntu/Experiments/MVBench/video/sta/sta_video/TU9K1.mp4",
        "/home/ubuntu/Experiments/MVBench/video/sta/sta_video/DLOS7.mp4",
        "/home/ubuntu/Experiments/MVBench/video/star/Charades_v1_480/M3S4D.mp4",
        "/home/ubuntu/Experiments/MVBench/video/star/Charades_v1_480/UDGP2.mp4",
        "/home/ubuntu/Experiments/MVBench/video/vlnqa/stop/2141_frame54.mp4",
        "/home/ubuntu/Experiments/MVBench/video/vlnqa/forward/5644_frame30.mp4",
        "/home/ubuntu/Experiments/MVBench/video/vlnqa/left/4668_frame54.mp4",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Exp 1: encoder temporal/spatial redundancy tail indices. "
                    "Default clips = the 26 hardcoded MVBench probes; pass --data_root/--tasks "
                    "to run on another dataset (e.g. --tasks EgoSchema).")
    ap.add_argument("--data_root", default=None,
                    help="Dir holding json/ and video/. If set, clips are drawn from --tasks "
                         "instead of the hardcoded MVBench list.")
    ap.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                    help="task name(s) from inference.DATA_LIST (used only with --data_root).")
    ap.add_argument("--max_clips", type=int, default=26,
                    help="cap on number of clips (0 = all); matches the 26-clip MVBench probe.")
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max_pixels", type=int, default=200704,
                    help="per-frame pixel cap (default 448^2, matching Exp 2). Long-form clips "
                         "at full res OOM; set 0 to disable the cap.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.data_root:
        clips = load_clips_from_task(args.data_root, args.tasks, args.max_clips, args.seed)
        print(f"loaded {len(clips)} clip(s) from tasks {args.tasks} under {args.data_root}")
    else:
        clips = DEFAULT_MVBENCH_CLIPS
        print(f"using {len(clips)} hardcoded MVBench probe clips")

    run(clips, max_frames=args.max_frames, fps=args.fps,
        max_pixels=(args.max_pixels or None))
