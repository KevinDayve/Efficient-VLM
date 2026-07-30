"""
Encoder-level redundancy probe for Qwen2.5-VL video.

Question this answers, and nothing more:
  Is there spatial and/or temporal redundancy in the vision-encoder patch
  features that a pruning method could exploit -- and is it CONCENTRATED
  (some patches very redundant, some not) or FLAT (everything equally
  redundant)? Flat redundancy, however high, gives you nothing to rank on.

No CLS, no EOS, no text similarity. Importance = 1 - redundancy, defined
entirely on the patch features.

Two definitions, per post-merge visual token t at frame f, grid cell (r,c):
  temporal redundancy  = cosine( feat[t], feat[same cell, frame f-1] )
  
High cosine = redundant (a near-copy) = low importance.

What "concentrated" means here: if you rank patches by redundancy and the
top ones are MUCH more redundant than the median, a selector that drops the
top has something to do. We quantify that with the Gini of (1-redundancy)
and the gap between high and low deciles. Report these, not just the mean.

NOT executed where written (no GPU/model). Two spots marked # VERIFY decide
whether every number is real; check them on one clip first.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
model.eval(); model.requires_grad_(False)
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID)
MERGE = model.config.vision_config.spatial_merge_size   # 2 for Qwen2.5-VL


def grid_dims(grid_thw, merge=MERGE):
    """Post-merge (T, H, W) for one video. grid_thw rows are pre-merge patch
    counts (t, h, w); the spatial merge divides h and w by `merge`, not t.
    # VERIFY: print T*H*W and assert it equals the number of visual tokens the
    # LLM receives -- if it doesn't, the (f,r,c) bookkeeping below is wrong and
    # every redundancy number is garbage."""
    t, h, w = [int(x) for x in grid_thw]
    return t, h // merge, w // merge


def encoder_features(video_path, query, layer=-1, max_frames=32, fps=2.0):
    """Return post-merge visual token features (M, d) at one encoder layer,
    plus (T, H, W). layer=-1 is the final encoder layer (what the projector
    sees); pass an int to read an earlier block via hook.
    # VERIFY: confirm these are POST-merge tokens (M == T*H*W), not pre-merge
    # (which would be merge^2 larger). Print feats.shape[0] vs T*H*W."""
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames},
        {"type": "text", "text": query}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, vid = process_vision_info(messages)
    inputs = processor(text=[chat], videos=vid, return_tensors="pt").to(model.device)
    grid_thw = inputs["video_grid_thw"][0]
    T, H, W = grid_dims(grid_thw)

    pix = inputs["pixel_values_videos"]
    if layer == -1:
        out = model.model.visual(pix, grid_thw=inputs["video_grid_thw"])
        feats = out.pooler_output   # post-merge, un-windowed (raster) order; last_hidden_state is pre-merge and window-permuted
    else:
        cap = {}
        blocks = model.model.visual.blocks
        h = blocks[layer].register_forward_hook(lambda m, i, o: cap.__setitem__("x", o[0] if isinstance(o, tuple) else o))
        _ = model.model.visual(pix, grid_thw=inputs["video_grid_thw"])
        h.remove()
        feats = cap["x"]   # NOTE: pre-merge granularity at an intermediate block; see VERIFY
    return feats.float(), (T, H, W)


def redundancy_maps(feats, dims):
    """Per-token temporal and spatial redundancy. Assumes row-major (T,H,W)
    ordering: token index = f*(H*W) + r*W + c."""
    T, H, W = dims
    M = T * H * W
    assert feats.shape[0] == M, f"feature count {feats.shape[0]} != T*H*W {M} -- grid bookkeeping wrong"
    x = F.normalize(feats, dim=-1)
    x3 = x.view(T, H, W, -1)

    # temporal: cosine with same cell one frame back; frame 0 has no predecessor -> nan
    temp = torch.full((T, H, W), float("nan"), device=x.device)
    if T > 1:
        temp[1:] = (x3[1:] * x3[:-1]).sum(-1)

    # spatial: mean cosine with 4-neighbours in the same frame
    sims, cnt = torch.zeros(T, H, W, device=x.device), torch.zeros(T, H, W, device=x.device)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rs, re = max(0, dr), H + min(0, dr)
        cs, ce = max(0, dc), W + min(0, dc)
        s = (x3[:, rs:re, cs:ce] * x3[:, rs - dr:re - dr, cs - dc:ce - dc]).sum(-1)
        sims[:, rs:re, cs:ce] += s
        cnt[:, rs:re, cs:ce] += 1
    spat = sims / cnt.clamp_min(1)
    return temp.flatten(), spat.flatten()


def gini(v):
    v = np.sort(v[~np.isnan(v)])
    n = len(v)
    if n == 0 or v.min() < 0:
        v = v - min(0, v.min())
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum() + 1e-12)) if n else float("nan")


def summarise(temp, spat, dims):
    t, s = temp.cpu().numpy(), spat.cpu().numpy()
    tv, sv = t[~np.isnan(t)], s[~np.isnan(s)]
    def block(name, v):
        imp = 1 - v                          # importance = 1 - redundancy
        d = np.percentile(imp, [10, 50, 90])
        print(f"  {name:9s} redundancy: mean {v.mean():.3f}  "
              f"importance Gini {gini(imp):.3f}  "
              f"imp decile[10/50/90] {d[0]:.3f}/{d[1]:.3f}/{d[2]:.3f}  "
              f"high-low gap {d[2]-d[0]:.3f}")
    print(f"  dims T,H,W = {dims}   tokens = {dims[0]*dims[1]*dims[2]}")
    block("temporal", tv)
    block("spatial", sv)
    # the number that decides everything: is redundancy concentrated enough to rank on?
    print("  READ: high Gini (>~0.15) and wide high-low gap => structured, rankable redundancy.")
    print("        low Gini / narrow gap => redundancy is flat; nothing to prune on, pivot.")


def run(clips, query="Describe what happens in the video.", layer=-1):
    """clips: list of video paths. Prints per-clip and pooled redundancy structure."""
    all_t, all_s = [], []
    for i, path in enumerate(clips):
        feats, dims = encoder_features(path, query, layer=layer)
        temp, spat = redundancy_maps(feats, dims)
        print(f"\n[{i}] {path}")
        summarise(temp, spat, dims)
        all_t.append(temp[~torch.isnan(temp)]); all_s.append(spat[~torch.isnan(spat)])
        torch.cuda.empty_cache()
    print("\n=== POOLED ===")
    T = torch.cat(all_t); S = torch.cat(all_s)
    print(f"  temporal redundancy mean {T.mean():.3f}  importance Gini {gini((1-T).cpu().numpy()):.3f}")
    print(f"  spatial  redundancy mean {S.mean():.3f}  importance Gini {gini((1-S).cpu().numpy()):.3f}")


if __name__ == "__main__":
    clips = [
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
    run(clips)