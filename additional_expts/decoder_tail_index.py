"""
decoder_tail_index.py -- tail-index gamma vs. DECODER (LLM) layer (Section 5),
standalone single image/video companion to vision_encoder_tail_index.py.
=============================================================================================
Unlike the vision encoder (vision_encoder_tail_index.py), the text decoder already
exposes output_attentions cleanly, so no monkey-patching is needed at all here: the
diagnostic is already implemented end to end by anchor_layer_prune.plan(), which
runs ONE dense forward and returns tail_indices for EVERY layer in the candidate
band B, not just the chosen anchor L*. This is a thin CLI around plan() + input
building, printing/plotting the full gamma-vs-layer curve -- anchor_layer_prune.py's
own --video demo already prints this same table (see `_report_and_compare`), but
also runs the real prune and compares next-token output, which a pure diagnostic
doesn't need.

Backbone-agnostic (--backbone qwen|llava_video|auto) -- Section 4-5's decoder-side
scoring already works identically for both (see anchor_layer_prune.py's own
backbone plumbing), unlike the vision-encoder script which is Qwen2.5-VL only.

Usage:
    python decoder_tail_index.py --video /path/to/clip.mp4 --query "What is happening?"
    python decoder_tail_index.py --backbone llava_video --video /path/to/clip.mp4 --num_frames 8
"""
from __future__ import annotations

import os
import sys
import argparse

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anchor_layer_prune import (
    plan, resolve_backbone, default_model_id, build_messages, build_llava_video_prompt,
    sample_video_frames, QWEN_MODEL_ID, LLAVA_VIDEO_MODEL_ID,
)


# --------------------------------------------------------------------------- #
# Input building -- mirrors anchor_layer_prune.run_demo_qwen /
# run_demo_llava_video's own input construction, minus the real-prune /
# next-token-compare tail this script doesn't need.
# --------------------------------------------------------------------------- #
def _qwen_inputs(model, processor, args, device):
    from qwen_vl_utils import process_vision_info
    from oracle_check import build_full_positions

    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    messages = build_messages(args.video, args.query, args.max_frames, args.max_pixels, args.fps)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    video_idx = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    grid_thw = inputs["video_grid_thw"].to(device)
    pix = inputs["pixel_values_videos"].to(device)

    with torch.no_grad():
        ve = model.get_video_features(pix, grid_thw).pooler_output
        ve = torch.cat(ve, dim=0).to(device)
        base_embeds = model.get_input_embeddings()(input_ids).clone()
        base_embeds[0, video_idx] = ve.to(base_embeds.dtype)
    position_ids = build_full_positions(model, input_ids, video_idx, grid_thw, attn_mask)
    return base_embeds, position_ids, attn_mask, video_idx


def _llava_video_inputs(model, processor, args, device, dtype):
    video_token_id = model.config.video_token_id
    vcfg = model.config.vision_config
    patches_side = vcfg.image_size // vcfg.patch_size
    pooled_side = -(-patches_side // 2)              # ceil(side/2): apply_pooling's 2x downsample

    frames = sample_video_frames(args.video, args.num_frames)
    prompt = build_llava_video_prompt(processor, args.query)
    inputs = processor(text=prompt, videos=[frames], return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    pixel_values_videos = inputs["pixel_values_videos"].to(device=device, dtype=dtype)

    video_idx_full = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    T = len(frames)
    S = pooled_side * pooled_side
    if video_idx_full.numel() != T * S + 1:
        raise RuntimeError(f"got {video_idx_full.numel()} video-token positions, expected "
                          f"{T * S + 1} ({T} frames x {S} pooled tokens + 1 newline).")
    video_idx = video_idx_full[:-1]                  # drop the trailing image_newline slot

    with torch.no_grad():
        merged = model(input_ids=input_ids, attention_mask=attn_mask,
                      pixel_values_videos=pixel_values_videos,
                      use_cache=False, output_hidden_states=True)
        base_embeds = merged.hidden_states[0].detach()
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)   # plain 1D RoPE
    return base_embeds, position_ids, attn_mask, video_idx


def main(args):
    backbone = resolve_backbone(args.backbone, args.model_name)
    if not args.model_name:
        args.model_name = default_model_id(backbone)
    if args.dtype == "auto":
        args.dtype = "bf16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backbone] {backbone}  model={args.model_name}  dtype={args.dtype}")

    if backbone == "llava_video":
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
        processor = AutoProcessor.from_pretrained(args.model_name)
        model.eval(); model.requires_grad_(False)
        base_embeds, position_ids, attn_mask, video_idx = _llava_video_inputs(
            model, processor, args, device, dtype)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
        model.eval(); model.requires_grad_(False)
        base_embeds, position_ids, attn_mask, video_idx = _qwen_inputs(model, processor, args, device)

    band = [int(x) for x in args.band.split(",")]
    out = plan(model, base_embeds, position_ids, attn_mask, video_idx, band,
              k_frac=args.k_frac, estimator=args.estimator)

    gammas, Lstar = out["tail_indices"], out["L_star"]
    print(f"\nvisual tokens M={video_idx.numel()}   band={band}   "
         f"estimator={args.estimator}   k_frac={args.k_frac}\n")
    for l in band:
        g = gammas[l]
        mark = "  <- L*" if l == Lstar else ""
        print(f"  layer {l:>3}: gamma = {g:+.4f}{mark}" if g == g else f"  layer {l:>3}: gamma = NaN{mark}")
    print(f"\nheaviest-tailed decoder layer L* = {Lstar}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ys = [gammas[l] for l in band]
        plt.figure(figsize=(6, 4))
        plt.plot(band, ys, marker="o")
        plt.axvline(Lstar, color="red", linestyle="--", label=f"L*={Lstar}")
        plt.xlabel("decoder layer")
        plt.ylabel("tail index (gamma)")
        plt.title(f"{os.path.basename(args.model_name)}: decoder tail index vs. layer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Tail index gamma vs. decoder layer (Section 5).")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_video"], default="auto")
    p.add_argument("--video", required=True, help="path to a video file.")
    p.add_argument("--query", default="Describe what is happening in the video.")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, "
                        f"llava_video={LLAVA_VIDEO_MODEL_ID}.")
    p.add_argument("--max_frames", type=int, default=32, help="qwen only.")
    p.add_argument("--fps", type=float, default=2.0, help="qwen only.")
    p.add_argument("--max_pixels", type=int, default=None, help="qwen only.")
    p.add_argument("--num_frames", type=int, default=8, help="llava_video only.")
    p.add_argument("--band", default="2,3,4,5,6,7,8", help="anchor-candidate layer band B.")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--plot", default=None, help="path to save a gamma-vs-layer PNG (optional).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
