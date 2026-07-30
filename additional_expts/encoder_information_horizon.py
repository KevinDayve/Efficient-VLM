"""
encoder_information_horizon.py -- vision-ENCODER-side "information horizon" check.
=============================================================================================
vision_encoder_tail_index.py asks WHICH encoder layer has the heaviest-tailed
(most concentrated) importance distribution. This asks the complementary
question directly: how DEEP does the vision encoder actually need to run before
the merged features it hands to the LLM stop mattering? I.e. drop (stop updating)
every vision token after some layer c -- literally freeze the residual stream
there and feed THAT straight to the patch merger -- and see how much the
decoder's final output changes relative to the full 32-layer encoder.

Unlike the anchor-layer token-BUDGET prune (anchor_layer_prune.py: keep K
important tokens, drop the rest), this drops NO tokens by count -- every visual
token survives -- it just stops ALL of them from being refined further, uniformly,
past layer c. If cutting early doesn't hurt output, that IS the information
horizon: whatever the deeper layers were doing had already propagated into the
per-token features (or wasn't needed) by layer c.

Implementation: `install_early_exit` patches every `Qwen2_5_VLVisionBlock`'s
forward (instance-level, not the class) to become a no-op identity once its own
index exceeds `visual.exit_layer`. This deliberately does NOT touch
`Qwen2_5_VisionTransformerPretrainedModel.forward` itself (position ids,
window-reorder, cu_seqlens, the merger call) at all -- unlike the encoder
tail-index script's attention capture, which HAD to reach into version-specific
internals, truncating at the per-block level is version-proof: a block that
returns its input unchanged is a valid no-op forward under any transformers
version's surrounding plumbing.

Usage:
    python encoder_information_horizon.py --video /path/to/clip.mp4 \
        --query "What is happening in the video?" --exit_layers 3,7,11,15,19,23,27,31
"""
from __future__ import annotations

import os
import sys
import types
import argparse

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anchor_layer_prune import QWEN_MODEL_ID, build_messages
from vision_encoder_tail_index import _visual_model


# --------------------------------------------------------------------------- #
# Early-exit patch: every block past `visual.exit_layer` becomes a no-op.
# --------------------------------------------------------------------------- #
def _maybe_skip_forward(self, hidden_states, *args, **kwargs):
    exit_layer = self._exit_owner.exit_layer
    if exit_layer is not None and self._exit_idx > exit_layer:
        return hidden_states                     # frozen: this token stream stops changing here
    return self._orig_forward(hidden_states, *args, **kwargs)


def install_early_exit(visual):
    """Patch every block's forward (idempotent)."""
    if not hasattr(visual, "exit_layer"):
        visual.exit_layer = None
    for i, blk in enumerate(visual.blocks):
        blk._exit_idx = i
        blk._exit_owner = visual
        if getattr(blk, "_orig_forward", None) is None:
            blk._orig_forward = blk.forward
            blk.forward = types.MethodType(_maybe_skip_forward, blk)


def uninstall_early_exit(visual):
    for blk in visual.blocks:
        if getattr(blk, "_orig_forward", None) is not None:
            blk.forward = blk._orig_forward
            blk._orig_forward = None
    visual.exit_layer = None


def set_exit_layer(visual, exit_layer):
    """exit_layer=None runs the full encoder depth; otherwise blocks 0..exit_layer
    run normally and every later block is a no-op (0-indexed, inclusive)."""
    visual.exit_layer = exit_layer


# --------------------------------------------------------------------------- #
# Input building (mirrors anchor_layer_prune.run_demo_qwen's Qwen2.5-VL path)
# --------------------------------------------------------------------------- #
def load_qwen_inputs(model, processor, args, device):
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
    position_ids = build_full_positions(model, input_ids, video_idx, grid_thw, attn_mask)
    return input_ids, attn_mask, video_idx, grid_thw, pix, position_ids


@torch.no_grad()
def run_at_exit_layer(model, input_ids, attn_mask, video_idx, grid_thw, pix, position_ids):
    """One full forward (video encoder at whatever `visual.exit_layer` is
    currently set to -> merge -> dense decoder) -> (merged_features, logits)."""
    ve = model.get_video_features(pix, grid_thw).pooler_output
    ve = torch.cat(ve, dim=0).to(input_ids.device)
    base_embeds = model.get_input_embeddings()(input_ids).clone()
    base_embeds[0, video_idx] = ve.to(base_embeds.dtype)
    out = model(inputs_embeds=base_embeds, position_ids=position_ids,
               attention_mask=attn_mask, use_cache=False)
    return ve, out.logits[0, -1]


def main(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)

    visual = _visual_model(model)
    depth = len(visual.blocks)
    install_early_exit(visual)

    input_ids, attn_mask, video_idx, grid_thw, pix, position_ids = load_qwen_inputs(
        model, processor, args, device)

    exit_layers = [int(x) for x in args.exit_layers.split(",")] if args.exit_layers else list(range(depth))
    for e in exit_layers:
        if not (0 <= e < depth):
            raise SystemExit(f"--exit_layers: {e} out of range [0, {depth - 1}]")

    set_exit_layer(visual, None)
    ve_full, logits_full = run_at_exit_layer(model, input_ids, attn_mask, video_idx, grid_thw, pix, position_ids)
    top_full = processor.tokenizer.decode(logits_full.argmax())

    print(f"visual tokens M={video_idx.numel()}   encoder depth={depth}\n")
    print(f"{'exit_layer':>10}{'feat_cos_vs_full':>18}{'next_token':>16}{'agrees':>8}")
    print("-" * 52)
    print(f"{'full':>10}{1.0:>18.4f}{top_full!r:>16}{'--':>8}")

    cosines = []
    for e in exit_layers:
        set_exit_layer(visual, e)
        ve_e, logits_e = run_at_exit_layer(model, input_ids, attn_mask, video_idx, grid_thw, pix, position_ids)
        cos = F.cosine_similarity(ve_e.float(), ve_full.float(), dim=-1).mean().item()
        cosines.append(cos)
        top_e = processor.tokenizer.decode(logits_e.argmax())
        agree = int(logits_e.argmax() == logits_full.argmax())
        print(f"{e:>10}{cos:>18.4f}{top_e!r:>16}{('yes' if agree else 'NO'):>8}")

    set_exit_layer(visual, None)                # leave the model clean
    uninstall_early_exit(visual)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(exit_layers, cosines, marker="o")
        plt.axhline(1.0, color="gray", linestyle=":")
        plt.xlabel("encoder exit layer c (blocks 0..c run, rest frozen)")
        plt.ylabel("mean cosine(merged features, full-depth)")
        plt.title(f"{os.path.basename(args.model_name)}: encoder information horizon")
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Vision-encoder depth-truncation ('information horizon') check.")
    p.add_argument("--video", required=True, help="path to a video file.")
    p.add_argument("--query", default="Describe what is happening in the video.")
    p.add_argument("--model_name", default=QWEN_MODEL_ID)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--exit_layers", default=None,
                   help="comma-separated encoder cut layers to sweep (0-indexed, inclusive). "
                        "Default: every layer.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--plot", default=None, help="path to save a cosine-vs-exit-layer PNG (optional).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
