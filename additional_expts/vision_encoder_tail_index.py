"""
vision_encoder_tail_index.py -- tail-index gamma vs. VISION ENCODER layer (Section 11).

Section 11 asks the same question anchor_layer_prune.py's Section 5 asks for the
decoder -- which layer's attention has the heaviest upper tail -- but on the vision
encoder's own self-attention. Qwen2.5-VL's ViT has no CLS token, so importance is
Section 11.1's "mean-incoming" variant for SigLIP-style encoders: recv_t = mean over
queries q of A_{q,t} ("how much the rest of the image attends to this patch"),
value-norm debiased exactly like the decoder side (Section 4).

Qwen2.5-VL's vision blocks don't expose output_attentions at all (unlike the text
side): `Qwen2_5_VLVisionAttention.forward` calls the SHARED module-level
`eager_attention_forward` (also used by the text decoder's own attention) once per
`cu_seqlens` chunk (each window / full-attention frame-group is a separate call),
and only keeps the `attn_output` half of its `(attn_output, attn_weights)` return --
attn_weights are computed and immediately discarded. So this monkey-patches that
ONE shared module-level function (not the per-instance forward -- there is no
single per-instance forward call to patch that would see the whole sequence at
once) to stash `(attn_weights, value)` on whichever `Qwen2_5_VLVisionAttention`
instance made the call, filtering out the text-decoder's calls to the same
function. `install_capture` also wraps `visual.forward` itself, purely to clear
those per-instance buffers at the start of every fresh encoder pass (otherwise
chunks from consecutive clips/images would keep piling up).

Tail index is permutation-invariant over tokens, so this never needs to undo the
window-attention token reorder (`get_window_index`) or worry about the exact
concatenation order of chunks -- gamma only depends on the score DISTRIBUTION,
not which patch (or which chunk) each score came from.

Targets transformers's unified-attention-interface refactor (`ALL_ATTENTION_FUNCTIONS`
+ shared `eager_attention_forward`), which is what this repo pins
(`requirements_latest.txt`: transformers==5.10.2) and where the vision tower moved
to `model.model.visual` (see `_visual_model`). Only Qwen2.5-VL is supported
(LLaVA-OneVision's SigLIP encoder is a different attention module, out of scope
here). Requires attn_implementation='eager' (the default in this script), since
flash/sdpa never materialize attn_weights at all.

Usage:
    python vision_encoder_tail_index.py --image /path/to/frame.jpg
    python vision_encoder_tail_index.py --video /path/to/clip.mp4 --max_frames 16
"""
from __future__ import annotations

import os
import sys
import types
import argparse

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from efficient_vlm.utils import einmahlHaan, hill_tail_index
from anchor_layer_prune import QWEN_MODEL_ID


def _visual_model(model):
    """The vision tower, across the transformers layout that put it (and the
    text decoder) inside an inner `Qwen2_5_VLModel` (`model.model.visual`) and
    the older flat layout (`model.visual` directly) -- same idea as
    anchor_layer_prune._text_model for the text side."""
    base = model.model if hasattr(model, "model") and hasattr(model.model, "visual") else model
    return base.visual


# --------------------------------------------------------------------------- #
# Capture attn_weights + value from the SHARED eager_attention_forward, per
# Qwen2_5_VLVisionAttention instance, per cu_seqlens chunk.
# --------------------------------------------------------------------------- #
def _wrap_eager_attention_forward(modeling_module, VisionAttnCls):
    orig = modeling_module.eager_attention_forward
    if getattr(orig, "_is_capture_wrapper", False):
        return          # already wrapped (idempotent)

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        attn_output, attn_weights = orig(module, query, key, value, attention_mask,
                                         scaling, dropout, **kwargs)
        if isinstance(module, VisionAttnCls) and getattr(module, "_capturing", False):
            module._captured_chunks.append((attn_weights.detach(), value.detach()))
        return attn_output, attn_weights

    wrapped._is_capture_wrapper = True
    wrapped._orig = orig
    modeling_module.eager_attention_forward = wrapped


def _unwrap_eager_attention_forward(modeling_module):
    current = modeling_module.eager_attention_forward
    if getattr(current, "_is_capture_wrapper", False):
        modeling_module.eager_attention_forward = current._orig


def _reset_and_forward(self, *args, **kwargs):
    for blk in self.blocks:
        blk.attn._captured_chunks = []
        blk.attn._capturing = True
    return self._orig_forward(*args, **kwargs)


def install_capture(visual):
    """Patch (idempotent): the shared module-level eager_attention_forward
    (stashes per-chunk attn_weights/value on the calling vision-attention
    instance) and `visual.forward` itself (resets those buffers per pass)."""
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _m
    _wrap_eager_attention_forward(_m, _m.Qwen2_5_VLVisionAttention)
    if getattr(visual, "_orig_forward", None) is None:
        visual._orig_forward = visual.forward
        visual.forward = types.MethodType(_reset_and_forward, visual)


def uninstall_capture(visual):
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _m
    if getattr(visual, "_orig_forward", None) is not None:
        visual.forward = visual._orig_forward
        visual._orig_forward = None
    for blk in visual.blocks:
        blk.attn._capturing = False
    _unwrap_eager_attention_forward(_m)


# --------------------------------------------------------------------------- #
# Per-layer tail index
# --------------------------------------------------------------------------- #
def read_captured_gammas(visual, k_frac=0.10, estimator="moment"):
    """{layer_idx: gamma} from whatever `visual`'s blocks last captured --
    install_capture must already be installed and a forward already run (by
    the caller, or by anything else that invoked `visual(...)`, e.g. a
    benchmark harness's own `model.get_video_features` call). Split out from
    `encoder_tail_index_by_layer` so a caller that ALREADY runs the encoder
    forward for its own purposes (e.g. building merged embeddings) can reuse
    that exact pass instead of paying for a second one.

    Uses the mean-incoming, value-norm debiased score, pooled across every
    cu_seqlens chunk the block was called with (order doesn't matter -- see
    the module docstring). recv scaling caveat: each chunk is exactly one
    window/frame-group, so averaging a column over just its OWN chunk's
    queries (not the whole sequence) is already the correct normalization --
    unlike a single whole-sequence masked matrix, there's no cross-chunk
    dilution to worry about here."""
    fn = einmahlHaan if estimator == "moment" else hill_tail_index
    gammas = {}
    for l, blk in enumerate(visual.blocks):
        chunks = blk.attn._captured_chunks
        if not chunks:
            raise RuntimeError(f"layer {l}: no captured attention -- did install_capture "
                              "run and a forward happen before read_captured_gammas?")
        scores = []
        for attn_weights, value in chunks:
            A = attn_weights.float()                    # (1, H, Sq, Sk) -- one chunk
            v = value.float()                             # (1, H, Sk, hd)
            vnorm = v.norm(dim=-1)                          # (1, H, Sk)
            recv = A.mean(dim=2)                             # (1, H, Sk) mean over queries
            s = (recv * vnorm).mean(dim=1)                    # (1, Sk) mean over heads
            scores.append(s.reshape(-1))
        gammas[l] = fn(torch.cat(scores), k_frac)
    return gammas


@torch.no_grad()
def encoder_tail_index_by_layer(model, pixel_values, grid_thw, k_frac=0.10, estimator="moment"):
    """{layer_idx: gamma}: installs the capture, runs ONE encoder forward over
    `pixel_values`/`grid_thw`, and reads it off via `read_captured_gammas`."""
    visual = _visual_model(model)
    install_capture(visual)
    try:
        visual(pixel_values, grid_thw=grid_thw)
        return read_captured_gammas(visual, k_frac, estimator)
    finally:
        uninstall_capture(visual)


# --------------------------------------------------------------------------- #
# Input building (mirrors anchor_layer_prune.run_demo_qwen's Qwen2.5-VL path)
# --------------------------------------------------------------------------- #
def load_pixel_inputs(processor, args, device, dtype):
    from qwen_vl_utils import process_vision_info

    if args.image:
        content = {"type": "image", "image": args.image}
        if args.max_pixels:
            content["max_pixels"] = args.max_pixels
    else:
        content = {"type": "video", "video": args.video, "fps": args.fps}
        if args.max_frames:
            content["max_frames"] = args.max_frames
        if args.max_pixels:
            content["max_pixels"] = args.max_pixels

    messages = [{"role": "user", "content": [content, {"type": "text", "text": "Describe this."}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")

    if args.image:
        return (inputs["pixel_values"].to(device, dtype), inputs["image_grid_thw"].to(device))
    return (inputs["pixel_values_videos"].to(device, dtype), inputs["video_grid_thw"].to(device))


def main(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    if not args.image and not args.video:
        raise SystemExit("pass --image or --video")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)

    pixel_values, grid_thw = load_pixel_inputs(processor, args, device, dtype)
    gammas = encoder_tail_index_by_layer(model, pixel_values, grid_thw,
                                         k_frac=args.k_frac, estimator=args.estimator)

    valid = {l: g for l, g in gammas.items() if g == g}
    Estar = max(valid, key=valid.get) if valid else None
    print(f"encoder depth = {len(gammas)}   estimator = {args.estimator}   k_frac = {args.k_frac}\n")
    for l, g in gammas.items():
        mark = "  <- E*" if l == Estar else ""
        print(f"  layer {l:>2}: gamma = {g:+.4f}{mark}" if g == g else f"  layer {l:>2}: gamma = NaN{mark}")
    print(f"\nheaviest-tailed encoder layer E* = {Estar}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = sorted(gammas)
        ys = [gammas[x] for x in xs]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        if Estar is not None:
            plt.axvline(Estar, color="red", linestyle="--", label=f"E*={Estar}")
            plt.legend()
        plt.xlabel("vision encoder layer")
        plt.ylabel("tail index (gamma)")
        plt.title(f"{os.path.basename(args.model_name)}: encoder tail index vs. layer")
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Tail index gamma vs. vision-encoder layer (Qwen2.5-VL).")
    p.add_argument("--image", default=None, help="path to a single image.")
    p.add_argument("--video", default=None, help="path to a video file.")
    p.add_argument("--model_name", default=QWEN_MODEL_ID)
    p.add_argument("--max_frames", type=int, default=16, help="video only.")
    p.add_argument("--fps", type=float, default=2.0, help="video only.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--plot", default=None, help="path to save a gamma-vs-layer PNG (optional).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
