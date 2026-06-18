"""Simple gated video inference with the native Qwen2.5-VL token scorer.

Give it a video and a prompt; it generates an answer with the trained scorer
dropping low-scoring video tokens before the LLM (paper Eq. 14). The gating lives
in the patched transformers ``modeling_qwen2_5_vl.py`` (``model.load_token_scorer``
/ ``disable_token_gating``).

Decoding is a plain greedy loop with ``use_cache=False``: each step is a fresh
single forward, which is the path the gating supports (it shortens the sequence on
every forward, so there is no KV-cache/attention-mask length mismatch to manage).
That makes it correct and simple, at the cost of re-running the forward per token
-- keep ``--max_new_tokens`` modest.

Example:
    python inference.py \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --scorer_ckpt checkpoints/scorer_best.pt \
        --video ~/NeXTVideo/NExTVideo/0000/2440175990.mp4 \
        --prompt "Describe what happens in the video." \
        --keep_ratio 0.5
"""

from __future__ import annotations

import argparse

import torch
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def build_inputs(processor, model, video, prompt, max_frames, max_pixels):
    video_item = {"type": "video", "video": video, "nframes": max_frames}
    if max_pixels is not None:
        video_item["max_pixels"] = max_pixels
    messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt")
    return inputs.to(model.device)


def token_stats(model, inputs, keep_ratio):
    """Original vs post-gating token counts. Kept video tokens are deterministic:
    the stratified selector retains exactly max(1, round(rho * tokens)) per clip."""
    ids = inputs["input_ids"][0]
    seq_len = int(ids.numel())
    n_video = int((ids == model.config.video_token_id).sum())
    n_text = seq_len - n_video  # everything that is not a video placeholder
    merge = model.config.vision_config.spatial_merge_size**2
    kept_video = sum(max(1, round(keep_ratio * (int(row.prod()) // merge))) for row in inputs["video_grid_thw"])
    kept_video = min(kept_video, n_video)
    kept_seq = seq_len - (n_video - kept_video)
    return {"video": n_video, "text": n_text, "seq": seq_len, "kept_video": kept_video, "kept_seq": kept_seq}


@torch.no_grad()
def gated_greedy(model, processor, inputs, max_new_tokens, eos_ids):
    """Greedy decode with use_cache=False so every step is a fresh gated forward."""
    cur = dict(inputs)
    cur["use_cache"] = False
    dev = cur["input_ids"].device
    out_ids = []
    for _ in range(max_new_tokens):
        next_id = int(model(**cur).logits[0, -1].argmax())
        if next_id in eos_ids:
            break
        out_ids.append(next_id)
        # append the new (text) token to every per-token input
        cur["input_ids"] = torch.cat([cur["input_ids"], torch.tensor([[next_id]], device=dev)], dim=1)
        if cur.get("attention_mask") is not None:
            cur["attention_mask"] = torch.cat(
                [cur["attention_mask"], torch.ones((1, 1), dtype=cur["attention_mask"].dtype, device=dev)], dim=1
            )
        if cur.get("mm_token_type_ids") is not None:  # 0 == text, needed by get_rope_index
            cur["mm_token_type_ids"] = torch.cat(
                [cur["mm_token_type_ids"], torch.zeros((1, 1), dtype=cur["mm_token_type_ids"].dtype, device=dev)], dim=1
            )
    return processor.tokenizer.decode(out_ids, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description="Simple gated video inference for Qwen2.5-VL.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="Must match the scorer's training size (3B: hidden 2048, 7B: hidden 3584).")
    p.add_argument("--scorer_ckpt", default="checkpoints/scorer_best.pt")
    p.add_argument("--video", required=True)
    p.add_argument("--prompt", default="Describe what happens in the video.")
    p.add_argument("--keep_ratio", type=float, default=0.5, help="rho: fraction of video tokens to retain.")
    p.add_argument("--hidden_dim", type=int, default=256, help="Scorer hidden width (must match training).")
    p.add_argument("--k_min", type=int, default=1)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--max_pixels", type=int, default=None, help="Cap per-frame resolution (e.g. 100352) on small GPUs.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Use bf16 for Qwen2.5-VL; fp16 overflows the vision tower and produces garbage.")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)
    model.load_token_scorer(args.scorer_ckpt, keep_ratio=args.keep_ratio, hidden_dim=args.hidden_dim,
                            k_min=args.k_min, temp=args.temp, beta_max=args.beta_max)

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    inputs = build_inputs(processor, model, args.video, args.prompt, args.max_frames, args.max_pixels)

    s = token_stats(model, inputs, args.keep_ratio)
    print(f"video tokens: {s['video']} -> {s['kept_video']}  (keep_ratio={args.keep_ratio}, "
          f"dropped {s['video'] - s['kept_video']})")
    print(f"text tokens:  {s['text']}")
    print(f"seq length:   {s['seq']} -> {s['kept_seq']}\n")

    answer = gated_greedy(model, processor, inputs, args.max_new_tokens, eos_ids)
    print(f"prompt:    {args.prompt}")
    print(f"answer:    {answer.strip()}")


if __name__ == "__main__":
    main()