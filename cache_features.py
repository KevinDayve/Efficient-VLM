"""Offline feature cache for scorer training.

Runs the frozen VLM once per sample and writes, to disk, exactly the two tensors
the scorer training loop consumes online today (see train.py):

  * vision_feats : (n_video, D) merged visual tokens -- the scorer MLP input.
                   D is the LLM token width (2048 for the 3B model, 3584 for 7B),
                   NOT the pre-merger ViT width.
  * teacher_raw  : (n_video,)   the *un-normalised* layer-12..16 language->video
                   attention score -- the ListMLE ranking target. We store raw
                   (not min-max normalised) because ListMLE / Spearman / recall /
                   NDCG are all rank-based and so invariant to the monotone min-max
                   map, while the raw heavy tail is what the EVT estimator needs.

Caching decouples the expensive VLM forward (3B/7B, two passes per sample) from
the cheap scorer training (~100-500K params): once cached, you can run hundreds of
scorer epochs -- and the upcoming contrastive variants -- without touching the GPU
backbone. It is also what makes batched InfoNCE feasible (a DataLoader can form
real B>1 batches from the cache; the online loop is effectively batch-size-1).

Output layout (out_dir):
    meta.json              cache key: model, layers, max_frames, max_pixels, dtype
    manifest.jsonl         one line per sample: {path, n_video, t, n_per_frame, ...}
    samples/000000.pt      per-sample dict (vision_feats, teacher_raw, t, ...)

Each sample also stores ``lang_feat`` (n_layers, D): the pooled layer-12..16 language
hidden states (mean over the language query rows), the language side of the
contrastive pair. The vision feature dim D equals the LLM hidden size, so it matches
the language embedding dim -- both live in the same LLM token space.
"""
import os
import json
import argparse

import torch

from train import (
    get_patch_embeds,
    make_conversation,
    load_local_jsonl,
)
from efficient_vlm.attention_extractor import AttentionExtractor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


def prompt_meta(prompt) -> dict:
    """Pull the video path and question text out of a chat-format prompt, for
    traceability in the manifest (lets you map a cached sample back to its video)."""
    video_path, question = None, None
    for turn in prompt:
        if turn.get("role") != "user":
            continue
        for item in turn.get("content", []):
            if item.get("type") == "video":
                video_path = item.get("video")
            elif item.get("type") == "text" and question is None:
                question = item.get("text")
    return {"video": video_path, "question": question}


def build_dataset(args):
    if args.data_file:
        print(f"Loading local jsonl: {args.data_file}")
        ds = load_local_jsonl(args.data_file, args.video_root, args.seed,
                              max_frames=args.max_frames, max_pixels=args.max_pixels)
        print(f"Loaded {len(ds)} local samples.")
        return ds
    ds = load_dataset(args.dataset_name, split="train")
    ds = ds.map(lambda x: make_conversation(x, args.video_root,
                                            max_frames=args.max_frames, max_pixels=args.max_pixels))
    ds = ds.shuffle(seed=args.seed)
    return ds


def cache(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name, use_fast=True)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    special_ids = set(processor.tokenizer.all_special_ids)
    attn_extractor = AttentionExtractor(model, Layers=args.layers)

    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    # The cache key: any of these changing invalidates the cache (different token
    # counts / features / teacher). Training asserts the loaded meta matches.
    meta = {
        "model_name": args.model_name,
        "layers": list(args.layers),
        "max_frames": args.max_frames,
        "max_pixels": args.max_pixels,
        "dtype": args.dtype,
        "schema": 2,
        "has_lang": True,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    dataset = build_dataset(args)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    written, skipped = 0, 0
    with open(manifest_path, "w") as manifest:
        for i, sample in enumerate(dataset):
            if args.limit and written >= args.limit:
                break
            prompt = sample["prompt"]
            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(prompt)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(device)
            input_ids = inputs["input_ids"].to(device)
            video_grid_thw = inputs["video_grid_thw"].to(device)
            video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
            if video_positions.numel() == 0:
                skipped += 1
                continue

            attn_extractor.set_sample(input_ids, video_token_id, special_ids)
            patch_embeds = get_patch_embeds(
                model, pixel_values, video_grid_thw, n_video=video_positions.numel()
            ).to(device)
            with torch.no_grad():
                attentions = attn_extractor.truncated_forward(
                    capture_hidden=True,
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"].to(device),
                    pixel_values_videos=pixel_values,
                    video_grid_thw=video_grid_thw,
                    output_attentions=True,
                    use_cache=False,
                )
            teacher_raw = attn_extractor.scores_from_attentions(attentions, normalise=False)
            lang_feat = attn_extractor.lang_features()  # (1, n_layers, D) or None
            del attentions
            if teacher_raw is None or lang_feat is None:
                skipped += 1
                continue

            vision_feats = patch_embeds.squeeze(0).to(torch.float16).cpu()  # (n_video, D)
            teacher_raw = teacher_raw.squeeze(0).float().cpu()              # (n_video,)
            lang_feat = lang_feat.squeeze(0).to(torch.float16).cpu()        # (n_layers, D)
            n_video = vision_feats.shape[0]
            t = int(video_grid_thw[0][0].item())
            n_per_frame = n_video // t if t > 0 else n_video

            rel = os.path.join("samples", f"{written:06d}.pt")
            record = {
                "vision_feats": vision_feats,
                "teacher_raw": teacher_raw,
                "lang_feat": lang_feat,
                "t": t,
                "n_per_frame": n_per_frame,
                **prompt_meta(prompt),
            }
            torch.save(record, os.path.join(args.out_dir, rel))
            manifest.write(json.dumps({
                "path": rel,
                "n_video": n_video,
                "t": t,
                "n_per_frame": n_per_frame,
                **prompt_meta(prompt),
            }) + "\n")
            manifest.flush()
            written += 1
            if written % args.log_every == 0:
                print(f"cached {written} samples (skipped {skipped}) | last n_video={n_video}, t={t}")

    print(f"Done. Wrote {written} samples to {args.out_dir} (skipped {skipped}).")


def parse_args():
    p = argparse.ArgumentParser(description="Cache merged visual features + teacher attention scores for offline scorer training.")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", type=str, default="lmms-lab/NExTVideo", help="HF dataset; ignored when --data_file is set.")
    p.add_argument("--data_file", type=str, default=None, help="Local jsonl (rhymes-ai/NeXTVideo format).")
    p.add_argument("--video_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write the cache into.")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--limit", type=int, default=None, help="Cache at most this many samples (for smoke tests).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    cache(parse_args())
