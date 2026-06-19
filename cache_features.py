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
    samples/<key>.pt       per-sample dict (vision_feats, teacher_raw, t, ...)

Sample files are named by a content hash of (video, question) rather than a running
counter, so re-running the script resumes: any sample whose .pt already exists is
skipped without touching the GPU. --limit counts toward the *total* cached, so a run
that died at 120/200 finishes the remaining 80 on the next invocation. Pass
--overwrite to ignore the existing cache and recompute everything.

Each sample also stores ``lang_feat`` (n_layers, D): the pooled layer-12..16 language
hidden states (mean over the language query rows), the language side of the
contrastive pair. The vision feature dim D equals the LLM hidden size, so it matches
the language embedding dim -- both live in the same LLM token space.
"""
import os
import json
import hashlib
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


def sample_key(meta_dict) -> str:
    """Stable content hash of (video, question) used as the cache filename, so a
    given sample maps to the same .pt across runs and can be skipped on resume."""
    raw = json.dumps([meta_dict.get("video"), meta_dict.get("question")], sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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
    meta_path = os.path.join(args.out_dir, "meta.json")
    # Guard: resuming into a cache built with different settings would silently mix
    # incompatible samples. Refuse unless the user explicitly asks to overwrite.
    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path) as fh:
            old_meta = json.load(fh)
        if old_meta != meta:
            raise SystemExit(
                f"Existing cache meta in {args.out_dir} differs from this run's settings:\n"
                f"  existing: {old_meta}\n  requested: {meta}\n"
                "Pass --overwrite to rebuild, or point --out_dir at a fresh directory."
            )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    # Resume: keep manifest entries whose .pt still exists; these count toward --limit
    # and let us skip recomputing those samples. --overwrite ignores the old cache.
    existing = {}
    if os.path.exists(manifest_path) and not args.overwrite:
        with open(manifest_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if os.path.exists(os.path.join(args.out_dir, rec["path"])):
                    existing[rec["path"]] = rec
        print(f"Resuming: {len(existing)} samples already cached in {args.out_dir}.")

    dataset = build_dataset(args)
    written, skipped = len(existing), 0
    with open(manifest_path, "w") as manifest:
        for rec in existing.values():  # replay valid prior entries, then append new ones
            manifest.write(json.dumps(rec) + "\n")
        manifest.flush()
        for i, sample in enumerate(dataset):
            if args.limit and written >= args.limit:
                break
            prompt = sample["prompt"]
            meta_d = prompt_meta(prompt)
            rel = os.path.join("samples", f"{sample_key(meta_d)}.pt")
            if rel in existing:
                continue  # already cached on a previous run
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

            # detach: get_patch_embeds runs the VLM merger outside no_grad, so the
            # features track grad. Detaching keeps the cache free of grad state
            # (non-leaf requires_grad tensors can't be pickled by DataLoader workers).
            vision_feats = patch_embeds.squeeze(0).detach().to(torch.float16).cpu()  # (n_video, D)
            teacher_raw = teacher_raw.squeeze(0).detach().float().cpu()              # (n_video,)
            lang_feat = lang_feat.squeeze(0).detach().to(torch.float16).cpu()        # (n_layers, D)
            n_video = vision_feats.shape[0]
            t = int(video_grid_thw[0][0].item())
            n_per_frame = n_video // t if t > 0 else n_video

            record = {
                "vision_feats": vision_feats,
                "teacher_raw": teacher_raw,
                "lang_feat": lang_feat,
                "t": t,
                "n_per_frame": n_per_frame,
                **meta_d,
            }
            torch.save(record, os.path.join(args.out_dir, rel))
            manifest.write(json.dumps({
                "path": rel,
                "n_video": n_video,
                "t": t,
                "n_per_frame": n_per_frame,
                **meta_d,
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
    p.add_argument("--limit", type=int, default=None, help="Cache at most this many samples total (counts already-cached samples on resume).")
    p.add_argument("--overwrite", action="store_true", help="Ignore any existing cache and recompute every sample from scratch.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    cache(parse_args())
