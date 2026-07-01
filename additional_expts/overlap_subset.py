"""
Does the set of "important" visual tokens change when you change which answer
you score against?

For one multiple-choice question:
  - For each candidate option, score every visual token by how much it pushes up
    the log-probability the model assigns to that option's letter.
    (score = input x gradient: the token's input embedding dotted with the
     gradient of that option's log-probability w.r.t. the embedding. This is a
     first-order estimate of how much that option's log-probability would drop
     if the token were removed.)
  - Keep the top 25% of visual tokens by that score  ->  one token set per option.
  - Measure how much these per-option sets overlap.

High overlap  ->  the important tokens are the same regardless of which answer is
                  right; the QUESTION picks them, not the answer. So a method that
                  never sees the answer can find them.            (World 1)
Low  overlap  ->  which tokens matter flips with the answer; nothing but the answer
                  tells you which set to keep.                    (World 2)

Read everything against the chance level: what two random equal-size sets would
share. With a 25% budget, chance overlap is ~0.25, NOT 0.

Model: Qwen2.5-VL-3B-Instruct.   Benchmark: MVBench or Video-MME (official protocols).

Run (MVBench):
    python overlap_subset.py \
        --benchmark mvbench \
        --data_root ~/MVBench \
        --tasks "Action Sequence" "Scene Transition" \
        --max_samples 50 \
        --out results_overlap_subset.json

Run (Video-MME):
    python overlap_subset.py \
        --benchmark videomme \
        --data_root ~/VideoMME \
        --durations short medium long \
        --max_samples 50 \
        --out results_overlap_subset_videomme.json
"""

import torch
from itertools import combinations
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
RETAIN   = 0.25      # keep the top 25% of visual tokens
N_FRAMES = 16        # match whatever your MVBench eval uses

# model, processor, VIDEO_TOKEN_ID, DEVICE are set in __main__ before use.


def build_inputs(messages):
    """Process a pre-built chat messages list (from make_mvbench_prompt) into model inputs.
    The messages list must end at the point where the model's next token is the answer letter."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(DEVICE)
    return inputs


def token_attribution(inputs, target_token_id, noise_sigma=0.0):
    """Per-visual-token importance for making the model emit `target_token_id` first.
    importance_t = sum_d  emb[t,d] * d log p(target) / d emb[t,d].   Returns one
    signed score per sequence position (visual + text); we slice out visual later.

    noise_sigma > 0 perturbs the input embeddings (used only by the stability check)."""
    captured = {}

    def pre_hook(module, args, kwargs):
        if "inputs_embeds" not in kwargs:
            raise RuntimeError(
                "inputs_embeds not in kwargs. Hook must be placed on model.model.language_model "
                "(Qwen2_5_VLTextModel), which receives the merged inputs_embeds after the "
                "vision encoder. model.model (Qwen2_5_VLModel) only receives input_ids."
            )
        emb = kwargs["inputs_embeds"].detach()
        if noise_sigma > 0:
            emb = emb + noise_sigma * emb.std() * torch.randn_like(emb)
        emb = emb.requires_grad_(True)
        captured["emb"] = emb
        kwargs["inputs_embeds"] = emb
        return args, kwargs

    # model.model.language_model is Qwen2_5_VLTextModel; it receives inputs_embeds
    # after the vision merge inside Qwen2_5_VLModel.forward.
    handle = model.model.language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        out = model(**inputs)                      # vision merge + mRoPE done internally
    finally:
        handle.remove()

    logits = out.logits[:, -1, :]                  # first answer-token logits
    logp   = torch.log_softmax(logits.float(), dim=-1)[0, target_token_id]
    grad   = torch.autograd.grad(logp, captured["emb"])[0][0]   # [seq, hidden]
    emb    = captured["emb"][0]
    return (emb.float() * grad.float()).sum(-1)    # [seq], signed


def topk_set(score, visual_mask, retain=RETAIN):
    """Top-k visual-token indices by score. k = retain * (#visual tokens)."""
    vis_idx = visual_mask.nonzero(as_tuple=True)[0]
    k = max(1, round(retain * vis_idx.numel()))
    top = score[vis_idx].topk(k).indices
    return set(vis_idx[top].tolist()), k, vis_idx.numel()


def overlap_for_sample(messages, option_token_ids):
    """option_token_ids: first-answer-token ids for each candidate option, in the same
    form your accuracy eval compares (e.g. the ids of 'A','B','C','D').
    messages: pre-built chat messages list from make_mvbench_prompt."""
    inputs = build_inputs(messages)
    visual_mask = (inputs.input_ids[0] == VIDEO_TOKEN_ID)

    sets, k, M = [], None, None
    for tid in option_token_ids:
        score = token_attribution(inputs, tid)
        S, k, M = topk_set(score, visual_mask)
        sets.append(S)

    N = len(sets)
    pair_ov = [len(a & b) / k for a, b in combinations(sets, 2)]
    mean_pairwise = sum(pair_ov) / len(pair_ov)
    all_inter = len(set.intersection(*sets)) / k
    chance = k / M                                       # two random equal-size sets
    normalized = (mean_pairwise - chance) / (1 - chance) # 1 = World 1, 0 = World 2

    return {
        "M": M, "k": k, "n_options": N,
        "mean_pairwise_overlap": mean_pairwise,
        "all_option_intersection": all_inter,
        "chance_overlap": chance,
        "normalized_overlap": normalized,
    }


def stability_under_noise(messages, target_token_id, sigma=0.02):
    """Control for 'are the sets just noisy?'. Score the SAME option twice - once
    clean, once with a tiny input perturbation - and measure self-overlap. If a single
    option's top-k set is itself unstable (low self-overlap), then low CROSS-option
    overlap is partly attribution noise, not answer-dependence. Run on a subsample."""
    inputs = build_inputs(messages)
    visual_mask = (inputs.input_ids[0] == VIDEO_TOKEN_ID)
    S1, _, _ = topk_set(token_attribution(inputs, target_token_id), visual_mask)
    S2, _, _ = topk_set(token_attribution(inputs, target_token_id, noise_sigma=sigma), visual_mask)
    return len(S1 & S2) / len(S1)                        # ~1.0 = stable, signal is real


def run(mvbench_items, build_prompt, get_option_token_ids):
    """
    mvbench_items: iterable from the MVBench loader.
    build_prompt(item)            -> pre-built messages list (from make_mvbench_prompt)
    get_option_token_ids(item)    -> list of first-answer-token ids for the options
    Each item must also carry item['task'].
    """
    by_task, rows = defaultdict(list), []
    for item in mvbench_items:
        opt_ids = get_option_token_ids(item)
        if len(opt_ids) < 2:
            continue
        try:
            r = overlap_for_sample(build_prompt(item), opt_ids)
        except Exception as e:
            print(f"skip [{item['task']}] {item.get('video_path', '')}: {e}")
            continue
        r["task"] = item["task"]
        rows.append(r)
        by_task[item["task"]].append(r["normalized_overlap"])
        torch.cuda.empty_cache()

    print(f"\n{'task':28s}  n    norm_overlap")
    print(f"{'(low = answer-dependent)':28s}")
    for task in sorted(by_task, key=lambda t: sum(by_task[t]) / len(by_task[t])):
        v = by_task[task]
        print(f"{task:28s} {len(v):3d}    {sum(v)/len(v):+.3f}")
    overall = [r["normalized_overlap"] for r in rows]
    if overall:
        print(f"\noverall normalized overlap: {sum(overall)/len(overall):+.3f}  (n={len(overall)})")
    return rows


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    from tqdm import tqdm

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # make_mvbench_prompt is the shared clip->Qwen prompt builder; Video-MME reuses it
    # with data_type="video", has_bound=False (see accuracy_videomme.py).
    from accuracy_mvbench import DATA_LIST, make_mvbench_prompt, build_prompt as _build_prompt_mvbench

    p = argparse.ArgumentParser(description="Per-option visual-token overlap on MVBench or Video-MME.")
    p.add_argument("--benchmark", default="mvbench", choices=["mvbench", "videomme"],
                   help="Which benchmark to run.")
    p.add_argument("--data_root", required=True,
                   help="MVBench: dir with json/ and video/. "
                        "Video-MME: dir holding videomme/, data/, subtitle/.")
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="MVBench task names, or 'all' (MVBench only).")
    p.add_argument("--durations", nargs="+", default=["all"],
                   help="Video-MME duration splits (short/medium/long), or 'all' (Video-MME only).")
    p.add_argument("--use_subs", action="store_true",
                   help="Inject frame-aligned .srt subtitles (Video-MME w/ subs).")
    p.add_argument("--model_name", default=MODEL_ID)
    p.add_argument("--retain", type=float, default=RETAIN,
                   help="Top-k retention fraction (default 0.25).")
    p.add_argument("--max_frames", type=int, default=N_FRAMES,
                   help="Frames per clip.")
    p.add_argument("--fps", type=float, default=2.0,
                   help="FPS for video sampling.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples per task (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="results_overlap_subset.json",
                   help="Output JSON path.")
    args = p.parse_args()

    RETAIN = args.retain

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    VIDEO_TOKEN_ID = model.config.video_token_id
    DEVICE = model.device

    # Both branches populate `items`; each item carries messages/letters/gt_idx and a
    # `task` key (the MVBench task or, for Video-MME, the duration split) that run()
    # groups by.
    items = []
    if args.benchmark == "mvbench":
        tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
        unknown = [t for t in tasks if t not in DATA_LIST]
        if unknown:
            raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

        json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
        video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

        for task in tasks:
            fname, subdir, data_type, has_bound = DATA_LIST[task]
            with open(os.path.join(json_dir, fname)) as fh:
                records = json.load(fh)
            if args.max_samples:
                records = records[:args.max_samples]
            for rec in tqdm(records, desc=f"Loading {task}"):
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = _build_prompt_mvbench(rec)
                messages = make_mvbench_prompt(
                    path, data_type, has_bound, rec, text,
                    args.max_frames, args.max_pixels, args.fps,
                )
                items.append({
                    "video_path": path,
                    "task": task,
                    "letters": letters,
                    "gt_idx": gt_idx,
                    "messages": messages,
                })
    else:  # videomme
        # Video-MME data layer (parquet load, subtitle timestamps, prompt).
        from evaluate_videomme import (
            DURATIONS, build_prompt as _build_prompt_videomme,
            load_records, frame_timestamps, subtitles_for_frames,
        )

        durations = list(DURATIONS) if args.durations == ["all"] else args.durations
        unknown = [d for d in durations if d not in DURATIONS]
        if unknown:
            raise ValueError(f"unknown durations {unknown}; choices: {list(DURATIONS)}")

        data_root = os.path.expanduser(args.data_root)
        video_dir = os.path.join(data_root, "data")
        sub_dir = os.path.join(data_root, "subtitle")
        records = load_records(data_root)

        for duration in durations:
            recs = [r for r in records if str(r["duration"]) == duration]
            if args.max_samples:
                recs = recs[:args.max_samples]
            for rec in tqdm(recs, desc=f"Loading {duration}"):
                path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
                subs = None
                if args.use_subs:
                    timestamps = frame_timestamps(path, args.max_frames)
                    subs = subtitles_for_frames(
                        os.path.join(sub_dir, f"{rec['videoID']}.srt"), timestamps)
                text, letters, gt_idx = _build_prompt_videomme(rec, subs)
                messages = make_mvbench_prompt(
                    path, "video", False, rec, text,
                    args.max_frames, args.max_pixels, args.fps,
                )
                items.append({
                    "video_path": path,
                    "task": duration,
                    "letters": letters,
                    "gt_idx": gt_idx,
                    "messages": messages,
                })

    def build_prompt_fn(item):
        return item["messages"]

    def get_option_token_ids_fn(item):
        return [processor.tokenizer.encode(L, add_special_tokens=False)[0]
                for L in item["letters"]]

    rows = run(items, build_prompt_fn, get_option_token_ids_fn)

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nsaved -> {args.out}")
