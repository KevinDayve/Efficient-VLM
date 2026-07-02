"""
Does the set of "important" visual tokens change when you change which answer
you score against?  (MMBench / single-image version of overlap_subset.py.)

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

Model: Qwen2.5-VL-3B-Instruct.   Benchmark: MMBench (lmms-lab/MMBench, dev split).
This is the single-image analogue of the video overlap_subset.py: the visual mask
now selects IMAGE tokens (model.config.image_token_id), samples are grouped by
MMBench's `l2-category` (the per-task axis), and the image comes from the parquet's
embedded PNG bytes rather than a decoded video clip.

Run:
    python overlap_subset_mmbench.py \
        --data_root ~/datasets/MMBench \
        --split dev \
        --max_samples 200 \
        --out results_overlap_subset_mmbench.json
"""

import torch
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
RETAIN   = 0.25      # keep the top 25% of visual tokens

SYSTEM_PROMPT = (
    "Carefully look at the image and pay attention to the objects, their "
    "attributes, spatial relations, and any text present. Based on your "
    "observations, select the best option that accurately addresses the question."
)

# model, processor, IMAGE_TOKEN_ID, DEVICE are set in __main__ before use.


def build_inputs(messages):
    """Process a pre-built chat messages list into model inputs.
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
    messages: pre-built chat messages list (single image + question)."""
    inputs = build_inputs(messages)
    visual_mask = (inputs.input_ids[0] == IMAGE_TOKEN_ID)

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
    visual_mask = (inputs.input_ids[0] == IMAGE_TOKEN_ID)
    S1, _, _ = topk_set(token_attribution(inputs, target_token_id), visual_mask)
    S2, _, _ = topk_set(token_attribution(inputs, target_token_id, noise_sigma=sigma), visual_mask)
    return len(S1 & S2) / len(S1)                        # ~1.0 = stable, signal is real


def run(items, build_prompt, get_option_token_ids):
    """
    items: iterable from the MMBench loader.
    build_prompt(item)            -> pre-built messages list (single image + question)
    get_option_token_ids(item)    -> list of first-answer-token ids for the options
    Each item must also carry item['task'] (MMBench l2-category).
    """
    by_task, rows = defaultdict(list), []
    total = len(items) if hasattr(items, "__len__") else None
    pbar = tqdm(items, total=total, desc="scoring examples", unit="ex")
    for item in pbar:
        opt_ids = get_option_token_ids(item)
        if len(opt_ids) < 2:
            continue
        try:
            r = overlap_for_sample(build_prompt(item), opt_ids)
        except Exception as e:
            tqdm.write(f"skip [{item['task']}] idx={item.get('index', '')}: {e}")
            continue
        r["task"] = item["task"]
        rows.append(r)
        by_task[item["task"]].append(r["normalized_overlap"])
        running = sum(r["normalized_overlap"] for r in rows) / len(rows)
        pbar.set_postfix(done=len(rows), mean_overlap=f"{running:+.3f}")
        torch.cuda.empty_cache()

    print(f"\n{'task (l2-category)':28s}  n    norm_overlap")
    print(f"{'(low = answer-dependent)':28s}")
    for task in sorted(by_task, key=lambda t: sum(by_task[t]) / len(by_task[t])):
        v = by_task[task]
        print(f"{task:28s} {len(v):3d}    {sum(v)/len(v):+.3f}")
    overall = [r["normalized_overlap"] for r in rows]
    if overall:
        print(f"\noverall normalized overlap: {sum(overall)/len(overall):+.3f}  (n={len(overall)})")
    return rows


# ----------------------------------------------------------------------------- data


OPTION_LETTERS = ["A", "B", "C", "D"]


def _is_present(v):
    """MMBench pads absent options with NaN/None/empty string."""
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s.lower() != "nan"


def build_mmbench_prompt(image, text, max_pixels=None):
    """Single-image Qwen chat prompt whose image item drives process_vision_info.
    `image` is a PIL.Image; qwen_vl_utils.fetch_image accepts PIL objects directly."""
    img = {"type": "image", "image": image}
    if max_pixels is not None:
        img["max_pixels"] = max_pixels
    return [{"role": "user", "content": [img, {"type": "text", "text": text}]}]


def build_prompt_text(record):
    """MMBench option block + the letters present, and the ground-truth index.
    Mirrors accuracy_mvbench.build_prompt but for image MC with an optional hint."""
    letters, texts = [], []
    for L in OPTION_LETTERS:
        if _is_present(record.get(L)):
            letters.append(L)
            texts.append(str(record[L]).strip())
    lines = SYSTEM_PROMPT + "\n"
    if _is_present(record.get("hint")):
        lines += f"Hint: {str(record['hint']).strip()}\n"
    lines += f"Question: {str(record['question']).strip()}\nOptions:\n"
    lines += "".join(f"({L}) {t}\n" for L, t in zip(letters, texts))
    lines += "Answer with the option's letter (A, B, C, ...) directly."
    gt = record.get("answer")
    gt_idx = letters.index(gt) if gt in letters else -1
    return lines, letters, gt_idx


if __name__ == "__main__":
    import argparse
    import glob
    import io
    import json
    import os

    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    p = argparse.ArgumentParser(description="Per-option visual-token overlap on MMBench (single image).")
    p.add_argument("--data_root", required=True,
                   help="MMBench dir holding data/ with <split>-*.parquet (lmms-lab/MMBench).")
    p.add_argument("--split", default="dev", choices=["dev", "test"],
                   help="MMBench split. Use 'dev' (test answers are withheld).")
    p.add_argument("--l2_categories", nargs="+", default=["all"],
                   help="MMBench l2-category names to keep, or 'all'.")
    p.add_argument("--model_name", default=MODEL_ID)
    p.add_argument("--retain", type=float, default=RETAIN,
                   help="Top-k retention fraction (default 0.25).")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total samples after category filtering (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="results_overlap_subset_mmbench.json",
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
    IMAGE_TOKEN_ID = model.config.image_token_id
    DEVICE = model.device

    # Load the MMBench parquet for the chosen split (HF stores the image as a
    # struct {bytes, path}; we decode the PNG bytes to a PIL image).
    data_dir = os.path.join(os.path.expanduser(args.data_root), "data")
    matches = sorted(glob.glob(os.path.join(data_dir, f"{args.split}-*.parquet")))
    if not matches:
        raise FileNotFoundError(f"no {args.split}-*.parquet under {data_dir}")
    df = pd.read_parquet(matches[0])

    if args.l2_categories != ["all"]:
        keep = set(args.l2_categories)
        df = df[df["l2-category"].isin(keep)]
    if args.max_samples:
        df = df.iloc[:args.max_samples]

    def _decode_image(cell):
        # HF Image feature in parquet -> dict with 'bytes'; be tolerant of raw bytes.
        b = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(b)).convert("RGB")

    items = []
    for _, rec in tqdm(df.iterrows(), total=len(df), desc=f"Loading MMBench/{args.split}"):
        text, letters, gt_idx = build_prompt_text(rec)
        if len(letters) < 2:
            continue
        items.append({
            "index": rec.get("index"),
            "task": rec.get("l2-category", "unknown"),
            "image": _decode_image(rec["image"]),
            "text": text,
            "letters": letters,
            "gt_idx": gt_idx,
        })

    def build_prompt_fn(item):
        return build_mmbench_prompt(item["image"], item["text"], args.max_pixels)

    def get_option_token_ids_fn(item):
        return [processor.tokenizer.encode(L, add_special_tokens=False)[0]
                for L in item["letters"]]

    rows = run(items, build_prompt_fn, get_option_token_ids_fn)

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nsaved -> {args.out}")
