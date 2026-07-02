"""
Per-option visual-token overlap on the easy-VQA benchmark.

Same question as overlap_subset_mmbench.py, on a different dataset: when you change
which answer you score against, does the set of "important" visual tokens change?

easy-VQA (https://github.com/vzhou842/easy-VQA) is a 13-way VQA task over 64x64
images of coloured shapes. There are no A/B/C/D letters; the fixed answer set
(get_answers(), e.g. "yes", "no", "red", "circle", ...) plays the role of the
"options". For each question we:
  - Prompt Qwen with the image + question + the allowed answer words, so the next
    token the model emits is the answer's first token.
  - For each candidate answer, run input x gradient attribution on that answer's
    FIRST token (same mechanism as the MMBench version) and keep the top-25% visual
    tokens  ->  one token set per candidate answer.
  - Measure how much these per-answer sets overlap, normalized against chance.

High normalized overlap -> the question picks the important tokens, not the answer
(World 1: an answer-blind method can find them). Low / negative -> which tokens
matter flips with the candidate answer (World 2: answer-dependent).

Samples are grouped (the `task` axis) by the category of the ground-truth answer:
shape / colour / presence(yes-no).

NOTE on candidate answers: attribution scores only the first answer token (as in the
MMBench version). Two answers that share a first token would get identical scores and
inflate overlap, so we de-duplicate candidates by first-token id and warn if any drop.
The core scoring functions are imported unchanged from overlap_subset_mmbench.py.

Run (after `pip install easy-vqa`):
    python overlap_subset_easyvqa.py \
        --split test \
        --max_samples 200 \
        --out results_overlap_subset_easyvqa.json

The output JSON matches the MMBench schema, so plot_overlap_subset_mmbench.py works:
    python plot_overlap_subset_mmbench.py results_overlap_subset_easyvqa.json
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Reuse the (model/prompt-agnostic) scoring core. Importing this module only
# *defines* functions; its heavy work is guarded by __main__.
import overlap_subset_mmbench as ov

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
RETAIN   = 0.25

SYSTEM_PROMPT = (
    "Look carefully at the image, which shows a single coloured shape on a black "
    "background. Answer the question about the shape's identity, colour, or presence."
)

# easy-VQA answer categories -> used as the per-sample `task` grouping axis.
SHAPES = {"circle", "rectangle", "triangle"}
COLORS = {"red", "green", "blue", "black", "gray", "teal", "brown", "yellow"}
YESNO  = {"yes", "no"}


def answer_category(ans):
    a = str(ans).strip().lower()
    if a in YESNO:
        return "presence"
    if a in COLORS:
        return "color"
    if a in SHAPES:
        return "shape"
    return "other"


def build_prompt_text(question, all_answers):
    """Image + question + the allowed answer words; the model's next token is the
    answer. Mirrors the MMBench prompt (system + question + constrained answer set)."""
    return (
        SYSTEM_PROMPT + "\n"
        f"Question: {str(question).strip()}\n"
        f"Answer with exactly one word from: {', '.join(all_answers)}.\n"
        "Answer:"
    )


if __name__ == "__main__":
    import argparse
    import json

    from PIL import Image
    from tqdm import tqdm
    from easy_vqa import (
        get_train_questions, get_test_questions,
        get_train_image_paths, get_test_image_paths, get_answers,
    )

    p = argparse.ArgumentParser(description="Per-option visual-token overlap on easy-VQA.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--model_name", default=MODEL_ID)
    p.add_argument("--retain", type=float, default=RETAIN,
                   help="Top-k retention fraction (default 0.25).")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total samples (debug / cost).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="results_overlap_subset_easyvqa.json")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Inject the globals the imported scoring functions read from their own module.
    ov.model = model
    ov.processor = processor
    ov.IMAGE_TOKEN_ID = model.config.image_token_id
    ov.DEVICE = model.device
    ov.RETAIN = args.retain

    # --- easy-VQA data (via the pip package) ---
    if args.split == "test":
        questions, answers, image_ids = get_test_questions()
        image_paths = get_test_image_paths()
    else:
        questions, answers, image_ids = get_train_questions()
        image_paths = get_train_image_paths()
    all_answers = get_answers()

    # Candidate answers -> first-token ids, de-duplicated (see module docstring).
    answer_tok = {}
    for a in all_answers:
        tid = processor.tokenizer.encode(a, add_special_tokens=False)[0]
        answer_tok.setdefault(tid, a)
    option_token_ids = list(answer_tok.keys())
    if len(option_token_ids) < len(all_answers):
        dropped = len(all_answers) - len(option_token_ids)
        print(f"WARNING: {dropped} answer(s) collide on their first token and were "
              f"merged; scoring {len(option_token_ids)} distinct candidates.")

    n = len(questions) if args.max_samples is None else min(args.max_samples, len(questions))
    items = []
    for i in tqdm(range(n), desc=f"Loading easy-VQA/{args.split}"):
        img = Image.open(image_paths[image_ids[i]]).convert("RGB")
        items.append({
            "index": i,
            "task": answer_category(answers[i]),
            "image": img,
            "text": build_prompt_text(questions[i], all_answers),
            "gt_answer": str(answers[i]).strip().lower(),
        })

    def build_prompt_fn(item):
        # build_mmbench_prompt is generic (single image + text).
        return ov.build_mmbench_prompt(item["image"], item["text"], args.max_pixels)

    # Every sample shares the same 13-way candidate set.
    def get_option_token_ids_fn(item):
        return option_token_ids

    rows = ov.run(items, build_prompt_fn, get_option_token_ids_fn)

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nsaved -> {args.out}")
