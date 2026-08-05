"""
lv_knockout_accuracy.py -- accuracy under text->vision attention knockout, for
Qwen2.5-VL or LLaVA-OneVision, on EgoSchema or MVBench.

This is the Language-to-Video Knockout (LV-K) probe of "An Empirical Study on How
Video-LLMs Answer Video Questions" (arXiv 2508.15360), run on OUR backbones and OUR
two benchmarks -- the paper reports LongVA, InternVideo2.5, LLaVA-OneVision and
LLaVA-Video, so the LLaVA-OneVision runs here overlap with it (same model, our
prompt/scoring protocol and our clips) while Qwen2.5-VL is new.

What is knocked out
-------------------
In a knocked-out layer, all attention from TEXT query positions to VISUAL key
positions is masked out:

    A[q, t] = 0    for q in text positions, t in visual positions

Visual tokens keep attending to each other and to text throughout; only the
text->vision direction is cut, which is what makes this a read-out probe rather
than a token-dropping ablation.

Which layers, and what that answers  (--setting)
------------------------------------------------
cumulative  (default; the paper's Global Setting 1, eq. 6, fig. 3)
    Every layer >= i is knocked out at once, sweeping the cutoff i in steps of
    --layer_step (2, as in the paper). Visual information can then only enter the
    language stream through the first i layers. i = L is the untouched baseline;
    i = 0 blocks the whole stack, so the model answers from language priors alone
    (which is also the paper's Global Setting 2 for LV-K). Reported as

        layer ratio        i / L            (x, in %)
        performance ratio  acc(i) / acc(L)  (y, in %)

    A curve that saturates at ~100% well before layer ratio 100 says the later
    layers never read the video -- the visual read-out has already finished.

window  (the paper's Fine-grained Setting, eq. 8, fig. 6)
    Exactly --window consecutive layers are knocked out and every other layer is
    left alone, stepping the window across the stack. This asks WHICH layers carry
    the read-out rather than how deep it goes, and is the setting behind the
    paper's finding that a few layers (12-16 in their models) are critical
    outliers while most layers barely matter. Reported as the absolute accuracy
    change vs. the untouched baseline, in points -- the y axis of fig. 6.

    The paper's windows are 4 layers and are drawn non-overlapping (1-4, 5-8, ...),
    which is what --window_step defaults to; set it to 1 for a true sliding window.
    Labels here are 0-indexed, so the paper's "1-4" is this script's "0-3".

How the knockout is applied
---------------------------
A forward pre-hook on every decoder layer's self_attn rewrites the `attention_mask`
kwarg for the knocked-out layers, adding -inf at the (text query, visual key)
entries of the 4D additive mask. Nothing about the model is otherwise changed, and
attention still runs through the normal SDPA path. If the model hands the layer a
None mask (SDPA's is_causal fast path), the hook materialises the causal mask
itself -- SDPA only ignores `is_causal` when an explicit mask is present, so the
causality has to be carried in the mask we substitute.

Accuracy
--------
One forward per (clip, config); no generation. The prompt ends with the
answer-forcing prefix "Best option:(" so the last position is exactly the slot the
option letter is read from, and the prediction is the argmax over the option-letter
token ids at that position. This is the standard letter-scoring protocol and is
NOT identical to free-form decoding, so absolute numbers can differ a little from
the published accuracies -- the ratios and deltas, which is all this probe is
about, are unaffected.

Frame sampling, prompt construction and the input builders are imported from
tail_vs_layer.py in this directory, so the clips and prompts are bit-identical to
the tail-index runs and the two experiments can be read side by side.

Cost
----
n_clips x (n_configs + 1) forwards: L/step + 1 configs for cumulative, L/window_step
+ 1 for window. EgoSchema (500 clips) on Qwen (28 layers, step 2) is ~7.5k forwards;
all 20 MVBench tasks at --max_samples 200 is ~57k. Use --max_samples to scale it,
and --layer_step 4 for a first look.

Run
---
    # Qwen, EgoSchema, cutoff sweep (fig. 3)
    python lv_knockout_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --num_segments 16 --max_pixels 200704 \
        --out lvk_ego_qwen.json

    # LLaVA-OneVision, MVBench, 4-layer window (fig. 6)
    python lv_knockout_accuracy.py --data_root ~/Experiments/MVBench --tasks mvbench \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --num_segments 16 \
        --max_samples 50 --setting window --window 4 --out lvk_win_mvb_llavaov.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from tail_vs_layer import (ANSWER_PREFIX, DATA_LIST, DEFAULT_SEGMENTS, LLAVA_FAMILY,
                           MVBENCH_TASKS, build_inputs, context_limit, infer_backbone,
                           iter_clips, sample_frames, text_model_of)

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")


# --------------------------------------------------------------------------- #
# 1. Model
# --------------------------------------------------------------------------- #
def load_model(args, dtype):
    """(model, processor, visual_token_id).

    SDPA, not eager: this probe never reads attention weights, it only substitutes
    the mask, and SDPA honours an explicit additive mask at a fraction of eager's
    memory. FlashAttention-2 is NOT usable here -- it takes no arbitrary mask."""
    from transformers import AutoProcessor

    if args.backbone == "llava":
        from transformers import LlavaForConditionalGeneration as ModelCls
        visual_token = "<image>"
    elif args.backbone == "llava_ov":
        from transformers import LlavaOnevisionForConditionalGeneration as ModelCls
        visual_token = "<video>"           # the clip is one placeholder, not one per frame
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        visual_token = "<|video_pad|>"

    model = ModelCls.from_pretrained(args.model_name, torch_dtype=dtype,
                                     device_map="auto", attn_implementation=args.attn)
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    return model, processor, processor.tokenizer.convert_tokens_to_ids(visual_token)


def option_token_ids(tokenizer, n_options: int):
    """Token id of each option letter AS IT CONTINUES the answer prefix.

    Tokenising "A" on its own is not the same token: LLaVA's SentencePiece
    tokenizer prepends a word boundary and returns "_A". The letter is therefore
    read off the tokenisation of ANSWER_PREFIX + letter, which is the exact
    sequence the model is being asked to continue."""
    base = tokenizer(ANSWER_PREFIX, add_special_tokens=False).input_ids
    ids = []
    for i in range(n_options):
        full = tokenizer(ANSWER_PREFIX + chr(ord("A") + i), add_special_tokens=False).input_ids
        # A clean extension of the prefix is the normal case; if the tokenizer
        # re-segments the boundary, the letter is still the final token.
        ids.append(full[len(base)] if full[:len(base)] == base else full[-1])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"option letters are not distinct tokens: {ids}")
    return ids


# --------------------------------------------------------------------------- #
# 2. The knockout
# --------------------------------------------------------------------------- #
def lv_block_mask(ids: torch.Tensor, visual_token_id: int) -> torch.Tensor:
    """(S, S) bool: True exactly where a TEXT query reads a VISUAL key.

    Visual->visual, visual->text and text->text all stay open; only the direction
    the answer is read out through is cut."""
    is_visual = ids == visual_token_id
    return (~is_visual)[:, None] & is_visual[None, :]


def attach_lv_knockout(model, ctx: dict):
    """One forward pre-hook per decoder layer's self_attn. Returns an uninstaller.

    ctx["ko_layers"] selects which layers are knocked out on the current forward,
    ctx["block"] is the (1,1,S,S) bool mask for the current clip, and ctx["mask"]
    caches the substituted mask so it is built once per forward rather than once
    per layer."""
    handles = []

    def make_hook(layer_idx):
        def pre_hook(module, args, kwargs):
            if layer_idx not in ctx["ko_layers"]:
                return None
            if "attention_mask" not in kwargs:
                raise RuntimeError("this transformers version passes attention_mask "
                                   "positionally to self_attn -- the hook cannot rewrite it.")
            if ctx["mask"] is None:
                h = kwargs.get("hidden_states")
                h = args[0] if h is None else h
                dtype, device = h.dtype, h.device
                block = ctx["block"].to(device)
                base = kwargs["attention_mask"]
                if base is None:
                    # SDPA's is_causal fast path handed us nothing; once we pass a
                    # mask, is_causal goes False, so causality must live in it.
                    S = block.shape[-1]
                    base = torch.zeros((1, 1, S, S), dtype=dtype, device=device)
                    base.masked_fill_(torch.ones(S, S, dtype=torch.bool, device=device).triu(1),
                                      torch.finfo(dtype).min)
                ctx["mask"] = base.to(dtype).masked_fill(block, torch.finfo(dtype).min)
            kwargs["attention_mask"] = ctx["mask"]
            return args, kwargs
        return pre_hook

    for i, layer in enumerate(text_model_of(model).layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True))
    return lambda: [h.remove() for h in handles]


def build_configs(args, n_layers: int):
    """[(label, knocked-out layers)]. The untouched baseline is always included.

    cumulative: cutoff i knocks out every layer >= i, so the baseline is i = L.
    window:     a block of --window layers is knocked out and nothing else, so the
                baseline is a separate 'full' config with an empty set."""
    if args.setting == "cumulative":
        cuts = sorted(set(list(range(0, n_layers + 1, args.layer_step)) + [n_layers]))
        return [(str(i), frozenset(range(i, n_layers))) for i in cuts], str(n_layers)

    starts = list(range(0, n_layers, args.window_step))
    configs = [("full", frozenset())]
    seen = set()
    for s in starts:
        layers = frozenset(range(s, min(s + args.window, n_layers)))
        if layers and layers not in seen:          # a short trailing window can repeat
            seen.add(layers)
            configs.append((f"{min(layers)}-{max(layers)}", layers))
    return configs, "full"


# --------------------------------------------------------------------------- #
# 3. One clip -> a prediction per config
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_for_configs(model, inputs, letter_ids, configs, ctx):
    """Predicted option index under each knockout config."""
    preds = {}
    for label, ko_layers in configs:
        ctx["ko_layers"] = ko_layers
        ctx["mask"] = None                      # rebuilt once on the first knocked-out layer
        logits = model(**inputs, use_cache=False).logits[0, -1]
        preds[label] = int(torch.argmax(logits[letter_ids]).item())
    return preds


def gold_index(record) -> int:
    """The answer is stored as the correct option TEXT (MVBench's own convention,
    which make_egoschema_json.py reproduces)."""
    return record["candidates"].index(record["answer"])


# --------------------------------------------------------------------------- #
# 4. Dataset sweep
# --------------------------------------------------------------------------- #
def main(args):
    if args.backbone == "auto":
        args.backbone = infer_backbone(args.model_name)
    if args.num_segments is None:
        args.num_segments = DEFAULT_SEGMENTS[args.backbone]
    if args.window_step is None:
        args.window_step = args.window          # non-overlapping, as drawn in the paper

    if args.tasks == ["all"]:
        args.tasks = list(DATA_LIST)
    elif args.tasks == ["mvbench"]:
        args.tasks = MVBENCH_TASKS
    unknown = [t for t in args.tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[model] {args.model_name}  backbone={args.backbone}  attn={args.attn}  "
          f"dtype={args.dtype}  device={device}")
    model, processor, visual_token_id = load_model(args, dtype)
    max_positions = context_limit(model)
    n_layers = len(text_model_of(model).layers)

    # Same pixel-budget handling as tail_vs_layer.py: both LLaVA towers have a fixed
    # frame size (576 tokens/frame on 1.5, 196 on OneVision), and on Qwen min_pixels
    # defaults to max_pixels so every frame costs the same number of tokens.
    if args.backbone in LLAVA_FAMILY:
        if args.max_pixels is not None or args.min_pixels is not None:
            fixed = "576" if args.backbone == "llava" else "196"
            print(f"[warn] --max_pixels/--min_pixels are ignored on {args.backbone} "
                  f"(fixed {fixed} decoder tokens/frame).")
        args.max_pixels = args.min_pixels = None
    else:
        if args.min_pixels is None:
            args.min_pixels = args.max_pixels
        if args.min_pixels is not None and args.min_pixels <= 0:
            args.min_pixels = None

    configs, base_label = build_configs(args, n_layers)
    labels = [lab for lab, _ in configs]
    print(f"[model] {n_layers} decoder layers, {max_positions} LM positions")
    if args.setting == "cumulative":
        print(f"[knockout] cumulative, cutoffs {labels} "
              f"(cutoff {n_layers} = untouched baseline)")
    else:
        print(f"[knockout] window of {args.window} layers, step {args.window_step}: "
              f"{labels[1:]}  (+ untouched baseline)")
    print(f"[knockout] {len(configs)} forwards/clip")
    print(f"[data] tasks={args.tasks}  {args.num_segments} frames/clip")

    ctx = {"ko_layers": frozenset(), "block": None, "mask": None}
    uninstall = attach_lv_knockout(model, ctx)

    letter_cache = {}
    correct = {lab: 0 for lab in labels}                    # config -> n correct
    per_task = {}                                           # task -> config -> n correct
    seen_by_task, per_sample, skipped = {}, [], []
    n_seen = 0
    try:
        for task, rec, path, data_type, bound in tqdm(list(iter_clips(args)), unit="clip"):
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": rec["video"], "reason": "missing file"})
                continue
            try:
                gold = gold_index(rec)
                n_opt = len(rec["candidates"])
                if n_opt not in letter_cache:
                    letter_cache[n_opt] = torch.tensor(
                        option_token_ids(processor.tokenizer, n_opt), device=device)
                frames = sample_frames(path, data_type, bound, args.num_segments)
                inputs = build_inputs(processor, frames, rec, args, device, dtype)

                ids = inputs["input_ids"][0]
                S = ids.numel()
                if max_positions is not None and S > max_positions:
                    raise RuntimeError(f"sequence is {S} tokens but the LM holds "
                                       f"{max_positions} -- lower --num_segments")
                block = lv_block_mask(ids, visual_token_id)
                n_visual = int((ids == visual_token_id).sum())
                if n_visual < 50:
                    raise RuntimeError(f"too few visual tokens ({n_visual} < 50)")
                ctx["block"] = block[None, None]

                preds = predict_for_configs(model, inputs, letter_cache[n_opt], configs, ctx)
            except Exception as e:
                skipped.append({"task": task, "video": rec["video"],
                                "reason": f"{type(e).__name__}: {e}"})
                tqdm.write(f"skip [{task}] {rec['video']}: {type(e).__name__}: {e}")
                continue

            n_seen += 1
            seen_by_task[task] = seen_by_task.get(task, 0) + 1
            tc = per_task.setdefault(task, {lab: 0 for lab in labels})
            for lab, p in preds.items():
                hit = int(p == gold)
                correct[lab] += hit
                tc[lab] += hit
            per_sample.append({"task": task, "question_idx": rec.get("question_idx"),
                               "video": rec["video"], "gold": gold,
                               "n_options": n_opt, "n_visual_tokens": n_visual,
                               "pred_by_config": preds})
            del inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        uninstall()

    if not n_seen:
        print("no usable clips -- check --data_root layout (json/ and video/).")
        return

    acc = {lab: correct[lab] / n_seen for lab in labels}
    base = acc[base_label]

    out = {"experiment": "lv_knockout_accuracy", "knockout": "language_to_video",
           "setting": args.setting,
           "backbone": args.backbone, "model_name": args.model_name,
           "data_root": args.data_root, "tasks": args.tasks,
           "num_segments": args.num_segments, "max_pixels": args.max_pixels,
           "min_pixels": args.min_pixels, "n_layers": n_layers,
           "configs": [{"label": lab, "layers": sorted(ko)} for lab, ko in configs],
           "baseline_config": base_label,
           "scoring": "argmax over option-letter tokens", "answer_prefix": ANSWER_PREFIX,
           "n_clips": n_seen, "baseline_accuracy": base,
           "accuracy_by_config": acc,
           "per_task_seen": seen_by_task,
           "accuracy_by_task": {t: {lab: c[lab] / seen_by_task[t] for lab in labels}
                                for t, c in sorted(per_task.items())},
           "skipped": skipped}

    if args.setting == "cumulative":
        cuts = [int(lab) for lab in labels]
        out.update({"layer_step": args.layer_step, "cutoffs": cuts,
                    "accuracy_by_cutoff": {str(i): acc[str(i)] for i in cuts},
                    "performance_ratio_by_cutoff": {
                        str(i): (acc[str(i)] / base if base > 0 else float("nan")) for i in cuts},
                    "layer_ratio_by_cutoff": {str(i): i / n_layers for i in cuts}})
    else:
        out.update({"window": args.window, "window_step": args.window_step,
                    "windows": labels[1:],
                    "accuracy_change_by_window": {lab: acc[lab] - base for lab in labels[1:]}})

    print(f"\n==== LV-K accuracy, {args.setting} setting ({n_seen} clips) ====")
    print(f"{args.backbone}: {args.model_name}, {args.num_segments} frames, "
          f"{len(seen_by_task)} task(s)")
    print(f"baseline (no knockout) accuracy = {base:.4f}")
    if args.setting == "cumulative":
        for i in [int(l) for l in labels]:
            ratio = acc[str(i)] / base if base > 0 else float("nan")
            bar = "#" * int(round(40 * min(ratio, 1.2) / 1.2))
            print(f"  cutoff {i:>3} (layer ratio {100 * i / n_layers:5.1f}%): "
                  f"acc = {acc[str(i)]:.4f}  ratio = {100 * ratio:6.2f}%  {bar}")
    else:
        for lab in labels[1:]:
            d = 100 * (acc[lab] - base)
            bar = ("#" if d < 0 else "+") * min(40, int(round(abs(d) * 2)))
            print(f"  layers {lab:>7}: acc = {acc[lab]:.4f}  change = {d:+6.2f} pts  {bar}")
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s)")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        title = (f"{os.path.basename(args.model_name)} LV-K "
                 f"({n_seen} clips, {args.num_segments}f)")
        if args.setting == "cumulative":
            xs = np.array([100 * int(l) / n_layers for l in labels])
            ys = np.array([100 * acc[l] / base for l in labels])
            plt.plot(xs, ys, marker="o", ms=3, color="C0")
            plt.axhline(100, color="gray", lw=0.8, ls=":")
            plt.axhline(95, color="gray", lw=0.8, ls="--")
            plt.xlabel("layer ratio (%)")
            plt.ylabel("performance ratio (%)")
        else:
            wins = labels[1:]
            ys = np.array([100 * (acc[l] - base) for l in wins])
            plt.plot(np.arange(len(wins)), ys, marker="o", ms=4, color="C2")
            plt.axhline(0, color="gray", lw=0.8, ls=":")
            plt.xticks(np.arange(len(wins)), wins, rotation=45, ha="right")
            plt.xlabel("knocked-out layers")
            plt.ylabel("absolute accuracy change (pts)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Accuracy under text->vision (LV-K) attention "
                                            "knockout, on MVBench / EgoSchema.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' (all 20 MVBench tasks) / 'all' (+ EgoSchema). "
                        "MVBench and EgoSchema live under different roots -- don't mix in one run.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_ov", "llava"], default="auto",
                   help="auto infers from --model_name (onevision/-ov- -> llava_ov, else "
                        "'llava' -> llava-1.5, else qwen).")
    p.add_argument("--setting", choices=["cumulative", "window"], default="cumulative",
                   help="cumulative: knock out every layer beyond a cutoff and sweep the cutoff "
                        "(paper's Global Setting 1, fig. 3). window: knock out --window "
                        "consecutive layers only, and step the window across the stack "
                        "(paper's Fine-grained Setting, fig. 6 -- the layers-12-16 result).")
    p.add_argument("--layer_step", type=int, default=2,
                   help="cumulative only: stride of the cutoff sweep in layers (the paper uses 2). "
                        "The untouched baseline is always included.")
    p.add_argument("--window", type=int, default=4,
                   help="window only: how many consecutive layers are knocked out (paper: 4).")
    p.add_argument("--window_step", type=int, default=None,
                   help="window only: stride between windows. Default = --window, i.e. "
                        "non-overlapping (0-3, 4-7, ...) as the paper draws them; 1 slides.")
    p.add_argument("--num_segments", type=int, default=None,
                   help=f"frames sampled per clip. Default per backbone: {DEFAULT_SEGMENTS}.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="qwen only: per-frame pixel cap, e.g. 200704. Both LLaVA backbones "
                        "have a fixed frame size and ignore this.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="qwen only: floor on per-frame pixels. Defaults to --max_pixels; 0 opts out.")
    p.add_argument("--max_samples", type=int, default=None, help="cap on records PER TASK.")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa",
                   help="attention kernel. flash_attention_2 cannot take the substituted mask.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="lv_knockout_accuracy.json")
    p.add_argument("--per_sample_out", default="", help="default: <--out stem>_per_sample.json")
    p.add_argument("--plot", default=None, help="optional PNG of the curve.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
