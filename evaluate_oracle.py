"""Standalone evaluation of the gradient-oracle token scorer: ours vs uniform vs random.

Self-contained on top of ``train_oracle.py`` alone -- it reuses that file's data
loading (``make_prompt`` / ``options_and_answer``), its merged-token feature path
(``model.get_video_features(...).pooler_output``, i.e. exactly the features the
scorer was trained on), and its M-RoPE construction (``build_full_positions``).
No dependency on the ``experiments/`` suite or ``efficient_vlm.gating``.

For each retention ratio rho it keeps K = round(rho * n_video) video tokens with
three strategies and runs the multiple-choice answer forward on the *physically
shortened* sequence (dropped video tokens removed; survivors keep their original
M-RoPE ids -- the deployable gating path):

  * ours    -- learned scorer + stratified Pareto selection (paper section 2.4)
  * uniform -- evenly spaced video tokens (matched retention, no importance)
  * random  -- random video tokens (matched-retention chance baseline)

The full model (keep all tokens) is always run as the accuracy reference and the
LLM-forward speedup denominator. Accuracy is multiple-choice top-1: we compare the
logits of the option-letter tokens at the answer position and take the argmax,
mirroring ``train_oracle``'s ``gt_token`` log-prob.

Example:
    python evaluate_oracle.py \
        --scorer_ckpt checkpoints_oracle/oracle_scorer_best.pt \
        --hidden_dim 512 \
        --data_file /path/to/val.jsonl \
        --video_root /path/to/videos \
        --rhos 0.25 0.50 0.75 \
        --strategies ours uniform random
"""

import os
import json
import time
import random
import argparse
import warnings

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

from efficient_vlm.scorer import Scorer
from efficient_vlm.utils import select_pareto_stratified
# Reuse train_oracle's helpers verbatim so eval matches training exactly.
from train_oracle import make_prompt, options_and_answer, build_full_positions

STRATEGIES = ("ours", "uniform", "random")


# --------------------------------------------------------------------------- #
# Per-sample preparation (no_grad twin of train_oracle.compute_oracle)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prepare_sample(model, processor, rec, video_token_id, device, args):
    """Build everything a gated forward needs for one record.

    Returns a dict with the full ``input_embeds`` (video slots filled with the
    merged visual features), the full-sequence M-RoPE ``position_ids``, the
    absolute ``video_pos`` of the video tokens, the fp32 ``features`` (scorer
    input), ``n_frames`` (temporal bins for stratified selection), and the MC
    ``choices`` / ``correct_idx``. Returns ``None`` if the record is unusable.
    """
    choices, correct_idx = options_and_answer(rec)
    if choices is None:
        return None
    prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels)
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention = inputs["attention_mask"].to(device)
    video_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    if video_pos.numel() == 0:
        warnings.warn("[Warning] found zero video tokens. Skipping")
        return None
    video_grid_thw = inputs["video_grid_thw"].to(device)
    pixels = inputs["pixel_values_videos"].to(device)

    # Merged visual tokens -- the exact scorer input used in training.
    visual_feats = model.get_video_features(pixels, video_grid_thw).pooler_output
    visual_feats = torch.cat(visual_feats, dim=0).to(device)
    features = visual_feats.float()

    input_embeds = model.get_input_embeddings()(input_ids).clone()
    input_embeds[0, video_pos] = visual_feats.to(input_embeds.dtype)
    position_ids = build_full_positions(model, input_ids, video_pos, video_grid_thw, attention)

    n_frames = int(video_grid_thw[0][0].item())  # temporal bins T (post temporal patch)
    return {
        "input_embeds": input_embeds,
        "position_ids": position_ids,
        "video_pos": video_pos,
        "features": features,
        "n_frames": n_frames,
        "choices": choices,
        "correct_idx": correct_idx,
    }


# --------------------------------------------------------------------------- #
# Selection strategies -> kept local video-token indices (0..n_video-1)
# --------------------------------------------------------------------------- #
def select_ours(scores, k, n_frames, args):
    return select_pareto_stratified(
        scores, k, n_frames, k_min=args.k_min, temp=args.temp, beta_max=args.beta_max
    )


def select_uniform(n_video, k, device):
    k = min(k, n_video)
    idx = torch.linspace(0, n_video - 1, steps=k).round().long().unique()
    return idx.to(device)


def select_random(n_video, k, device, generator):
    k = min(k, n_video)
    idx = torch.randperm(n_video, generator=generator)[:k]
    return torch.sort(idx).values.to(device)


def k_from_rho(n_video, rho):
    return max(1, int(round(rho * n_video)))


# --------------------------------------------------------------------------- #
# Gated forward + multiple-choice scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gated_answer_logits(model, sample, kept_local, time_llm=False):
    """Run the LLM on the sequence with non-kept video tokens removed.

    Survivors keep their original M-RoPE position ids (we slice ``position_ids``
    with the keep mask), so this is the real deployable gating path, not a mask.
    Returns ``(last_token_logits, llm_ms)``.
    """
    input_embeds = sample["input_embeds"]
    position_ids = sample["position_ids"]
    video_pos = sample["video_pos"]
    device = input_embeds.device
    seq_len = input_embeds.shape[1]
    n_video = video_pos.numel()

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    local_keep = torch.zeros(n_video, dtype=torch.bool, device=device)
    local_keep[kept_local.to(device)] = True
    keep_mask[video_pos[~local_keep]] = False  # drop the non-kept video tokens

    emb = input_embeds[:, keep_mask, :]
    pos = position_ids[:, :, keep_mask]
    attn = torch.ones((1, int(keep_mask.sum())), dtype=torch.long, device=device)

    if time_llm and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(inputs_embeds=emb, position_ids=pos, attention_mask=attn, use_cache=False)
    if time_llm and device.type == "cuda":
        torch.cuda.synchronize()
    llm_ms = (time.perf_counter() - t0) * 1e3
    return out.logits[0, -1, :].float(), llm_ms


def mc_correct(logits, choices, correct_idx, tokenizer):
    """True if the argmax over the option-letter logits is the gold option.

    Letter token ids are taken exactly as train_oracle builds ``gt_token``."""
    letter_ids = [tokenizer(c, add_special_tokens=False).input_ids[0] for c in choices]
    pred = int(torch.stack([logits[i] for i in letter_ids]).argmax().item())
    return pred == correct_idx


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)  # cpu generator for 'random' strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategies = list(args.strategies)
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}; choices: {list(STRATEGIES)}")
    if "ours" in strategies and not args.scorer_ckpt:
        raise ValueError("strategy 'ours' requires --scorer_ckpt (trained via train_oracle.py).")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa"
    )
    model.eval()
    model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    with open(args.data_file) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    random.Random(args.seed).shuffle(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    print(f"Eval: {len(records)} records | strategies={strategies} | rhos={args.rhos}")

    scorer = None
    correct = {s: {r: 0 for r in args.rhos} for s in strategies}
    llm_ms = {s: {r: 0.0 for r in args.rhos} for s in strategies}
    n_timed = {s: {r: 0 for r in args.rhos} for s in strategies}
    full_correct = 0
    full_llm_ms = 0.0
    full_n_timed = 0
    evaluated = 0

    for i, rec in enumerate(records):
        try:
            sample = prepare_sample(model, processor, rec, video_token_id, device, args)
        except Exception as e:
            print(f"skip record {i}: {e}")
            continue
        if sample is None:
            continue

        features = sample["features"]
        n_video = features.shape[0]
        n_frames = sample["n_frames"]
        warm = evaluated >= args.warmup

        # Lazily build the scorer once we know the feature width.
        scores = None
        if "ours" in strategies:
            if scorer is None:
                scorer = load_scorer(args.scorer_ckpt, features.shape[-1], args.hidden_dim, device)
            scores = scorer(features.unsqueeze(0))[0]

        # Full-model reference (keep all tokens).
        keep_all = torch.arange(n_video, device=device)
        logits, ms = gated_answer_logits(model, sample, keep_all, time_llm=True)
        full_correct += int(mc_correct(logits, sample["choices"], sample["correct_idx"], processor.tokenizer))
        if warm:
            full_llm_ms += ms
            full_n_timed += 1

        for rho in args.rhos:
            k = k_from_rho(n_video, rho)
            for strat in strategies:
                if strat == "ours":
                    kept = select_ours(scores, k, n_frames, args)
                elif strat == "uniform":
                    kept = select_uniform(n_video, k, device)
                else:  # random
                    kept = select_random(n_video, k, device, rng)
                logits, ms = gated_answer_logits(model, sample, kept, time_llm=True)
                correct[strat][rho] += int(
                    mc_correct(logits, sample["choices"], sample["correct_idx"], processor.tokenizer)
                )
                if warm:
                    llm_ms[strat][rho] += ms
                    n_timed[strat][rho] += 1
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples evaluated -- check --data_file / --video_root.")

    _report(args, strategies, evaluated, correct, llm_ms, n_timed,
            full_correct, full_llm_ms, full_n_timed)


def load_scorer(ckpt_path, input_dim, hidden_dim, device):
    """Load a scorer checkpoint written by train_oracle.save_checkpoint."""
    state = torch.load(ckpt_path, map_location=device)
    scorer = Scorer(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    scorer.load_state_dict(state.get("model_state", state))
    scorer.eval()
    print(f"Loaded scorer from {ckpt_path} (input_dim={input_dim}, hidden_dim={hidden_dim})")
    return scorer


def _report(args, strategies, evaluated, correct, llm_ms, n_timed,
            full_correct, full_llm_ms, full_n_timed):
    full_acc = full_correct / evaluated
    full_llm = full_llm_ms / max(1, full_n_timed)

    table = {}
    for s in strategies:
        table[s] = {}
        for r in args.rhos:
            nt = max(1, n_timed[s][r])
            acc = correct[s][r] / evaluated
            s_llm = llm_ms[s][r] / nt
            table[s][str(r)] = {
                "accuracy": acc,
                "accuracy_retention": (acc / full_acc) if full_acc > 0 else float("nan"),
                "compute_saving": 1.0 - r,
                "speedup_llm": (full_llm / s_llm) if s_llm > 0 else float("nan"),
                "llm_ms": s_llm,
            }

    result = {
        "experiment": "oracle_eval",
        "model_name": args.model_name,
        "scorer_ckpt": args.scorer_ckpt,
        "data_file": args.data_file,
        "n_evaluated": evaluated,
        "n_timed": full_n_timed,
        "rhos": args.rhos,
        "k_min": args.k_min,
        "full_accuracy": full_acc,
        "full_llm_ms": full_llm,
        "table": table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull model: acc={full_acc:.4f} | llm={full_llm:.2f}ms  "
          f"(timed over {full_n_timed} samples)\n")
    hdr = f"{'strat':<9}{'rho':>6}{'acc':>9}{'ret%':>8}{'save%':>8}{'sp_llm':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in strategies:
        for r in args.rhos:
            t = table[s][str(r)]
            print(f"{s:<9}{r:>6.2f}{t['accuracy']:>9.4f}{t['accuracy_retention'] * 100:>8.1f}"
                  f"{t['compute_saving'] * 100:>8.1f}{t['speedup_llm']:>9.2f}")
    print(f"\nSaved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Oracle-scorer gated eval: ours vs uniform vs random.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--scorer_ckpt", default="", help="Scorer checkpoint from train_oracle.py (oracle_scorer_*.pt).")
    p.add_argument("--hidden_dim", type=int, default=512, help="Must match the trained scorer (train_oracle default 512).")
    p.add_argument("--data_file", required=True, help="Eval jsonl (NExT-QA format; same as train_oracle --val_file).")
    p.add_argument("--video_root", required=True)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.50, 0.75], help="retention ratios")
    p.add_argument("--strategies", nargs="+", default=["ours", "uniform", "random"],
                   help=f"gating strategies to sweep: {list(STRATEGIES)} (full is always the baseline)")
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of records evaluated.")
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None, help="Cap per-frame resolution (e.g. 100352) on small GPUs.")
    # stratified-Pareto selection knobs (must match how you intend to deploy 'ours')
    p.add_argument("--k_min", type=int, default=1, help="per-frame coverage floor for stratified selection")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--warmup", type=int, default=3, help="samples excluded from timing stats (CUDA warm-up)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_oracle_eval.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())