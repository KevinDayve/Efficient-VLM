"""Experiment G - Main results table (Phase 2).

Produces the accuracy-and-wallclock table across retention ratios rho for a set
of token-selection strategies. Per the plan, wallclock includes the *selection*
overhead (scoring + sort + budgeting), not just LLM FLOPs -- so we time the
selection function separately from the (identical-cost) masked LLM forward.

What is real vs. stubbed
------------------------
Implemented here directly:
  full, uniform, random, norm_high, norm_low, kitoke(approx), attention_oracle,
  fastv, ours(trained scorer).
Registered as explicit stubs (raise NotImplementedError with a pointer) because
they require the authors' reference code and should not be faked for a
head-to-head number:
  l1_delta, dycoke, learnpruner, score.

Drop in the official implementations at the marked seams, then re-run. FastV and
LearnPruner are the non-optional comparisons (FastV is the stated foil,
LearnPruner the closest methodological neighbour).

Note on accuracy semantics: as elsewhere, "keeping k tokens" is implemented by
masking the dropped video tokens as attention keys, so this measures the
*information* ceiling of each selection. Real deployable wallclock (physically
shorter sequences) is a separate engineering measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Dict

import torch
from tqdm import tqdm

from experiments import common
from experiments.exp_b_oracle_ceiling import evaluate_with_kept


class SelectionContext:
    """Everything a strategy might need to pick k tokens for one sample."""

    def __init__(self, prepared, teacher_scores, fastv_scores, features, n_frames, scorer):
        self.prepared = prepared
        self.teacher_scores = teacher_scores      # language->video attention (oracle)
        self.fastv_scores = fastv_scores
        self.features = features
        self.n_frames = n_frames
        self.scorer = scorer
        self.n_video = prepared.n_video
        self.device = features.device


def _stub(name, ref):
    def fn(ctx, k):
        raise NotImplementedError(
            f"'{name}' needs the authors' reference implementation ({ref}). "
            f"Wire it in at the seam in exp_g_main_table.py before reporting a number."
        )
    return fn


# name -> (callable(ctx, k) -> kept_local_idx, deployable?)
STRATEGIES: Dict[str, Callable] = {
    "uniform":          lambda ctx, k: common.select_uniform(ctx.n_video, k, ctx.device),
    "random":           lambda ctx, k: common.select_topk_scores(
                            common.random_scores(ctx.n_video, ctx.device), k),
    "norm_high":        lambda ctx, k: common.select_l2norm(ctx.features, k, largest=True),
    "norm_low":         lambda ctx, k: common.select_l2norm(ctx.features, k, largest=False),
    "kitoke":           lambda ctx, k: common.select_kitoke(ctx.features, k),
    "fastv":            lambda ctx, k: common.select_topk_scores(ctx.fastv_scores, k),
    "attention_oracle": lambda ctx, k: common.select_topk_scores(ctx.teacher_scores, k),
    "ours":             lambda ctx, k: common.select_pareto_stratified(
                            ctx.scorer(ctx.features.unsqueeze(0)).squeeze(0), k, ctx.n_frames),
    # --- stubs: official code required ---
    "l1_delta":     _stub("l1_delta", "L1-delta token redundancy"),
    "dycoke":       _stub("dycoke", "DyCoke"),
    "learnpruner":  _stub("learnpruner", "LearnPruner"),
    "score":        _stub("score", "SCORE"),
}

# strategies that are not deployable (need the full teacher forward to even score)
NON_DEPLOYABLE = {"attention_oracle"}


def load_scorer(ckpt_path, input_dim, hidden_dim, device):
    from efficient_vlm.scorer import Scorer
    scorer = Scorer(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    scorer.load_state_dict(state.get("model_state", state))
    scorer.eval()
    return scorer


def run(args):
    common.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)

    strategies = args.strategies
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}. choices: {sorted(STRATEGIES)}")
    needs_attn = bool({"attention_oracle", "fastv"} & set(strategies))
    needs_scorer = "ours" in strategies
    if needs_scorer and not args.scorer_ckpt:
        raise ValueError("strategy 'ours' requires --scorer_ckpt (train it with exp_f).")

    samples = common.load_mc_samples(args)
    print(f"Benchmark: {len(samples)} pairs | strategies={strategies} | rhos={args.rhos}")

    correct = {s: {r: 0 for r in args.rhos} for s in strategies}
    select_ms = {s: 0.0 for s in strategies}    # cumulative selection wallclock
    select_n = {s: 0 for s in strategies}
    full_correct = 0
    evaluated = 0
    scorer = None

    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue

        full_res, outputs = common.mc_evaluate(
            model, processor, prepared, sample.answer_idx, len(sample.options),
            output_attentions=needs_attn,
        )
        full_correct += int(full_res.correct)
        teacher = common.language_to_video_scores(outputs, prepared, args.layers) if needs_attn else None
        fastv = common.fastv_scores(outputs, prepared, args.fastv_layer) if needs_attn else None
        del outputs

        features = common.get_video_features(model, prepared)
        if needs_scorer and scorer is None:
            scorer = load_scorer(args.scorer_ckpt, features.shape[-1], args.hidden_dim, device)

        ctx = SelectionContext(prepared, teacher, fastv, features, common.n_frames_of(prepared), scorer)

        for rho in args.rhos:
            k = common.k_from_rho(ctx.n_video, rho)
            for strat in strategies:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                kept = STRATEGIES[strat](ctx, k)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                select_ms[strat] += (time.perf_counter() - t0) * 1e3
                select_n[strat] += 1
                correct[strat][rho] += int(
                    evaluate_with_kept(model, processor, prepared, sample, kept)
                )
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples were evaluated -- check video_root / dataset fields.")

    full_acc = full_correct / evaluated
    acc_table = {s: {str(r): correct[s][r] / evaluated for r in args.rhos} for s in strategies}
    overhead = {s: (select_ms[s] / select_n[s] if select_n[s] else float("nan")) for s in strategies}
    result = {
        "experiment": "G_main_table",
        "model_name": args.model_name,
        "dataset": args.dataset_name,
        "n_evaluated": evaluated,
        "rhos": args.rhos,
        "layers": args.layers,
        "full_accuracy": full_acc,
        "accuracy": acc_table,
        "selection_overhead_ms": overhead,
        "non_deployable": sorted(NON_DEPLOYABLE & set(strategies)),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull (no drop) accuracy: {full_acc:.4f}\n")
    print(f"{'strategy':<18}" + "".join(f"rho={r:<7.2f}" for r in args.rhos) + f"{'sel_ms':>9}")
    for s in strategies:
        flag = "*" if s in NON_DEPLOYABLE else " "
        row = "".join(f"{acc_table[s][str(r)]:<11.4f}" for r in args.rhos)
        print(f"{s:<18}{row}{overhead[s]:>9.3f}{flag}")
    print("\n* = not deployable (needs the full teacher forward to score).")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment G: main results table (accuracy + wallclock).")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA",
                   help="swap for VideoMME / MVBench / EgoSchema mirrors per the plan")
    p.add_argument("--split", default="test")
    p.add_argument("--data_file", type=str, default=None,
                   help="Path to a local jsonl (rhymes-ai/NeXTVideo format). When set, "
                        "overrides --dataset_name and resolves nested video paths against --video_root.")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    p.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.50])
    p.add_argument("--strategies", nargs="+",
                   default=["uniform", "fastv", "kitoke", "attention_oracle", "ours"],
                   help=f"any of: {sorted(STRATEGIES)}")
    p.add_argument("--fastv_layer", type=int, default=2)
    p.add_argument("--scorer_ckpt", default="", help="trained scorer for strategy 'ours' (from exp_f)")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_g.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
