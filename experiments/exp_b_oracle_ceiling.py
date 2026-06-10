"""Experiment B - The oracle ceiling (most important early test).

Goal
----
Establish the upper bound a distilled scorer could ever reach, and confirm the
supervision signal beats trivial baselines. We use the *teacher scores directly*
(full language->video attention from the critical layers found in A) to select
the top-k video tokens at inference, and measure accuracy at each retention ratio
rho. We compare against uniform sampling and a KiToke-style baseline at matched
retention.

This is not deployable -- it needs the full forward pass to even compute the
scores. That is the point: it is a *ceiling*.

Kill criterion
--------------
If the attention-oracle barely beats uniform sampling, the signal carries no
usable information and no amount of distillation will save it -- STOP and pivot.
If it clearly beats uniform (and KiToke), there is a real bound to chase.

Output
------
JSON with accuracy vs rho for: attention-oracle, uniform, kitoke (and norm if
enabled). Also a "full" (no dropping) reference accuracy.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from tqdm import tqdm

from experiments import common


def evaluate_with_kept(model, processor, prepared, sample, kept_local_idx):
    """MC accuracy when only ``kept_local_idx`` video tokens are visible."""
    dropped = common.dropped_positions(prepared.video_positions, kept_local_idx)
    with common.block_keys(model, key_positions=dropped, query_positions=None, layers=None):
        res, _ = common.mc_evaluate(model, processor, prepared, sample.answer_idx, len(sample.options))
    return res.correct


def run(args):
    common.set_seed(args.seed)
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)

    strategies = ["oracle", "uniform", "kitoke"]
    if args.include_norm:
        strategies.append("norm")

    samples = common.load_nextqa_dev(
        args.dataset_name, args.split, args.video_root, args.max_pairs, args.video_ext, args.seed
    )
    print(f"Evaluating {len(samples)} pairs at rho={args.rhos}, layers={args.layers}")

    full_correct = 0
    # correct[strategy][rho] = count
    correct: Dict[str, Dict[float, int]] = {s: {r: 0 for r in args.rhos} for s in strategies}
    evaluated = 0

    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue

        # teacher forward pass (gives clean accuracy + oracle scores)
        full_res, outputs = common.mc_evaluate(
            model, processor, prepared, sample.answer_idx, len(sample.options), output_attentions=True
        )
        full_correct += int(full_res.correct)
        scores = common.language_to_video_scores(outputs, prepared, args.layers)
        del outputs  # free attention tensors before the many masked forward passes

        features = None
        if "kitoke" in strategies or "norm" in strategies:
            features = common.get_video_features(model, prepared)

        n_video = prepared.n_video
        for rho in args.rhos:
            k = common.k_from_rho(n_video, rho)
            for strat in strategies:
                if strat == "oracle":
                    kept = common.select_topk_scores(scores, k)
                elif strat == "uniform":
                    kept = common.select_uniform(n_video, k, device=scores.device)
                elif strat == "kitoke":
                    kept = common.select_kitoke(features, k)
                elif strat == "norm":
                    kept = common.select_l2norm(features, k, largest=True)
                else:
                    continue
                correct[strat][rho] += int(
                    evaluate_with_kept(model, processor, prepared, sample, kept)
                )
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples were evaluated -- check video_root / dataset fields.")

    full_acc = full_correct / evaluated
    table: Dict[str, Dict[str, float]] = {}
    for strat in strategies:
        table[strat] = {str(r): correct[strat][r] / evaluated for r in args.rhos}

    result = {
        "experiment": "B_oracle_ceiling",
        "model_name": args.model_name,
        "n_evaluated": evaluated,
        "layers": args.layers,
        "rhos": args.rhos,
        "full_accuracy": full_acc,
        "accuracy_by_strategy": table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull (no drop) accuracy: {full_acc:.4f}")
    header = "rho     " + "".join(f"{s:>10}" for s in strategies)
    print(header)
    for rho in args.rhos:
        row = f"{rho:<8.2f}" + "".join(f"{table[s][str(rho)]:>10.4f}" for s in strategies)
        print(row)

    # kill-criterion summary: oracle-minus-uniform gap, averaged over rho
    gaps = [table["oracle"][str(r)] - table["uniform"][str(r)] for r in args.rhos]
    mean_gap = sum(gaps) / len(gaps)
    print(f"\nMean(oracle - uniform) over rho = {mean_gap:+.4f}")
    if mean_gap <= args.kill_threshold:
        print(f"KILL CRITERION: gap <= {args.kill_threshold}. Signal looks too weak -- consider pivoting.")
    else:
        print("Oracle beats uniform -- there is a real ceiling to chase. Proceed.")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment B: attention-oracle ceiling vs baselines.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA")
    p.add_argument("--split", default="test")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16],
                   help="critical layers from Experiment A")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.50])
    p.add_argument("--include_norm", action="store_true", help="also evaluate L2-norm top-k")
    p.add_argument("--kill_threshold", type=float, default=0.01,
                   help="if mean(oracle-uniform) <= this, flag the kill criterion")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_b.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
