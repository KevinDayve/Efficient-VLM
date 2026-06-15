"""Experiment E - Norm-asymmetry (Phase 1, contribution 4).

Cheap, standalone, and it motivates the learned scorer in the paper's own
narrative.

Goal
----
Show that L2 norm of the ViT feature is an *asymmetric* importance signal:
  * keeping the **high-norm** tokens (equivalently, discarding the low-norm ones)
    is expected to be relatively stable -- low-norm tokens are largely
    uninformative, so dropping them costs little accuracy;
  * keeping the **low-norm** tokens (discarding the high-norm ones) is expected
    to degrade sharply.
If a single scalar (norm) could rank importance both ways, you would not need a
learned scorer. The gap between the two curves is the motivation.

This also stress-tests the Pareto budgeting, which derives per-frame budgets from
norms: if norms cannot rank importance, that component needs separate
justification.

Method
------
Training-free. At each rho we keep k = rho * n_video tokens by:
  * ``norm_high``  -- top-k by L2 norm (discard low-norm)
  * ``norm_low``   -- bottom-k by L2 norm (discard high-norm)
  * ``uniform``    -- evenly spaced (reference)
  * ``oracle``     -- top-k by language->video attention (upper bound, optional)
and measure multiple-choice accuracy via key-masking the dropped tokens.

Output
------
Accuracy vs rho per strategy, and the norm_high - norm_low asymmetry gap.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

from tqdm import tqdm

from experiments import common
from experiments.exp_b_oracle_ceiling import evaluate_with_kept


def run(args):
    common.set_seed(args.seed)
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)

    strategies = ["norm_high", "norm_low", "uniform"]
    if args.include_oracle:
        strategies.append("oracle")

    samples = common.load_mc_samples(args)
    print(f"Evaluating {len(samples)} pairs at rho={args.rhos}")

    full_correct = 0
    correct: Dict[str, Dict[float, int]] = {s: {r: 0 for r in args.rhos} for s in strategies}
    evaluated = 0

    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue

        need_attn = "oracle" in strategies
        full_res, outputs = common.mc_evaluate(
            model, processor, prepared, sample.answer_idx, len(sample.options),
            output_attentions=need_attn,
        )
        full_correct += int(full_res.correct)
        scores = common.language_to_video_scores(outputs, prepared, args.layers) if need_attn else None
        del outputs

        features = common.get_video_features(model, prepared)
        n_video = prepared.n_video

        for rho in args.rhos:
            k = common.k_from_rho(n_video, rho)
            for strat in strategies:
                if strat == "norm_high":
                    kept = common.select_l2norm(features, k, largest=True)
                elif strat == "norm_low":
                    kept = common.select_l2norm(features, k, largest=False)
                elif strat == "uniform":
                    kept = common.select_uniform(n_video, k, device=features.device)
                elif strat == "oracle":
                    kept = common.select_topk_scores(scores, k)
                else:
                    continue
                correct[strat][rho] += int(
                    evaluate_with_kept(model, processor, prepared, sample, kept)
                )
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples were evaluated -- check video_root / dataset fields.")

    full_acc = full_correct / evaluated
    table = {s: {str(r): correct[s][r] / evaluated for r in args.rhos} for s in strategies}
    asymmetry = {
        str(r): table["norm_high"][str(r)] - table["norm_low"][str(r)] for r in args.rhos
    }
    result = {
        "experiment": "E_norm_asymmetry",
        "model_name": args.model_name,
        "n_evaluated": evaluated,
        "rhos": args.rhos,
        "full_accuracy": full_acc,
        "accuracy_by_strategy": table,
        "asymmetry_gap_high_minus_low": asymmetry,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull (no drop) accuracy: {full_acc:.4f}")
    header = "rho     " + "".join(f"{s:>11}" for s in strategies)
    print(header)
    for rho in args.rhos:
        print(f"{rho:<8.2f}" + "".join(f"{table[s][str(rho)]:>11.4f}" for s in strategies))
    mean_gap = sum(asymmetry.values()) / len(asymmetry)
    print(f"\nMean asymmetry (norm_high - norm_low) over rho = {mean_gap:+.4f}")
    if mean_gap > 0.05:
        print("Positive gap: norm_high > norm_low -- norm is a partial importance signal, but likely insufficient alone. Learned scorer is justified.")
    elif mean_gap < -0.05:
        print("Negative gap: norm_low > norm_high -- high-norm tokens may be redundant structure. Norm is not a useful ranker.")
    else:
        print(f"Weak asymmetry ({mean_gap:+.4f}): norm does not reliably rank token importance. Learned scorer is justified.")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment E: ViT-norm asymmetry.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA")
    p.add_argument("--split", default="test")
    p.add_argument("--data_file", type=str, default=None,
                   help="Path to a local jsonl (rhymes-ai/NeXTVideo format). When set, "
                        "overrides --dataset_name and resolves nested video paths against --video_root.")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.50])
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16],
                   help="critical layers, only used when --include_oracle is set")
    p.add_argument("--include_oracle", action="store_true",
                   help="also evaluate the attention oracle as an upper-bound reference")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_e.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
