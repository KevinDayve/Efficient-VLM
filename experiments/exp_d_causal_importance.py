"""Experiment D - Does attention magnitude track causal importance?

Goal
----
Validate the premise that the quantity we distill (language->video attention
weight) is the quantity we actually care about (effect on the answer).

Method
------
On a small set, compare each token group's teacher attention score to a
leave-one-out causal effect: the drop in the gold-answer logit when that group of
video tokens is removed. We use *grouped* ablations (contiguous chunks of video
tokens) to stay tractable. For each sample we get, over its groups:
  * attention importance  = mean teacher score in the group
  * causal importance     = gold_logit(full) - gold_logit(group ablated)
and report the Spearman rank correlation between them.

Output
------
Per-sample Spearman correlations (mean +/- std) and a pooled correlation over all
(group) points.

Decision
--------
Weak correlation => we are distilling a correlational proxy that doesn't move
accuracy; switch the supervision target (e.g. to a knockout-derived signal) or
rethink the layers.
"""

from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from experiments import common


def make_groups(n_video: int, group_size: int) -> List[torch.Tensor]:
    """Contiguous chunks of local video-token indices (a grouped ablation)."""
    groups = []
    for start in range(0, n_video, group_size):
        groups.append(torch.arange(start, min(start + group_size, n_video)))
    return groups


def run(args):
    common.set_seed(args.seed)
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)

    samples = common.load_mc_samples(args)
    print(f"Evaluating up to {len(samples)} pairs, group_size={args.group_size}, layers={args.layers}")

    per_sample_rho: List[float] = []
    pooled_attn: List[float] = []
    pooled_causal: List[float] = []
    evaluated = 0

    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue

        full_res, outputs = common.mc_evaluate(
            model, processor, prepared, sample.answer_idx, len(sample.options),
            output_attentions=True,
        )
        scores = common.language_to_video_scores(outputs, prepared, args.layers)
        del outputs
        full_gold = full_res.gold_logit

        groups = make_groups(prepared.n_video, args.group_size)
        attn_imp, causal_imp = [], []
        for g in groups:
            attn_imp.append(float(scores[g.to(scores.device)].mean().item()))
            key_pos = prepared.video_positions[g.to(prepared.video_positions.device)]
            with common.block_keys(model, key_positions=key_pos, query_positions=None, layers=None):
                res, _ = common.mc_evaluate(
                    model, processor, prepared, sample.answer_idx, len(sample.options)
                )
            causal_imp.append(full_gold - res.gold_logit)  # logit drop when group removed

        if len(groups) >= 2:
            per_sample_rho.append(common.spearman(attn_imp, causal_imp))
            pooled_attn.extend(attn_imp)
            pooled_causal.extend(causal_imp)
        evaluated += 1

    if evaluated == 0 or not per_sample_rho:
        raise RuntimeError("No usable samples -- check video_root / dataset fields.")

    rho_arr = np.asarray([r for r in per_sample_rho if not np.isnan(r)], dtype=float)
    result = {
        "experiment": "D_causal_importance",
        "model_name": args.model_name,
        "n_evaluated": evaluated,
        "group_size": args.group_size,
        "layers": args.layers,
        "per_sample_spearman_mean": float(rho_arr.mean()) if rho_arr.size else float("nan"),
        "per_sample_spearman_std": float(rho_arr.std()) if rho_arr.size else float("nan"),
        "per_sample_spearman_median": float(np.median(rho_arr)) if rho_arr.size else float("nan"),
        "pooled_spearman": common.spearman(pooled_attn, pooled_causal),
        "n_points": len(pooled_attn),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nPer-sample Spearman: mean={result['per_sample_spearman_mean']:.4f} "
          f"+/- {result['per_sample_spearman_std']:.4f} "
          f"(median={result['per_sample_spearman_median']:.4f})")
    print(f"Pooled Spearman over {result['n_points']} points: {result['pooled_spearman']:.4f}")
    if result["per_sample_spearman_mean"] < args.weak_threshold:
        print(f"WEAK correlation (< {args.weak_threshold}): attention may be a poor proxy for "
              "causal importance -- consider changing the supervision target.")
    else:
        print("Attention tracks causal importance -- the premise holds.")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment D: attention vs causal importance.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA")
    p.add_argument("--split", default="test")
    p.add_argument("--data_file", type=str, default=None,
                   help="Path to a local jsonl (rhymes-ai/NeXTVideo format). When set, "
                        "overrides --dataset_name and resolves nested video paths against --video_root.")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=100, help="keep small; ablations are O(groups)")
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    p.add_argument("--group_size", type=int, default=16,
                   help="video tokens per ablated group (larger = cheaper, coarser)")
    p.add_argument("--weak_threshold", type=float, default=0.2,
                   help="mean Spearman below this flags a weak-correlation warning")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_d.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
