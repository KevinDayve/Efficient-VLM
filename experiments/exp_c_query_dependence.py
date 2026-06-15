"""Experiment C - How query-dependent is the optimal selection?

Goal
----
Settle the central design question: must the scorer see the question, or can it
stay query-blind? NExT-QA has multiple questions per video, which makes this
directly measurable.

Method
------
For each video with >= 2 questions:
  1. Per question, compute the teacher top-k video-token set (oracle scores at
     retention rho). Measure pairwise Jaccard overlap of these sets across the
     video's questions -> cross-question overlap distribution.
  2. Build a *query-blind* oracle: average the teacher scores over all of the
     video's questions and pick ONE fixed top-k set. Measure its per-question MC
     accuracy, and compare against the *per-question* oracle accuracy (each
     question keeps its own top-k). The accuracy gap is the value of query
     conditioning.

Decision
--------
High overlap / small gap  => a query-blind MLP on patch embeddings is fine;
                             reframe honestly as a distilled saliency predictor.
Low overlap / large gap   => the scorer must be conditioned on the question.

This reuses Experiment B's forward passes -- near-zero extra cost.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from experiments import common
from experiments.exp_b_oracle_ceiling import evaluate_with_kept


def run(args):
    common.set_seed(args.seed)
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)

    samples = common.load_mc_samples(args)
    by_video: Dict[str, List] = defaultdict(list)
    for s in samples:
        by_video[s.video_id].append(s)
    multi = {v: qs for v, qs in by_video.items() if len(qs) >= 2}
    print(f"{len(multi)} videos with >= 2 questions (of {len(by_video)} total)")

    rho = args.rho
    jaccards: List[float] = []
    blind_correct = 0
    perq_correct = 0
    n_questions = 0

    for vid, questions in tqdm(list(multi.items())):
        prepared_list = []
        scores_list = []
        ok_questions = []
        n_video_ref = None
        for sample in questions:
            try:
                prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
            except Exception as e:
                tqdm.write(f"skip {sample.qid}: {e}")
                continue
            if n_video_ref is None:
                n_video_ref = prepared.n_video
            elif prepared.n_video != n_video_ref:
                # token counts must align to average scores / compare sets
                tqdm.write(f"skip {sample.qid}: n_video {prepared.n_video} != {n_video_ref}")
                continue
            _, outputs = common.mc_evaluate(
                model, processor, prepared, sample.answer_idx, len(sample.options),
                output_attentions=True,
            )
            scores_list.append(common.language_to_video_scores(outputs, prepared, args.layers))
            del outputs
            prepared_list.append(prepared)
            ok_questions.append(sample)

        if len(ok_questions) < 2:
            continue

        n_video = n_video_ref
        k = common.k_from_rho(n_video, rho)

        # per-question top-k sets + pairwise Jaccard
        topk_sets = [common.select_topk_scores(s, k).tolist() for s in scores_list]
        for a, b in combinations(topk_sets, 2):
            jaccards.append(common.jaccard(a, b))

        # query-blind set: average scores across questions, one fixed top-k
        mean_scores = torch.stack(scores_list, dim=0).mean(dim=0)
        blind_kept = common.select_topk_scores(mean_scores, k)

        for prepared, sample, kept in zip(prepared_list, ok_questions, topk_sets):
            perq_kept = torch.tensor(kept, dtype=torch.long)
            perq_correct += int(evaluate_with_kept(model, processor, prepared, sample, perq_kept))
            blind_correct += int(evaluate_with_kept(model, processor, prepared, sample, blind_kept))
            n_questions += 1

    if n_questions == 0:
        raise RuntimeError("No multi-question videos evaluated -- check the dataset slice.")

    perq_acc = perq_correct / n_questions
    blind_acc = blind_correct / n_questions
    jac = np.asarray(jaccards, dtype=float)
    result = {
        "experiment": "C_query_dependence",
        "model_name": args.model_name,
        "rho": rho,
        "layers": args.layers,
        "n_videos": len(multi),
        "n_questions_evaluated": n_questions,
        "n_question_pairs": int(jac.size),
        "jaccard_mean": float(jac.mean()) if jac.size else float("nan"),
        "jaccard_std": float(jac.std()) if jac.size else float("nan"),
        "jaccard_percentiles": {
            p: float(np.percentile(jac, p)) for p in (10, 25, 50, 75, 90)
        } if jac.size else {},
        "per_question_oracle_accuracy": perq_acc,
        "query_blind_oracle_accuracy": blind_acc,
        "accuracy_gap": perq_acc - blind_acc,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCross-question Jaccard (rho={rho}): "
          f"mean={result['jaccard_mean']:.4f} +/- {result['jaccard_std']:.4f} "
          f"(median={result['jaccard_percentiles'].get(50, float('nan')):.4f})")
    print(f"Per-question oracle accuracy: {perq_acc:.4f}")
    print(f"Query-blind oracle accuracy: {blind_acc:.4f}")
    print(f"Accuracy gap (value of query conditioning): {result['accuracy_gap']:+.4f}")
    if result["accuracy_gap"] >= args.gap_threshold:
        print("Large gap => the scorer MUST be query-conditioned.")
    else:
        print("Small gap => a query-blind MLP is fine; reframe as a saliency predictor.")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment C: query-dependence of the optimal token set.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA")
    p.add_argument("--split", default="test")
    p.add_argument("--data_file", type=str, default=None,
                   help="Path to a local jsonl (rhymes-ai/NeXTVideo format). When set, "
                        "overrides --dataset_name and resolves nested video paths against --video_root.")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=500)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    p.add_argument("--rho", type=float, default=0.20)
    p.add_argument("--gap_threshold", type=float, default=0.02,
                   help="accuracy gap above which query-conditioning is deemed necessary")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_c.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
