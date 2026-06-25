"""Experiment H - Ablations (Phase 2).

Three ablations, each sharing one teacher forward pass per sample:

1. **Supervision source** -- where the importance signal comes from:
   language->video vs CLS(final-row) vs all-token-average vs random. Validates
   that the paper's language->video attention is the right target, not just any
   attention pooling.

2. **Layer range** -- oracle accuracy when reading the signal from different
   decoder layer sets (informed by Experiment A). Confirms the chosen critical
   layers actually carry the most usable signal.

3. **Three-way selection** -- given one score, how you spend the budget:
   global top-k vs uniform-per-bin vs Pareto-adaptive (norm-derived per-frame
   budgets). Validates the budgeting component rather than assuming it.

All arms are evaluated as oracle ceilings (teacher scores, key-masking) at each
rho, so no training is required. Run whichever you need via --ablation.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

from tqdm import tqdm

from experiments import common
from experiments.exp_b_oracle_ceiling import evaluate_with_kept


def parse_layer_sets(specs):
    return [[int(x) for x in spec.split(",")] for spec in specs]


def run(args):
    common.set_seed(args.seed)
    model, processor = common.load_model_and_processor(args.model_name, dtype=args.dtype)

    do = {a: (args.ablation in ("all", a)) for a in ("supervision", "layers", "selection")}
    layer_sets = parse_layer_sets(args.layer_sets)

    sup_sources = ["language", "cls", "all", "random"]
    sel_variants = ["global_topk", "uniform_per_bin", "pareto", "norm_energy"]

    # accuracy[ablation][variant][rho] = count
    def fresh(variants):
        return {v: {r: 0 for r in args.rhos} for v in variants}
    acc = {
        "supervision": fresh(sup_sources),
        "layers": fresh([",".join(map(str, ls)) for ls in layer_sets]),
        "selection": fresh(sel_variants),
    }

    samples = common.load_mc_samples(args)
    print(f"Ablation={args.ablation} | {len(samples)} pairs | rhos={args.rhos}")

    evaluated = 0
    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames, max_pixels=args.max_pixels)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue

        _, outputs = common.mc_evaluate(
            model, processor, prepared, sample.answer_idx, len(sample.options),
            output_attentions=True,
        )

        # Precompute every score we need from this single forward before freeing it.
        sup_scores, layer_scores = {}, {}
        if do["supervision"]:
            for src in sup_sources:
                if src == "random":
                    sup_scores[src] = common.random_scores(prepared.n_video, prepared.video_positions.device)
                else:
                    sup_scores[src] = common.attention_scores(outputs, prepared, args.layers, source=src)
        if do["layers"]:
            for ls in layer_sets:
                layer_scores[",".join(map(str, ls))] = common.language_to_video_scores(outputs, prepared, ls)
        base_scores = common.language_to_video_scores(outputs, prepared, args.layers) if do["selection"] else None
        del outputs

        features = common.get_video_features(model, prepared) if do["selection"] else None
        n_frames = common.n_frames_of(prepared)
        n_video = prepared.n_video

        for rho in args.rhos:
            k = common.k_from_rho(n_video, rho)
            if do["supervision"]:
                for src, sc in sup_scores.items():
                    kept = common.select_topk_scores(sc, k)
                    acc["supervision"][src][rho] += int(
                        evaluate_with_kept(model, processor, prepared, sample, kept))
            if do["layers"]:
                for name, sc in layer_scores.items():
                    kept = common.select_topk_scores(sc, k)
                    acc["layers"][name][rho] += int(
                        evaluate_with_kept(model, processor, prepared, sample, kept))
            if do["selection"]:
                variants = {
                    "global_topk": common.select_topk_scores(base_scores, k),
                    "uniform_per_bin": common.select_uniform_per_bin(n_video, k, n_frames, features.device),
                    # The paper's 2.4 method: stratified per-bin top-k with
                    # tail-index-adaptive budgets.
                    "pareto": common.select_pareto_stratified(base_scores, k, n_frames),
                    # Baseline contrast: per-frame budgets from L2-norm energy.
                    "norm_energy": common.select_pareto_adaptive(base_scores, features, k, n_frames),
                }
                for v, kept in variants.items():
                    acc["selection"][v][rho] += int(
                        evaluate_with_kept(model, processor, prepared, sample, kept))
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples were evaluated -- check video_root / dataset fields.")

    result = {"experiment": "H_ablations", "model_name": args.model_name,
              "n_evaluated": evaluated, "rhos": args.rhos, "layers": args.layers}
    for ablation, enabled in do.items():
        if not enabled:
            continue
        result[ablation] = {
            v: {str(r): acc[ablation][v][r] / evaluated for r in args.rhos}
            for v in acc[ablation]
        }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    for ablation, enabled in do.items():
        if not enabled:
            continue
        print(f"\n=== {ablation} ===")
        print(f"{'variant':<18}" + "".join(f"rho={r:<7.2f}" for r in args.rhos))
        for v in acc[ablation]:
            print(f"{v:<18}" + "".join(f"{result[ablation][v][str(r)]:<11.4f}" for r in args.rhos))
    print(f"\nSaved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment H: Phase 2 ablations.")
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
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Cap per-frame resolution (pixels) to bound video-token count "
                        "and memory. e.g. 50176 (224x224). None = Qwen dynamic resolution.")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16],
                   help="critical layers (from Exp A) used by supervision + selection ablations")
    p.add_argument("--layer_sets", nargs="+", default=["4,5,6", "12,13,14,15,16", "20,21,22"],
                   help="comma-separated layer sets to compare in the layer-range ablation")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.50])
    p.add_argument("--ablation", choices=["all", "supervision", "layers", "selection"], default="all")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Compute dtype for the frozen VLM. bf16 (default) is correct on "
                        "Ampere+/Ada (RTX 6000 Ada, A100, H100); fp16 risks NaN attention "
                        "on Qwen2.5-VL; fp32 for max precision.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_h.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
