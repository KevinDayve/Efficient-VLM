"""Experiment A - Critical-layer localization on Qwen2.5-VL itself.

Goal
----
Replace the *imported* "layers 12-16" with a measurement on the actual backbone.
We slide a small window (2-4 layers) across all decoder layers, knock out
language->video attention inside the window, and record the multiple-choice
accuracy drop. The window(s) with the largest drop are this model's critical
layer set L -- the layers from which the teacher/oracle scores should be read in
Experiments B-D.

Method
------
"Knockout" = block every language query row from attending to any video key
column, for the layers in the window only (additive -inf pre-softmax). The rest
of the forward pass is untouched, so the drop isolates that window's
contribution to visual information retrieval.

Output
------
A JSON file with, for each window, the mean accuracy and the drop relative to the
clean baseline. Prints the accuracy-drop-vs-depth curve and the argmax window.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from tqdm import tqdm

from experiments import common


def sliding_windows(n_layers: int, window: int, stride: int = 1) -> List[List[int]]:
    return [list(range(s, s + window)) for s in range(0, n_layers - window + 1, stride)]


def run(args):
    common.set_seed(args.seed)
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)
    n_layers = len(common.get_decoder_layers(model))
    windows = sliding_windows(n_layers, args.window, args.stride)
    print(f"{n_layers} decoder layers -> {len(windows)} windows of size {args.window}")

    samples = common.load_nextqa_dev(
        args.dataset_name, args.split, args.video_root, args.max_pairs, args.video_ext, args.seed
    )
    print(f"Evaluating on {len(samples)} QA pairs")

    baseline_correct = 0
    window_correct = [0] * len(windows)
    evaluated = 0

    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
        except Exception as e:  # missing video / no video tokens
            tqdm.write(f"skip {sample.qid}: {e}")
            continue
        n_opt = len(sample.options)

        # clean baseline
        base_res, _ = common.mc_evaluate(model, processor, prepared, sample.answer_idx, n_opt)
        baseline_correct += int(base_res.correct)

        # each window knocked out
        for w, layers in enumerate(windows):
            with common.block_keys(
                model,
                key_positions=prepared.video_positions,
                query_positions=prepared.lang_positions,
                layers=layers,
            ):
                res, _ = common.mc_evaluate(model, processor, prepared, sample.answer_idx, n_opt)
            window_correct[w] += int(res.correct)
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples were evaluated -- check video_root / dataset fields.")

    baseline_acc = baseline_correct / evaluated
    curve = []
    for w, layers in enumerate(windows):
        acc = window_correct[w] / evaluated
        curve.append(
            {
                "window": layers,
                "center_layer": sum(layers) / len(layers),
                "accuracy": acc,
                "accuracy_drop": baseline_acc - acc,
            }
        )

    curve_sorted = sorted(curve, key=lambda c: c["accuracy_drop"], reverse=True)
    result = {
        "experiment": "A_layer_localization",
        "model_name": args.model_name,
        "n_evaluated": evaluated,
        "window_size": args.window,
        "baseline_accuracy": baseline_acc,
        "curve": curve,
        "top_windows": curve_sorted[: args.top_k],
        "recommended_layers": curve_sorted[0]["window"],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print("Largest-drop windows:")
    for c in curve_sorted[: args.top_k]:
        print(f"  layers {c['window']}: acc={c['accuracy']:.4f}  drop={c['accuracy_drop']:.4f}")
    print(f"\nRecommended critical layers L = {result['recommended_layers']}")
    print(f"Decision: if L is far from 12-16, update the imported default. Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment A: critical-layer localization via knockout.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA")
    p.add_argument("--split", default="test")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--window", type=int, default=3, help="sliding window size (2-4)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--top_k", type=int, default=5, help="how many top windows to report")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_a.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
