"""Deployable evaluation of the trained token scorer (paper Eq. 14 + section 3.3).

Unlike the experiment suite -- which masks dropped tokens to measure an
*information ceiling* and explicitly defers wallclock -- this script runs the real
pre-projector gating pipeline: low-scoring video tokens are physically removed
before the LLM, surviving tokens keep their original M-RoPE ids, and the frozen
decoder runs on a genuinely shorter sequence. That makes wallclock speedup
measurable, which is the point of an evaluation entrypoint.

Reported per retention ratio rho:
  * VideoQA multiple-choice top-1 accuracy
  * accuracy retention vs. the full model
  * wallclock speedup -- BOTH the LLM-only forward (isolates the section 3.3 claim,
    since visual tokens drive the quadratic cost) and end-to-end including the fixed
    ViT + scorer + projector overhead (the honest net speedup at this frame budget)
  * theoretical compute saving 1 - K/(T*N)

Strategies swept over rho: ``ours`` (learned scorer + stratified Pareto selection)
and ``uniform`` (matched-retention baseline). The full model is always run as the
accuracy reference and the speedup denominator.

Dataset: NExT-QA multiple-choice via common.load_nextqa_dev. VideoMME (the paper's
primary benchmark) and MVBench need their field layouts added to the ``_*_KEYS``
constants in experiments/common.py -- a marked, one-function follow-up.

Example:
    python evaluate.py --video_root /path/to/nextqa/videos \
        --scorer_ckpt checkpoints/scorer_best.pt \
        --rhos 0.25 0.50 0.75
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tqdm import tqdm

from efficient_vlm import gating
from efficient_vlm.utils import select_pareto_stratified
from experiments import common

GATING_STRATEGIES = ("ours", "uniform")


def _select(strat, scores, k, n_frames, n_video, device, args):
    """Kept local video-token indices for a gating strategy at budget k."""
    if strat == "ours":
        return select_pareto_stratified(
            scores, k, n_frames, k_min=args.k_min, temp=args.temp, beta_max=args.beta_max
        )
    if strat == "uniform":
        return common.select_uniform(n_video, k, device=device)
    raise ValueError(f"unknown gating strategy {strat!r}")


def run(args):
    common.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = common.load_model_and_processor(args.model_name, dtype=args.dtype)

    strategies = [s for s in args.strategies if s in GATING_STRATEGIES]
    unknown = [s for s in args.strategies if s not in GATING_STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}; choices: {list(GATING_STRATEGIES)}")
    needs_scorer = "ours" in strategies
    if needs_scorer and not args.scorer_ckpt:
        raise ValueError("strategy 'ours' requires --scorer_ckpt (train it with exp_f / train.py).")

    if args.data_file:
        samples = common.load_local_mc_jsonl(
            args.data_file, args.video_root, args.max_pairs, args.seed
        )
        print(f"Eval (local): {len(samples)} pairs from {args.data_file}")
    else:
        samples = common.load_nextqa_dev(
            args.dataset_name, args.split, args.video_root, args.max_pairs, args.video_ext, args.seed
        )
    print(f"Eval: {len(samples)} pairs | strategies={strategies} | rhos={args.rhos} "
          f"| k_min={args.k_min}")

    scorer = None
    # accuracy + timing accumulators
    correct = {s: {r: 0 for r in args.rhos} for s in strategies}
    llm_ms = {s: {r: 0.0 for r in args.rhos} for s in strategies}
    e2e_ms = {s: {r: 0.0 for r in args.rhos} for s in strategies}
    n_timed = {s: {r: 0 for r in args.rhos} for s in strategies}
    full_correct = full_llm_ms = full_e2e_ms = 0.0
    full_n_timed = 0
    evaluated = 0
    self_tested = False

    for sample in tqdm(samples):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames,
                                             max_pixels=args.max_pixels)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue

        # One-time correctness gate: the M-RoPE plumbing must reproduce the full
        # model when nothing is dropped, else every gated number is meaningless.
        if not self_tested and not args.skip_self_test:
            ok, diff = gating.self_test(model, processor, prepared, sample, atol=args.self_test_atol)
            if not ok:
                raise RuntimeError(
                    f"Gated-forward self-test FAILED (max|delta logit|={diff:.4g} > "
                    f"{args.self_test_atol}). M-RoPE / inputs_embeds plumbing is wrong; "
                    f"fix efficient_vlm/gating.py before trusting any number."
                )
            tqdm.write(f"self-test ok (max|delta|={diff:.4g})")
            self_tested = True

        # Shared per-sample work: ViT + projector features (also the scorer input)
        # and the full-sequence M-RoPE ids. Timed once; reused by every strategy.
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        video_embeds = common.get_video_features(model, prepared)
        if device.type == "cuda":
            torch.cuda.synchronize()
        feature_ms = (time.perf_counter() - t0) * 1e3

        position_ids = gating.compute_rope_index(model, prepared)
        n_video = prepared.n_video
        n_frames = common.n_frames_of(prepared)
        warm = evaluated >= args.warmup   # exclude warm-up samples from timing stats

        # scorer scores (timed; only charged to 'ours' in the e2e total)
        score_ms = 0.0
        scores = None
        if needs_scorer:
            if scorer is None:
                scorer = gating.load_scorer(
                    args.scorer_ckpt, video_embeds.shape[-1], args.hidden_dim, device
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            scores = scorer(video_embeds.float().unsqueeze(0)).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            score_ms = (time.perf_counter() - t0) * 1e3

        # Full-model baseline (keep all tokens): accuracy reference + speedup denom.
        keep_all = torch.arange(n_video, device=prepared.video_positions.device)
        full_mc, full_res = gating.mc_evaluate_gated(
            model, processor, prepared, sample, video_embeds, keep_all, position_ids,
            time_llm=True,
        )
        full_correct += int(full_mc.correct)
        if warm:
            full_llm_ms += full_res.llm_ms
            full_e2e_ms += feature_ms + full_res.llm_ms
            full_n_timed += 1

        for rho in args.rhos:
            k = common.k_from_rho(n_video, rho)
            for strat in strategies:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                kept = _select(strat, scores, k, n_frames, n_video, prepared.video_positions.device, args)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                select_ms = (time.perf_counter() - t0) * 1e3

                mc, res = gating.mc_evaluate_gated(
                    model, processor, prepared, sample, video_embeds, kept, position_ids,
                    time_llm=True,
                )
                correct[strat][rho] += int(mc.correct)
                if warm:
                    overhead = feature_ms + select_ms + (score_ms if strat == "ours" else 0.0)
                    llm_ms[strat][rho] += res.llm_ms
                    e2e_ms[strat][rho] += overhead + res.llm_ms
                    n_timed[strat][rho] += 1
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples evaluated -- check video_root / dataset fields.")

    _report(args, strategies, evaluated, correct, llm_ms, e2e_ms, n_timed,
            full_correct, full_llm_ms, full_e2e_ms, full_n_timed)


def _report(args, strategies, evaluated, correct, llm_ms, e2e_ms, n_timed,
            full_correct, full_llm_ms, full_e2e_ms, full_n_timed):
    full_acc = full_correct / evaluated
    full_llm = full_llm_ms / max(1, full_n_timed)
    full_e2e = full_e2e_ms / max(1, full_n_timed)

    table = {}
    for s in strategies:
        table[s] = {}
        for r in args.rhos:
            nt = max(1, n_timed[s][r])
            acc = correct[s][r] / evaluated
            s_llm = llm_ms[s][r] / nt
            s_e2e = e2e_ms[s][r] / nt
            table[s][str(r)] = {
                "accuracy": acc,
                "accuracy_retention": (acc / full_acc) if full_acc > 0 else float("nan"),
                "compute_saving": 1.0 - r,   # 1 - K/(T*N), K = round(r * n_video)
                "speedup_llm": full_llm / s_llm if s_llm > 0 else float("nan"),
                "speedup_e2e": full_e2e / s_e2e if s_e2e > 0 else float("nan"),
                "llm_ms": s_llm,
                "e2e_ms": s_e2e,
            }

    result = {
        "experiment": "deployable_eval",
        "model_name": args.model_name,
        "dataset": args.dataset_name,
        "n_evaluated": evaluated,
        "n_timed": full_n_timed,
        "rhos": args.rhos,
        "k_min": args.k_min,
        "full_accuracy": full_acc,
        "full_llm_ms": full_llm,
        "full_e2e_ms": full_e2e,
        "table": table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull model: acc={full_acc:.4f} | llm={full_llm:.2f}ms | e2e={full_e2e:.2f}ms"
          f"  (timed over {full_n_timed} samples)\n")
    hdr = f"{'strat':<8}{'rho':>6}{'acc':>9}{'ret%':>8}{'save%':>8}{'sp_llm':>9}{'sp_e2e':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in strategies:
        for r in args.rhos:
            t = table[s][str(r)]
            print(f"{s:<8}{r:>6.2f}{t['accuracy']:>9.4f}{t['accuracy_retention']*100:>8.1f}"
                  f"{t['compute_saving']*100:>8.1f}{t['speedup_llm']:>9.2f}{t['speedup_e2e']:>9.2f}")
    print("\nsp_llm = LLM-forward-only speedup (the section 3.3 claim); "
          "sp_e2e = end-to-end incl. fixed ViT+scorer cost.")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Deployable gated-inference evaluation (accuracy + wallclock).")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA",
                   help="NExT-QA MC mirror; VideoMME/MVBench need a field adapter in common.py")
    p.add_argument("--data_file", default=None,
                   help="Local MC jsonl (train/val format). Overrides --dataset_name; resolves nested videos under --video_root.")
    p.add_argument("--split", default="test")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Cap per-frame resolution (e.g. 100352) to bound attention memory. Use this on a T4.")
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.50, 0.75],
                   help="retention ratios (paper section 3.2)")
    p.add_argument("--strategies", nargs="+", default=["ours", "uniform"],
                   help=f"gating strategies to sweep: {list(GATING_STRATEGIES)} (full is always the baseline)")
    p.add_argument("--scorer_ckpt", default="", help="trained scorer checkpoint (e.g. scorer_best.pt)")
    p.add_argument("--hidden_dim", type=int, default=256)
    # stratified-Pareto selection knobs (paper section 2.4); k_min default 1 matches exp_g's 'ours'
    p.add_argument("--k_min", type=int, default=1, help="per-frame coverage floor for stratified selection")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    # timing / validation
    p.add_argument("--warmup", type=int, default=3, help="samples excluded from timing stats (CUDA warm-up)")
    p.add_argument("--skip_self_test", action="store_true", help="skip the K=n_video plumbing check")
    p.add_argument("--self_test_atol", type=float, default=0.5,
                   help="fp16-realistic logit bound; the gate also requires the predicted option to match")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Compute dtype for the frozen VLM. Use bf16 for Qwen2.5-VL; fp16 overflows "
                        "the vision tower and produces garbage. fp32 doubles memory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_eval.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
