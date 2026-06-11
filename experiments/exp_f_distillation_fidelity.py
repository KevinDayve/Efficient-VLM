"""Experiment F - Distillation fidelity (Phase 1).

Before spending compute on downstream evaluation, confirm the MLP can actually
*reproduce the teacher ranking*. This isolates "does the scorer learn the signal"
from "does the signal help" (Experiment B) -- so a later downstream miss is
diagnosable.

Method
------
1. Cache, for a slice of QA pairs, the frozen ViT per-patch features and the
   teacher language->video attention scores (the supervision target).
2. Train the query-blind ``Scorer`` (efficient_vlm.scorer) on those pairs with
   the ListMLE ranking loss.
3. On a held-out split, measure how well the scorer reproduces the teacher
   ranking: top-k recall and NDCG@k at each retention ratio rho.

This is the query-blind variant (the paper's default). The query-conditioned
variant is only needed if Experiment C shows a large per-question gap; it would
require feeding pooled question embeddings into the scorer and is left as a
follow-up (see README).

Output
------
Per-rho top-k recall and NDCG against the teacher, on the held-out split.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import torch
from tqdm import tqdm

from efficient_vlm.loss import listmle_loss
from efficient_vlm.scorer import Scorer
from experiments import common


def build_cache(model, processor, samples, args) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run the teacher once per sample; cache (features, teacher_scores) on CPU."""
    feats_cache, score_cache = [], []
    for sample in tqdm(samples, desc="caching teacher"):
        try:
            prepared = common.prepare_inputs(model, processor, sample, args.max_frames)
        except Exception as e:
            tqdm.write(f"skip {sample.qid}: {e}")
            continue
        _, outputs = common.mc_evaluate(
            model, processor, prepared, sample.answer_idx, len(sample.options),
            output_attentions=True,
        )
        scores = common.language_to_video_scores(outputs, prepared, args.layers)
        del outputs
        features = common.get_video_features(model, prepared)
        feats_cache.append(features.detach().to("cpu", torch.float32))
        score_cache.append(scores.detach().to("cpu", torch.float32))
    return feats_cache, score_cache


def train_scorer(scorer, feats, scores, args, device):
    opt = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scorer.train()
    n = len(feats)
    for epoch in range(args.epochs):
        running = 0.0
        rng = torch.Generator().manual_seed(args.seed + epoch)
        perm = torch.randperm(n, generator=rng).tolist()
        for i in perm:
            x = feats[i].to(device).unsqueeze(0)          # (1, L, D)
            tgt = scores[i].to(device).unsqueeze(0)       # (1, L)
            logits = scorer(x)                            # (1, L)
            loss = listmle_loss(logits, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            opt.step()
            running += loss.item()
        print(f"  epoch {epoch + 1}/{args.epochs}  listmle={running / max(1, n):.4f}")
    scorer.eval()


@torch.no_grad()
def evaluate_fidelity(scorer, feats, scores, rhos, device):
    recall = {r: [] for r in rhos}
    ndcg = {r: [] for r in rhos}
    for x_cpu, t_cpu in zip(feats, scores):
        x = x_cpu.to(device).unsqueeze(0)
        pred = scorer(x).squeeze(0).float().cpu()   # (L,)
        teacher = t_cpu.float()
        n_video = teacher.numel()
        for r in rhos:
            k = common.k_from_rho(n_video, r)
            recall[r].append(common.topk_recall(pred, teacher, k))
            ndcg[r].append(common.ndcg_at_k(pred, teacher, k))
    return recall, ndcg


def run(args):
    common.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = common.load_model_and_processor(args.model_name, fp16=args.fp16)

    samples = common.load_nextqa_dev(
        args.dataset_name, args.split, args.video_root, args.max_pairs, args.video_ext, args.seed
    )
    feats, scores = build_cache(model, processor, samples, args)
    if len(feats) < 4:
        raise RuntimeError(f"Only cached {len(feats)} samples -- need more for a train/val split.")

    n_val = max(1, int(len(feats) * args.val_frac))
    val_feats, val_scores = feats[:n_val], scores[:n_val]
    train_feats, train_scores = feats[n_val:], scores[n_val:]
    print(f"Cached {len(feats)} samples -> {len(train_feats)} train / {len(val_feats)} val")

    input_dim = feats[0].shape[-1]
    scorer = Scorer(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"Scorer: input_dim={input_dim}, ~{n_params/1e3:.0f}K params")

    train_scorer(scorer, train_feats, train_scores, args, device)
    recall, ndcg = evaluate_fidelity(scorer, val_feats, val_scores, args.rhos, device)

    def mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    result = {
        "experiment": "F_distillation_fidelity",
        "model_name": args.model_name,
        "n_train": len(train_feats),
        "n_val": len(val_feats),
        "scorer_params": n_params,
        "layers": args.layers,
        "rhos": args.rhos,
        "topk_recall": {str(r): mean(recall[r]) for r in args.rhos},
        "ndcg_at_k": {str(r): mean(ndcg[r]) for r in args.rhos},
    }
    if args.checkpoint:
        torch.save(scorer.state_dict(), args.checkpoint)
        result["checkpoint"] = args.checkpoint
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\nHeld-out fidelity (scorer vs teacher ranking):")
    print(f"{'rho':<8}{'top-k recall':>14}{'NDCG@k':>10}")
    for r in args.rhos:
        print(f"{r:<8.2f}{result['topk_recall'][str(r)]:>14.4f}{result['ndcg_at_k'][str(r)]:>10.4f}")
    print("\nHigh recall/NDCG => the MLP learns the signal; a downstream miss would be the "
          "signal's fault, not the scorer's.")
    print(f"Saved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Experiment F: distillation fidelity of the query-blind scorer.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="lmms-lab/NExTQA")
    p.add_argument("--split", default="test")
    p.add_argument("--video_root", required=True)
    p.add_argument("--video_ext", default="mp4")
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    p.add_argument("--rhos", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.50])
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--checkpoint", default="", help="optional path to save the trained scorer")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_exp_f.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
