"""Offline scorer training from a feature cache (cache_features.py).

This is the GPU-free twin of train.py: instead of running the frozen VLM every
step, it streams pre-extracted (vision_feats, teacher_raw) pairs from disk. The
scorer, loss, budget allocator and validation metrics are identical to train.py,
so this should reproduce the online ListMLE numbers (the teacher scores are cached
raw, and ListMLE / Spearman / recall / NDCG are all rank-based -> invariant to the
min-max normalisation train.py applies). Once that parity holds, this is where the
batched contrastive term lands (phase 3): a real B>1 batch is what InfoNCE needs.

Validation needs no model -- the teacher target is cached -- so a held-out cache
gives the same recall@k / NDCG@k / sel_recall@k / Spearman the online loop reports.
"""
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from efficient_vlm.scorer import Scorer
from efficient_vlm.loss import listmle_loss, info_nce
from efficient_vlm.cached_dataset import CachedFeatureDataset, ragged_collate
from efficient_vlm.utils import einmahlHaan, topk_recall, ndcg_at_k, select_pareto_stratified


def save_checkpoint(scorer, optimiser, scheduler, step, checkpoint_dir, loss, tag=None):
    """Checkpoint the scorer/optimiser/scheduler. Inlined (not imported from train.py)
    so the offline trainer has no dependency on the VLM stack train.py pulls in."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    name = f"scorer_{tag}.pt" if tag else f"scorer_step_{step}.pt"
    path = os.path.join(checkpoint_dir, name)
    torch.save({
        "step": step,
        "model_state": scorer.state_dict(),
        "loss": loss,
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, path)
    print(f"Checkpoint saved at step {step} to path: {path} with loss: {loss}")


@torch.no_grad()
def run_validation(scorer, val_loader, args, device, val_ratios):
    """Model-free validation over a held-out cache. Mirrors train.run_validation:
    mean ListMLE, full-ranking Spearman rho, and selection-aligned recall@k /
    NDCG@k / sel_recall@k at each retention ratio."""
    was_training = scorer.training
    scorer.eval()
    losses, rhos, n = [], [], 0
    recalls = {r: [] for r in val_ratios}
    ndcgs = {r: [] for r in val_ratios}
    sel_recalls = {r: [] for r in val_ratios}
    for batch in val_loader:
        for sample in batch:
            if n >= args.val_samples:
                break
            feats = sample["vision_feats"].unsqueeze(0).to(device)
            targets = sample["teacher_raw"].unsqueeze(0).to(device)
            logits = scorer(feats)
            losses.append(listmle_loss(logits, targets, top_m=args.top_m).item())
            pred, teach = logits[0].float(), targets[0].float()
            rho = spearmanr(pred.cpu().numpy(), teach.cpu().numpy()).statistic
            if rho == rho:
                rhos.append(rho)
            n_video = pred.numel()
            n_frames = sample["t"]
            for r in val_ratios:
                k = max(1, int(round(r * n_video)))
                recalls[r].append(topk_recall(pred, teach, k))
                nd = ndcg_at_k(pred, teach, k)
                if nd == nd:
                    ndcgs[r].append(nd)
                kept = set(select_pareto_stratified(pred, k, n_frames).tolist())
                teach_top = set(torch.topk(teach, min(k, n_video)).indices.tolist())
                sel_recalls[r].append(len(kept & teach_top) / max(1, len(teach_top)))
            n += 1
        if n >= args.val_samples:
            break
    if was_training:
        scorer.train()

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    metrics = {"val/loss": _mean(losses), "val/spearman_rho": _mean(rhos)}
    for r in val_ratios:
        pct = int(round(r * 100))
        metrics[f"val/recall@{pct}"] = _mean(recalls[r])
        metrics[f"val/ndcg@{pct}"] = _mean(ndcgs[r])
        metrics[f"val/sel_recall@{pct}"] = _mean(sel_recalls[r])
    return metrics, n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                         entity=args.wandb_entity, config=vars(args))

    dataset = CachedFeatureDataset(args.cache_dir)
    print(f"Loaded cache {args.cache_dir}: {len(dataset)} samples, D={dataset.feature_dim}, meta={dataset.meta}")
    if args.contrastive and not dataset.has_lang:
        raise ValueError(
            f"--contrastive requires language features, but cache {args.cache_dir} has none "
            "(has_lang=False). Re-run cache_features.py to add lang_feat (schema 2)."
        )
    # drop_last so the contrastive batch always has >=2 samples for in-batch negatives.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=ragged_collate,
                        drop_last=args.contrastive)

    val_loader = None
    if args.val_cache_dir:
        val_set = CachedFeatureDataset(args.val_cache_dir)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=ragged_collate)
        print(f"Loaded val cache {args.val_cache_dir}: {len(val_set)} samples")

    scorer = Scorer(input_dim=dataset.feature_dim, hidden_dim=args.hidden_dim,
                    proj_dim=args.proj_dim if args.contrastive else None).to(device)
    scorer.train()
    print(f"Scorer built: input_dim={dataset.feature_dim}, "
          f"{sum(p.numel() for p in scorer.parameters()) / 1e3:.0f}K params, "
          f"contrastive={args.contrastive}")
    optimiser = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.max_steps,
                                                           eta_min=args.learning_rate * 0.1)

    step = 0
    cumulativeLoss = 0.0
    cumulativeListmle = 0.0
    cumulativeCon = 0.0
    last_loss = 0.0
    last_logits = last_targets = None
    best_val_metric = float("-inf")
    raw_score_history = []
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            # One optimiser step per batch. ListMLE is per-sample (ragged token
            # counts), averaged over the batch. The contrastive term needs all the
            # batch's pooled video/language embeddings together, so we keep every
            # sample's graph alive and do a single backward on the combined loss.
            optimiser.zero_grad()
            listmle_sum = 0.0
            video_embs, lang_embs = [], []
            for sample in batch:
                feats = sample["vision_feats"].unsqueeze(0).to(device)
                targets = sample["teacher_raw"].unsqueeze(0).to(device)
                logits = scorer(feats)
                listmle_sum = listmle_sum + listmle_loss(logits, targets, top_m=args.top_m)
                if args.contrastive:
                    lang = sample["lang_feat"].unsqueeze(0).to(device)
                    video_embs.append(scorer.project_video(feats, logits, tau_pool=args.tau_pool))
                    lang_embs.append(scorer.project_lang(lang))
                last_logits, last_targets = logits.detach(), targets.detach()
                raw_score_history.append(targets.detach().flatten().cpu())
            listmle = listmle_sum / len(batch)
            con = torch.zeros((), device=device)
            if args.contrastive and len(video_embs) > 1:
                con = info_nce(torch.cat(video_embs), torch.cat(lang_embs), tau=args.tau)
            loss = listmle + args.lambda_con * con
            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()

            last_loss = loss.item()
            cumulativeLoss += last_loss
            cumulativeListmle += listmle.item()
            cumulativeCon += float(con)
            step += 1
            if step % args.log_interval == 0:
                rho = spearmanr(last_logits[0].float().cpu().numpy(),
                                last_targets[0].float().cpu().numpy()).statistic
                avg_loss = cumulativeLoss / args.log_interval
                avg_listmle = cumulativeListmle / args.log_interval
                avg_con = cumulativeCon / args.log_interval
                msg = f"Step {step} / {args.max_steps}, Loss: {avg_loss:.4f}, Correlation: {rho}"
                metrics = {"train/loss": avg_loss, "train/listmle": avg_listmle,
                           "train/spearman_rho": rho, "train/lr": scheduler.get_last_lr()[0]}
                if args.contrastive:
                    msg += f", ListMLE: {avg_listmle:.4f}, InfoNCE: {avg_con:.4f}"
                    metrics["train/info_nce"] = avg_con
                if raw_score_history:
                    all_scores = torch.cat(raw_score_history)
                    tail_index = einmahlHaan(all_scores)
                    median_score = all_scores.median().item()
                    msg += f", EVT tail index (eps): {tail_index:.4f}, median raw score: {median_score:.4e}"
                    metrics["train/evt_tail_index"] = tail_index
                    metrics["train/median_raw_score"] = median_score
                    raw_score_history.clear()
                print(msg)
                if run is not None:
                    run.log(metrics, step=step)
                cumulativeLoss = cumulativeListmle = cumulativeCon = 0.0
            if step % args.save_every == 0:
                save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, last_loss)
            if val_loader is not None and step % args.val_interval == 0:
                val_metrics, n_val = run_validation(scorer, val_loader, args, device, args.val_ratios)
                rec_str = ", ".join(
                    f"R@{int(round(r*100))} {val_metrics[f'val/recall@{int(round(r*100))}']:.3f}"
                    for r in args.val_ratios)
                print(f"  [val] step {step}: loss {val_metrics['val/loss']:.4f}, "
                      f"rho {val_metrics['val/spearman_rho']:.4f}, {rec_str} over {n_val} samples")
                if run is not None:
                    run.log(val_metrics, step=step)
                mval = val_metrics.get(f"val/{args.best_metric}")
                if mval is not None and mval == mval and mval > best_val_metric:
                    best_val_metric = mval
                    save_checkpoint(scorer, optimiser, scheduler, step,
                                    args.checkpoint_dir, val_metrics["val/loss"], tag="best")

    if step % args.save_every != 0:
        save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, last_loss)
    if run is not None:
        run.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train the scorer offline from a feature cache.")
    p.add_argument("--cache_dir", type=str, required=True, help="Cache directory written by cache_features.py.")
    p.add_argument("--val_cache_dir", type=str, default=None, help="Held-out cache for validation.")
    p.add_argument("--batch_size", type=int, default=8, help="Samples per optimiser step (effective batch).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--top_m", type=int, default=None)
    # Contrastive (InfoNCE) term. Requires a cache with language features (schema 2).
    p.add_argument("--contrastive", action="store_true",
                   help="Add lambda_con * InfoNCE(video, language) to the ListMLE loss.")
    p.add_argument("--lambda_con", type=float, default=0.1, help="Weight on the InfoNCE term.")
    p.add_argument("--tau", type=float, default=0.07, help="InfoNCE temperature.")
    p.add_argument("--tau_pool", type=float, default=1.0,
                   help="Temperature for the score-weighted video pooling (softmax over scorer logits).")
    p.add_argument("--proj_dim", type=int, default=256, help="Contrastive projection dimension.")
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--val_samples", type=int, default=100)
    p.add_argument("--val_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--best_metric", type=str, default="recall@25")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_cached")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="efficientvlm")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
