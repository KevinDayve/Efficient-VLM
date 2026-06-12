"""
Script to monitor the Dekkers Einmahl de Haan moment estimator of the EVT tail index, epislon.
"""
import torch

def einmahlHaan(scores: torch.Tensor, k_frac: float = 0.10) -> float:
    x = scores.flatten().float()
    x = x[x > 0].sort().values
    n = x.numel()
    k = max(10, int(k_frac * n))
    logs = torch.log(x[n-k:]) - torch.log(x[n-k-1])
    M1 = logs.mean()
    M2 = (logs ** 2).mean()
    return (M1 + 1.0 - 0.5 / (1.0 - M1**2 / M2)).item()


def topk_recall(pred_scores: torch.Tensor, teacher_scores: torch.Tensor, k: int) -> float:
    """Fraction of the teacher's top-k tokens that the prediction also ranks top-k.

    This is the selection-aligned metric: at retention ratio r, the scorer keeps
    the top-k = round(r * n_video) tokens, so what matters is how many of the
    teacher's most-important tokens survive -- not the full-ranking correlation.
    """
    k = min(k, pred_scores.numel())
    pred_top = set(torch.topk(pred_scores, k).indices.tolist())
    teach_top = set(torch.topk(teacher_scores, k).indices.tolist())
    return len(pred_top & teach_top) / max(1, len(teach_top))


def ndcg_at_k(pred_scores: torch.Tensor, teacher_scores: torch.Tensor, k: int) -> float:
    """NDCG@k using the teacher score as graded relevance (gains = teacher score).

    Rewards ranking the tokens the teacher cares about most near the top, not just
    set overlap. Returns NaN when the ideal DCG is zero (degenerate target).
    """
    k = min(k, pred_scores.numel())
    rel = teacher_scores.float()
    rel = rel - rel.min()  # non-negative gains
    order = torch.argsort(pred_scores, descending=True)[:k]
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=rel.device).float())
    dcg = float((rel[order] * discounts).sum().item())
    ideal = torch.sort(rel, descending=True).values[:k]
    idcg = float((ideal * discounts).sum().item())
    return dcg / idcg if idcg > 0 else float("nan")
