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


def hill_tail_index(scores: torch.Tensor, k_frac: float = 0.10) -> float:
    """Hill estimator of the EVT tail index gamma = 1/alpha (paper Eq 9).

    This is exactly the ``M1`` term inside :func:`einmahlHaan` -- the conditional
    MLE of the Pareto tail over the top-k order statistics:
        gamma_hat = (1/k) * sum_i log( x_(n-i+1) / x_(n-k) ).
    Larger gamma => heavier tail => importance is more concentrated in a few
    tokens. Returns NaN if there aren't enough positive samples to estimate.
    """
    x = scores.flatten().float()
    x = x[x > 0].sort().values
    n = x.numel()
    if n < 12:
        return float("nan")
    k = max(10, int(k_frac * n))
    k = min(k, n - 1)
    logs = torch.log(x[n - k:]) - torch.log(x[n - k - 1])
    return logs.mean().item()


def pareto_budget(
    scores: torch.Tensor,
    n_frames: int,
    K: int,
    k_min: int = 1,
    temp: float = 1.0,
    beta_max: float = 3.0,
) -> torch.Tensor:
    """Per-frame token budgets ``k_t`` that sum to ``K`` (paper section 2.4).

    Sets each temporal bin's budget adaptively from the Pareto tail index of the
    predicted importance: a tempered power-law allocation
        w_t  proportional to  m_t ** beta,   beta = clamp(temp * gamma, 0, beta_max)
    where ``m_t`` is the bin's predicted-importance mass and ``gamma`` is the Hill
    tail index over all tokens. ``beta -> 0`` recovers uniform-per-bin; large
    ``beta`` approaches global top-k. The ``k_min`` floor guarantees temporal
    coverage. Budgets are clamped to the per-bin token count ``N`` and any budget
    lost to that cap is redistributed, so the result sums to ``K`` whenever K <= n.

    Args:
        scores: 1-D predicted importance for one video, shape ``(n_video,)``. Raw
                scorer logits are fine; they're softmaxed internally for the mass.
        n_frames: number of temporal bins ``T`` (``n_video`` must divide by it).
        K: total token budget to retain.
    Returns:
        1-D long tensor ``(n_frames,)`` of per-bin budgets.
    """
    n_video = scores.numel()
    T = n_frames
    N = n_video // T
    device = scores.device
    K = int(min(K, n_video))

    p = torch.softmax(scores.float(), dim=-1)          # positive, heavy-tailed
    gamma = hill_tail_index(p)
    beta = 0.0 if gamma != gamma else max(0.0, min(temp * gamma, beta_max))

    m = p.view(T, N).sum(dim=1)                        # per-bin mass (T,)
    w = m.pow(beta)
    w = w / w.sum().clamp_min(1e-12)

    k_min = max(0, min(k_min, K // T))                 # never floor above budget
    spendable = K - k_min * T
    frac = spendable * w
    k_t = k_min + frac.floor().long()

    deficit = K - int(k_t.sum().item())                # from flooring
    if deficit > 0:
        rem = frac - frac.floor()
        top = torch.topk(rem, min(deficit, T)).indices
        k_t[top] += 1
    k_t = k_t.clamp(max=N)

    # Redistribute any budget lost to the per-bin N cap to bins with room.
    for _ in range(T):
        short = K - int(k_t.sum().item())
        if short <= 0:
            break
        room = k_t < N
        if not bool(room.any()):
            break
        cand = torch.where(room)[0]
        order = cand[torch.argsort(w[cand], descending=True)]
        for idx in order[:short]:
            k_t[idx] += 1
    return k_t.to(device)


def select_pareto_stratified(
    scores: torch.Tensor, k: int, n_frames: int,
    k_min: int = 1, temp: float = 1.0, beta_max: float = 3.0,
) -> torch.Tensor:
    """Paper section 2.4 selector: stratified per-bin top-k with Pareto-tail-index
    adaptive per-bin budgets. Returns the kept token indices (sorted ascending).

    Splits the n_video tokens into n_frames temporal bins, derives per-bin budgets
    via :func:`pareto_budget`, then keeps the top-scoring tokens within each bin.
    Falls back to plain global top-k when the tokens don't divide evenly into bins.
    """
    n_video = scores.numel()
    if n_frames <= 0 or n_video % n_frames != 0:
        kk = min(k, n_video)
        return torch.sort(torch.topk(scores, kk).indices).values
    T = n_frames
    N = n_video // T
    budgets = pareto_budget(scores, n_frames=T, K=k, k_min=k_min, temp=temp, beta_max=beta_max)
    kept = []
    for t in range(T):
        bt = int(budgets[t].item())
        if bt == 0:
            continue
        offset = t * N
        local = torch.topk(scores[offset: offset + N], k=bt).indices
        kept.append(local + offset)
    if not kept:
        kk = min(k, n_video)
        return torch.sort(torch.topk(scores, kk).indices).values
    return torch.sort(torch.cat(kept)).values


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
