import torch
import torch.nn.functional as F

def listmle_loss(logits: torch.Tensor, targets: torch.Tensor, top_m: int = None) -> torch.Tensor:
    """
    Compute the ListMLE loss for a batch of predictions and targets.
    Args:
        logits: A tensor of shape (B, L) containing the predicted scores for each item in the list.
        targets: A tensor of shape (B, L) containing the ground truth relevance labels for each item in the list.
        top_m: the integer value after which we don't use the Plackett-Luce to avoid noisy gradients.
    Returns:
        A scalar tensor representing the average ListMLE loss over the batch.
    """
    indices = torch.argsort(targets, dim=-1, descending=True)
    sorted_logits = torch.gather(logits, dim=-1, index=indices)
    cumsum_exp_logits = torch.logcumsumexp(
        sorted_logits.flip(dims=[-1]), dim=-1
    ).flip(dims=[-1])
    per_position = cumsum_exp_logits - sorted_logits
    if top_m is not None:
        per_position = per_position[:, :top_m]
    loss = per_position.mean()
    return loss


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft-label binary cross-entropy, as in LITE (arXiv:2411.13626 Eq. 4 / Fig. 6).

    LITE trains the selector with ``s_hat = Sigmoid(MLP(p))`` against the oracle's
    min-max-normalised [0,1] token values, using BCE. We mirror that here: ``logits``
    are the scorer's raw outputs (sigmoid applied internally for numerical stability)
    and ``targets`` are the [0,1] relevance scores used as *soft* labels -- no
    binarisation, so the full graded relevance signal is preserved.

    Args:
        logits: (B, L) raw scorer outputs (pre-sigmoid).
        targets: (B, L) relevance scores in [0, 1].
    Returns:
        Scalar mean BCE loss over the batch.
    """
    return F.binary_cross_entropy_with_logits(logits, targets)