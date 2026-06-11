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