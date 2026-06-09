import torch
import torch.nn.functional as F

def listmle_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the ListMLE loss for a batch of predictions and targets.
    Args:
        logits: A tensor of shape (B, L) containing the predicted scores for each item in the list.
        targets: A tensor of shape (B, L) containing the ground truth relevance labels for each item in the list.
    Returns:
        A scalar tensor representing the average ListMLE loss over the batch.
    """
    indices = torch.argsort(targets, dim=-1, descending=True)
    sorted_logits = torch.gather(logits, dim=-1, index=indices)
    cumsum_exp_logits = torch.logcumsumexp(
        sorted_logits.flip(dims=[-1]), dim=-1
    ).flip(dims=[-1])
    loss = (cumsum_exp_logits - sorted_logits).mean()
    return loss