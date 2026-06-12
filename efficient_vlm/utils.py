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
