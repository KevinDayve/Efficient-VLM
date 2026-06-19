import torch
import torch.nn as nn
from typing import Tuple

from efficient_vlm.utils import pareto_budget

class Scorer(nn.Module):
    """
    A small MLP that outputs a score for a feature vector R^{T x N x D}
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Initialise the scorer
        Args:
            input_dim: The dimension of the input feature vector (example for VIT it would be 768)
            hidden_dim: The dimennsion of the hidden layer. Default is 256
            dropout: the dropout rate for the network (which node to drop). Default is 0.1
        """
        super().__init__()
        self.input_dim = input_dim
        self.inputNorm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden_dim, 1)
        self._init_weights()
    
    def _init_weights(self):
        """
        Small init on the head keeps initial logits near zero,
        which prevents the ListMLE loss from being dominated by
        arbitrary large scores at the start of training.
        """
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scorer module
        Args:
            x: The input feature vector of shape (T, N, D)
        Returns:
            A score for each feature vector of shape (T, N)
        """
        x = self.inputNorm(x)
        xhat = self.mlp(x)
        logits = self.head(xhat).squeeze(-1) # B, L
        return logits
    
    def allocate_budget(self, logits: torch.Tensor, T: int, K: int, k_min: int = 4,
                        temp: float = 1.0, beta_max: float = 3.0) -> torch.Tensor:
        """Per-bin budgets (B, T) summing to K, set by the Pareto tail index.

        Delegates to :func:`efficient_vlm.utils.pareto_budget` (paper section 2.4):
        the per-bin allocation concentrates with the estimated tail index gamma.

        Args:
            logits: (B, T*N) predicted importance.
        Returns:
            integer budgets (B, T) that sum to K, floored at k_min, capped at N.
        """
        B, _ = logits.shape
        budgets = [
            pareto_budget(logits[b], n_frames=T, K=K, k_min=k_min, temp=temp, beta_max=beta_max)
            for b in range(B)
        ]
        return torch.stack(budgets, dim=0)
    
    def select_stratified(self, x: torch.Tensor, T: int, K: int, k_min: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given embeddings, returns kept embeddings and original indicies for M-RoPE positional ids. Derives per-bin topK with scorer-derived budgets.
        """
        B, L, dim = x.shape
        N = L // T
        logits = self.forward(x)
        k_t = self.allocate_budget(logits, T, K, k_min)
        kept_idx = []
        for b in range(B):
            idx_b = []
            for t in range(T):
                seg = logits[b, t*N:(t+1)*N]
                top = torch.topk(seg, int(k_t[b, t])).indices + t * N
                idx_b.append(top)
            kept_idx.append(torch.cat(idx_b).sort().values)
        kept_idx = torch.stack(kept_idx)
        feat_kept = torch.gather(x, 1, kept_idx.unsqueeze(-1).expand(-1, -1, dim))
        return feat_kept, kept_idx
    
    def select_topk(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for inference convenience.
        Return the top-k embeddings from the input feature vector.
        Args:
            x: The input feature vector of shape (T, N, D)
            k: The number of top-k embeddings to select
        Returns:
            The top-k embeddings which are kept.
        """
        logits = self.forward(x)
        indices = torch.topk(logits, k=k, dim=-1).indices
        indices_sorted = torch.sort(indices, dim=-1).values
        f_kept = torch.gather(
            x,
            dim=1,
            index=indices_sorted.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        return f_kept, indices_sorted
    
if __name__ == "__main__":
    # Test
    B, T, N, d = 2, 8, 196, 768   # Standard ViT feature shape (8 frames, 196 patches per frame)
    f = torch.randn(B, T * N, d)
 
    scorer = Scorer(input_dim=d, hidden_dim=256)
    print(scorer)
 
    logits = scorer(f)
    print(f"logits shape : {logits.shape}")
    assert logits.shape == (B, T * N)
    f_kept, idx = scorer.select_topk(f, k=int(T * N * 0.5))
    print(f"kept shape   : {f_kept.shape}")
    print(f"indices shape: {idx.shape}") 
    print("Looks solid!")