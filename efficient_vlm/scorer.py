import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from efficient_vlm.utils import pareto_budget

class Scorer(nn.Module):
    """
    A small MLP that outputs a score for a feature vector R^{T x N x D}

    When ``proj_dim`` is set, the scorer also carries a contrastive head: it pools
    the visual tokens into a single video embedding *weighted by its own scores*
    (:meth:`project_video`) and projects the language features into the same space
    (:meth:`project_lang`). Score-weighted pooling is the point -- the contrastive
    gradient flows back through the pooling weights into the ranking logits, so the
    InfoNCE term trains the scorer to upweight tokens that align with the question,
    not just to match the teacher ranking.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1,
                 proj_dim: Optional[int] = None, lang_dim: Optional[int] = None):
        """
        Initialise the scorer
        Args:
            input_dim: The dimension of the input feature vector (example for VIT it would be 768)
            hidden_dim: The dimennsion of the hidden layer. Default is 256
            dropout: the dropout rate for the network (which node to drop). Default is 0.1
            proj_dim: If set, build the contrastive projection heads into this dim.
                      None (default) disables them -- pure ListMLE, unchanged behaviour.
            lang_dim: Dimension of the language features. Defaults to ``input_dim``
                      (vision and language both live in the LLM token space).
        """
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
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
        # Contrastive projection heads (optional). proj_v projects the pooled video
        # embedding, proj_l the pooled language embedding, into a shared proj_dim.
        if proj_dim is not None:
            lang_dim = lang_dim or input_dim
            self.proj_v = nn.Linear(input_dim, proj_dim)
            self.proj_l = nn.Linear(lang_dim, proj_dim)
        else:
            self.proj_v = self.proj_l = None
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
        for proj in (self.proj_v, self.proj_l):
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)

    def project_video(self, x: torch.Tensor, logits: torch.Tensor,
                      tau_pool: float = 1.0) -> torch.Tensor:
        """Score-weighted video embedding: pool tokens by softmax(logits/tau_pool),
        project, L2-normalise. ``x`` is (B, L, D), ``logits`` is (B, L) from
        :meth:`forward`. Returns (B, proj_dim). The pooling weights ARE the scorer's
        own logits, so contrastive gradient reaches the ranking head."""
        assert self.proj_v is not None, "Scorer built without proj_dim; no contrastive head."
        w = torch.softmax(logits / tau_pool, dim=-1)        # (B, L)
        pooled = torch.einsum("bl,bld->bd", w, x)            # (B, D)
        return F.normalize(self.proj_v(pooled), dim=-1)     # (B, proj_dim)

    def project_lang(self, lang_feat: torch.Tensor) -> torch.Tensor:
        """Language embedding: mean over the captured layers, project, L2-normalise.
        ``lang_feat`` is (B, n_layers, D). Returns (B, proj_dim)."""
        assert self.proj_l is not None, "Scorer built without proj_dim; no contrastive head."
        pooled = lang_feat.mean(dim=1)                       # (B, D)
        return F.normalize(self.proj_l(pooled), dim=-1)     # (B, proj_dim)

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