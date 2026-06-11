import torch
from typing import Optional, List
import torch.nn as nn


class AttentionExtractor(nn.Module):
    """Compute language->video attention scores from a model's returned attentions.

    Newer Transformers capture attention weights into ``outputs.attentions`` (when
    the model is run with ``output_attentions=True`` and
    ``attn_implementation="eager"``) rather than exposing them via module forward
    hooks. So we read them from the model output instead of hooking.

    Usage:
        extractor = AttentionExtractor(model, Layers=[12, 13, 14, 15, 16])
        extractor.video_positions = <1-D LongTensor of video-token indices>
        with torch.no_grad():
            outputs = model(..., output_attentions=True)
        targets = extractor.scores_from_attentions(outputs.attentions)
    """

    def __init__(
        self,
        model,
        Layers: List[int] = [12, 13, 14, 15, 16],
        video_positions: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.Layers = list(Layers)
        # Absolute indices (into the input sequence) of the video tokens. Must be
        # set before each forward; the video block is contiguous in Qwen2.5-VL and
        # these indices are in the same order as the (merged) video tokens.
        self.video_positions = video_positions

    def scores_from_attentions(self, attentions) -> Optional[torch.Tensor]:
        """Aggregate language->video attention into a per-video-token score.

        ``attentions`` is the tuple returned in ``outputs.attentions`` -- one
        ``(B, heads, S, S)`` tensor per decoder layer. For each critical layer we
        take attention from the post-video text query rows to the video key
        columns, average over heads and those rows, average across layers, and
        min-max normalise per sample. Returns ``(B, n_video)`` or ``None``.
        """
        if attentions is None or self.video_positions is None:
            return None
        vid = self.video_positions.flatten()
        if vid.numel() == 0:
            return None
        end = int(vid.max().item()) + 1  # first position after the (contiguous) video block

        scores = []
        for L in self.Layers:
            if L >= len(attentions) or attentions[L] is None:
                continue
            attn = attentions[L]                          # (B, heads, S, S)
            v = vid.to(attn.device)
            S = attn.shape[-1]
            rows = torch.arange(end, S, device=attn.device)  # language (post-video) queries
            if rows.numel() == 0:
                return None
            block = attn[:, :, rows][:, :, :, v]          # (B, heads, |rows|, n_video)
            scores.append(block.mean(dim=1).mean(dim=1).float())  # (B, n_video)

        if not scores:
            return None
        scores = torch.stack(scores, dim=0).mean(dim=0)   # (B, n_video)
        scores = (scores - scores.min(dim=-1, keepdim=True).values) / \
                 (scores.max(dim=-1, keepdim=True).values -
                  scores.min(dim=-1, keepdim=True).values + 1e-8)
        return scores.detach()
