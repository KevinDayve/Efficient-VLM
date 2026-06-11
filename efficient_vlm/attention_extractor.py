import torch
from typing import Optional, List, Dict
import torch.nn as nn


class StopForwardPass(Exception):
    """Raised from a forward hook to abort the decoder once the critical layers'
    attention has been captured.

    The paper (sec 2.2 / 2.3) truncates the frozen VLM forward at the last
    critical layer during training -- there is no reason to run layers beyond the
    ones we read attention from. We implement that truncation by raising this
    exception from a hook on the last critical layer; ``train.py`` catches it.
    """
    pass


class AttentionExtractor(nn.Module):
    """Capture language->video attention from a set of critical decoder layers.

    Usage:
        extractor = AttentionExtractor(model, Layers=[12, 13, 14, 15, 16])
        extractor.video_positions = <1-D LongTensor of video-token indices>
        with extractor:
            try:
                model(..., output_attentions=True)   # needs eager attention
            except StopForwardPass:
                pass
        targets = extractor.get_scores()
    """

    def __init__(
        self,
        model,
        Layers: List[int] = [12, 13, 14, 15, 16],
        video_positions: Optional[torch.Tensor] = None,
        truncate: bool = True,
    ):
        super().__init__()
        self.model = model
        self.Layers = list(Layers)
        self.truncate_at = max(self.Layers)
        self.truncate = truncate
        # Absolute indices (into the input sequence) of the video tokens. Must be
        # set before each forward; the video block is contiguous in Qwen2.5-VL and
        # these indices are in the same order as the ViT patch embeddings.
        self.video_positions = video_positions
        self._store: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def _decoder_layers(self) -> nn.ModuleList:
        """Return the LLM decoder layers, robust to HF layout changes.

        Newer Qwen2.5-VL nests the text stack under ``model.language_model``.
        """
        base = getattr(self.model, "model", self.model)
        if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
            return base.language_model.layers
        return base.layers

    def __enter__(self):
        self._register()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    # --- hooks ------------------------------------------------------------- #
    def _get_store_hook(self, index: int):
        def hook(module, _input, output):
            # Eager attention returns (attn_output, attn_weights, ...). Other
            # backends (sdpa/flash) return attn_weights=None -- caught upstream.
            if isinstance(output, (tuple, list)) and len(output) > 1 and output[1] is not None:
                self._store[index] = output[1].detach()
        return hook

    def _stop_hook(self, module, _input, _output):
        # Fires after the self_attn submodule hook of the last critical layer has
        # already stored its weights, so it is safe to abort here.
        raise StopForwardPass

    def _register(self):
        layers = self._decoder_layers()
        for index in self.Layers:
            h = layers[index].self_attn.register_forward_hook(self._get_store_hook(index))
            self._hooks.append(h)
        if self.truncate:
            h = layers[self.truncate_at].register_forward_hook(self._stop_hook)
            self._hooks.append(h)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._store.clear()

    # --- scoring ----------------------------------------------------------- #
    def get_scores(self) -> Optional[torch.Tensor]:
        """Aggregate language->video attention into a per-video-token score.

        For each critical layer we take attention from the post-video text query
        rows to the video key columns, average over heads and over those query
        rows, then average across layers and min-max normalise per sample.
        Returns ``(B, n_video)`` or ``None`` if nothing usable was captured.
        """
        if len(self._store) == 0 or self.video_positions is None:
            return None

        any_attn = next(iter(self._store.values()))
        vid = self.video_positions.to(any_attn.device).flatten()
        if vid.numel() == 0:
            return None
        end = int(vid.max().item()) + 1  # first position after the (contiguous) video block

        scores = []
        for index in self.Layers:
            if index not in self._store:
                continue
            attn = self._store[index]                 # (B, heads, S, S)
            S = attn.shape[-1]
            rows = torch.arange(end, S, device=attn.device)  # language (post-video) queries
            if rows.numel() == 0:
                return None
            block = attn[:, :, rows][:, :, :, vid]    # (B, heads, |rows|, n_video)
            scores.append(block.mean(dim=1).mean(dim=1))  # (B, n_video)

        if not scores:
            return None
        scores = torch.stack(scores, dim=0).mean(dim=0)   # (B, n_video)
        scores = (scores - scores.min(dim=-1, keepdim=True).values) / \
                 (scores.max(dim=-1, keepdim=True).values -
                  scores.min(dim=-1, keepdim=True).values + 1e-8)
        return scores.detach()
