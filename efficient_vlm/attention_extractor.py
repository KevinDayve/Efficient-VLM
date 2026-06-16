import torch
from typing import Optional, List
import torch.nn as nn


class _StopForward(Exception):
    """Sentinel raised inside a forward hook to abort the VLM forward early."""


class AttentionExtractor(nn.Module):
    """Compute language->video attention scores from a model's internal attention.

    The supervision signal only needs attention from a few critical layers
    (``Layers``, e.g. 12-16). Rather than run the whole decoder, ``truncated_forward``
    captures attention weights at those layers via forward hooks and aborts the
    forward right after the last needed layer -- so the model never computes the
    layers above ``max(Layers)``. The captured attentions are then consumed by
    ``scores_from_attentions`` exactly like ``outputs.attentions`` would be.

    Capturing requires ``output_attentions=True`` and ``attn_implementation="eager"``
    so the self-attention modules emit weights.

    Usage:
        extractor = AttentionExtractor(model, Layers=[12, 13, 14, 15, 16])
        extractor.set_sample(input_ids, video_token_id, special_ids)
        with torch.no_grad():
            attentions = extractor.truncated_forward(
                input_ids=..., output_attentions=True, ...
            )
        targets = extractor.scores_from_attentions(attentions)
    """

    def __init__(
        self,
        model,
        Layers: List[int] = [12, 13, 14, 15, 16],
        video_positions: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.Layers = list(Layers)
        # Absolute indices (into the input sequence) of the video tokens. Must be
        # set before each forward; the video block is contiguous in Qwen2.5-VL and
        # these indices are in the same order as the (merged) video tokens.
        self.video_positions = video_positions
        # Optional boolean mask (1-D, length S) selecting the language query rows
        # to read attention from. When set, special tokens are excluded so they
        # don't add noise to the per-video-token scores. If None, all post-video
        # rows are used.
        self.query_mask = query_mask
        # Per-layer attention weights captured during the most recent forward.
        self._captured: dict = {}

    # ------------------------------------------------------------------ #
    # Truncated capture
    # ------------------------------------------------------------------ #
    def _decoder_layers(self):
        """Locate the language decoder layer list across Qwen2.5-VL versions."""
        base = getattr(self.model, "model", self.model)
        if hasattr(base, "layers"):                       # model.model.layers
            return base.layers
        lm = getattr(base, "language_model", None)        # newer split text model
        if lm is not None and hasattr(lm, "layers"):
            return lm.layers
        raise AttributeError("Could not locate decoder layers on the model.")

    def _make_capture_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Eager self-attention returns (attn_output, attn_weights, ...) when
            # output_attentions=True; attn_weights is (B, heads, S, S).
            attn = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            if attn is not None:
                # Reduce to (B, n_video) here and drop the (B, heads, S, S) map
                # immediately, so only one full attention tensor is alive at a
                # time instead of one per captured layer (~5x less peak memory).
                self._captured[layer_idx] = self._reduce(attn.detach())
        return hook

    def _reduce(self, attn: torch.Tensor) -> Optional[torch.Tensor]:
        """Collapse one layer's ``(B, heads, S, S)`` attention to a per-video-token
        score ``(B, n_video)``: language(query)->video(key) attention averaged over
        heads and over the language query rows. Returns ``None`` if there are no
        video tokens or no valid query rows (handled the same as a skipped layer).
        """
        vid = self.video_positions
        if vid is None:
            return None
        vid = vid.flatten()
        if vid.numel() == 0:
            return None
        v = vid.to(attn.device)
        S = attn.shape[-1]
        if self.query_mask is not None:
            rows = self.query_mask.to(attn.device)    # bool mask, excludes special tokens
            if not bool(rows.any()):
                return None
        else:
            # Only needed when no query mask is set; compute lazily to avoid a
            # per-layer GPU->CPU sync (.item()) in the common masked path.
            end = int(v.max().item()) + 1  # first position after the (contiguous) video block
            rows = torch.arange(end, S, device=attn.device)  # language (post-video) queries
            if rows.numel() == 0:
                return None
        block = attn[:, :, rows][:, :, :, v]          # (B, heads, |rows|, n_video)
        return block.mean(dim=1).mean(dim=1).float()  # (B, n_video)

    def _make_stop_hook(self):
        def hook(module, inputs, output):
            raise _StopForward
        return hook

    def captured_attentions(self) -> Optional[list]:
        """Captured per-layer reduced scores as a layer-indexed list (None where
        not captured), consumed by ``scores_from_attentions``. Each entry is the
        ``(B, n_video)`` score from :meth:`_reduce`, not the full attention map."""
        if not self._captured:
            return None
        n = max(self._captured) + 1
        return [self._captured.get(i) for i in range(n)]

    def truncated_forward(self, **forward_kwargs) -> Optional[list]:
        """Run the VLM forward but abort right after ``max(Layers)``.

        Captures attention at ``self.Layers`` via hooks, reducing each layer to a
        ``(B, n_video)`` score on the fly (the full ``(B, heads, S, S)`` maps are
        never retained), and returns them as a layer-indexed list. ``forward_kwargs``
        must include ``output_attentions=True`` so the eager attention modules emit
        weights.
        """
        self._captured = {}
        layers = self._decoder_layers()
        last = max(self.Layers)
        hooks = []
        for L in self.Layers:
            hooks.append(layers[L].self_attn.register_forward_hook(self._make_capture_hook(L)))
        # Fires after the last needed decoder layer completes -> stop the forward.
        hooks.append(layers[last].register_forward_hook(self._make_stop_hook()))
        try:
            self.model(**forward_kwargs)
        except _StopForward:
            pass
        finally:
            for h in hooks:
                h.remove()
        return self.captured_attentions()

    def set_sample(self, input_ids: torch.Tensor, video_token_id: int, special_ids: set):
        """Compute ``video_positions`` and the language-query mask for one sample.

        Call once per sample before the forward pass. The query mask covers all
        positions after the (contiguous) video block, excluding special tokens
        (e.g. ``<|im_end|>``, padding) so they don't pollute the attention scores.
        """
        ids = input_ids[0]
        self.video_positions = (ids == video_token_id).nonzero(as_tuple=False).flatten()
        if self.video_positions.numel() == 0:
            self.query_mask = None
            return
        vid_end = int(self.video_positions.max().item()) + 1
        positions = torch.arange(ids.numel(), device=ids.device)
        is_special = torch.isin(
            ids, torch.tensor(list(special_ids), device=ids.device)
        )
        self.query_mask = (positions >= vid_end) & (~is_special)

    def scores_from_attentions(self, attentions, normalise: bool = True) -> Optional[torch.Tensor]:
        """Aggregate the per-layer language->video scores into a per-video-token score.

        ``attentions`` is the list returned by :meth:`truncated_forward` -- one
        reduced ``(B, n_video)`` score tensor per captured critical layer (the
        per-layer head/row averaging now happens in the capture hook, see
        :meth:`_reduce`, so the full ``(B, heads, S, S)`` maps are never retained).
        We average across layers and, when ``normalise``, min-max normalise per
        sample. Returns ``(B, n_video)`` or ``None``.

        Set ``normalise=False`` to get the raw scores -- the ListMLE target uses
        the normalised version, but EVT diagnostics (e.g. the Pareto tail index)
        need the un-scaled heavy-tailed distribution.
        """
        if attentions is None:
            return None
        scores = [a for a in attentions if a is not None]  # already (B, n_video) per layer
        if not scores:
            return None
        scores = torch.stack(scores, dim=0).mean(dim=0)   # (B, n_video)
        if not normalise:
            return scores.detach()
        scores = (scores - scores.min(dim=-1, keepdim=True).values) / \
                 (scores.max(dim=-1, keepdim=True).values -
                  scores.min(dim=-1, keepdim=True).values + 1e-8)
        return scores.detach()
