"""FastV inference-time visual-token pruning for Qwen2.5-VL (Chen et al., 2024).

FastV ("An Image is Worth 1/2 Tokens After Layer 2", arXiv:2403.06764) is a
plug-and-play, *training-free* pruning method. It is fundamentally different from
this repo's learned scorer, which drops tokens **before** the LLM (pre-projector,
section 2.5). FastV instead drops them **inside** the LLM:

    * Layers ``0 .. K``       run on the full sequence.
    * At layer ``K``, rank the visual tokens by the attention they *receive* from
      the last query position (head-averaged), keep the top ``rho`` fraction.
    * Layers ``K+1 .. end``   run on the shortened sequence (kept visual tokens +
      all text tokens). Surviving tokens keep their original M-RoPE positions.

The criterion ``phi_attn`` is the attention each visual token receives. The paper
text says "averaged over all other tokens"; the official implementation uses the
**last token's** attention row, which is also the most relevant signal for our
single-forward MC readout (the last position is the one that produces the answer
logit). We follow the official implementation.

Why a monkey-patch instead of editing the vendored ``transformers`` fork: the
decoder-layer loop we must intervene in lives in ``Qwen2_5_VLTextModel.forward``,
which is *generated* and not expressed in ``modular_qwen2_5_vl.py`` -- a
``make fix-repo`` would revert any edit there. Keeping FastV here makes it
self-contained, reviewable, and reversible. :func:`install_fastv` swaps the text
model's ``forward`` for :func:`_fastv_text_forward` (behaviourally identical to the
original whenever ``fastv_k is None``), and the caller flips the per-forward knobs.

Usage (single forward, no KV-cache -- prefill only; see accuracy_mvbench_fastv.py)::

    from efficient_vlm.fastv import install_fastv, set_fastv, set_visual_mask
    tm = install_fastv(model)                 # patch text-model forward (idempotent)
    model.disable_token_gating()              # FastV and the scorer must not both prune
    set_visual_mask(tm, inputs["input_ids"][0] == model.config.video_token_id)
    set_fastv(tm, k=2, keep_ratio=0.5)        # or set_fastv(tm, None) for the full model
    logits = model(**inputs).logits[0, -1]
"""

from __future__ import annotations

import types

import torch

from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _m


# --------------------------------------------------------------------------- #
# Token ranking at layer K
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _fastv_keep_indices(decoder_layer, hidden_states, position_embeddings, visual_idx, keep_ratio):
    """Sequence indices (sorted ascending) to keep for layers after ``K``.

    Recomputes layer ``K``'s last-token attention row directly from this layer's
    input ``hidden_states`` -- exactly the attention layer ``K`` itself would
    compute -- so we never need to plumb ``output_attentions`` through the
    refactored attention stack, and the result is identical for sdpa / flash /
    eager. The last token attends causally to every position, so its row needs no
    masking. All non-visual tokens are always kept; only visual tokens compete for
    the ``keep_ratio`` budget (FastV ranks them globally, not per frame).
    """
    attn = decoder_layer.self_attn
    seq_len = hidden_states.shape[1]
    hn = decoder_layer.input_layernorm(hidden_states)

    q = attn.q_proj(hn).view(1, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hn).view(1, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = _m.apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.config.rope_parameters["mrope_section"])
    k = _m.repeat_kv(k, attn.num_key_value_groups)  # (1, num_heads, seq, head_dim)

    q_last = q[:, :, -1:, :]                                    # (1, H, 1, head_dim)
    scores = torch.matmul(q_last, k.transpose(-1, -2)) * attn.scaling  # (1, H, 1, seq)
    scores = scores.float().softmax(dim=-1)
    recv = scores.mean(dim=1).flatten()                        # (seq,) head-averaged

    n_visual = visual_idx.numel()
    k_keep = min(n_visual, max(1, int(round(keep_ratio * n_visual))))
    top = torch.topk(recv[visual_idx], k_keep).indices
    keep_visual = visual_idx[top]

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
    keep_mask[visual_idx] = False
    keep_mask[keep_visual] = True
    return keep_mask.nonzero(as_tuple=False).flatten()


# --------------------------------------------------------------------------- #
# FastV-aware copy of Qwen2_5_VLTextModel.forward
# --------------------------------------------------------------------------- #
def _fastv_text_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """Drop-in replacement for ``Qwen2_5_VLTextModel.forward`` with FastV pruning.

    The preamble (cache, M-RoPE position handling, mask construction, rotary
    embeddings) is copied verbatim from the original so the *full-model* path --
    taken whenever ``self.fastv_k is None`` -- is bit-identical. The only change is
    inside the layer loop: at layer ``K`` we select the visual tokens to keep, run
    layer ``K`` on the full sequence, then shrink ``hidden_states`` / position
    embeddings / attention masks for every later layer.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = _m.DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    masks_prebuilt = isinstance(attention_mask, dict)  # True only on the .generate() path
    if not masks_prebuilt:
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {"full_attention": _m.create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = _m.create_sliding_window_causal_mask(**mask_kwargs)
    else:
        causal_mask_mapping = attention_mask
    attn_mask_2d = attention_mask  # raw (1, S) padding mask (or None); for rebuilding after a prune

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # ----- FastV setup (inert unless a budget is configured for this forward) ---
    fastv_k = getattr(self, "fastv_k", None)
    keep_ratio = getattr(self, "fastv_keep_ratio", None)
    visual_mask = getattr(self, "_fastv_visual_mask", None)
    fastv_active = (
        fastv_k is not None
        and keep_ratio is not None
        and visual_mask is not None
        and inputs_embeds.shape[0] == 1
        and 0 <= fastv_k < len(self.layers)
        and not masks_prebuilt  # prepared-mask path (.generate()) can't be re-sliced
    )
    visual_idx = None
    if fastv_active:
        vm = visual_mask.to(hidden_states.device).flatten()
        if vm.numel() == hidden_states.shape[1] and bool(vm.any()):
            visual_idx = vm.nonzero(as_tuple=False).flatten()
        else:
            fastv_active = False

    for i, decoder_layer in enumerate(self.layers):
        if fastv_active and i == fastv_k:
            keep_idx = _fastv_keep_indices(
                decoder_layer, hidden_states, position_embeddings, visual_idx, keep_ratio
            )

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
            position_embeddings=position_embeddings,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if fastv_active and i == fastv_k:
            self._fastv_last_kept = int(keep_idx.numel())
            hidden_states = hidden_states[:, keep_idx, :]
            cos, sin = position_embeddings
            position_embeddings = (cos[:, :, keep_idx, :], sin[:, :, keep_idx, :])
            if text_position_ids is not None:
                text_position_ids = text_position_ids[:, keep_idx]
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attn_mask_2d[:, keep_idx] if attn_mask_2d is not None else None,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {"full_attention": _m.create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = _m.create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = self.norm(hidden_states)
    return _m.BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


# --------------------------------------------------------------------------- #
# Install / configure
# --------------------------------------------------------------------------- #
def _text_model(model):
    """The Qwen2_5_VLTextModel, reachable from either the LM-head wrapper or the base model."""
    base = model.model if hasattr(model, "model") and hasattr(model.model, "language_model") else model
    return base.language_model


def install_fastv(model):
    """Patch the text model's ``forward`` for FastV (idempotent). Returns the text model.

    The patched forward reproduces the full model exactly while ``fastv_k is None``,
    so it is safe to install once and toggle FastV per forward with :func:`set_fastv`.
    """
    tm = _text_model(model)
    if getattr(tm, "_fastv_orig_forward", None) is None:
        tm._fastv_orig_forward = tm.forward
        tm.forward = types.MethodType(_fastv_text_forward, tm)
        tm.fastv_k = None
        tm.fastv_keep_ratio = None
        tm._fastv_visual_mask = None
    return tm


def uninstall_fastv(model):
    """Restore the original text-model forward."""
    tm = _text_model(model)
    if getattr(tm, "_fastv_orig_forward", None) is not None:
        tm.forward = tm._fastv_orig_forward
        tm._fastv_orig_forward = None
    return tm


def set_fastv(text_model, k, keep_ratio=None):
    """Configure FastV for subsequent forwards. ``k=None`` runs the full model."""
    text_model.fastv_k = k
    text_model.fastv_keep_ratio = keep_ratio
    return text_model


def set_visual_mask(text_model, mask):
    """Set the boolean visual-token mask (length = full sequence) for the next forward."""
    text_model._fastv_visual_mask = mask
    return text_model
