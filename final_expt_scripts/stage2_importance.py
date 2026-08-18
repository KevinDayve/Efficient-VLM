"""
stage2_importance.py -- stage two of the two-stage method: score the surviving visual
tokens by text-to-visual attention at one decoder layer, from ONE query row, rebuilt
from that layer's W_Q and W_K rather than read off a materialised attention matrix.

Why it is rebuilt rather than read
----------------------------------
FlashAttention never materialises the attention matrix, so a decoder-side scorer cannot
read one out of it. It does not need to. The one row that matters can be rebuilt from
parts that are always available: take the query token's hidden state at the pruning
layer, push it through that layer's W_Q, take the visual tokens through W_K, and do the
softmax by hand. That is a (1 x d) @ (d x N) matmul, and during prefill the keys are
already in the KV cache, so the only new work is computing q for a single token. The
real forward pass runs on FlashAttention untouched and the scoring happens beside it.

This is why the score comes from one query row rather than an average over the whole
instruction. One row keeps the recompute trivial. It is also why the analysis scorer --
five layers averaged, every text position after the visual block -- is not the method:
the scores at layers 12 through 15 do not exist until those layers have run on the full
token set, so a band-averaged scorer cannot prune before 15, which is where its saving
would have gone.

    prune at 14:  (14 + 14 * rho) / 28  =  55%  at rho = 0.1
    prune at 15:  (15 + 13 * rho) / 28  =  58%  at rho = 0.1

Layer 14 of 28 is taken rather than tuned. VScan publishes k=14 for Qwen2.5-VL-7B, both
target backbones have 28 decoder layers, and 14 sits inside the 11-15 band the attention
knockout identifies. Taking the published layer removes a degree of freedom a reviewer
could otherwise call tuning.

What this module does and does not do
-------------------------------------
It computes the score and turns it into a keep set. It does NOT do the dropping --
that is `stage_topk_accuracy.attach_midforward_drop`, which is already written and
already handles slicing the causal mask, the position ids, `cache_position` and
`position_embeddings` consistently across every layer above the prune point. Reusing it
means the two-stage runs and the single-stage top-K runs prune through the identical
mechanism, so their numbers sit on one scale.

Exactness, and the one place the cheap form differs
---------------------------------------------------
`score_from_row` reproduces the eager attention weights at the same layer and row to
floating-point tolerance -- it applies the layer's own `input_layernorm`, its own
projections, and the library's own RoPE function to the layer's own `position_embeddings`
rather than a reimplementation that could drift. `verify_against_eager` is that check,
and the runner asserts it before trusting any pruned number. If it fails, stage two is
not measuring the quantity the analysis measured and every downstream number describes
something else.

    norm="all"     softmax over every key, then read the visual columns. This is the
                   quantity the analysis measured, and the one the gate checks. Cost is
                   O(S*d), and S = O(N) whenever visual tokens dominate the sequence,
                   which on a 16-frame clip they do by an order of magnitude.
    norm="visual"  softmax over the visual keys only -- the strict O(N*d) form.

The two are NOT rank-equivalent, and it is worth being precise about why, because the
obvious argument that softmax is monotone is wrong here. Within a single head it is
monotone and the two forms rank identically. But the score averages over heads, and each
head's softmax carries its own denominator: restricting the support changes each head's
denominator by a different factor, which reweights the heads against each other in the
average. So norm="visual" is a genuine approximation, not a free simplification, and
whether it costs anything is a measurement rather than an assumption.

Causality
---------
The query row must sit after every visual token, or the ones following it are masked out
of the real attention and the rebuilt row would score tokens the model cannot see. The
default row is the last position -- the answer slot, which is also the only row a
deployed prefill has finished computing when it reaches the prune point. `score_from_row`
asserts this rather than trusting it.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1. Capturing a layer's input without running the rest of the stack
# --------------------------------------------------------------------------- #
class _StopForward(Exception):
    """Raised by the capture hook to abandon the forward once layer k's input is in
    hand. Layers k+1..N-1 never run, so the capture costs a partial prefill rather than
    a whole one."""


def capture_layer_input(text_model, layer_idx: int, forward_fn) -> dict:
    """Run `forward_fn()` and return layer `layer_idx`'s inputs, stopping there.

    Returns {"hidden_states": (1,S,D), "position_embeddings": (cos, sin)} -- everything
    `score_from_row` needs. The forward is abandoned by exception, which is safe here
    because nothing is being trained and no cache is being written (`use_cache=False`).

    In deployment this capture does not exist: the score is computed inside the single
    prefill, at the moment layer k receives its input. It is a separate pass here only
    so that the scoring pass and the pruned pass can be compared against each other.
    """
    grabbed = {}

    def pre(module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        grabbed["hidden_states"] = h
        grabbed["position_embeddings"] = kwargs.get("position_embeddings")
        raise _StopForward

    handle = text_model.layers[layer_idx].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        forward_fn()
    except _StopForward:
        pass
    finally:
        handle.remove()

    if "hidden_states" not in grabbed:
        raise RuntimeError(f"layer {layer_idx} never ran -- nothing captured")
    if grabbed.get("position_embeddings") is None:
        raise RuntimeError(
            "this transformers version does not hand decoder layers their "
            "position_embeddings; the rebuilt row cannot apply RoPE correctly")
    return grabbed


# --------------------------------------------------------------------------- #
# 2. The rebuilt attention row
# --------------------------------------------------------------------------- #
def _apply_rope(attn, q, k, cos_q, sin_q, cos_k, sin_k):
    """Apply the layer's own RoPE to q and k, using the library's own implementation.

    Qwen2.5-VL splits the channel dimension into temporal / height / width sections
    (mRoPE); the Qwen2 text model behind LLaVA-OneVision uses plain 1D RoPE. Which one
    applies is read off the attention module rather than passed in, so a backbone whose
    text tower changes cannot silently get the wrong rotation.

    q and k carry DIFFERENT sequence positions here -- one query row against N visual
    keys -- so they cannot share a cos/sin and are rotated in separate calls. The
    library function rotates a q/k pair, so each call passes its tensor twice and takes
    only the first output. The duplicated argument is deliberate: it keeps the rotation
    the library's own code rather than a reimplementation of it that could drift from
    the forward the model actually runs.
    """
    ms = _mrope_section(attn)
    if ms is not None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb as rope)
        return rope(q, q, cos_q, sin_q, ms)[0], rope(k, k, cos_k, sin_k, ms)[0]
    # A 3D mRoPE cos is (3, B, S, d) against the 1D form's (B, S, d). Handing the former
    # to the 1D rotation does not fail -- it broadcasts, and the caller only notices three
    # frames later when an unpack sees five axes. Refuse it here, where the cause is legible.
    if cos_k.dim() == 4:
        raise RuntimeError(
            f"position_embeddings are 3D mRoPE {tuple(cos_k.shape)} but no mrope_section "
            f"was found on {type(attn).__name__} or its config; the rebuilt row would be "
            "rotated as plain 1D RoPE and would not be the quantity the model computes")
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as rope
    return rope(q, q, cos_q, sin_q)[0], rope(k, k, cos_k, sin_k)[0]


def _mrope_section(attn):
    """The mrope_section for this attention module, or None if its RoPE is plain 1D.

    Qwen2.5-VL splits the channel dimension into temporal / height / width sections; the
    Qwen2 text model behind LLaVA-OneVision does not. Where that split is recorded has
    moved across transformers versions -- `config.rope_scaling` became
    `config.rope_parameters` -- and it lives on the CONFIG, not on the attention module,
    so a single getattr on `attn` silently reports "no mrope" for every Qwen build.
    """
    for holder in (attn, getattr(attn, "config", None)):
        for name in ("rope_parameters", "rope_scaling"):
            cfg = getattr(holder, name, None)
            if isinstance(cfg, dict) and "mrope_section" in cfg:
                return cfg["mrope_section"]
    return None


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, H_kv, S, d) -> (B, H_kv * n_rep, S, d). Inlined rather than imported: the
    function's home module has moved between transformers versions, and it is three
    lines."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@torch.no_grad()
def score_from_row(layer, hidden_states: torch.Tensor,
                   position_embeddings: tuple[torch.Tensor, torch.Tensor],
                   visual_idx: torch.Tensor, query_row: int = -1,
                   norm: str = "all") -> torch.Tensor:
    """(N,) attention mass each visual token receives from ONE query row at this layer.

    Args:
        layer:                the decoder layer to score at (the pruning layer).
        hidden_states:        (1, S, D) that layer's INPUT -- pre-layernorm, as the
                              layer itself receives it.
        position_embeddings:  the (cos, sin) the layer was handed, so the rebuilt row
                              is rotated exactly as the real one will be.
        visual_idx:           (N,) absolute positions of the visual tokens.
        query_row:            which row to score from; -1 (default) is the last
                              position, the answer slot.
        norm:                 "all" softmaxes over every key then reads the visual
                              columns; "visual" softmaxes over the visual keys alone.
                              See the module docstring -- these are not rank-equivalent.

    Returns a float32 CPU tensor, head-averaged, matching the convention the rest of
    this family scores with.
    """
    attn = layer.self_attn
    S = hidden_states.shape[1]
    qrow = query_row + S if query_row < 0 else query_row
    if not 0 <= qrow < S:
        raise ValueError(f"query row {query_row} outside a length-{S} sequence")
    if qrow < int(visual_idx.max()):
        raise ValueError(
            f"query row {qrow} sits before visual token {int(visual_idx.max())}; under a "
            "causal mask that token is invisible to this row and scoring it is meaningless")

    head_dim = attn.head_dim
    scaling = getattr(attn, "scaling", head_dim ** -0.5)

    h = layer.input_layernorm(hidden_states)          # the layer's own norm, not a copy
    # Take the device from h, not from what we were handed. Under device_map="auto" the
    # caller's hidden_states come from a forward PRE-hook on this layer, and accelerate
    # aligns devices inside the patched forward -- which runs after pre-hooks -- so the
    # captured tensor is still on the previous layer's GPU. The norm above is the first
    # hooked submodule to touch it, so h is the tensor already on this layer's execution
    # device, and every index and rotary tensor below has to follow h rather than it.
    dev = h.device
    visual_idx = visual_idx.to(dev)
    cos, sin = (t.to(dev) for t in position_embeddings)
    key_idx = (torch.arange(S, device=dev) if norm == "all" else visual_idx)

    q = attn.q_proj(h[:, qrow:qrow + 1, :]).view(1, 1, -1, head_dim).transpose(1, 2)
    k = attn.k_proj(h.index_select(1, key_idx)).view(
        1, key_idx.numel(), -1, head_dim).transpose(1, 2)

    # cos/sin carry the sequence on axis -2 for both the 3D mRoPE (3, B, S, d) and the
    # 1D form (B, S, d), which is the same axis attach_midforward_drop slices them on.
    qi = torch.tensor([qrow], device=dev)
    q, k = _apply_rope(attn, q, k,
                       cos.index_select(-2, qi), sin.index_select(-2, qi),
                       cos.index_select(-2, key_idx), sin.index_select(-2, key_idx))

    k = _repeat_kv(k, attn.num_key_value_groups)
    logits = (q.float() @ k.float().transpose(-1, -2)) * scaling      # (1, H, 1, n_keys)
    p = torch.softmax(logits, dim=-1)[0, :, 0, :]                     # (H, n_keys)

    if norm == "all":
        p = p.index_select(1, visual_idx)
    elif norm != "visual":
        raise ValueError(f"unknown norm {norm!r}; choose 'all' or 'visual'")
    return p.mean(dim=0).float().cpu()


# --------------------------------------------------------------------------- #
# 3. Keep set
# --------------------------------------------------------------------------- #
def stage2_keep(scores: torch.Tensor | np.ndarray, rho: float) -> np.ndarray:
    """Top-K local indices by score, ASCENDING. K = max(1, round(rho * N)).

    Sorted rather than left in rank order so the surviving block stays in sequence
    order and position ids stay monotone -- the same convention stage one returns.
    """
    s = scores.numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    N = s.size
    K = int(max(1, min(N, round(rho * N))))
    return np.sort(np.argsort(-s, kind="stable")[:K])


# --------------------------------------------------------------------------- #
# 4. The gate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_against_eager(rebuilt: torch.Tensor, eager_row: torch.Tensor,
                         visual_idx: torch.Tensor, atol: float = 2e-3) -> dict:
    """Does the hand-rebuilt row equal the row the model actually computed?

    Args:
        rebuilt:    (N,) the output of `score_from_row(..., norm="all")`.
        eager_row:  (H, S) the layer's eager attention weights for the SAME query row,
                    already sliced out of the (H, S, S) tensor by the caller's hook --
                    the full matrix is hundreds of MB and only this row is needed.
        visual_idx: (N,) absolute positions of the visual tokens.

    Returns a dict with the max absolute difference, the top-10% rank overlap, and a
    pass flag. The runner asserts on `ok` before any pruned number is reported.

    bf16 hidden states make an exact match impossible, hence a tolerance rather than
    `allclose` at machine epsilon. The ranking agreement is the check that actually
    matters, because the ranking is the only thing the method consumes -- a uniform
    scale error would leave every keep set identical.
    """
    ref = eager_row.float().mean(dim=0)[visual_idx.to(eager_row.device)].cpu()
    a, b = rebuilt.float(), ref.float()
    max_abs = float((a - b).abs().max())
    N = a.numel()
    k = max(1, N // 10)
    top_a = set(torch.argsort(-a)[:k].tolist())
    top_b = set(torch.argsort(-b)[:k].tolist())
    overlap = len(top_a & top_b) / k
    return {"max_abs_diff": max_abs, "top10pct_overlap": overlap,
            "n_visual": int(N), "ok": bool(max_abs <= atol and overlap >= 0.98)}


def average_retention(rho1: float, rho2: float, prune_layer: int, n_layers: int) -> dict:
    """Visual retention averaged over ALL decoder layers -- the quantity baselines are
    matched on, rather than the final token count.

        rho1 * (k + (N - k) * rho2) / N

    Stage one is input-side, so its saving applies to every layer; stage two only
    applies from the prune layer up. Reporting the final token count instead would
    credit stage two with a saving it does not make in layers 0..k-1.

    `stage1` is reported separately because whether it runs inside the vision tower
    (saving encoder compute too) or after it (saving decoder compute only) is a
    deployment choice this number does not capture.
    """
    k, N = prune_layer, n_layers
    return {"stage1": rho1,
            "stage2": rho2,
            "decoder_layer_fraction": (k + (N - k) * rho2) / N,
            "average_visual_retention": rho1 * (k + (N - k) * rho2) / N}
