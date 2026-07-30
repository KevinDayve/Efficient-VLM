"""
anchor_layer_prune.py -- Anchor-Layer Importance-Diversity Token Reduction for VLMs.
============================================================================================
Implements Efficient_VLMs_Method.pdf end to end. A single anchor layer L* is chosen
dynamically per-sample, K visual tokens are kept at L* under a combined
importance-diversity objective, and the rest are DROPPED from the sequence for every
deeper layer -- a real reduction in attention/KV cost, not just a scoring exercise.

Two backbones (--backbone, auto-detected from --model_name): Qwen2.5-VL (dynamic
resolution, mRoPE, Qwen2_5_VLTextModel) and LLaVA-OneVision / LLaVA-Video (real
multi-frame video, 1D RoPE, Qwen2Model text backbone -- LLaVA-Video-7B-Qwen2 shares this
architecture). Both are video backbones; the method itself -- Sections 1-7 below -- is
fully backbone-agnostic, and only grid construction, model loading, and the real-prune
monkey-patch's target class differ.

Pipeline (PDF Sections 3-8):
  1. Dense forward through a shallow layer band B, with output_attentions +
     output_hidden_states (this is the one pass that "buys" everything below).
  2. Importance: value-norm debiased attention score s_t^(l) for every l in B
     (Section 4) -- sinks (high attention, ~zero value norm) are scaled down for free.
  3. Anchor L*: the layer in B whose s_t has the heaviest upper tail, via the sign-aware
     Dekkers-Einmahl-de Haan moment estimator (Section 5; reuses efficient_vlm.utils
     .einmahlHaan, the same estimator already used by pareto_budget elsewhere here).
  4. Diversity: a cross-shaped spatio-temporal redundancy R(t;S) = max(R_spatial,
     R_temporal), the temporal arm gated by a per-token stability scalar so static
     background gets temporally subsampled while motion survives (Section 6).
  5. Greedy MMR selection of K tokens at L* under s_t - lambda(rho)*R(t;S) (Section 7,
     Algorithm 1).
  6. Real prune: a FastV-style (efficient_vlm/fastv.py) monkey-patch of the text model's
     forward restricts layers L*+1..end to the kept K tokens (Section 8).

Run a demo on one video + query (Qwen2.5-VL):
    python anchor_layer_prune.py --video /path/to/clip.mp4 --query "What is happening?" \
        --rho 0.1 --band 2,3,4,5,6,7,8

Run a demo on one video + query (LLaVA-OneVision / LLaVA-Video, REAL multi-frame video --
the temporal arm applies exactly as it does for Qwen):
    python anchor_layer_prune.py --backbone llava_video --video /path/to/clip.mp4 \
        --query "What is happening?" --num_frames 8 --rho 0.1

Run the (fast, no model/GPU) correctness self-checks:
    python anchor_layer_prune.py --self_test
"""

from __future__ import annotations

import os
import sys
import types
import argparse

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from efficient_vlm.utils import einmahlHaan, hill_tail_index          # Section 5 estimator

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLAVA_VIDEO_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


def resolve_backbone(backbone, model_name):
    """'auto' -> infer from the model name: 'llava'/'onevision'/'video' substring
    => the LLaVA-OneVision/LLaVA-Video backbone; else qwen."""
    if backbone != "auto":
        return backbone
    name = (model_name or "").lower()
    return "llava_video" if any(k in name for k in ("llava", "onevision", "video")) else "qwen"


def default_model_id(backbone):
    return LLAVA_VIDEO_MODEL_ID if backbone == "llava_video" else QWEN_MODEL_ID


# --------------------------------------------------------------------------- #
# 1. Grid positions p(t) = (frame, row, col)  (Section 6.2)
# --------------------------------------------------------------------------- #
def token_grid(video_grid_thw, merge_size):
    """(frame, row, col) LongTensors of length M = T*S, plus (T, S).

    Qwen2.5-VL pairs adjacent frames (temporal_patch_size=2) and merges 2x2
    spatially, so T = video_grid_thw[0,0] is already the finest temporal unit the
    model exposes, and S = (H/merge)*(W/merge) tokens are laid out row-major
    within each frame group."""
    T = int(video_grid_thw[0, 0])
    Hs = int(video_grid_thw[0, 1]) // merge_size
    Ws = int(video_grid_thw[0, 2]) // merge_size
    S = Hs * Ws
    device = video_grid_thw.device
    idx = torch.arange(T * S, device=device)
    f = idx // S
    within = idx % S
    r = within // Ws
    c = within % Ws
    return f, r, c, T, S


def token_grid_video(num_frames, pooled_side, device):
    """(frame, row, col, T, S) for LLaVA-OneVision/LLaVA-Video real multi-frame
    video: T=num_frames, S=pooled_side^2, tokens laid out frame-major (matching
    LlavaOnevisionModel.get_video_features' own
    `reshape(batch, frames * pooled_h*pooled_w, -1)`). The model appends ONE
    extra "image_newline" token after ALL frame tokens (see
    LlavaOnevisionModel.forward: `torch.cat((video_features, image_newline), dim=1)`,
    matching the processor's `num_video_tokens = frames*pooled*pooled + 1`) --
    that trailing position is NOT part of this grid and must be excluded from the
    candidate set by the caller (see `run_demo_llava_video`), the same positional
    guard Section 4 uses for BOS/register tokens."""
    S = pooled_side * pooled_side
    idx = torch.arange(num_frames * S, device=device)
    f = idx // S
    within = idx % S
    r = within // pooled_side
    c = within % pooled_side
    return f, r, c, num_frames, S


# --------------------------------------------------------------------------- #
# 2. Stability gate g_t = cos(P h_t, P h_prev(t))  (Section 6.4)
# --------------------------------------------------------------------------- #
# def stability_gates(feats, f, S, proj_dim=64, seed=0):
#     """g_t in [0,1]: how much visual token t changed from the SAME grid cell one
#     frame back. Because tokens are laid out (T,S) row-major, prev(t) = t - S.
#     g_t ~ 1 for a static patch (temporal penalty fires later); g_t ~ 0 for a
#     moving one (penalty vanishes). Frame-0 tokens have no predecessor -> g=0
#     (no information to penalize on, so they never get temporally suppressed
#     from the FIRST frame alone).

#     Negative cosine (token rotated away from its predecessor, not just changed
#     magnitude) is clamped to 0: it must never turn the temporal arm into a
#     *reward* for keeping a "duplicate"."""
#     N = feats.shape[0]
#     g = torch.zeros(N, device=feats.device, dtype=feats.dtype)
#     has_prev = f >= 1
#     if not bool(has_prev.any()):
#         return g
#     gen = torch.Generator(device="cpu").manual_seed(seed)
#     P = torch.randn(feats.shape[1], proj_dim, generator=gen).to(feats.device, feats.dtype)
#     proj = feats.float() @ P.float()                       # (N, d')
#     idx = torch.nonzero(has_prev, as_tuple=False).flatten()
#     cur = proj[idx]
#     prev = proj[idx - S]
#     cos = F.cosine_similarity(cur, prev, dim=-1)
#     g[idx] = cos.clamp_min(0.0).to(g.dtype)
#     return g

def stability_gates(feats, f, S, proj_dim=64, seed=0, eps=1e-6):
    """g_t in [0,1]: how much visual token t's *appearance* changed from the SAME
    grid cell one frame back. High g_t (~1) = static patch (temporal penalty will
    fire, so we don't spend budget on near-duplicates across frames); low g_t (~0)
    = moving patch (penalty vanishes, we keep it because it carries new content).

    CRITICAL -- what `feats` must be:
        Pass the PRE-DECODER visual embeddings (the projector output that first
        enters the language model, i.e. base_embeds[0, video_idx]), NOT hidden
        states read from some decoder layer L*. The gate is meant to measure a
        physical question -- did the pixels in this cell move -- and only the
        pre-decoder embedding still corresponds to "what this patch looks like".
        After even a few decoder layers, each token has mixed in its neighbours',
        other frames', and the text query's information via self-attention, so
        cosine similarity there measures POST-MIXING representational similarity,
        not temporal change: a static patch can look "moved" (its context bled in
        differently) and a moving patch can look "static" (attention homogenises).
        Feeding contextualised hidden states here does not error -- it silently
        returns a gate that means the wrong thing.

    Layout: tokens are (T, S) row-major (T frames, S cells per frame), so the same
    cell one frame back is prev(t) = t - S. Frame-0 tokens have no predecessor and
    get g=0 (never temporally suppressed on the strength of the first frame alone).

    Robustness:
      * Negative cosine (token rotated away from its predecessor) is clamped to 0:
        the temporal arm must never turn into a *reward* for keeping a token.
      * Near-zero-norm patches (blank background, common in synthetic/low-detail
        video) have undefined direction; cosine_similarity would divide by ~0 and
        return noise. We treat any cell whose current OR previous embedding is
        effectively zero-norm as g=0 (no reliable appearance to compare, so don't
        temporally penalize on it).

    Args:
        feats: (N, d) pre-decoder visual embeddings, one row per visual token,
               ordered to match f (i.e. base_embeds[0, video_idx]).
        f:     (N,) long, frame index of each visual token.
        S:     int, number of grid cells per frame (so prev(t) = t - S).
        proj_dim: random-projection width; JL preserves cosine well at 64.
        seed:  fixed so the projection (and thus the gate) is deterministic
               across the whole run -- do NOT reseed per call.
        eps:   norm below which a patch is treated as contentless.

    Returns:
        g: (N,) in [0,1], same order as feats.
    """
    N = feats.shape[0]
    g = torch.zeros(N, device=feats.device, dtype=feats.dtype)

    has_prev = f >= 1                      # frame-0 tokens (f==0) have no predecessor
    # also require the predecessor index to actually exist in-range (defensive:
    # guards against any f/S mismatch rather than trusting T*S == N upstream)
    idx_all = torch.nonzero(has_prev, as_tuple=False).flatten()
    idx_all = idx_all[idx_all - S >= 0]
    if idx_all.numel() == 0:
        return g

    # fixed random projection (JL) -- built on CPU with a seeded generator so the
    # gate is identical run-to-run, then moved to the feats device/dtype.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    P = torch.randn(feats.shape[1], proj_dim, generator=gen).to(feats.device, torch.float32)
    proj = feats.float() @ P                                  # (N, proj_dim)

    cur = proj[idx_all]                                       # (m, proj_dim)
    prev = proj[idx_all - S]                                  # (m, proj_dim)

    # mask out cells with no reliable appearance in either frame (blank patches):
    # their direction is undefined, so cosine there is noise -> g stays 0.
    cur_n = cur.norm(dim=-1)
    prev_n = prev.norm(dim=-1)
    reliable = (cur_n > eps) & (prev_n > eps)
    if reliable.any():
        keep = idx_all[reliable]
        cos = F.cosine_similarity(cur[reliable], prev[reliable], dim=-1)
        g[keep] = cos.clamp_min(0.0).to(g.dtype)
    return g


# --------------------------------------------------------------------------- #
# 3. Value-norm debiased importance score s_t^(l)  (Section 4)
# --------------------------------------------------------------------------- #
def _repeat_kv(hidden_states, n_rep):
    """Backbone-agnostic copy of the standard HF repeat_kv (byte-identical in
    every modeling_*.py this script touches -- Qwen2.5-VL and Qwen2 alike), so
    `debiased_scores_by_layer` never has to import a model-specific internal
    module just for this one helper."""
    if n_rep == 1:
        return hidden_states
    b, kvh, s, hd = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, kvh, n_rep, s, hd).reshape(b, kvh * n_rep, s, hd)


@torch.no_grad()
def debiased_scores_by_layer(text_model, hidden_states, attentions, video_idx,
                              text_query_mask, layers):
    """{layer -> (M,)} debiased score for the visual tokens, for every `layer` in
    `layers`. s_t = mean over text-query positions q, mean over heads e, of
    A^(l,e)_{q,t} * ||v^(l,e)_t||. Sinks (high A, ~0 value norm) are scaled down
    with no threshold or detection -- they simply carry little value mass.

    Backbone-agnostic: only touches `v_proj`/`input_layernorm`/`head_dim`/
    `num_key_value_groups`, present under those exact names on both Qwen2.5-VL's
    and Qwen2's attention module. `num_key_value_heads` is deliberately NOT read
    as an attribute (Qwen2.5-VL sets it directly; Qwen2Attention does not --
    only `config.num_key_value_heads` exists there) -- it is derived instead
    from v_proj's own output width, which is correct for either."""
    out = {}
    for l in layers:
        layer = text_model.layers[l]
        attn = layer.self_attn
        hn = layer.input_layernorm(hidden_states[l])                 # input to layer l
        v_flat = attn.v_proj(hn)                                     # (1,S,kv_heads*head_dim)
        num_kv_heads = v_flat.shape[-1] // attn.head_dim
        v = v_flat.view(1, -1, num_kv_heads, attn.head_dim).transpose(1, 2)
        v = _repeat_kv(v, attn.num_key_value_groups)                  # (1,H,S,hd) -> query heads
        vnorm = v[0].float().norm(dim=-1)                             # (H,S)

        A = attentions[l][0].float()                                  # (H,Sq,Sk)
        recv = A[:, text_query_mask, :].mean(dim=1)                   # (H,Sk) mean over queries
        s = (recv * vnorm).mean(dim=0)                                # (Sk,) mean over heads
        out[l] = s[video_idx]
    return out


# --------------------------------------------------------------------------- #
# 4. Anchor-layer selection L*  (Section 5)
# --------------------------------------------------------------------------- #
def select_anchor_layer(scores_by_layer, k_frac=0.10, estimator="moment"):
    """Layer in the band with the heaviest upper tail of its debiased scores.
    estimator='moment' (default) is the sign-aware Dekkers-Einmahl-de Haan
    estimator (paper's lower-variance upgrade over Hill); 'hill' uses the plain
    Hill estimator instead. Returns (L*, {layer: tail_index})."""
    fn = einmahlHaan if estimator == "moment" else hill_tail_index
    taus = {l: fn(s, k_frac) for l, s in scores_by_layer.items()}
    valid = {l: g for l, g in taus.items() if g == g}      # drop NaN (too few tokens)
    if not valid:
        raise RuntimeError("no layer in the band produced a usable tail-index estimate")
    Lstar = max(valid, key=valid.get)
    return Lstar, taus


# --------------------------------------------------------------------------- #
# 5. Greedy importance-diversity selection with cross-shaped redundancy
#    (Section 6, Algorithm 1)
# --------------------------------------------------------------------------- #
def greedy_select(scores, f, r, c, gate, K, sigma_s=1.5, sigma_tau=1.0, window=2, lam=1.0):
    """Algorithm 1. Greedily grows the kept set S, each step adding
        t* = argmax_{t not in S} [ s_t - lam * R(t; S) ],
    where R(t;S) = max(R_spatial, R_temporal) is tracked as a single running
    array (max-updated by whichever arm fires), so a token's redundancy is
    always the stronger of the two arms without ever materializing them
    separately. Returns sorted LOCAL indices into `scores` (size min(K, N))."""
    device = scores.device
    N = scores.numel()
    K = min(K, N)
    s = scores.float()
    f, r, c = f.long(), r.long(), c.long()
    r_run = torch.zeros(N, device=device, dtype=torch.float32)        # running R(t;S)
    sel_mask = torch.zeros(N, dtype=torch.bool, device=device)
    order = []

    for _ in range(K):
        obj = s - lam * r_run
        obj = obj.masked_fill(sel_mask, float("-inf"))
        t = int(obj.argmax())
        order.append(t)
        sel_mask[t] = True

        # spatial arm: coverage penalty against every OTHER token in the same frame.
        same_frame = f == f[t]
        d2 = (r[same_frame] - r[t]).float() ** 2 + (c[same_frame] - c[t]).float() ** 2
        contrib = torch.exp(-d2 / (2.0 * sigma_s ** 2))
        r_run[same_frame] = torch.maximum(r_run[same_frame], contrib)

        # temporal arm: duplication penalty against the same grid cell within +/- window
        # frames, gated by how STABLE that cell is (gate ~1 static, ~0 moving).
        same_cell = (r == r[t]) & (c == c[t]) & ((f - f[t]).abs() <= window)
        dtau = (f[same_cell] - f[t]).float().abs()
        contrib_t = gate[same_cell] * torch.exp(-dtau ** 2 / (2.0 * sigma_tau ** 2))
        r_run[same_cell] = torch.maximum(r_run[same_cell], contrib_t)

    return torch.tensor(sorted(order), device=device, dtype=torch.long)


# --------------------------------------------------------------------------- #
# 6. Real in-LLM prune: FastV-style monkey-patch of the text model's forward
#    (Section 8). Mirrors efficient_vlm/fastv.py's pattern, but the keep set is
#    computed ONCE up front (Sections 1-5 above) rather than recomputed inside
#    the patched layer loop, since our criterion needs the FULL attention band,
#    not just the last query row.
# --------------------------------------------------------------------------- #
def _text_model(model):
    base = model.model if hasattr(model, "model") and hasattr(model.model, "language_model") else model
    return base.language_model


def _anchor_prune_text_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    """Drop-in replacement for Qwen2_5_VLTextModel.forward. Identical to the
    original whenever `self.anchor_layer` is None; otherwise restricts every
    layer after `self.anchor_layer` to `self.anchor_keep_idx` (absolute sequence
    positions, precomputed by `plan()`)."""
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _m

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

    masks_prebuilt = isinstance(attention_mask, dict)
    if not masks_prebuilt:
        mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds,
                       "attention_mask": attention_mask, "past_key_values": past_key_values,
                       "position_ids": text_position_ids}
        causal_mask_mapping = {"full_attention": _m.create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = _m.create_sliding_window_causal_mask(**mask_kwargs)
    else:
        causal_mask_mapping = attention_mask
    attn_mask_2d = attention_mask

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    anchor_layer = getattr(self, "anchor_layer", None)
    keep_idx = getattr(self, "anchor_keep_idx", None)
    active = (anchor_layer is not None and keep_idx is not None
              and inputs_embeds.shape[0] == 1 and 0 <= anchor_layer < len(self.layers)
              and not masks_prebuilt)

    for i, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states, attention_mask=causal_mask_mapping[self.config.layer_types[i]],
            position_embeddings=position_embeddings, position_ids=text_position_ids,
            past_key_values=past_key_values, use_cache=use_cache, **kwargs)

        if active and i == anchor_layer:
            hidden_states = hidden_states[:, keep_idx, :]
            cos, sin = position_embeddings
            position_embeddings = (cos[:, :, keep_idx, :], sin[:, :, keep_idx, :])
            if text_position_ids is not None:
                text_position_ids = text_position_ids[:, keep_idx]
            mask_kwargs = {"config": self.config, "inputs_embeds": hidden_states,
                           "attention_mask": attn_mask_2d[:, keep_idx] if attn_mask_2d is not None else None,
                           "past_key_values": past_key_values, "position_ids": text_position_ids}
            causal_mask_mapping = {"full_attention": _m.create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = _m.create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = self.norm(hidden_states)
    return _m.BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def _anchor_prune_plain_lm_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                                    past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    """Drop-in replacement for a plain 1D-RoPE causal LM's forward -- currently
    only Qwen2Model (LLaVA-OneVision/LLaVA-Video's text backbone), but written
    generically (any decoder-only 1D-RoPE LM with the same layer-loop shape would
    work identically) rather than hardcoded to one class. Same contract as
    `_anchor_prune_text_forward` (identical when `self.anchor_layer` is None),
    but simpler than the Qwen2.5-VL mRoPE patch: plain (1,S) 1D RoPE (no 3-way
    split), and no prebuilt-mask dict case to handle (this repo never runs this
    backbone through `.generate()`).

    Resolves `create_causal_mask`/`DynamicCache`/`BaseModelOutputWithPast` from
    THIS model's own modeling module (`type(self).__module__`) rather than a
    hardcoded import, so the same function would serve any such backbone."""
    import importlib
    _m = importlib.import_module(type(self).__module__)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None:
        past_key_values = _m.DynamicCache(config=self.config)
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    has_sliding = getattr(self, "has_sliding_layers", False)
    mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds,
                   "attention_mask": attention_mask, "past_key_values": past_key_values,
                   "position_ids": position_ids}
    causal_mask_mapping = {"full_attention": _m.create_causal_mask(**mask_kwargs)}
    if has_sliding:
        causal_mask_mapping["sliding_attention"] = _m.create_sliding_window_causal_mask(**mask_kwargs)
    attn_mask_2d = attention_mask

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    anchor_layer = getattr(self, "anchor_layer", None)
    keep_idx = getattr(self, "anchor_keep_idx", None)
    active = (anchor_layer is not None and keep_idx is not None
             and inputs_embeds.shape[0] == 1 and 0 <= anchor_layer < len(self.layers))

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_type = self.config.layer_types[i] if has_sliding else "full_attention"
        hidden_states = decoder_layer(
            hidden_states, attention_mask=causal_mask_mapping[layer_type],
            position_embeddings=position_embeddings, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache, **kwargs)

        if active and i == anchor_layer:
            hidden_states = hidden_states[:, keep_idx, :]
            cos, sin = position_embeddings                            # (batch, seq, head_dim)
            position_embeddings = (cos[:, keep_idx, :], sin[:, keep_idx, :])
            position_ids = position_ids[:, keep_idx]
            mask_kwargs = {"config": self.config, "inputs_embeds": hidden_states,
                           "attention_mask": attn_mask_2d[:, keep_idx] if attn_mask_2d is not None else None,
                           "past_key_values": past_key_values, "position_ids": position_ids}
            causal_mask_mapping = {"full_attention": _m.create_causal_mask(**mask_kwargs)}
            if has_sliding:
                causal_mask_mapping["sliding_attention"] = _m.create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = self.norm(hidden_states)
    return _m.BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def install_anchor_prune(model):
    """Patch the text model's forward (idempotent). Works for either backbone --
    Qwen2.5-VL's Qwen2_5_VLTextModel (mRoPE) uses the dedicated patch;
    LLaVA-OneVision/Video's Qwen2Model (plain 1D RoPE) uses
    `_anchor_prune_plain_lm_forward` -- dispatched by class name, since
    `_text_model` resolves to either one generically. Returns the text model."""
    tm = _text_model(model)
    if getattr(tm, "_anchor_orig_forward", None) is None:
        mrope = type(tm).__name__ == "Qwen2_5_VLTextModel"
        patched = _anchor_prune_text_forward if mrope else _anchor_prune_plain_lm_forward
        tm._anchor_orig_forward = tm.forward
        tm.forward = types.MethodType(patched, tm)
        tm.anchor_layer = None
        tm.anchor_keep_idx = None
    return tm


def uninstall_anchor_prune(model):
    tm = _text_model(model)
    if getattr(tm, "_anchor_orig_forward", None) is not None:
        tm.forward = tm._anchor_orig_forward
        tm._anchor_orig_forward = None
    return tm


def set_anchor_prune(text_model, layer, keep_idx_abs):
    """layer=None runs the full model; otherwise restrict layers > layer to
    `keep_idx_abs` (absolute, sorted sequence positions)."""
    text_model.anchor_layer = layer
    text_model.anchor_keep_idx = keep_idx_abs
    return text_model


# --------------------------------------------------------------------------- #
# 7. End-to-end plan: analysis pass -> (L*, kept video-token indices, diagnostics)
# --------------------------------------------------------------------------- #
def plan(model, base_embeds, position_ids, attn_mask, video_idx, band,
         k_frac=0.10, estimator="moment"):
    """One dense forward through the whole model (cheap relative to generation,
    and it is what BUYS the per-layer scores) + Sections 4-5: debiased per-band
    importance and anchor-layer selection. Returns L*, its tail-index diagnostics
    per band layer, its debiased scores (feeds Section 7's greedy step), and the
    layer's input hidden states (feeds the Section 6.4 stability gate). Diversity
    selection itself (Sections 6-7) is a separate step -- see
    `keep_indices_from_plan` -- since it needs the additional grid/budget/knob
    arguments that this analysis pass does not.

    If `install_anchor_prune` has already patched this model's text-model
    forward (e.g. a benchmark harness installs it once up front, before ever
    calling `plan`), this ALWAYS bypasses that patch for the analysis forward:
    both `_anchor_prune_text_forward` and `_anchor_prune_plain_lm_forward` are
    plain functions with none of the original forward's
    `output_hidden_states`/`output_attentions` capture machinery (newer
    transformers wires that up via a decorator on the ORIGINAL method), so
    calling either with those flags silently returns `hidden_states=None` /
    `attentions=None` -- not an error, just a `None` that fails a few lines
    below at `out.hidden_states[Lstar]`. Restored immediately after."""
    text_model = _text_model(model)
    orig_forward = getattr(text_model, "_anchor_orig_forward", None)
    if orig_forward is not None:
        mrope = type(text_model).__name__ == "Qwen2_5_VLTextModel"
        patched = _anchor_prune_text_forward if mrope else _anchor_prune_plain_lm_forward
        text_model.forward = orig_forward
    try:
        with torch.no_grad():
            out = model(inputs_embeds=base_embeds, position_ids=position_ids,
                        attention_mask=attn_mask, use_cache=False,
                        output_attentions=True, output_hidden_states=True)
    finally:
        if orig_forward is not None:
            text_model.forward = types.MethodType(patched, text_model)

    S = base_embeds.shape[1]
    text_query_mask = torch.ones(S, dtype=torch.bool, device=base_embeds.device)
    text_query_mask[video_idx] = False                      # Section 4's Q: every non-visual position

    scores_by_layer = debiased_scores_by_layer(
        text_model, out.hidden_states, out.attentions, video_idx, text_query_mask, band)
    Lstar, taus = select_anchor_layer(scores_by_layer, k_frac, estimator)
    s_star = scores_by_layer[Lstar]

    return {
        "L_star": Lstar, "tail_indices": taus, "scores": s_star,
        "hidden_states_L": out.hidden_states[Lstar][0, video_idx],   # (M,d), input to layer L*
    }


def full_sequence_keep_idx(seq_len, video_idx, keep_local):
    """Every non-visual position (kept verbatim) plus the visual positions at
    `keep_local` (local indices into `video_idx`). This is what the patched
    forward's `keep_idx` must cover: only visual tokens ever compete for the
    budget, so any selection strategy -- ours or a baseline -- funnels through
    this same full-sequence construction before being handed to the prune."""
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=video_idx.device)
    keep_mask[video_idx] = False
    keep_mask[video_idx[keep_local]] = True
    return keep_mask.nonzero(as_tuple=False).flatten()


def keep_indices_from_plan(plan_out, video_idx, grid, rho, seq_len,
                           sigma_s=1.5, sigma_tau=1.0, window=2, lambda0=1.0, beta=0.5,
                           proj_dim=64):
    """Sections 6-7: build the stability gate and run the greedy selection at L*
    over the given `grid` = (f, r, c, T, S) (see `token_grid` for Qwen video,
    `token_grid_video` for LLaVA-OneVision/Video -- this function itself is
    backbone-agnostic, only consuming the grid tuple, never how it was built).
    Returns (keep_idx_abs, keep_idx_local, K), where keep_idx_abs covers the
    FULL sequence (see `full_sequence_keep_idx`)."""
    f, r, c, T, Sframe = grid
    M = video_idx.numel()
    assert T * Sframe == M, f"grid T*S={T * Sframe} != {M} video tokens"
    gate = stability_gates(plan_out["hidden_states_L"], f, Sframe, proj_dim=proj_dim)
    K = max(1, int(round(rho * M)))
    lam = lambda0 * (rho ** (-beta))
    local = greedy_select(plan_out["scores"], f, r, c, gate, K,             # already sorted
                          sigma_s=sigma_s, sigma_tau=sigma_tau, window=window, lam=lam)
    abs_idx = full_sequence_keep_idx(seq_len, video_idx, local)
    return abs_idx, local, K


# --------------------------------------------------------------------------- #
# 8. Demo
# --------------------------------------------------------------------------- #
def build_messages(video_path, query, max_frames=32, max_pixels=None, fps=2.0):
    content = {"type": "video", "video": video_path, "fps": fps}
    if max_frames:
        content["max_frames"] = max_frames
    if max_pixels:
        content["max_pixels"] = max_pixels
    return [{"role": "user", "content": [content, {"type": "text", "text": query}]}]


def _report_and_compare(model, base_embeds, position_ids, attn_mask,
                        video_idx, out, keep_abs, K, band, n_layers, decode_fn):
    """Shared tail-of-the-demo: print diagnostics, run dense vs. pruned forward,
    compare next tokens. Identical for both backbones once inputs are built."""
    M = video_idx.numel()
    print(f"\nvisual tokens M={M}  band={band}")
    print("tail index (moment estimator) per band layer:")
    for l in band:
        mark = "  <- L*" if l == out["L_star"] else ""
        print(f"  layer {l:>3}: gamma = {out['tail_indices'][l]:+.3f}{mark}")
    print(f"anchor L* = {out['L_star']}   kept K = {K} / {M}  (rho given)")
    print(f"realized reduction proxy (N-K)*(n_layers-L*) = "
         f"{(M - K) * (n_layers - out['L_star'])}")

    with torch.no_grad():
        logits_full = model(inputs_embeds=base_embeds, position_ids=position_ids,
                            attention_mask=attn_mask, use_cache=False).logits[0, -1]

    tm = install_anchor_prune(model)
    set_anchor_prune(tm, out["L_star"], keep_abs)
    with torch.no_grad():
        logits_pruned = model(inputs_embeds=base_embeds, position_ids=position_ids,
                              attention_mask=attn_mask, use_cache=False).logits[0, -1]
    set_anchor_prune(tm, None, None)                          # leave the model clean

    top_full, top_pruned = decode_fn(logits_full.argmax()), decode_fn(logits_pruned.argmax())
    agree = int(logits_full.argmax() == logits_pruned.argmax())
    print(f"\nnext-token (dense):  {top_full!r}")
    print(f"next-token (pruned): {top_pruned!r}   {'(agrees)' if agree else '(DIFFERS)'}")


def run_demo_qwen(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from qwen_vl_utils import process_vision_info
    from oracle_check import build_full_positions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    merge_size = model.config.vision_config.spatial_merge_size

    messages = build_messages(args.video, args.query, args.max_frames, args.max_pixels, args.fps)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    video_idx = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    grid_thw = inputs["video_grid_thw"].to(device)
    pix = inputs["pixel_values_videos"].to(device)

    with torch.no_grad():
        ve = model.get_video_features(pix, grid_thw).pooler_output
        ve = torch.cat(ve, dim=0).to(device)
        base_embeds = model.get_input_embeddings()(input_ids).clone()
        base_embeds[0, video_idx] = ve.to(base_embeds.dtype)
    position_ids = build_full_positions(model, input_ids, video_idx, grid_thw, attn_mask)

    band = [int(x) for x in args.band.split(",")]
    out = plan(model, base_embeds, position_ids, attn_mask, video_idx, band,
              k_frac=args.k_frac, estimator=args.estimator)
    grid = token_grid(grid_thw, merge_size)
    keep_abs, _, K = keep_indices_from_plan(
        out, video_idx, grid, args.rho, input_ids.shape[1],
        sigma_s=args.sigma_s, sigma_tau=args.sigma_tau, window=args.window,
        lambda0=args.lambda0, beta=args.beta, proj_dim=args.proj_dim)

    n_layers = len(_text_model(model).layers)
    _report_and_compare(model, base_embeds, position_ids, attn_mask,
                        video_idx, out, keep_abs, K, band, n_layers, processor.tokenizer.decode)


def sample_video_frames(path, num_frames):
    """Uniformly sample `num_frames` PIL frames from a video file (decord), at
    segment midpoints -- the same convention as inference.official_frames, minus
    the MVBench record/bound it needs (this is a standalone --video demo, not a
    benchmark clip)."""
    from decord import VideoReader, cpu
    from PIL import Image
    from inference import get_index
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idxs = get_index(None, float(vr.get_avg_fps()), len(vr) - 1, num_frames, first_idx=0)
    return [Image.fromarray(f) for f in vr.get_batch(idxs).asnumpy()]


def build_llava_video_prompt(processor, query):
    """A single literal <video> placeholder + the query; LlavaOnevisionProcessor's
    own __call__ expands that one placeholder to `frames*pooled*pooled + 1` copies
    automatically once `videos=` is given (see processing_llava_onevision.py), so
    no manual token-count math is needed here. Tries the shipped chat template
    first (Qwen2-Instruct ships one), falls back to a literal Qwen2 chat format
    otherwise."""
    conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": query}]}]
    try:
        return processor.apply_chat_template(conv, add_generation_prompt=True)
    except (AttributeError, ValueError, TypeError):
        vt = getattr(processor, "video_token", "<video>")
        return f"<|im_start|>user\n{vt}\n{query}<|im_end|>\n<|im_start|>assistant\n"


def run_demo_llava_video(args):
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    video_token_id = model.config.video_token_id
    vcfg = model.config.vision_config
    patches_side = vcfg.image_size // vcfg.patch_size
    pooled_side = -(-patches_side // 2)              # ceil(side/2): apply_pooling's 2x downsample

    frames = sample_video_frames(args.video, args.num_frames)
    prompt = build_llava_video_prompt(processor, args.query)
    inputs = processor(text=prompt, videos=[frames], return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    pixel_values_videos = inputs["pixel_values_videos"].to(device=device, dtype=dtype)

    video_idx_full = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    T = len(frames)
    S = pooled_side * pooled_side
    if video_idx_full.numel() != T * S + 1:
        raise RuntimeError(f"got {video_idx_full.numel()} video-token positions, expected "
                          f"{T * S + 1} ({T} frames x {S} pooled tokens + 1 newline).")
    # The LAST video-token position is the model's single trailing "image_newline"
    # slot appended after ALL frame tokens (LlavaOnevisionModel.forward), not a real
    # grid cell -- exclude it from the candidate set (Section 4's positional-guard
    # idea): it is then automatically treated as an always-kept "text" position by
    # `full_sequence_keep_idx`, exactly like BOS/register tokens would be.
    video_idx = video_idx_full[:-1]

    with torch.no_grad():
        merged = model(input_ids=input_ids, attention_mask=attn_mask,
                      pixel_values_videos=pixel_values_videos,
                      use_cache=False, output_hidden_states=True)
        base_embeds = merged.hidden_states[0].detach()
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)   # plain 1D RoPE

    band = [int(x) for x in args.band.split(",")]
    out = plan(model, base_embeds, position_ids, attn_mask, video_idx, band,
              k_frac=args.k_frac, estimator=args.estimator)
    grid = token_grid_video(T, pooled_side, device)
    keep_abs, _, K = keep_indices_from_plan(
        out, video_idx, grid, args.rho, input_ids.shape[1],
        sigma_s=args.sigma_s, sigma_tau=args.sigma_tau, window=args.window,
        lambda0=args.lambda0, beta=args.beta, proj_dim=args.proj_dim)

    n_layers = len(_text_model(model).layers)
    _report_and_compare(model, base_embeds, position_ids, attn_mask,
                        video_idx, out, keep_abs, K, band, n_layers, processor.tokenizer.decode)


def run_demo(args):
    backbone = resolve_backbone(args.backbone, args.model_name)
    if not args.model_name:
        args.model_name = default_model_id(backbone)
    if args.dtype == "auto":
        args.dtype = "bf16"                      # both backbones ship bf16
    print(f"[backbone] {backbone}  model={args.model_name}  dtype={args.dtype}")
    if not args.video:
        raise SystemExit("pass --video <path> (or --self_test to run the correctness checks).")
    if backbone == "llava_video":
        run_demo_llava_video(args)
    else:
        run_demo_qwen(args)


# --------------------------------------------------------------------------- #
# 9. Self-tests (fast, CPU-only, no model) -- correctness of the core math
# --------------------------------------------------------------------------- #
def self_test():
    torch.manual_seed(0)

    # --- greedy_select: a static background cell should get temporally
    # subsampled while an equally-salient but non-static cell survives. ---
    T, Hh, Ww = 4, 4, 4
    S = Hh * Ww
    N = T * S
    idx = torch.arange(N)
    f = idx // S
    within = idx % S
    r = within // Ww
    c = within % Ww

    scores = torch.rand(N) * 0.05
    gate = torch.zeros(N)
    static_cell = (r == 0) & (c == 0)
    moving_cell = (r == 1) & (c == 1)
    scores[static_cell] = 1.0
    scores[moving_cell] = 1.0
    gate[static_cell] = 1.0          # unchanged frame-to-frame -> temporal penalty fires
    gate[moving_cell] = 0.0          # changes every frame -> temporal penalty vanishes

    K = 6                            # < the 8 candidates across the two cells -> real competition
    keep = greedy_select(scores, f, r, c, gate, K, sigma_s=1.0, sigma_tau=1.0, window=3, lam=5.0)
    kept_static = int(static_cell[keep].sum())
    kept_moving = int(moving_cell[keep].sum())
    print(f"[self_test] static-cell kept {kept_static}/4   moving-cell kept {kept_moving}/4")
    assert kept_static >= 1, "the first pick at a cell has zero prior redundancy and must survive"
    assert kept_moving > kept_static, "a moving cell must survive more than an equally-salient static one"
    print("[self_test] PASSED: static backgrounds are temporally subsampled, motion is preserved.\n")

    # --- select_anchor_layer: a Pareto-tailed layer among near-uniform ones must win ---
    band = [2, 3, 4]
    scores_by_layer = {
        2: torch.rand(4000) * 0.01,
        3: torch.distributions.Pareto(1.0, 1.5).sample((4000,)),
        4: torch.rand(4000) * 0.01,
    }
    Lstar, taus = select_anchor_layer(scores_by_layer, k_frac=0.10)
    print(f"[self_test] tail indices: { {l: round(g, 3) for l, g in taus.items()} }  L*={Lstar}")
    assert Lstar == 3, "the anchor must land on the heaviest-tailed layer"
    print("[self_test] PASSED: anchor selection finds the heaviest-tailed layer.\n")

    print("ALL SELF-TESTS PASSED")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Anchor-layer importance-diversity token reduction (demo).")
    p.add_argument("--self_test", action="store_true", help="run fast CPU-only correctness checks and exit.")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_video"], default="auto",
                   help="VLM backbone. 'auto' infers from --model_name ('llava'/'onevision'/"
                        "'video' substring => llava_video, else Qwen2.5-VL). Both are video "
                        "backbones. qwen: dynamic resolution, mRoPE, Qwen2_5_VLTextModel. "
                        "llava_video: LLaVA-OneVision/LLaVA-Video, REAL multi-frame video (uniform "
                        "--num_frames sampling), 1D RoPE, Qwen2Model.")
    p.add_argument("--video", help="path to a video file.")
    p.add_argument("--query", default="Describe what is happening in the video.")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, "
                        f"llava_video={LLAVA_VIDEO_MODEL_ID} (LLaVA-Video-7B-Qwen2 shares this class).")
    p.add_argument("--max_frames", type=int, default=32, help="qwen only.")
    p.add_argument("--fps", type=float, default=2.0, help="qwen only.")
    p.add_argument("--num_frames", type=int, default=8,
                   help="llava_video only: uniform frame count sampled per clip (each frame costs "
                        "pooled_side^2 tokens, so this drives the token budget directly).")
    p.add_argument("--max_pixels", type=int, default=None, help="qwen only.")
    p.add_argument("--rho", type=float, default=0.10, help="keep-ratio K/N.")
    p.add_argument("--band", default="2,3,4,5,6,7,8", help="anchor-candidate layer band B.")
    p.add_argument("--k_frac", type=float, default=0.10, help="top-order-statistic fraction for the tail estimator.")
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--sigma_s", type=float, default=1.5, help="spatial-arm radius (patch units).")
    p.add_argument("--sigma_tau", type=float, default=1.0, help="temporal-arm decay (frame-group units).")
    p.add_argument("--window", type=int, default=2, help="temporal-arm window W (frame groups).")
    p.add_argument("--lambda0", type=float, default=1.0, help="lambda(rho) = lambda0 * rho^-beta.")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--proj_dim", type=int, default=64, help="stability-gate projection dim d'.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 for both backbones.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run_demo(args)
