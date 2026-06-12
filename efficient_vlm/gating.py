"""Deployable pre-projector token gating (paper section 2.5, Eq. 14).

This is the *real* inference pipeline, as distinct from the attention-masking used
by the Phase-0/2 experiments. There, a dropped token keeps its position id and is
masked as an attention key -- it still occupies a row in every forward, so the
measurement is an *information* ceiling, not a speedup. Here we instead remove the
dropped video tokens from the sequence entirely:

    V --Enc_v--> F --Scorer--> s --stratified top-k--> F_I --Proj--> LLM --> Answer

The surviving tokens are passed to the frozen LLM as a *physically shorter*
sequence, so attention and MLP compute actually drop by ~1 - K/(T*N). Critically,
each surviving token keeps its **original** M-RoPE (t, x, y) position id -- there is
no positional re-indexing (section 2.5) -- so the model still knows where and when
each retained token came from. We achieve this by computing the full-sequence
M-RoPE ids once, then gathering the kept columns.

The one piece of model-internal surgery is ``get_rope_index`` + feeding
``inputs_embeds``/``position_ids`` directly. We call ``get_rope_index`` defensively
(its kwargs vary across transformers versions) and provide :func:`self_test` so the
plumbing can be validated against a plain forward before any number is trusted:
with K = n_video the gated forward must reproduce the full-model logits.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from experiments import common
from experiments.common import MCResult, MCSample, PreparedInputs


# --------------------------------------------------------------------------- #
# Scorer checkpoint loading
# --------------------------------------------------------------------------- #
def load_scorer(ckpt_path: str, input_dim: int, hidden_dim: int, device) -> torch.nn.Module:
    """Load a trained scorer. Matches train.save_checkpoint's ``model_state`` key."""
    from efficient_vlm.scorer import Scorer

    scorer = Scorer(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    scorer.load_state_dict(state.get("model_state", state))
    scorer.eval()
    return scorer


# --------------------------------------------------------------------------- #
# M-RoPE position ids for the full (un-gated) sequence
# --------------------------------------------------------------------------- #
def compute_rope_index(model, prepared: PreparedInputs) -> Optional[torch.Tensor]:
    """Full-sequence M-RoPE position ids, shape ``(3, B, S)``.

    Qwen2.5-VL derives 3-axis (temporal, height, width) position ids from the
    grid layout via ``get_rope_index``. We compute them for the *whole* sequence
    here; the gated forward then keeps the columns of the surviving tokens, so
    each retained token carries its original coordinate unchanged.

    Returns ``None`` if the model exposes no ``get_rope_index`` (then the caller
    lets the model build default ids -- correct only when nothing is dropped).
    """
    # get_rope_index lives on the model in older transformers but moved to the
    # inner model (model.model) in newer ones (e.g. 5.x). Search both.
    fn = None
    for obj in (model, getattr(model, "model", None),
                getattr(getattr(model, "model", None), "language_model", None)):
        if obj is not None and hasattr(obj, "get_rope_index"):
            fn = obj.get_rope_index
            break
    if fn is None:
        return None
    enc = prepared.inputs
    # Only pass kwargs this version's signature actually accepts. ``mm_token_type_ids``
    # is required by newer signatures and present in the processor output.
    candidates = {
        "input_ids": enc.get("input_ids"),
        "mm_token_type_ids": enc.get("mm_token_type_ids"),
        "image_grid_thw": enc.get("image_grid_thw"),
        "video_grid_thw": enc.get("video_grid_thw"),
        "second_per_grid_ts": enc.get("second_per_grid_ts"),
        "attention_mask": enc.get("attention_mask"),
    }
    accepted = set(inspect.signature(fn).parameters)
    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}
    out = fn(**kwargs)
    position_ids = out[0] if isinstance(out, (tuple, list)) else out
    return position_ids  # (3, B, S)


# --------------------------------------------------------------------------- #
# The deployable gated forward
# --------------------------------------------------------------------------- #
@dataclass
class GatedForwardResult:
    logits: torch.Tensor       # (vocab,) at the final position
    kept_seq_len: int          # length of the shortened sequence actually run
    full_seq_len: int
    llm_ms: float              # wallclock of just the LLM forward (ms)


def _embed_tokens(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(input_ids)


@torch.no_grad()
def gated_forward(
    model,
    prepared: PreparedInputs,
    video_embeds: torch.Tensor,
    kept_local_idx: torch.Tensor,
    position_ids_full: Optional[torch.Tensor],
    time_llm: bool = False,
) -> GatedForwardResult:
    """Run the LLM on a physically shortened sequence (Eq. 14).

    Args:
        video_embeds: merged (post-projector) video token embeddings, ``(n_video, D)``
            -- exactly the tensors that would be scattered into the video
            placeholder positions. These are also the scorer's input features.
        kept_local_idx: indices (0..n_video-1) of the video tokens to keep.
        position_ids_full: ``(3, B, S)`` M-RoPE ids for the full sequence, from
            :func:`compute_rope_index`. ``None`` is only valid when nothing is
            dropped (kept == all video tokens).
    """
    enc = prepared.inputs
    input_ids = enc["input_ids"]                       # (1, S)
    device = input_ids.device
    S = input_ids.shape[1]

    # 1. Embed text + scatter the (merged) video embeddings into their positions.
    inputs_embeds = _embed_tokens(model, input_ids)    # (1, S, D)
    vpos = prepared.video_positions.to(device)
    inputs_embeds[0, vpos] = video_embeds.to(inputs_embeds.dtype)

    # 2. Build the keep-mask over the full sequence: every non-video token plus
    #    the kept video tokens. Dropped = the video tokens not in kept_local_idx.
    dropped = common.dropped_positions(prepared.video_positions, kept_local_idx).to(device)
    keep_mask = torch.ones(S, dtype=torch.bool, device=device)
    keep_mask[dropped] = False

    # 3. Slice every per-token input by the keep-mask. Surviving tokens keep their
    #    ORIGINAL M-RoPE ids -- no re-indexing (section 2.5).
    embeds_kept = inputs_embeds[:, keep_mask, :]
    attn_mask = enc.get("attention_mask")
    attn_kept = attn_mask[:, keep_mask] if attn_mask is not None else None

    fwd_kwargs = dict(inputs_embeds=embeds_kept, use_cache=False)
    if attn_kept is not None:
        fwd_kwargs["attention_mask"] = attn_kept
    if position_ids_full is not None:
        fwd_kwargs["position_ids"] = position_ids_full[:, :, keep_mask]

    # 4. LLM forward on the shortened sequence (this is the cost we save on).
    if time_llm and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model(**fwd_kwargs)
    if time_llm and device.type == "cuda":
        torch.cuda.synchronize()
    llm_ms = (time.perf_counter() - t0) * 1e3

    return GatedForwardResult(
        logits=outputs.logits[0, -1, :],
        kept_seq_len=int(keep_mask.sum().item()),
        full_seq_len=S,
        llm_ms=llm_ms,
    )


# --------------------------------------------------------------------------- #
# Multiple-choice readout on top of the gated forward
# --------------------------------------------------------------------------- #
def mc_from_logits(
    processor, last_logits: torch.Tensor, answer_idx: int, n_options: int
) -> MCResult:
    """Option-letter MC prediction from final-position logits (mirrors
    common.mc_evaluate's readout so gated and full paths score identically)."""
    letter_ids = common._letter_token_ids(processor, n_options)
    option_logits = [
        max(last_logits[i].item() for i in ids) if ids else float("-inf")
        for ids in letter_ids
    ]
    pred_idx = int(np.argmax(option_logits))
    return MCResult(
        pred_idx=pred_idx,
        correct=(pred_idx == answer_idx),
        gold_logit=option_logits[answer_idx],
        option_logits=option_logits,
    )


@torch.no_grad()
def mc_evaluate_gated(
    model,
    processor,
    prepared: PreparedInputs,
    sample: MCSample,
    video_embeds: torch.Tensor,
    kept_local_idx: torch.Tensor,
    position_ids_full: Optional[torch.Tensor],
    time_llm: bool = False,
) -> Tuple[MCResult, GatedForwardResult]:
    res = gated_forward(
        model, prepared, video_embeds, kept_local_idx, position_ids_full, time_llm=time_llm
    )
    mc = mc_from_logits(processor, res.logits, sample.answer_idx, len(sample.options))
    return mc, res


# --------------------------------------------------------------------------- #
# Self-test: gated forward with K = n_video must match the plain forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def self_test(model, processor, prepared: PreparedInputs, sample: MCSample,
              atol: float = 0.5) -> Tuple[bool, float]:
    """Validate the M-RoPE plumbing: keeping *all* video tokens must reproduce the
    full-model **prediction**. Returns ``(passed, max_abs_diff)``.

    The gated path necessarily re-runs the visual tower and feeds the result via
    ``inputs_embeds``, so in fp16 the option logits drift from a native forward by
    ~0.1 even when the rope ids are bit-exact -- that drift does not change the
    argmax. So the gate is: the predicted option must be unchanged, AND the logit
    delta must stay within a generous fp16 bound (``atol``) to still catch gross
    wiring errors (a wrong rope shifts logits by >>0.5 and usually flips the
    prediction). A failure means ``get_rope_index`` / ``inputs_embeds`` injection
    is wrong and downstream numbers are untrustworthy.
    """
    ref, _ = common.mc_evaluate(model, processor, prepared, sample.answer_idx, len(sample.options))
    video_embeds = common.get_video_features(model, prepared)
    keep_all = torch.arange(prepared.n_video, device=prepared.video_positions.device)
    pos = compute_rope_index(model, prepared)
    gated = mc_from_logits(
        processor,
        gated_forward(model, prepared, video_embeds, keep_all, pos).logits,
        sample.answer_idx,
        len(sample.options),
    )
    diff = max(abs(a - b) for a, b in zip(ref.option_logits, gated.option_logits))
    passed = (gated.pred_idx == ref.pred_idx) and (diff <= atol)
    return passed, diff
