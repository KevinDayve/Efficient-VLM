"""
stage_topk_accuracy.py -- is the concentration exploitable? Accuracy under attention
top-K visual-token pruning, with the ranking read off THREE different stages of the
decoder, against random and evenly-spaced keeps, swept over the keep rate.

This is Claim 2 of the narrative: the tail index says importance is concentrated at
essentially every layer (tail_vs_layer.py), but concentration is only useful if
ranking by it beats not ranking at all. Here every selector prunes through the SAME
mechanism at the SAME budget, so any accuracy difference is the ranking's and nothing
else's.

Selectors  (--selectors)
------------------------
attn_early  top-K by text->vision attention in the EARLY band (--early_band, default
            layers 0-6, i.e. the paper's 1-7), taken at the single HEAVIEST-TAILED
            layer of the band (--early_agg heaviest: the layer with the largest EVT
            index gamma, which is where top-K has the most to grab). This is the
            FastV-style read: prune on what an early layer says.
attn_mid    top-K by attention averaged over the MID band (--mid_band, default layers
            11-15, i.e. the paper's 12-16 -- the attention-knockout / STAR / VTW band,
            chosen because early attention is corruptible by sinks).
attn_late   top-K by attention averaged over the LAST five layers (--late_band).
            Included as the "as late as a scorer could possibly be read" control.
div_mid     the same MID-band attention score, but selected greedily under a Maximal
            Marginal Relevance objective instead of by rank alone: at each step keep the
            token maximising  lambda * s_i - (1 - lambda) * max_{j in kept} cos(e_i, e_j),
            where s is the min-max normalised attention score and e is that token's merged
            input embedding (--div_lambda, default 0.5). Same band, same score, same
            budget as attn_mid -- the ONLY difference is the redundancy penalty, so the
            div_mid - attn_mid gap is attributable to diversity and nothing else.
            div_early / div_late do the same on the other two bands.
            Rationale: raw top-K is free to spend the whole budget on one salient object
            repeated across all 16 frames. The penalty makes a token that duplicates an
            already-kept one cheap to skip, so the keep set buys coverage the way
            uniform/uniform_stagger do, but chooses WHERE to cover by attention. This is
            the DivPrune / CDPruner-style read.
            Redundancy is measured in the projected LM embedding space (the same vectors
            the pruned forwards consume), on features CENTERED per clip first
            (--no-div_center opts out): LM embeddings are strongly anisotropic, so
            uncentered cosines sit near a large common value and the penalty would be
            near-constant -- i.e. it would silently collapse back to top-K.
random      K visual tokens drawn uniformly at random (the distribution-free floor).
uniform     K evenly spaced visual tokens, one per M/K block, midpoint of the block
            (a content-free but structured floor -- it buys coverage, not ranking).
bot_*       (--include_bottom) BOTTOM-K by the same score. The sharpest control there
            is: if bot_x ~ attn_x, the score carries no usable order at all, not even
            a reversed one.

Averaging over a band L1-normalises each layer's score vector first. Attention rows
sum to one over ALL keys, so the visual-token mass at a layer sits at whatever scale
that layer's text/sink split leaves it; an unnormalised mean is just whichever layer
is loudest.

The text-only floor  (--text_only, on by default)
-------------------------------------------------
One extra forward per clip in which EVERY visual token is dropped -- K = 0, the rho -> 0
limit of the same prune. Language priors, the question and the option strings are all
still there, so this is what the model scores by reading the multiple choice and
guessing, and no keep rate is worth reporting unless it sits above it. It is the true
lower bound of the sweep, and the thing that says whether a curve that looks flat down
to 1% is evidence that 1% of the tokens suffice or evidence that the benchmark barely
needs the video. It is a single number per clip, not a curve: it does not depend on
rho, so it is reported as a floor next to `full_accuracy` and drawn as a horizontal
line, and every selector at every rho is McNemar-tested against it.

Dropped, not re-prompted: the text tokens keep the position ids they had when the
visual block was present (so on Qwen the mRoPE coordinates still carry the gap), and
the prompt is byte-identical to every other config's. That makes the floor the same
operation as the rest of the sweep at K = 0 rather than a differently-built prompt that
would confound the comparison with a tokenisation change. Note that it therefore still
contains whatever the chat template says about a video being present.

The score is raw attention mass -- no sink handling, no value-norm reweighting, no
positional correction -- matching tail_vs_layer.py's `--stack decoder` score exactly,
so the gamma curves and these accuracies describe the same quantity.

How the pruning is done
-----------------------
Input-space drop. One dense forward per clip captures (a) the merged input embeddings,
(b) the position ids the model built, and (c) the per-layer attention scores. Each
pruned config then re-runs the LM on that same embedding sequence minus the dropped
visual positions, with every surviving token keeping its ORIGINAL position id (no
re-indexing -- for Qwen that means the mRoPE time/height/width coordinates still say
where each patch came from). Text tokens are never dropped. On LLaVA-OneVision the
clip's trailing newline embedding is itself a <video> position, so it sits in the
visual block and is prunable like any other -- one token in M, at every keep rate.

Two consequences worth stating before a reviewer does:
  * The scores come from a forward in which every visual token was still present, so
    each selector is graded on the best ranking it could possibly have. If top-K still
    fails to beat random here, it is not because the score was measured under a
    degraded sequence.
  * Dropping at the input is stricter than FastV's drop-after-layer-K, which lets the
    doomed tokens contribute to layers 0..K first. It is, however, the only prune site
    at which attn_early, attn_mid, attn_late, random and uniform are all the same
    operation -- which is the comparison this experiment exists to make.

Where the prune happens  (--drop_after)
---------------------------------------
--drop_after L additionally runs EVERY selector with the same keep set deleted at the
input of 0-indexed layer L instead of before layer 0: layers 0..L-1 run on the full
sequence and the doomed tokens contribute their keys and values there, layers L..N-1 never
see them. This is the FastV prune site, and the second bullet above is exactly the claim it
puts a number on -- how much of what input-space pruning costs is bought back by letting
the tokens live through the early layers first.

The pair is controlled: same score, same ranking, same K, same tokens (the keep set is
computed ONCE per (clip, selector, rho) and reused at every site, random draws included),
same original position ids. Only the site differs, so the per-clip McNemar in
`mcnemar_by_prune_site` is attributable to the site alone. --drop_after 0 IS the input drop
and must reproduce the bare selector exactly -- a free correctness check on the mechanism.

Read the gap with the compute in mind: the two sites are NOT iso-FLOP. The input drop saves
the pruned tokens' cost in all N layers; --drop_after L saves it in only N-L. At L=16 of 28
that is roughly 43% of the decoder rather than 100%, so a drop-after-16 curve sitting above
an input-drop curve is a statement about where the information is read, not a better
efficiency result. The layer-drop forward also costs FULL dense attention for layers 0..L-1,
which makes it the expensive config in the sweep, not the cheap one.

Applies to the mask as well as the states: the hooks slice the causal mask's rows and
columns with the same ascending index list, so causality among the survivors is exactly
what it was, and each surviving token keeps its original position id.

Accuracy
--------
One forward per (clip, config); no generation. The prompt ends with "Best option:(" so
the last position is exactly the slot the option letter is read from, and the
prediction is the argmax over the option-letter token ids there. Identical protocol to
lv_knockout_accuracy.py (the letter scorer is imported from it), so the two
experiments' numbers sit on the same scale.

Per pair of selectors at each keep rate the summary reports an exact McNemar test on
the paired per-clip correctness, so "top-K did not beat random" can be stated as a
test result rather than as a gap that happens to be small.

Frame sampling, prompts and the input builders come from tail_vs_layer.py, so the
clips are bit-identical to the tail-index and knockout runs.

Cost
----
n_clips x (1 + n_selectors x n_rhos + text_only) forwards. The dense one is eager-
attention (the weights have to be materialised to be scored); the pruned ones are short.
Defaults are 6 selectors x 5 rhos + the floor = 32 forwards/clip, and the floor is the
cheapest of them (no visual tokens at all), so use --max_samples for a first look.
The div_* selectors add a greedy O(M x K x D) matvec loop per (clip, rho) on top of their
forward -- GPU-side and small next to the forward itself, but it is the one selector whose
cost grows with the keep rate.

Run
---
    # Qwen, EgoSchema
    python stage_topk_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --num_segments 16 --max_pixels 200704 \
        --out topk_ego_qwen.json --plot topk_ego_qwen.png

    # LLaVA-OneVision, MVBench, 50 clips/task, with the bottom-K controls
    python stage_topk_accuracy.py --data_root ~/Experiments/MVBench --tasks mvbench \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --num_segments 16 \
        --max_samples 50 --include_bottom --out topk_mvb_llavaov.json

    # just the text-only floor, no sweep -- one dense + one blind forward per clip
    python stage_topk_accuracy.py --data_root ~/Experiments/MVBench --tasks mvbench \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --num_segments 16 \
        --rhos --selectors --out floor_mvb_llavaov.json

    # input-space drop vs. the FastV site: mid-band top-K, dropped before layer 0 and
    # dropped after layer 16, on the same keep set
    python stage_topk_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --num_segments 16 \
        --selectors attn_mid --drop_after 16 --out site_ego_llavaov.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from lv_knockout_accuracy import gold_index, option_token_ids
from tail_vs_layer import (ANSWER_PREFIX, DATA_LIST, DEFAULT_SEGMENTS, LLAVA_FAMILY,
                           MVBENCH_TASKS, build_inputs, context_limit, infer_backbone,
                           iter_clips, load_model, moment_tail_index, sample_frames,
                           text_model_of, visual_query_masks)

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

# stage -> (default band, default aggregation). Bands are 0-INDEXED; the papers'
# 1-indexed "1-7" and "12-16" are "0:6" and "11:15" here. Negative endpoints count
# from the end, so the late band follows whatever depth the backbone has.
STAGE_DEFAULTS = {"early": ("0:6", "heaviest"),
                  "mid": ("11:15", "mean"),
                  "late": ("-5:-1", "mean")}
DEFAULT_SELECTORS = ["attn_early", "attn_mid", "attn_late", "div_mid", "random", "uniform"]
# Every non-baseline selector is "<prefix>_<stage>": the prefix says how the stage's score
# is turned into a keep set, the stage says which band the score is read from.
SCORE_PREFIXES = ("attn", "bot", "div")
BASELINE_SELECTORS = ["random", "uniform", "uniform_stagger"]  # This list is moot anyway because `is_baseline` is used to check for baseline selectors. Retained for the sake of posterity.


# --------------------------------------------------------------------------- #
# 1. Bands
# --------------------------------------------------------------------------- #
def _layer_index(tok: str, n_layers: int) -> int:
    i = int(tok)
    i = i + n_layers if i < 0 else i
    if not 0 <= i < n_layers:
        raise ValueError(f"layer {tok} is outside a {n_layers}-layer stack")
    return i

def variant_label(sel: str, site) -> str:
    """A selector run at a prune site. site=None is the input-space drop and keeps the bare
    selector name, so every config string this script has ever written stays valid."""
    return sel if site is None else f"{sel}#L{site}"


def base_selector(label: str) -> str:
    return label.split("#", 1)[0]


def is_baseline(s):
    s = base_selector(s)
    return s.startswith("random") or s in ("uniform", "uniform_stagger")


def parse_band(spec: str, n_layers: int) -> list[int]:
    """
    A band is a comma-separated list of layer ranges, each of which is either a single layer or a colon or hyphen separated pair of layer indices.
    Args:
        spec (str): The band specification string i,e (11-15, 20:25, -5:-1 etc)
        n_layers (int): The total number of decoder layers in the model.
    Returns:
        list[int]: A sorted list of unique layer indices (inclusive) i.e., (11-15) -> [11, 12, 13, 14, 15]
    """
    layers: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
        else:
            # 'a-b' for non-negative endpoints; anything else (including a bare '-5')
            # is a single layer. The split starts at part[1:] so a leading minus sign
            # is never mistaken for the range separator.
            head, sep, tail = part[1:].partition("-")
            a, b = (part[0] + head, tail) if sep else (part, part)
        lo, hi = _layer_index(a, n_layers), _layer_index(b, n_layers)
        if lo > hi:
            raise ValueError(f"band {part!r} runs backwards ({lo} > {hi})")
        layers.update(range(lo, hi + 1))
    if not layers:
        raise ValueError(f"empty band {spec!r}")
    return sorted(layers)


# --------------------------------------------------------------------------- #
# 2. Capture: attention scores, merged embeddings, position ids -- one forward
# --------------------------------------------------------------------------- #
def _bound_arg(module, args, kwargs, name):
    """
    A helper function that abstracts how to retrieve the value of a named argument from a module's forward method.
    Example: passing "positional_ids" will retrieve the actual tensor, irrespective of how it was passed in the model's forward (either as a positional or keyword argument).
    Args:
        module: The module whose forward method is being inspected.
        args: The positional arguments passed to the forward method.
        kwargs: The keyword arguments passed to the forward method.
        name: Name of the argument to retrieve.
    Returns:
        The value of the named argument if found, otherwise None.
    """
    if name in kwargs:
        return kwargs[name]
    try:
        params = list(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):
        return None
    if name not in params:
        return None
    i = params.index(name)
    return args[i] if i < len(args) else None


def attach_capture(model, store: dict, ctx: dict):
    """Hooks for the dense forward. Returns an uninstaller.

    All three are gated on ctx["capture"], which is switched off for the pruned
    forwards: those run on a different sequence length, and re-entering the attention
    hook with the dense clip's visual indices would be both wrong and expensive.

        self_attn forward hook   layer -> (M,) attention score over visual tokens.
                                 Under eager the module returns its (1,H,Sq,Sk) weight
                                 tensor regardless of output_attentions; the text-query
                                 rows are sliced out BEFORE upcasting (the full tensor
                                 is hundreds of MB at these lengths) and the tensor then
                                 dies with the hook.
        layer[0] pre-hook        the merged input embeddings -- visual features already
                                 scattered into their placeholder positions, which is
                                 exactly what the pruned forwards need to slice.
        decoder pre-hook         the position ids the model built for this clip (3D
                                 mRoPE on Qwen, 1D on LLaVA)."""
    # Gets the language or decoder tower, which is where the attention lives.
    text_model = text_model_of(model)
    handles = []

    def stack_pre(module, args, kwargs):
        """
        This is the hook to capture the positional IDs
        """
        if ctx.get("capture"):
            # The position ids are needed for the pruned forwards, we use `_bound_arg` to get the values of the named parameter from the forward call.
            ctx["position_ids"] = _bound_arg(module, args, kwargs, "position_ids")
        return None

    def embed_pre(module, args, kwargs):
        """
        A hook to capture the merged input embeddings (visual features are already scattered into their placeholder positions).
        """
        if ctx.get("capture"):
            h = kwargs.get("hidden_states")
            ctx["embeds"] = (args[0] if h is None else h).detach()
        return None

    def make_attn_hook(layer_idx):
        """
        This is a pytorch hook to capture the attention scores from the self-attention layer of the model.
        """
        def hook(module, inputs, output):
            if not ctx.get("capture"):
                return None
            if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    "self_attn returned no attention weights -- load the model with "
                    "attn_implementation='eager' (sdpa/flash never materialize them).")
            # Under device_map="auto" this layer's weights (and so `output`) may live on
            # a different GPU than the input_ids the indices were built from, so both
            # index tensors follow the tensor they index.
            attn = output[1][0]
            text_q = ctx["text_q"].to(attn.device)
            recv = attn[:, text_q, :].float().mean(dim=1)                     # (H, Sk)
            store[layer_idx] = recv.mean(dim=0)[ctx["visual_idx"].to(recv.device)].detach().cpu()
        return hook

    handles.append(text_model.register_forward_pre_hook(stack_pre, with_kwargs=True))
    handles.append(text_model.layers[0].register_forward_pre_hook(embed_pre, with_kwargs=True))
    for i, layer in enumerate(text_model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
    return lambda: [h.remove() for h in handles]


# --------------------------------------------------------------------------- #
# 3. Scores -> a keep set
# --------------------------------------------------------------------------- #
def stage_score(store: dict, band: list[int], agg: str, k_frac: float):
    """(scores (M,), layer used or None, gamma at that layer or None).

    heaviest: the band's most heavy-tailed layer by the Dekkers-Einmahl-de Haan index,
              picked per clip -- the layer at which this clip's importance is most
              concentrated, i.e. the best case for top-K.
    mean:     the band average of L1-normalised per-layer scores."""
    if agg == "heaviest":
        gammas = [moment_tail_index(store[l], k_frac) for l in band]
        best = int(max(range(len(band)),
                       key=lambda i: gammas[i] if np.isfinite(gammas[i]) else -np.inf))
        return store[band[best]], band[best], gammas[best]
    normed = torch.stack([store[l] / store[l].sum().clamp_min(1e-12) for l in band])
    return normed.mean(dim=0), None, None


def uniform_stagger(M: int, K: int, F: int) -> np.ndarray:
    """
    A better function for uniformly spaced visual tokens, which emphasises more coverage. Evenly spaced in time, rotated in space so frames don't share the same
    spatial offset.
    Args:
        M (int): The total number of visual tokens in the sequence.
        K (int): The number of visual tokens to keep.
        F (int): The number of frames in the sequence.
    Returns:
        np.ndarray: The local indices of the K tokens to keep, staggered across frames for better coverage.
    """
    P = M // F # get the tokens per frame.
    base, remainder = divmod(K, F)
    out = []
    for frame in range(F):
        # Spread the remaining frames.
        kf = base + (1 if (frame * remainder) // F != ((frame + 1) * remainder) // F else 0)
        if kf == 0:
            continue
        step = P / kf # get the step size.
        phase = (frame / F) * step # stagger the phase across frames.
        out.append(frame * P + (np.arange(kf) * step + phase).astype(int))
    return np.concatenate(out)




def div_features(embeds: torch.Tensor, visual_idx: torch.Tensor, center: bool) -> torch.Tensor:
    """(M, D) unit-norm visual-token features, the space the MMR redundancy term lives in.

    These are the MERGED INPUT embeddings -- the visual features already projected into the
    LM's space, i.e. literally the vectors the pruned forwards consume. Cosine here is
    therefore "would the model see these two tokens as the same thing on the way in", not a
    similarity in some other encoder's space.

    center: subtract the clip's mean visual embedding before normalising. LM embedding
    clouds are anisotropic -- a large shared component makes every pairwise cosine sit near
    the same high value, so the max-similarity penalty would be near-constant across
    candidates and MMR would degenerate into plain top-K. Centering removes exactly that
    component and leaves the part that distinguishes tokens. Done ONCE per clip: the same
    features serve every keep rate, so the budgets are nested in the score, not in the
    feature space."""
    x = embeds[0, visual_idx.to(embeds.device)].float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(x, dim=1)


def mmr_local(scores: np.ndarray, feats: torch.Tensor, K: int, lam: float) -> np.ndarray:
    """Greedy Maximal Marginal Relevance over the visual tokens -> K local indices.

        pick argmax_i  lam * s_i - (1 - lam) * max_{j in kept} cos(e_i, e_j)

    s is min-max normalised to [0, 1] per clip so the two terms are on one scale and lam
    means the same thing at every clip and every band; cos is left raw in [-1, 1]. The
    first pick is the pure argmax of s (nothing is kept yet, so the penalty is undefined
    rather than zero), which makes lam=1 exactly attn_* and lam=0 a pure coverage selector
    seeded by the top-scoring token.

    The running max-similarity vector is updated with only the newest pick each step, so
    the full M x M gram matrix is never formed -- K matvecs of (M, D) @ (D,) instead."""
    M = feats.shape[0]
    K = int(min(K, M))
    s = torch.as_tensor(np.ascontiguousarray(scores), dtype=torch.float32, device=feats.device)
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo).clamp_min(1e-12)

    chosen = torch.empty(K, dtype=torch.long, device=feats.device)
    max_sim = torch.full((M,), -1.0, device=feats.device)
    taken = torch.zeros(M, dtype=torch.bool, device=feats.device)
    chosen[0] = last = int(torch.argmax(s).item())
    taken[last] = True
    for t in range(1, K):
        max_sim = torch.maximum(max_sim, feats @ feats[last])
        obj = lam * s - (1.0 - lam) * max_sim
        obj[taken] = -float("inf")
        chosen[t] = last = int(torch.argmax(obj).item())
        taken[last] = True
    return chosen.cpu().numpy()


def keep_local(selector: str, scores, M: int, K: int, rng, n_frames: int,
               feats: torch.Tensor | None = None, div_lambda: float = 0.5) -> np.ndarray:
    """Local indices (into the visual-token block) of the K tokens this selector keeps."""
    if selector.startswith("attn_"):
        return np.argsort(-scores, kind="stable")[:K]
    if selector.startswith("div_"):
        if feats is None:
            raise RuntimeError(f"{selector} needs the visual-token features; none captured")
        return mmr_local(scores, feats, K, div_lambda)
    if selector.startswith("bot_"):
        return np.argsort(scores, kind="stable")[:K]
    if selector == "random":
        return rng.choice(M, size=K, replace=False)
    if selector == "uniform":
        """
        This method picks one token every M/K position in the flattened token block but since the tokens are laid out frame by frame,
        it lands on the same spatial offset in every frame.
        """
        # Midpoint of each of K equal blocks -- distinct for every K <= M, and the same
        # even-coverage convention the frame sampler uses.
        return ((np.arange(K) + 0.5) * M / K).astype(int)
    if selector == "uniform_stagger":
        return uniform_stagger(M, K, n_frames)
    raise ValueError(f"unknown selector {selector!r}")


def keep_abs_idx(S: int, visual_idx: torch.Tensor, local: np.ndarray) -> torch.Tensor:
    """
    Absolute indices (into the full sequence) of the K tokens this selector keeps,
    for example. If the visual tokens are at positions 10, 11, 12, 13, 14 and the local indices are [0, 2], the absolute indices would be [10, 12].
    Args:
        S (int): The cardinality of the token set.
        visual_idx (torch.Tensor): The indices of the visual tokens in the full sequence.
        local (np.ndarray): The local indices of the tokens to keep (relative to the visual token block).
    Returns:
        torch.Tensor: The absolute indices of the tokens to keep in the full-sequence, along with the text tokens in ascending order. The result is of shape (K + T, ) where K = kept visual tokens and T = text tokens.
    """
    keep = torch.ones(S, dtype=torch.bool, device=visual_idx.device)
    keep[visual_idx] = False
    keep[visual_idx[torch.as_tensor(np.ascontiguousarray(local), device=visual_idx.device,
                                    dtype=torch.long)]] = True
    return keep.nonzero(as_tuple=False).flatten()


def attach_midforward_drop(model, layer_idx: int, keep_abs: torch.Tensor, S: int):
    """Delete the pruned positions from the residual stream at the INPUT of layer
    `layer_idx`, rather than before layer 0. Returns an uninstaller.

    This is the FastV prune site: layers 0..layer_idx-1 run on the full sequence, so the
    doomed visual tokens still contribute their keys and values there and the surviving
    tokens have already absorbed some of them; from layer_idx on they are gone. It is the
    strictly weaker intervention -- and the one real systems actually ship, because the
    early layers are where the tokens are believed to be read.

    Hooks go on EVERY layer from layer_idx up, not just layer_idx, because the decoder loop
    hands `attention_mask`, `position_ids`, `cache_position` and `position_embeddings` to
    each layer from ITS OWN full-length variables -- only `hidden_states` is threaded from
    the previous layer's output. So layer_idx shortens the hidden states, and every layer
    above it would otherwise get a full-length mask against a short sequence. Each tensor
    is sliced only while it is still at full length S, which makes the hook idempotent down
    the stack and keeps it working whether or not a given transformers version passes any
    particular one of them.

    Slicing rows and columns of the causal mask with the same ascending index list
    preserves causality exactly (kept i attends kept j iff j <= i), and every surviving
    token keeps its ORIGINAL position id -- same convention as the input-space drop, so the
    two prune sites differ in the site and nothing else."""
    text_model = text_model_of(model)
    handles = []

    def pre(module, args, kwargs):
        if len(args) > 1:
            raise RuntimeError(
                "this transformers version passes decoder-layer arguments positionally; "
                "the mid-forward drop slices them by keyword and would mis-slice them")
        h = args[0] if args else kwargs.get("hidden_states")
        kw = dict(kwargs)
        kw.pop("hidden_states", None)
        k_for = lambda t: keep_abs.to(t.device)

        # hidden_states: full length only at the first hooked layer.
        if h is not None and h.shape[1] == S:
            h = h.index_select(1, k_for(h))
        am = kw.get("attention_mask")
        if am is not None and torch.is_tensor(am) and am.shape[-1] == S:
            am = am.index_select(-1, k_for(am))
            if am.dim() >= 3 and am.shape[-2] == S:      # 4D causal mask: -2 is the query axis
                am = am.index_select(-2, k_for(am))
            kw["attention_mask"] = am
        for name in ("position_ids", "cache_position"):
            t = kw.get(name)
            if t is not None and t.shape[-1] == S:       # (B,S), (3,B,S) mRoPE, or (S,)
                kw[name] = t.index_select(-1, k_for(t))
        pe = kw.get("position_embeddings")
        if pe is not None:                               # (cos, sin), seq axis is -2
            kw["position_embeddings"] = tuple(
                t.index_select(-2, k_for(t)) if t.shape[-2] == S else t for t in pe)
        return (h,), kw

    for layer in text_model.layers[layer_idx:]:
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    return lambda: [hd.remove() for hd in handles]


@torch.no_grad()
def predict_midforward(model, embeds, position_ids, keep_abs, letter_ids,
                       layer_idx: int) -> int:
    """The same prediction as predict_pruned, but with the drop applied at layer_idx
    instead of at the input. The FULL sequence goes in -- the hooks do the deleting -- so
    layers 0..layer_idx-1 cost what the dense forward costs. The kept set always contains
    every text token, so the last position is still the answer slot."""
    S = embeds.shape[1]
    uninstall = attach_midforward_drop(model, layer_idx, keep_abs, S)
    try:
        out = model(inputs_embeds=embeds, position_ids=position_ids,
                    attention_mask=torch.ones(1, S, dtype=torch.long, device=embeds.device),
                    use_cache=False)
    finally:
        uninstall()
    return int(torch.argmax(out.logits[0, -1][letter_ids]).item())


@torch.no_grad()
def predict_pruned(model, embeds, position_ids, keep_abs, letter_ids) -> int:
    """
    The forward pass using the pruned set S' of tokens, returning the predicted option index. The kept tokens retain their original positional ID.
    The model is run with the embeddings and position ids corresponding to the kept tokens, and attn_mask is set to 1 for the kept tokens. The prediction is the argmax (greedy) over the option-letter token ids at the laast position of the sequence.
    Args:
        model (AutoModelForCausalLM): The language model to use for prediction.
        embeds (torch.Tensor): The merged input embeddings of shape (B, S, D). Where S = cardinality of the token set (sequence), D = embedding dimension.
        position_ids (torch.Tensor): The position ids of shape (3, B, S) corresponding to the embeddings. 3 because Qwen uses 3D mRoPE positioning (time, height, width).
        keep_abs (torch.Tensor): The absolute indices (received probably after running `keep_abs_idx` function) of the retained tokens in the full sequence.
        letter_ids (torch.Tensor): The token ids of the option letters (e.g., A, B, C, D) to consider for prediction.
    Returns:
        int: The predicted option index (0-based) corresponding to the argmax over the option-letter token ids at the last position of the sequence.
    """
    out = model(inputs_embeds=embeds[:, keep_abs],
                position_ids=position_ids[..., keep_abs],
                attention_mask=torch.ones(1, keep_abs.numel(), dtype=torch.long,
                                          device=embeds.device),
                use_cache=False)
    return int(torch.argmax(out.logits[0, -1][letter_ids]).item())


# --------------------------------------------------------------------------- #
# 4. McNemar
# --------------------------------------------------------------------------- #
def mcnemar(a_hits: list[int], b_hits: list[int]) -> dict:
    """Exact two-sided McNemar on paired correctness: is A's edge over B more than the
    discordant clips flipping a fair coin?

    b = A right & B wrong, c = A wrong & B right; concordant clips carry no information
    about the difference and drop out. p is the exact binomial tail (computed in logs,
    so the binomial coefficients never overflow), and is 1.0 when nothing is discordant."""
    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    log_half_n = -n * math.log(2.0)
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + log_half_n) for i in range(min(b, c) + 1))
    return {"b": b, "c": c, "n_discordant": n, "p_value": min(1.0, 2.0 * tail)}


# --------------------------------------------------------------------------- #
# 5. Dataset sweep
# --------------------------------------------------------------------------- #
def resolve_args(args):
    """Backbone, frame count, task list and pixel budget -- same handling as the rest
    of the family, so the clips match clip-for-clip."""
    if args.backbone == "auto":
        args.backbone = infer_backbone(args.model_name)
    if args.num_segments is None:
        args.num_segments = DEFAULT_SEGMENTS[args.backbone]

    if args.tasks == ["all"]:
        args.tasks = list(DATA_LIST)
    elif args.tasks == ["mvbench"]:
        args.tasks = MVBENCH_TASKS
    unknown = [t for t in args.tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    if args.backbone in LLAVA_FAMILY:
        # Both LLaVA towers have a fixed frame size (576 tokens/frame on 1.5, 196 on
        # OneVision), so M is already constant per frame; say the knobs are inert rather
        # than let them look like they took effect.
        if args.max_pixels is not None or args.min_pixels is not None:
            fixed = "576" if args.backbone == "llava" else "196"
            print(f"[warn] --max_pixels/--min_pixels are ignored on {args.backbone} "
                  f"(fixed {fixed} decoder tokens/frame).")
        args.max_pixels = args.min_pixels = None
    else:
        # Tie min to max so every frame costs the same number of tokens and M is
        # comparable across clips; --min_pixels 0 restores the library default.
        if args.min_pixels is None:
            args.min_pixels = args.max_pixels
        if args.min_pixels is not None and args.min_pixels <= 0:
            args.min_pixels = None

    selectors = list(args.selectors)
    if args.include_bottom:
        selectors += [s.replace("attn_", "bot_") for s in selectors if s.startswith("attn_")]
    # The prefix must be checked too, not just the stage suffix: keep_local raises on an
    # unknown selector from INSIDE the per-clip try, so a typo that slips through here is
    # swallowed as a skip on every clip rather than reported once, up front.
    bad = [s for s in selectors if not (
        is_baseline(s) or (s.split("_", 1)[0] in SCORE_PREFIXES
                           and s.split("_", 1)[-1] in STAGE_DEFAULTS)
    )]

    if bad:
        raise ValueError(f"unknown selectors {bad}; choices: "
                         f"{[f'{p}_{s}' for p in SCORE_PREFIXES for s in STAGE_DEFAULTS]}"
                         f" + {BASELINE_SELECTORS}")
    if not 0.0 <= args.div_lambda <= 1.0:
        raise ValueError(f"--div_lambda must be in [0, 1], got {args.div_lambda}")
    return selectors


def main(args):
    selectors = resolve_args(args)
    rhos = sorted(set(args.rhos))
    # Prune sites. None = the input-space drop this script has always done; an integer L =
    # the same keep set deleted at the input of layer L instead. Every selector runs at
    # every site, so the site is a free axis alongside the selector and the keep rate.
    sites = [None] + sorted(set(args.drop_after))
    labels = [variant_label(s, site) for site in sites for s in selectors]
    configs = [f"{lab}@{r:g}" for r in rhos for lab in labels]
    # Rho-independent references, carried through hits/per-task/per-sample exactly like
    # a config so nothing downstream has to special-case them: 'full' is the ceiling
    # (every visual token), 'text_only' the floor (none of them).
    refs = ["full"] + (["text_only"] if args.text_only else [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[model] {args.model_name}  backbone={args.backbone}  dtype={args.dtype}  device={device}")
    model, processor, visual_token_id = load_model(args, dtype)     # eager: scores need the weights
    max_positions = context_limit(model)
    n_layers = len(text_model_of(model).layers)

    stages = {name: (parse_band(getattr(args, f"{name}_band"), n_layers),
                     getattr(args, f"{name}_agg"))
              for name in STAGE_DEFAULTS
              if any(s.endswith(f"_{name}") for s in selectors)}

    bad_sites = [L for L in sites if L is not None and not 0 <= L < n_layers]
    if bad_sites:
        raise ValueError(f"--drop_after {bad_sites} outside a {n_layers}-layer stack "
                         f"(0 = drop before layer 0, i.e. the input-space drop)")

    print(f"[model] {n_layers} decoder layers, {max_positions} LM positions")
    for name, (band, agg) in stages.items():
        print(f"[stage] {name:<5} layers {band[0]}-{band[-1]} ({len(band)}), agg={agg}")
    print(f"[prune] original position ids kept, queries={args.queries}, budgets rho={rhos}")
    print(f"[prune] sites: " + ", ".join("input (before layer 0)" if L is None
                                         else f"layer {L} (layers 0-{L - 1} see everything)"
                                         for L in sites))
    print(f"[prune] selectors: {selectors}  ->  {len(refs) + len(configs)} forwards/clip")
    if any(s.startswith("div_") for s in selectors):
        print(f"[div]   greedy MMR, lambda={args.div_lambda:g} "
              f"(1=pure top-K, 0=pure coverage), features="
              f"{'centred ' if args.div_center else ''}merged input embeddings")
    if args.text_only:
        print("[floor] text-only: one forward per clip with every visual token dropped")
    print(f"[data] tasks={args.tasks}  {args.num_segments} frames/clip")

    store, ctx = {}, {"capture": False}
    uninstall = attach_capture(model, store, ctx)

    letter_cache = {}
    hits = {c: [] for c in refs + configs}          # config -> per-clip 0/1, clip-aligned
    per_task_hits = {}
    seen_by_task, per_sample, skipped = {}, [], []
    heaviest_layers = {name: {} for name, (_, agg) in stages.items() if agg == "heaviest"}
    n_seen = 0
    mass_sum = np.zeros(n_layers)
    try:
        for idx, (task, rec, path, data_type, bound) in enumerate(
                tqdm(list(iter_clips(args)), unit="clip")):
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": rec["video"], "reason": "missing file"})
                continue
            try:
                gold = gold_index(rec)
                n_opt = len(rec["candidates"])
                if n_opt not in letter_cache:
                    letter_cache[n_opt] = torch.tensor(
                        option_token_ids(processor.tokenizer, n_opt), device=device)
                letter_ids = letter_cache[n_opt]

                frames = sample_frames(path, data_type, bound, args.num_segments)
                inputs = build_inputs(processor, frames, rec, args, device, dtype)
                ids = inputs["input_ids"][0]
                S = ids.numel()
                if max_positions is not None and S > max_positions:
                    raise RuntimeError(f"sequence is {S} tokens but the LM holds "
                                       f"{max_positions} -- lower --num_segments")
                visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
                M = int(visual_idx.numel())
                if M < 50:
                    raise RuntimeError(f"too few visual tokens ({M} < 50)")

                # ---- the one dense forward: answer + scores + the sequence to prune ----
                store.clear()
                ctx.update({"capture": True, "visual_idx": visual_idx, "text_q": text_q,
                            "embeds": None, "position_ids": None})
                with torch.no_grad():
                    dense = model(**inputs, use_cache=False)
                full_pred = int(torch.argmax(dense.logits[0, -1][letter_ids]).item())
                ctx["capture"] = False
                del dense

                missing = [l for l in range(n_layers) if l not in store]
                if missing:
                    raise RuntimeError(f"no attention captured for layer(s) {missing[:5]}")
                embeds = ctx["embeds"]
                if embeds is None or embeds.shape[1] != S:
                    raise RuntimeError("did not capture the merged input embeddings")
                pos = ctx["position_ids"]
                if pos is None:                     # LM built them internally: plain 0..S-1
                    pos = torch.arange(S, device=device)[None]
                if pos.shape[-1] != S:
                    raise RuntimeError(f"captured position ids are {tuple(pos.shape)}, "
                                       f"expected last dim {S}")

                scored = {name: stage_score(store, band, agg, args.k_frac)
                          for name, (band, agg) in stages.items()}
                score_np = {name: s.numpy() for name, (s, _, _) in scored.items()}
                # One feature matrix per clip, shared by every div_* selector and every rho.
                feats = (div_features(embeds, visual_idx, args.div_center)
                         if any(s.startswith("div_") for s in selectors) else None)

                # ---- the floor: the same prune at K = 0, no visual tokens at all ----
                preds, kept = {}, {}
                if args.text_only:
                    preds["text_only"] = predict_pruned(
                        model, embeds, pos,
                        keep_abs_idx(S, visual_idx, np.empty(0, dtype=np.int64)),
                        letter_ids)

                # ---- one pruned forward per (rho, selector) ----
                rng = np.random.default_rng([args.seed, idx])
                for r in rhos:
                    K = int(min(M, max(1, round(r * M))))
                    for sel in selectors:
                        stage = sel.split("_", 1)[-1]
                        s = score_np.get(stage)
                        local = keep_local(sel, s, M, K, rng, n_frames=args.num_segments,
                                           feats=feats, div_lambda=args.div_lambda)
                        # ONE keep set, reused at every site: the sites are then compared
                        # on the identical token subset (identical random draw included),
                        # so the difference between them is the site alone.
                        keep_abs = keep_abs_idx(S, visual_idx, local)
                        for site in sites:
                            cfg = f"{variant_label(sel, site)}@{r:g}"
                            preds[cfg] = (
                                predict_pruned(model, embeds, pos, keep_abs, letter_ids)
                                if site is None else
                                predict_midforward(model, embeds, pos, keep_abs,
                                                   letter_ids, site))
                    kept[f"{r:g}"] = K
            except Exception as e:
                # A failure inside the dense forward leaves capture on; clear it here so a
                # stale clip's indices can never be applied to the next one.
                ctx.update({"capture": False, "embeds": None, "position_ids": None})
                skipped.append({"task": task, "video": rec["video"],
                                "reason": f"{type(e).__name__}: {e}"})
                tqdm.write(f"skip [{task}] {rec['video']}: {type(e).__name__}: {e}")
                continue

            mass_sum += np.array([float(store[l].sum()) for l in range(n_layers)])
            n_seen += 1
            seen_by_task[task] = seen_by_task.get(task, 0) + 1
            tc = per_task_hits.setdefault(task, {c: 0 for c in refs + configs})
            preds["full"] = full_pred
            for c in refs + configs:
                hit = int(preds[c] == gold)
                hits[c].append(hit)
                tc[c] += hit
            for name, (_, layer, gamma) in scored.items():
                if layer is not None:
                    heaviest_layers[name][layer] = heaviest_layers[name].get(layer, 0) + 1
            per_sample.append({"task": task, "question_idx": rec.get("question_idx"),
                               "video": rec["video"], "gold": gold, "n_options": n_opt,
                               "n_visual_tokens": M, "keep_budget": kept,
                               "heaviest_layer": {n: l for n, (_, l, _) in scored.items()
                                                  if l is not None},
                               "pred_by_config": preds})

            del inputs, embeds, feats
            ctx.update({"embeds": None, "position_ids": None})   # ctx holds the GPU refs
            store.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        uninstall()

    if not n_seen:
        print("no usable clips -- check --data_root layout (json/ and video/).")
        return

    acc = {c: sum(hits[c]) / n_seen for c in refs + configs}
    # Ranked vs unranked, WITHIN a prune site -- comparing a top-K at one site against a
    # random keep at another would confound the two axes.
    tests = [dict(rho=r, site=site, selector=variant_label(a, site),
                  baseline=variant_label(b, site),
                  delta_pts=100 * (acc[f"{variant_label(a, site)}@{r:g}"]
                                   - acc[f"{variant_label(b, site)}@{r:g}"]),
                  **mcnemar(hits[f"{variant_label(a, site)}@{r:g}"],
                            hits[f"{variant_label(b, site)}@{r:g}"]))
             for r in rhos for site in sites
             for a in selectors if not is_baseline(a)
             for b in selectors if is_baseline(b)]
    # The prune-site comparison: same selector, same rho, same keep set, dropped mid-stack
    # instead of at the input. This is the whole point of --drop_after.
    site_tests = [dict(rho=r, selector=base, drop_after=L,
                       delta_pts=100 * (acc[f"{variant_label(base, L)}@{r:g}"]
                                        - acc[f"{base}@{r:g}"]),
                       **mcnemar(hits[f"{variant_label(base, L)}@{r:g}"],
                                 hits[f"{base}@{r:g}"]))
                  for r in rhos for L in sites if L is not None
                  for base in selectors]
    # The test each div_* selector exists to pass: same band, same score, same budget as
    # its attn_* twin, with the redundancy penalty the single difference between them. Its
    # margin over random/uniform is inherited from the ranking; this is the part that is
    # the diversity's own.
    div_tests = [dict(rho=r, selector=a, baseline=b,
                      delta_pts=100 * (acc[f"{a}@{r:g}"] - acc[f"{b}@{r:g}"]),
                      **mcnemar(hits[f"{a}@{r:g}"], hits[f"{b}@{r:g}"]))
                 for r in rhos
                 for a in labels if a.startswith("div_")
                 for b in [a.replace("div_", "attn_", 1)] if b in labels]
    # Against the floor, EVERY selector is on trial -- the unranked ones included. A
    # keep rate at which random is no better than blind is a keep rate at which the
    # benchmark, not the selector, is doing the answering.
    floor_tests = [dict(rho=r, selector=a, baseline="text_only",
                        delta_pts=100 * (acc[f"{a}@{r:g}"] - acc["text_only"]),
                        **mcnemar(hits[f"{a}@{r:g}"], hits["text_only"]))
                   for r in rhos for a in labels] if args.text_only else []

    out = {"experiment": "stage_topk_accuracy",
           "prune": ("input_drop_visual_tokens" if sites == [None]
                     else "input_drop_visual_tokens + midforward_drop"),
           "backbone": args.backbone, "model_name": args.model_name,
           "data_root": args.data_root, "tasks": args.tasks,
           "num_segments": args.num_segments, "max_pixels": args.max_pixels,
           "min_pixels": args.min_pixels, "n_layers": n_layers,
           "queries": args.queries, "estimator": "moment", "k_frac": args.k_frac,
           "seed": args.seed, "rhos": rhos, "selectors": selectors,
           "prune_sites": sites, "variants": labels,
           "drop_after": sorted(set(args.drop_after)),
           "div_lambda": args.div_lambda, "div_center": args.div_center,
           "div_features": "merged_input_embeddings",
           "stages": {n: {"layers": b, "agg": a} for n, (b, a) in stages.items()},
           "scoring": "argmax over option-letter tokens", "answer_prefix": ANSWER_PREFIX,
           "n_clips": n_seen, "full_accuracy": acc["full"],
           "text_only": args.text_only,
           "text_only_accuracy": acc.get("text_only"),
           "chance_accuracy": (sum(1.0 / s["n_options"] for s in per_sample) / n_seen),
           "accuracy_by_config": acc,
           "accuracy_by_rho": {f"{r:g}": {s: acc[f"{s}@{r:g}"] for s in labels} for r in rhos},
           "retention_by_rho": {f"{r:g}": {s: (acc[f"{s}@{r:g}"] / acc["full"]
                                               if acc["full"] > 0 else float("nan"))
                                           for s in labels} for r in rhos},
           # Where a budget sits on the floor -> ceiling scale: 0 = the video bought it
           # nothing over answering blind, 1 = it kept everything the video was worth.
           # Undefined (nan) when the video buys the model nothing to begin with.
           "above_floor_by_rho": ({f"{r:g}": {s: ((acc[f"{s}@{r:g}"] - acc["text_only"])
                                                  / (acc["full"] - acc["text_only"])
                                                  if acc["full"] != acc["text_only"]
                                                  else float("nan"))
                                              for s in labels} for r in rhos}
                                  if args.text_only else None),
           "mcnemar": tests,
           "mcnemar_by_prune_site": site_tests,
           "mcnemar_div_vs_attn": div_tests,
           "mcnemar_vs_text_only": floor_tests,
           "mcnemar_full_vs_text_only": (dict(selector="full", baseline="text_only",
                                              delta_pts=100 * (acc["full"] - acc["text_only"]),
                                              **mcnemar(hits["full"], hits["text_only"]))
                                         if args.text_only else None),
           "heaviest_layer_histogram": {n: {str(k): v for k, v in sorted(h.items())}
                                        for n, h in heaviest_layers.items()},
            "visual_attention_mass_by_layer": (mass_sum / n_seen).tolist(),
           "per_task_seen": seen_by_task,
           "accuracy_by_task": {t: {c: h[c] / seen_by_task[t] for c in refs + configs}
                                for t, h in sorted(per_task_hits.items())},
           "skipped": skipped}

    # ---- report ----
    print(f"\n==== attention top-K vs. random/uniform ({n_seen} clips) ====")
    print(f"{args.backbone}: {args.model_name}, {args.num_segments} frames, "
          f"{len(seen_by_task)} task(s)")
    print(f"full (no pruning)   accuracy = {acc['full']:.4f}   <- ceiling")
    if args.text_only:
        ft = out["mcnemar_full_vs_text_only"]
        print(f"text only (0 tokens) accuracy = {acc['text_only']:.4f}   <- floor, "
              f"the video is worth {ft['delta_pts']:+.2f} pts (p={ft['p_value']:.3g})")
    print(f"chance (1/n_options)         = {out['chance_accuracy']:.4f}\n")
    w = max(13, max((len(s) for s in labels), default=13))
    print(f"{'rho':>6} " + " ".join(f"{s:>{w}s}" for s in labels))
    for r in rhos:
        print(f"{r:>6g} " + " ".join(f"{100 * acc[f'{s}@{r:g}']:{w - 1}.1f}%" for s in labels))
    if args.text_only:
        print(f"{'blind':>6} " + " ".join(f"{100 * acc['text_only']:{w - 1}.1f}%" for _ in labels))
    if site_tests:
        print("\nMcNemar, prune site (same selector, same keep set, dropped later):")
        for t in site_tests:
            flag = "significant" if t["p_value"] < 0.05 else "n.s."
            print(f"  rho={t['rho']:<5g} {t['selector']:>10s}  L{t['drop_after']} vs input "
                  f"{t['delta_pts']:+6.2f} pts  (b={t['b']:>4d} c={t['c']:>4d}, "
                  f"p={t['p_value']:.3g}, {flag})")
    print("\nMcNemar (exact, two-sided) against the unranked floors:")
    for t in tests:
        flag = "significant" if t["p_value"] < 0.05 else "n.s."
        print(f"  rho={t['rho']:<5g} {t['selector']:>14s} - {t['baseline']:<14s} "
              f"{t['delta_pts']:+6.2f} pts  (b={t['b']:>4d} c={t['c']:>4d}, "
              f"p={t['p_value']:.3g}, {flag})")
    if div_tests:
        print("\nMcNemar, diversity against its own ranking (does the penalty pay for itself?):")
        for t in div_tests:
            flag = "significant" if t["p_value"] < 0.05 else "n.s."
            print(f"  rho={t['rho']:<5g} {t['selector']:>14s} - {t['baseline']:<14s} "
                  f"{t['delta_pts']:+6.2f} pts  (b={t['b']:>4d} c={t['c']:>4d}, "
                  f"p={t['p_value']:.3g}, {flag})")
    if floor_tests:
        print("\nMcNemar against the text-only floor (does this budget beat seeing nothing?):")
        for t in floor_tests:
            flag = "significant" if t["p_value"] < 0.05 else "n.s."
            print(f"  rho={t['rho']:<5g} {t['selector']:>10s} - text_only "
                  f"{t['delta_pts']:+6.2f} pts  (b={t['b']:>4d} c={t['c']:>4d}, "
                  f"p={t['p_value']:.3g}, {flag})")
    for name, hist in out["heaviest_layer_histogram"].items():
        if hist:
            print(f"\nheaviest-tailed {name} layer, over clips: "
                  + ", ".join(f"L{k}x{v}" for k, v in hist.items()))
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s)")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")

    if args.plot and rhos and selectors:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        xs = np.array(rhos, dtype=float)
        # Colour carries the selector and linestyle carries the prune site, so a selector's
        # two sites sit on one colour and the gap between them is the thing you read.
        for i, sel in enumerate(selectors):
            for site in sites:
                lab = variant_label(sel, site)
                ys = np.array([100 * acc[f"{lab}@{r:g}"] for r in rhos])
                plt.plot(100 * xs, ys, "-" if site is None else "-.",
                         marker="o" if site is None else "s", ms=4, color=f"C{i}",
                         lw=1.2 if is_baseline(lab) else 1.8,
                         alpha=0.65 if is_baseline(lab) else 1.0, label=lab)
        plt.axhline(100 * acc["full"], color="gray", lw=0.8, ls=":", label="no pruning")
        if args.text_only:
            # The floor closes the band: anything between this line and the ceiling is
            # what the surviving visual tokens are actually buying.
            plt.axhline(100 * acc["text_only"], color="black", lw=0.8, ls="-.",
                        label="text only (0 visual tokens)")
        plt.xscale("log")
        plt.xlabel("visual tokens kept (%)")
        plt.ylabel("accuracy (%)")
        plt.title(f"{os.path.basename(args.model_name)} top-K vs. floors "
                  f"({n_seen} clips, {args.num_segments}f)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Accuracy under attention top-K visual-token pruning scored at three "
                    "decoder stages, vs. random / evenly-spaced keeps, on MVBench / EgoSchema.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' (all 20 MVBench tasks) / 'all' (+ EgoSchema). "
                        "MVBench and EgoSchema live under different roots -- don't mix in one run.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_ov", "llava"], default="auto",
                   help="auto infers from --model_name (onevision/-ov- -> llava_ov, else "
                        "'llava' -> llava-1.5, else qwen).")
    p.add_argument("--rhos", type=float, nargs="*", default=[0.01, 0.05, 0.1, 0.25, 0.5],
                   help="keep rates: K = round(rho * M) visual tokens survive. Pass with no "
                        "values to skip the sweep entirely (floor + ceiling only).")
    p.add_argument("--n_random", type=int, default=5, help="Independent random keeps per clip.")
    p.add_argument("--selectors", nargs="*", default=DEFAULT_SELECTORS,
                   help=f"any of "
                        f"{[f'{p_}_{s}' for p_ in SCORE_PREFIXES for s in STAGE_DEFAULTS]}"
                        f" + {BASELINE_SELECTORS}. attn_* = top-K by the band's score, "
                        f"bot_* = bottom-K, div_* = the same score under greedy MMR "
                        f"(top-K with a redundancy penalty). "
                        f"Pass with no values to run no selectors at all.")
    p.add_argument("--drop_after", type=int, nargs="*", default=[], metavar="LAYER",
                   help="ALSO run every selector with the drop applied at the input of "
                        "0-indexed LAYER instead of before layer 0 -- the FastV prune site: "
                        "layers 0..LAYER-1 still see every visual token, LAYER onward do "
                        "not. Same score, same keep set, same budget as the input-space "
                        "run, so the pair isolates the site. Repeatable. --drop_after 0 is "
                        "the input drop itself and is a useful self-check (it must "
                        "reproduce the bare selector exactly). Default: input drop only.")
    p.add_argument("--div_lambda", type=float, default=0.5,
                   help="div_* only: weight on the attention score against the redundancy "
                        "penalty, lam*score - (1-lam)*max cosine to an already-kept token. "
                        "1.0 reproduces attn_* exactly, 0.0 is pure coverage (default 0.5).")
    p.add_argument("--div_center", action=argparse.BooleanOptionalAction, default=True,
                   help="div_* only: centre the visual embeddings on the clip mean before "
                        "cosine. On by default -- uncentred LM embeddings are anisotropic "
                        "enough that the penalty goes near-constant and MMR collapses to "
                        "plain top-K.")
    p.add_argument("--include_bottom", action="store_true",
                   help="also run bottom-K by each attention score -- the control that says "
                        "whether the ranking carries any usable order at all.")
    p.add_argument("--text_only", action=argparse.BooleanOptionalAction, default=True,
                   help="one extra forward per clip with EVERY visual token dropped: the "
                        "rho->0 floor of the same prune, i.e. what the model scores from the "
                        "question and options alone. --no-text_only skips it.")
    for name, (band, agg) in STAGE_DEFAULTS.items():
        p.add_argument(f"--{name}_band", default=band,
                       help=f"0-indexed layers for the {name} stage (default {band}). "
                            f"'a:b' inclusive, negatives count from the end, comma-separated. "
                            f"A band starting with a minus needs the equals form, "
                            f"--{name}_band=-5:-1, or argparse reads it as an option.")
        p.add_argument(f"--{name}_agg", choices=["heaviest", "mean"], default=agg,
                       help=f"how the {name} band is reduced to one score (default {agg}). "
                            f"heaviest = the band's most heavy-tailed layer for this clip; "
                            f"mean = average of L1-normalised per-layer scores.")
    p.add_argument("--queries", choices=["post", "all", "last"], default="post",
                   help="text query rows the score averages over. post = non-visual positions "
                        "after the visual block, all = every non-visual position, last = the "
                        "answer slot only.")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="upper-tail fraction for the gamma used by '--*_agg heaviest'.")
    p.add_argument("--num_segments", type=int, default=None,
                   help=f"frames sampled per clip. Default per backbone: {DEFAULT_SEGMENTS}.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="qwen only: per-frame pixel cap, e.g. 200704. Both LLaVA backbones "
                        "have a fixed frame size and ignore this.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="qwen only: floor on per-frame pixels. Defaults to --max_pixels; 0 opts out.")
    p.add_argument("--max_samples", type=int, default=None, help="cap on records PER TASK.")
    p.add_argument("--seed", type=int, default=0, help="seeds the random selector, per clip.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="stage_topk_accuracy.json")
    p.add_argument("--per_sample_out", default="", help="default: <--out stem>_per_sample.json")
    p.add_argument("--plot", default=None, help="optional PNG of accuracy vs. keep rate.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
