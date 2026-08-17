"""
stage1_diversity.py -- stage one of the two-stage method: encoder/input-side token
reduction by greedy max-min diversity within temporal groups, with the unselected
tokens MERGED into their nearest survivor rather than discarded.

This is the half of the method that runs before the language model sees anything. It
consumes the merged input embeddings -- the visual features already projected into the
LM's space and scattered into their placeholder positions -- and returns a shorter
visual block. Nothing here needs a class token, a text tower, or a question: it is a
pure function of the clip, so it drops into any serving stack unchanged and its saving
applies to every decoder layer rather than to the layers above a prune point.

Why diversity rather than a score
---------------------------------
Selection here is by diversity, not by ranking, and that is a measurement result rather
than a preference. Positional spread on its own buys nothing: evenly spaced keeps,
whether strided along the token sequence or staggered across frames, track a random keep
in all four runs (stage_topk_accuracy.py's `uniform` / `uniform_stagger`). And ranking
has nothing to grip on at this stage -- score each visual token by one minus the cosine
similarity between its feature vector and the same grid cell in the previous frame, and
the tail index of that score comes out at -0.31 temporally and -0.22 spatially, at or
below zero on every clip. A tail index at or below zero says no small group of tokens
sits above a gap; redundancy is a smooth ramp, so wherever the keep line falls, the
tokens either side of it are worth about the same. What is left to exploit is not which
tokens are important but which tokens are duplicates, and that is a covering problem.

Grouping  (`frame_groups`)
--------------------------
Groups are windows of W adjacent frames, each group holding every spatial cell of those
frames. Max-min then ranges over space and time jointly inside the window: it is free to
answer "these two frames show the same thing, keep one copy of it" and "this frame has
two distinct regions, keep both" with the same mechanism.

The alternative reading -- one group per spatial cell across adjacent frames -- gives
groups of size W, which makes max-min a two-way dedup rather than a covering selection,
and is available as `group_by="cell"` for the ablation. The default is the windowed one:
at W=2 and 196 cells/frame a group holds a few hundred tokens, which is where greedy
max-min's O(|g| * k * D) is cheap, and the whole clip's few thousand is where it is not.

Selection  (`maxmin_select`)
----------------------------
Greedy farthest-point traversal: repeatedly take the token whose minimum cosine distance
to the already-selected set is largest, i.e.

    pick argmin_i  max_{j in S} cos(e_i, e_j)

The running max-similarity vector is updated with only the newest pick each step, so the
full |g| x |g| gram matrix is never formed -- k matvecs of (|g|, D) @ (D,) instead. This
is stage_topk_accuracy.py's `mmr_local` at lambda = 0, with one difference: MMR seeds at
the argmax of its attention score, and there is no score here. The seed is the group
medoid (the token closest to the group mean), which is deterministic, content-derived,
and keeps randomness out of the method entirely.

Merging  (`merge_into`)
-----------------------
Every unselected token is folded into the survivor it is closest to by cosine, and each
survivor becomes the plain mean of the raw embeddings assigned to it, itself included.

This is not a refinement. Dropping is the most likely explanation for the gap between
decoder-only ranking methods (~94% of dense at matched FLOPs) and the merging ones
(~97%+): at pooled temporal redundancy around 0.61, a dropped token usually has a
near-duplicate elsewhere in the clip that could have absorbed it. `merge=False` gives
the drop-only ablation, which is the comparison that says how much of stage one's
retention is the merge's doing rather than the selection's.

The centering trap
------------------
Cosine geometry on LM embeddings is anisotropic: a large shared component makes every
pairwise cosine sit near the same high value. Under MMR that collapses the redundancy
penalty to a near-constant and silently degenerates into plain top-K; under max-min it
is worse, because there is no score to fall back on -- the traversal just wanders. So
selection and parent assignment both run on features centered on the clip mean and then
L2-normalised, while the merge averages the RAW embeddings, because raw is what the
model consumes. Two feature sets, one clip; `center=False` opts out and is there to
demonstrate the collapse rather than to be used.

Positions
---------
Survivors keep their ORIGINAL position ids -- the same convention as every prune in this
family (stage_topk_accuracy.py's `predict_pruned`). A merged token therefore carries an
average of several patches at the mRoPE coordinate of the one that was kept. That is the
intended reading: the kept token is the representative of its cluster and sits where the
representative sat. Returned indices are ascending so the shortened block stays in
sequence order and position ids stay monotone.

Cost
----
Per group: O(|g| * k * D) for the traversal and O(|g| * k * D) for the parent assignment.
At W=2, P=196, D=3584, rho=0.5 that is roughly 0.14 GFLOP/group and ~1.1 GFLOP/clip
against a 7B decoder prefill -- negligible, but it is the one part of the method whose
cost grows with the keep rate, so it is reported rather than assumed.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1. Grouping
# --------------------------------------------------------------------------- #
def frame_groups(M: int, n_frames: int, window: int,
                 group_by: str = "window") -> tuple[list[np.ndarray], np.ndarray]:
    """Partition the visual block into groups, plus the pinned (never-grouped) tokens.

    Args:
        M:         number of visual tokens in the block.
        n_frames:  frames sampled for this clip.
        window:    W, how many adjacent frames a group spans.
        group_by:  "window" -- a group is every spatial cell of W adjacent frames
                              (|g| = W * P). The default; this is where max-min has
                              enough candidates to be a covering selection.
                   "cell"   -- a group is one spatial cell across W adjacent frames
                              (|g| = W). The literal "by spatial position" reading,
                              kept for the ablation.

    Returns:
        (groups, pinned). `groups` is a list of ascending int64 index arrays that
        partition the grouped tokens; `pinned` holds tokens that survive untouched.

    Tokens are laid out frame-major with P = M // n_frames per frame -- the same layout
    `uniform_stagger` assumes. M is not always F * P: on LLaVA-OneVision the clip's
    trailing newline embedding sits inside the visual block and has no spatial cell. Any
    such remainder is PINNED rather than folded into the last group, because it is not a
    patch and averaging it into one would corrupt a real token with a separator.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    P = M // n_frames
    if P == 0:
        raise ValueError(f"{M} visual tokens over {n_frames} frames leaves <1 token/frame")
    n_grid = P * n_frames
    pinned = np.arange(n_grid, M, dtype=np.int64)          # the remainder, if any

    groups: list[np.ndarray] = []
    if group_by == "window":
        for f0 in range(0, n_frames, window):
            f1 = min(f0 + window, n_frames)
            groups.append(np.arange(f0 * P, f1 * P, dtype=np.int64))
    elif group_by == "cell":
        for f0 in range(0, n_frames, window):
            f1 = min(f0 + window, n_frames)
            base = np.arange(f0, f1, dtype=np.int64) * P
            for p in range(P):
                groups.append(base + p)
    else:
        raise ValueError(f"unknown group_by {group_by!r}; choose 'window' or 'cell'")
    return groups, pinned


# --------------------------------------------------------------------------- #
# 2. Features
# --------------------------------------------------------------------------- #
def diversity_features(visual_embeds: torch.Tensor, center: bool = True) -> torch.Tensor:
    """(M, D) unit-norm features -- the space the cosine geometry lives in.

    Centering is done ONCE per clip, not per group, so every group's geometry is
    measured against the same origin and the groups stay comparable. See the module
    docstring on why uncentered cosines flatten the traversal.
    """
    x = visual_embeds.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(x, dim=1)


# --------------------------------------------------------------------------- #
# 3. Greedy max-min
# --------------------------------------------------------------------------- #
def maxmin_select(feats: torch.Tensor, k: int, seed: str = "centroid") -> torch.Tensor:
    """Greedy farthest-point traversal over `feats` (G, D) -> (k,) local indices.

        pick argmin_i  max_{j in S} cos(e_i, e_j)

    seed: "centroid" takes the token closest to the group mean -- deterministic and
          content-derived, so the method carries no RNG. "first" takes index 0 and
          exists only to show the traversal is insensitive to its seed.

    Indices are returned in PICK order (most-diverse-first). The caller sorts, because
    the pick order is what a budget sweep would truncate and it is worth keeping.
    """
    G = feats.shape[0]
    k = int(max(1, min(k, G)))
    if seed == "centroid":
        mean = torch.nn.functional.normalize(feats.mean(dim=0), dim=0)
        first = int(torch.argmax(feats @ mean).item())
    elif seed == "first":
        first = 0
    else:
        raise ValueError(f"unknown seed {seed!r}; choose 'centroid' or 'first'")

    chosen = torch.empty(k, dtype=torch.long, device=feats.device)
    max_sim = torch.full((G,), -float("inf"), device=feats.device)
    taken = torch.zeros(G, dtype=torch.bool, device=feats.device)
    chosen[0] = last = first
    taken[first] = True
    for t in range(1, k):
        # Only the newest pick can raise a candidate's max-similarity, so the running
        # vector needs one matvec per step and the G x G gram is never formed.
        max_sim = torch.maximum(max_sim, feats @ feats[last])
        obj = -max_sim                      # farthest from the set == least similar to it
        obj[taken] = -float("inf")
        chosen[t] = last = int(torch.argmax(obj).item())
        taken[last] = True
    return chosen


# --------------------------------------------------------------------------- #
# 4. Merge
# --------------------------------------------------------------------------- #
def merge_into(raw: torch.Tensor, feats: torch.Tensor,
               selected: torch.Tensor) -> torch.Tensor:
    """Fold every unselected token into its nearest survivor -> (k, D) merged features.

    Args:
        raw:      (G, D) the group's RAW embeddings -- what the model consumes, and so
                  what is averaged.
        feats:    (G, D) the centered unit-norm features -- what "nearest" is measured
                  in, the same geometry the selection used.
        selected: (k,)  local indices of the survivors.

    Each survivor comes back as the plain mean of the raw embeddings assigned to it,
    itself included. Assignment is a hard argmax, so every unselected token lands in
    exactly one cluster and no token's mass is counted twice or lost.
    """
    G = raw.shape[0]
    k = selected.numel()
    taken = torch.zeros(G, dtype=torch.bool, device=raw.device)
    taken[selected] = True
    unsel = (~taken).nonzero(as_tuple=False).flatten()

    sums = raw[selected].float().clone()
    counts = torch.ones(k, device=raw.device, dtype=torch.float32)
    if unsel.numel():
        # (n_unsel, k) cosines; feats are unit-norm so the matmul IS the cosine.
        parent = torch.argmax(feats[unsel] @ feats[selected].T, dim=1)
        sums.index_add_(0, parent, raw[unsel].float())
        counts.index_add_(0, parent, torch.ones_like(parent, dtype=torch.float32))
    return (sums / counts[:, None]).to(raw.dtype)


# --------------------------------------------------------------------------- #
# 5. The stage
# --------------------------------------------------------------------------- #
def stage1_reduce(visual_embeds: torch.Tensor, n_frames: int, rho: float,
                  window: int = 2, group_by: str = "window", merge: bool = True,
                  center: bool = True, seed: str = "centroid", select: str = "maxmin",
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, torch.Tensor]:
    """Run stage one over a clip's visual block.

    Args:
        visual_embeds: (M, D) merged input embeddings of the visual tokens only.
        n_frames:      frames sampled for this clip.
        rho:           keep fraction WITHIN each group. rho=1.0 is a no-op.
        window:        W adjacent frames per group.
        group_by:      "window" or "cell" -- see `frame_groups`.
        merge:         fold unselected tokens into their nearest survivor. False drops
                       them, which is the ablation the merge is measured against.
        center:        centre on the clip mean before cosine. See the module docstring.
        seed:          max-min seed, "centroid" or "first".
        select:        "maxmin" (the method) or "random" -- K drawn uniformly within
                       each group. The random arm is the control that says how much of
                       stage one's retention is the diversity selection's doing rather
                       than the grouping's and the merge's; it goes through the IDENTICAL
                       merge path, so the two differ in the selection and nothing else.
        rng:           required when select="random".

    Returns:
        (keep_local, new_embeds) -- ASCENDING local indices of the survivors, and the
        (len(keep_local), D) embeddings to substitute at those positions. With
        merge=False the embeddings are the originals, untouched.

    rho=1.0 short-circuits to an exact identity: same indices, same tensor. That is the
    self-test the runner asserts before it trusts any pruned number.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if select not in ("maxmin", "random"):
        raise ValueError(f"unknown select {select!r}; choose 'maxmin' or 'random'")
    if select == "random" and rng is None:
        raise ValueError("select='random' needs an rng")
    M = visual_embeds.shape[0]
    if rho == 1.0:
        return np.arange(M, dtype=np.int64), visual_embeds

    groups, pinned = frame_groups(M, n_frames, window, group_by)
    feats = diversity_features(visual_embeds, center)

    keep_parts, embed_parts = [], []
    if pinned.size:
        keep_parts.append(pinned)
        embed_parts.append(visual_embeds[torch.as_tensor(pinned, device=visual_embeds.device)])

    for g in groups:
        gi = torch.as_tensor(g, device=visual_embeds.device)
        k = int(max(1, round(rho * g.size)))
        if select == "maxmin":
            sel = maxmin_select(feats[gi], k, seed)
        else:
            sel = torch.as_tensor(rng.choice(g.size, size=k, replace=False),
                                  device=visual_embeds.device, dtype=torch.long)
        keep_parts.append(g[sel.cpu().numpy()])
        embed_parts.append(merge_into(visual_embeds[gi], feats[gi], sel) if merge
                           else visual_embeds[gi][sel])

    keep = np.concatenate(keep_parts)
    embeds = torch.cat(embed_parts, dim=0)
    # Sort into sequence order: the shortened block has to stay monotone in position id,
    # and the embeddings have to follow their indices.
    order = np.argsort(keep, kind="stable")
    return keep[order], embeds[torch.as_tensor(order, device=embeds.device)]


def stage1_retention(M: int, n_frames: int, rho: float, window: int = 2,
                     group_by: str = "window") -> float:
    """The keep fraction stage one ACTUALLY achieves, which is not rho.

    Per-group budgets are rounded and floored at one token, and pinned tokens always
    survive, so the realised fraction drifts above rho at small budgets. Retention
    accounting has to use this number, not the requested one -- matching baselines on a
    nominal rho that neither side hits is exactly the aggregation mismatch that costs a
    reviewer's trust.
    """
    if rho == 1.0:
        return 1.0
    groups, pinned = frame_groups(M, n_frames, window, group_by)
    kept = pinned.size + sum(max(1, round(rho * g.size)) for g in groups)
    return kept / M
