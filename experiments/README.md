# Phase 0 — Validation experiments (training-free)

These four experiments are forward-pass only. They run *before* training the
scorer and either green-light the project or tell you exactly what to fix: which
layers are critical, whether the supervision signal beats trivial baselines,
whether the scorer must see the question, and whether attention magnitude tracks
causal importance.

**Run B and C first** — they are the cheap deciders. A defines the layer set that
B/C/D read from, so in practice: run **A**, plug its layers into **B/C/D**.

## Shared setup (locked once, see the plan)

| Item | Value |
|------|-------|
| Backbone | `Qwen/Qwen2.5-VL-7B-Instruct`, frozen, **`attn_implementation="eager"`** |
| Dev set | ~300–500 NExT-QA val MC pairs |
| Retention ratios ρ | `{0.10, 0.20, 0.25, 0.50}` |
| Frame budget | `--max_frames 8` (report it; token counts depend on it) |
| Primary metric | Multiple-choice top-1 accuracy |

Why eager attention: every experiment both *reads* attention weights (teacher /
oracle scores) and *writes* an additive attention bias (knockout in A, token
dropping in B–D). Flash/SDPA don't expose that additive mask, so the masking
hooks in `common.py` fail loudly under those backends.

## How "dropping a token" is implemented

For the oracle ceiling and ablations we never re-plumb the model's position ids
or embeddings. We add `-inf` to the additive attention mask at the dropped video
**key** columns, for every query, at every layer (`common.block_keys`). The token
keeps its position id but contributes nothing downstream — exactly the
information ceiling we want. Wall-clock savings are a separate, later measurement
(the plan defers them).

## Running

```bash
# A — critical-layer localization (sliding-window knockout)
python -m experiments.exp_a_layer_localization \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --video_root /path/to/nextqa/videos --window 3 --max_pairs 400

# Plug A's recommended layers into B, C, D via --layers
# B — oracle ceiling vs uniform / KiToke
python -m experiments.exp_b_oracle_ceiling \
    --video_root /path/to/nextqa/videos --layers 12 13 14 15 16 \
    --rhos 0.10 0.20 0.25 0.50

# C — query dependence (reuses B's forward passes)
python -m experiments.exp_c_query_dependence \
    --video_root /path/to/nextqa/videos --layers 12 13 14 15 16 --rho 0.20

# D — attention vs causal importance (keep the sample small)
python -m experiments.exp_d_causal_importance \
    --video_root /path/to/nextqa/videos --layers 12 13 14 15 16 \
    --group_size 16 --max_pairs 100
```

Each script writes a `results_exp_*.json` and prints a summary plus the relevant
decision/kill-criterion.

## Decision gates

| Exp | Question | Kill / decision criterion |
|-----|----------|---------------------------|
| A | Which layers are critical in *this* model? | If the peak is far from 12–16, update the imported default. |
| B | What is the ceiling? Does the teacher beat baselines? | If oracle ≈ uniform → signal is unusable, **STOP and pivot**. |
| C | How query-dependent is the optimal token set? | Large per-question gap ⇒ scorer must be query-conditioned. |
| D | Does attention magnitude track causal importance? | Weak Spearman ⇒ change the supervision target. |

## Assumptions to verify on your machine

1. **Dataset fields.** `common.load_nextqa_dev` tries several known NExT-QA
   schemas (`a0..a4`, `option_0..4`, …). If your mirror differs it raises with
   the real key list — add your layout to the `_*_KEYS` constants in `common.py`.
2. **Video paths.** Built as `{video_root}/{video_id}.{video_ext}`; adjust
   `--video_ext` or the dataset's `video` field as needed.
3. **KiToke baseline** (`common.select_kitoke`) is a documented *best-effort*
   approximation (centroid-distinctiveness), not the authors' reference code.
   Swap in the official implementation before reporting a head-to-head number.
4. **language→video attention** = attention from text rows *after* the video
   block to video key columns, averaged over heads, query rows, then the
   critical layers, min-max normalized (`common.language_to_video_scores`).
