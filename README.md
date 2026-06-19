# Efficient VLM Token Selection

A lightweight learned token scorer for efficient video understanding in Video Large Language Models (Video-LLMs).

## Overview

Video-LLMs process all visual tokens uniformly, regardless of semantic importance. This work trains a small MLP scorer (~300K parameters) to identify and retain only the most important tokens before they enter the VLM's expensive transformer layers.

The scorer is supervised by language-to-video attention extracted from critical intermediate layers (12–16) of a frozen VLM — the layers empirically shown to perform the majority of visual information retrieval. No captions, no human annotations, and no reward model are required.

## Method

```
Frames → Frozen ViT → Scorer MLP → Top-k selection → Frozen Projector → Frozen VLM
                           ↑
              Supervised by language-to-video
              attention at layers 12–16
```

**Key properties:**
- Only the scorer MLP (~300K params) is trained
- Supervision is derived from the VLM's own internal attention — no external labels
- Backbone-agnostic: works with any continuous video encoder
- At inference, drops low-scoring tokens before the projector, saving compute proportional to retention ratio
- Features can be cached once and the scorer trained offline (GPU-free), with an optional contrastive (InfoNCE) term aligning each video to its question

## Installation

```bash
git clone https://github.com/yourusername/efficient-vlm
cd efficient-vlm
pip install -r requirements.txt
```

## Training

### Online (single pass over the frozen VLM per step)

```bash
python train.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name lmms-lab/NExTVideo \
    --video_root /path/to/videos \
    --max_steps 10000 \
    --checkpoint_dir checkpoints/
```

Check dataset field names before training:
```python
from datasets import load_dataset
ds = load_dataset("lmms-lab/NExTVideo", split="train")
print(ds[0].keys())
```

### Offline (cache features once, then train the scorer GPU-free)

The frozen VLM forward is the bottleneck and is recomputed identically every epoch.
`cache_features.py` runs it **once per sample** and writes the scorer's inputs and
targets to disk; `train_cached.py` then trains off the cache with no VLM loaded — so
you can run many epochs, sweep losses, and form real `B > 1` batches cheaply.

```bash
# 1. Cache merged visual features + teacher attention + language features (GPU)
python cache_features.py --data_file /path/to/train.jsonl --video_root /path/to/videos \
    --out_dir cache/train --max_frames 16 --max_pixels 200704
python cache_features.py --data_file /path/to/val.jsonl --video_root /path/to/videos \
    --out_dir cache/val   --max_frames 16 --max_pixels 200704

# 2. Train the scorer from the cache (no VLM; fast). Validation is model-free.
python train_cached.py --cache_dir cache/train --val_cache_dir cache/val \
    --batch_size 32 --max_steps 10000 --checkpoint_dir checkpoints/
```

Each cached sample stores `vision_feats (n_video, D)` (the scorer input, merged
visual tokens in LLM token space), `teacher_raw (n_video,)` (the **un-normalised**
layer-12–16 attention target — ListMLE/recall/NDCG are rank-based, so this
reproduces the online numbers), and `lang_feat (n_layers, D)` (pooled language
hidden states, used by the contrastive term). `meta.json` records the cache key
(`model`, `layers`, `max_frames`, `max_pixels`, `dtype`) — keep these identical to
your `evaluate.py` settings so token counts line up. Storage ≈ `n_video × 4 KB`
per sample; budget ~10–20 GB per 1k samples.

### Contrastive learning (optional)

Add a symmetric InfoNCE term that aligns each video with its question. The scorer's
own logits are the **score-weighted pooling** weights for the video embedding, so the
contrastive gradient flows back into the ranking head — training the scorer to upweight
tokens that align with the language, not just to match the teacher ranking. Requires a
cache with language features (the default) and lives in the offline trainer (InfoNCE
needs in-batch negatives):

```bash
python train_cached.py --cache_dir cache/train --val_cache_dir cache/val \
    --contrastive --lambda_con 0.1 --tau 0.07 --proj_dim 256 --batch_size 32
```

The objective is `L = ListMLE + λ · InfoNCE`; `train/listmle` and `train/info_nce`
are logged separately (add `--wandb` to stream them to Weights & Biases). The
`λ = 0` vs `λ > 0` comparison on `val/recall@k` is the ablation that shows whether
the contrastive term helps.

## Evaluation

`evaluate.py` runs the **deployable** pipeline (paper Eq. 14): low-scoring video
tokens are physically dropped before the LLM, surviving tokens keep their original
M-RoPE positions, and the decoder runs on a shorter sequence — so wallclock
speedup is real, not a masked information ceiling.

```bash
python evaluate.py \
    --video_root /path/to/nextqa/videos \
    --scorer_ckpt checkpoints/scorer_best.pt \
    --rhos 0.25 0.50 0.75
```

Per retention ratio ρ it reports MC accuracy, accuracy retention vs. the full
model, and wallclock speedup — both **LLM-only** (isolates the quadratic-cost
claim) and **end-to-end** (incl. fixed ViT + scorer overhead) — plus the
theoretical `1 − K/(T·N)` saving. A one-time self-test asserts the gated forward
with ρ = 1.0 reproduces the full-model logits before any number is trusted.
VideoMME / MVBench need their field layout added to `experiments/common.py`.

## Repository Structure

```
efficient_vlm/
├── __init__.py
├── scorer.py             # lightweight MLP scorer (+ optional contrastive proj heads)
├── attention_extractor.py # extracts language→video attention + language hidden states
├── loss.py               # ListMLE ranking loss, soft-label BCE, symmetric InfoNCE
├── cached_dataset.py     # dataset over a feature cache (ragged collate)
├── utils.py              # Pareto tail-index budgeting + stratified selection
└── gating.py             # deployable pre-projector token gating (M-RoPE preserved)
train.py                  # online training loop
cache_features.py         # one-pass feature/teacher/language cache writer (GPU)
train_cached.py           # offline scorer training from cache (+ contrastive), GPU-free
evaluate.py               # deployable gated-inference eval (accuracy + wallclock)
configs/
└── default.yaml          # default hyperparameters
requirements.txt
```