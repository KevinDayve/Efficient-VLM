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

## Installation

```bash
git clone https://github.com/yourusername/efficient-vlm
cd efficient-vlm
pip install -r requirements.txt
```

## Training

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
├── scorer.py             # lightweight MLP scorer
├── attention_extractor.py # extracts language-to-video attention from frozen VLM
├── loss.py               # ListMLE ranking loss
├── utils.py              # Pareto tail-index budgeting + stratified selection
└── gating.py             # deployable pre-projector token gating (M-RoPE preserved)
train.py                  # online training loop
evaluate.py               # deployable gated-inference eval (accuracy + wallclock)
configs/
└── default.yaml          # default hyperparameters
requirements.txt
```