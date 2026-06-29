#!/usr/bin/env bash
set -e

MODEL=Qwen/Qwen2.5-VL-3B-Instruct
FRAMES=32
MAXPIX=200704
RHOS="0.01 0.05 0.1 0.25"
# KS="2 3 5"

MVBENCH=~/MVBench          # <-- set to your MVBench data_root
VIDEOMME=~/VideoMME        # <-- set to your Video-MME data_root

# ATTN_CKPT=/home/ubuntu/Efficient-VLM/nextqa-3b-8f-10k-attn/scorer_best.pt
ORACLE_CKPT=/home/ubuntu/Efficient-VLM/nextqa-oracle-3b-8f-10k-bce/oracle_scorer_best.pt

# ---- attn scorer (train.py)  ->  accuracy_* scripts (hidden_dim 256) ----
# python accuracy_mvbench.py  --data_root "$MVBENCH"  --model_name "$MODEL" \
#     --scorer_ckpt "$ATTN_CKPT" --hidden_dim 256 --rhos $RHOS \
#     --max_frames $FRAMES --max_pixels $MAXPIX \
#     --out results_mvbench_attn.json

# python accuracy_videomme.py --data_root "$VIDEOMME" --model_name "$MODEL" \
#     --scorer_ckpt "$ATTN_CKPT" --hidden_dim 256 --rhos $RHOS \
#     --max_frames $FRAMES --max_pixels $MAXPIX \
#     --out results_videomme_attn.json

# ---- oracle scorer (train_oracle.py)  ->  evaluate_* scripts (hidden_dim 512) ----
python evaluate_mvbench.py  --data_root "$MVBENCH"  --model_name "$MODEL" \
    --scorer_ckpt "$ORACLE_CKPT" --hidden_dim 512 --rhos $RHOS \
    --max_frames $FRAMES --max_pixels $MAXPIX \
    --strategies ours uniform random --out results_mvbench_oracle_32.json

python evaluate_videomme.py --data_root "$VIDEOMME" --model_name "$MODEL" \
    --scorer_ckpt "$ORACLE_CKPT" --hidden_dim 512 --rhos $RHOS \
    --max_frames $FRAMES --max_pixels $MAXPIX \
    --strategies ours uniform random --out results_videomme_oracle_32.json

# python accuracy_mvbench_fastv.py  --data_root "$MVBENCH"  --model_name "$MODEL" \
#     --fastv_k $KS --rhos $RHOS --max_frames $FRAMES --max_pixels $MAXPIX

# python accuracy_videomme_fastv.py --data_root "$VIDEOMME" --model_name "$MODEL" \
#     --fastv_k $KS --rhos $RHOS --max_frames $FRAMES --max_pixels $MAXPIX
