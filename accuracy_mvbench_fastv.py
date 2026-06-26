"""FastV MC accuracy on OpenGVLab/MVBench (Chen et al., 2024, arXiv:2403.06764).

The FastV counterpart to ``accuracy_mvbench.py``: same frames, prompt and
option-letter readout, but instead of the learned pre-LLM scorer this drops visual
tokens *inside* the LLM. At layer ``K`` the visual tokens are ranked by the
attention they receive from the last query position (head-averaged); the top
``rho`` fraction survive and layers ``K+1 ..`` run on the shortened sequence. See
``efficient_vlm/fastv.py`` for the mechanism (a reversible monkey-patch of the
Qwen2.5-VL text-model forward -- no edits to the vendored transformers fork).

``rho`` here is the *keep* ratio (FastV's ``R`` is the prune ratio, ``R = 1 - rho``),
matching ``accuracy_mvbench.py`` so the two methods are directly comparable at the
same visual-token budget. The headline number is the mean over per-task accuracies.

Example:
    python accuracy_mvbench_fastv.py \
        --data_root ~/MVBench \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --fastv_k 2 3 5 --rhos 0.25 0.5 0.75

FastV's training-free design means no ``--scorer_ckpt`` is needed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Reuse the exact MVBench plumbing from the scorer eval so the comparison is fair.
from accuracy_mvbench import (
    DATA_LIST,
    build_inputs,
    build_prompt,
    letter_token_ids,
    make_mvbench_prompt,
    predict,
    video_text_counts,
)
from efficient_vlm.fastv import install_fastv, set_fastv, set_visual_mask


def fastv_flops_ratio(n_tokens, n_kept, k, n_layers, hidden, inter):
    """Theoretical FLOPs reduction vs. the full model (FastV Eq. 5).

    Per transformer layer with ``n`` tokens, hidden ``d`` and FFN width ``m`` the
    cost is ``4 n d^2 + 2 n^2 d + 2 n d m``. FastV runs ``k`` layers at full length
    ``n`` and ``T-k`` layers at the pruned length ``n_hat``.
    """
    d, m = hidden, inter

    def layer_cost(n):
        return 4 * n * d * d + 2 * n * n * d + 2 * n * d * m

    full = n_layers * layer_cost(n_tokens)
    fast = k * layer_cost(n_tokens) + (n_layers - k) * layer_cost(n_kept)
    return 1.0 - fast / full


def main():
    p = argparse.ArgumentParser(description="FastV MC accuracy on OpenGVLab/MVBench.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/ (see accuracy_mvbench.py).")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--fastv_k", type=int, nargs="+", default=[2],
                   help="FastV filtering layer(s) K: layers 0..K run full, layers >K run pruned.")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="Visual-token KEEP ratios (FastV prune ratio R = 1 - rho).")
    p.add_argument("--max_frames", type=int, default=16, help="frames sampled per clip (MVBench default 16).")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="cap samples PER TASK (debug).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model baseline.")
    args = p.parse_args()

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    # FastV and the learned scorer must not both prune; ensure pre-LLM gating is off,
    # then install the (reversible) FastV-aware text-model forward.
    if hasattr(model, "disable_token_gating"):
        model.disable_token_gating()
    text_model = install_fastv(model)
    video_token_id = model.config.video_token_id
    tcfg = model.config.text_config
    n_layers, hidden, inter = tcfg.num_hidden_layers, tcfg.hidden_size, tcfg.intermediate_size

    # (column name, K, rho). full = no pruning; then each (K, rho) FastV setting.
    settings = [] if args.no_full else [("full", None, None)]
    for k in args.fastv_k:
        for r in args.rhos:
            settings.append((f"K{k}@{r}", k, r))

    correct = {t: {name: 0 for name, _, _ in settings} for t in tasks}
    seen = {t: 0 for t in tasks}
    kept_vid = {name: 0 for name, _, _ in settings}
    flops_red = {name: 0.0 for name, _, _ in settings}
    vid_total = 0
    text_total = 0
    n = 0

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=task):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = build_prompt(rec)
                prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                             args.max_frames, args.max_pixels)
                inputs = build_inputs(processor, model, prompt)
                letter_ids = letter_token_ids(processor, letters)
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

            n_video, n_text = video_text_counts(model, inputs)
            seq_len = n_video + n_text
            visual = (inputs["input_ids"][0] == video_token_id)
            vid_total += n_video
            text_total += n_text

            for name, k, rho in settings:
                if k is None:  # full model
                    set_fastv(text_model, None)
                    kept = n_video
                else:
                    set_visual_mask(text_model, visual)
                    set_fastv(text_model, k, rho)
                    kept = min(n_video, max(1, round(rho * n_video)))
                    flops_red[name] += fastv_flops_ratio(seq_len, seq_len - (n_video - kept),
                                                          k, n_layers, hidden, inter)
                correct[task][name] += int(predict(model, inputs, letter_ids) == gt_idx)
                kept_vid[name] += kept
            set_fastv(text_model, None)  # leave the model in the full state
            seen[task] += 1
            n += 1

    if n == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (json/ and video/).")

    text_avg = text_total / n
    print(f"\nEvaluated {n} samples across {len(tasks)} task(s)  "
          f"(avg video tokens/clip = {vid_total / n:.0f}, avg text tokens = {text_avg:.0f})\n")

    col = [name for name, _, _ in settings]
    print(f"{'task':<26}{'n':>5}" + "".join(f"{c:>10}" for c in col))
    print("-" * (31 + 10 * len(col)))
    for task in tasks:
        if not seen[task]:
            continue
        accs = "".join(f"{correct[task][c] / seen[task]:>10.4f}" for c in col)
        print(f"{task:<26}{seen[task]:>5}{accs}")
    print("-" * (31 + 10 * len(col)))
    valid = [t for t in tasks if seen[t]]
    mean_row = "".join(f"{np.mean([correct[t][c] / seen[t] for t in valid]):>10.4f}" for c in col)
    micro_row = "".join(f"{sum(correct[t][c] for t in valid) / n:>10.4f}" for c in col)
    print(f"{'mean (per-task)':<26}{'':>5}{mean_row}")
    print(f"{'micro (per-sample)':<26}{n:>5}{micro_row}")

    # Per-setting visual-token budget and theoretical FLOPs reduction (Eq. 5).
    print(f"\n{'setting':<10}{'vid%':>8}{'vid_tok':>9}{'text_tok':>10}{'FLOPs-':>9}")
    print("-" * 46)
    for name, k, _ in settings:
        vid_pct = kept_vid[name] / vid_total * 100 if vid_total else 0.0
        fr = (flops_red[name] / n * 100) if k is not None else 0.0
        print(f"{name:<10}{vid_pct:>7.1f}%{kept_vid[name] / n:>9.0f}{text_avg:>10.0f}{fr:>8.1f}%")


if __name__ == "__main__":
    main()
