"""
prune_accuracy_mvbench.py -- the closer: does the decoder's concentration actually
translate into accuracy, and is the cheap encoder x query proxy good enough to prune on?

Every experiment so far measured WHERE importance is concentrated (tail index gamma).
None showed the concentration is EXPLOITABLE. This does: on MVBench, at a fixed token
budget rho, keep only K = round(rho * M) visual tokens chosen by each of four scorers,
prune them at the INPUT (drop those positions from the sequence, keep every other
token's original mRoPE position), and read the option-letter answer. Same prune
mechanism for all scorers -- only the SCORE differs -- so accuracy differences are the
scorer's, not the mechanism's.

Scorers:
  full      -- no pruning (reference accuracy)
  decoder   -- debiased decoder attention importance at L* (the real, in-LLM target;
               here applied as an input prune, a slightly conservative test for it)
  encquery  -- input-space visual . unit(question) projection (the cheap PRE-LLM proxy;
               token-level Spearman ~ +0.36 vs the decoder scorer)
  random    -- K random tokens (floor)
  uniform   -- K evenly-spaced tokens (content-free structured floor)

Reads (per rho):
  decoder >> random           => the concentration is real and exploitable.
  encquery ~ decoder          => the cheap pre-LLM proxy is good enough to prune on
                                 (FastV-free, prefill-saving, FlashAttention-compatible).
  decoder ~ random            => gamma>0 was necessary but NOT sufficient; attention
                                 concentration is not causal importance.

One dense forward per question (output_attentions -> decoder scores + full-model answer),
then one pruned forward per scorer x rho. Official MVBench protocol from inference.py.

Run:
    python prune_accuracy_mvbench.py --data_root ~/MVBench --num_frames 8 \
        --rhos 0.1 0.25 --max_samples 30 --max_pixels 200704
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))

from inference import DATA_LIST, ANSWER_PREFIX, build_prompt, make_mvbench_prompt, letter_token_ids
from anchor_layer_prune import (
    debiased_scores_by_layer, select_anchor_layer, full_sequence_keep_idx, _text_model,
    QWEN_MODEL_ID,
)
from oracle_check import build_full_positions

SCORERS = ["decoder", "encquery", "random", "uniform"]


def build_embeds(model, processor, path, data_type, has_bound, rec, text, args, device):
    """MVBench prompt (official frames) + ANSWER_PREFIX -> merged input embeddings,
    full mRoPE positions, mask, and the visual-token indices. Mirrors
    decoder_tail_index._qwen_inputs but for the MVBench answer-forcing prompt."""
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                 args.num_frames, args.max_pixels, args.fps,
                                 official=True, num_segments=args.num_frames)
    chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
    img_in, vid_in = process_vision_info(prompt)
    inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt").to(device)

    input_ids, attn_mask = inputs["input_ids"], inputs["attention_mask"]
    video_idx = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    grid_thw, pix = inputs["video_grid_thw"], inputs["pixel_values_videos"]
    with torch.no_grad():
        ve = model.get_video_features(pix, grid_thw).pooler_output
        ve = torch.cat(ve, dim=0)
        base = model.get_input_embeddings()(input_ids).clone()
        base[0, video_idx] = ve.to(base.dtype)
    pos = build_full_positions(model, input_ids, video_idx, grid_thw, attn_mask)
    return base, pos, attn_mask, video_idx


def question_direction(processor, model, question, device):
    ids = processor.tokenizer(question, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        e = model.get_input_embeddings()(ids)[0].float().mean(0)
    return torch.nn.functional.normalize(e, dim=-1)


def argmax_letter(last_logits, letter_ids):
    opt = [max(last_logits[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    return int(np.argmax(opt))


def keep_local(mode, dec_scores, q_proj, M, K, rng):
    if mode == "decoder":
        return np.argsort(dec_scores)[::-1][:K]
    if mode == "encquery":
        return np.argsort(q_proj)[::-1][:K]
    if mode == "random":
        return rng.choice(M, size=K, replace=False)
    if mode == "uniform":
        return np.unique(np.linspace(0, M - 1, K).round().astype(int))
    raise ValueError(mode)


@torch.no_grad()
def predict_pruned(model, base, pos, mask, video_idx, keep_loc, letter_ids):
    keep_loc = torch.as_tensor(np.ascontiguousarray(keep_loc), device=base.device, dtype=torch.long)
    keep_abs = full_sequence_keep_idx(base.shape[1], video_idx, keep_loc)
    out = model(inputs_embeds=base[:, keep_abs], position_ids=pos[..., keep_abs],
                attention_mask=mask[:, keep_abs], use_cache=False)
    return argmax_letter(out.logits[0, -1], letter_ids)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="gamma->accuracy prune test on MVBench.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--model_name", default=QWEN_MODEL_ID)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.25])
    p.add_argument("--band", default="2,3,4,5,6,7,8", help="decoder anchor-candidate band for L*.")
    p.add_argument("--k_frac", type=float, default=0.10)
    p.add_argument("--max_samples", type=int, default=30, help="per task.")
    p.add_argument("--max_pixels", type=int, default=200704)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="prune_accuracy_mvbench.json")
    args = p.parse_args()

    band = [int(x) for x in args.band.split(",")]
    rng = np.random.default_rng(args.seed)
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    text_model = _text_model(model)

    # settings: "full" + "<scorer>@<rho>"
    settings = ["full"] + [f"{s}@{r}" for r in args.rhos for s in SCORERS]
    C = defaultdict(lambda: defaultdict(int))
    seen = defaultdict(int)
    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=f"{task} (N={args.num_frames})"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt = build_prompt(rec)
                letter_ids = letter_token_ids(processor, letters)
                base, pos, mask, video_idx = build_embeds(
                    model, processor, path, data_type, has_bound, rec, text, args, device)
                M = int(video_idx.numel())
                q_hat = question_direction(processor, model, rec["question"], device)

                # one dense forward: full-model answer + decoder importance at L*
                out = model(inputs_embeds=base, position_ids=pos, attention_mask=mask,
                            use_cache=False, output_attentions=True, output_hidden_states=True)
                full_pred = argmax_letter(out.logits[0, -1], letter_ids)
                tqm = torch.ones(base.shape[1], dtype=torch.bool, device=device)
                tqm[video_idx] = False
                sbl = debiased_scores_by_layer(text_model, out.hidden_states, out.attentions,
                                               video_idx, tqm, band)
                Lstar, _ = select_anchor_layer(sbl, args.k_frac)
                dec_scores = sbl[Lstar].float().cpu().numpy()
                q_proj = (base[0, video_idx].float() @ q_hat).cpu().numpy()
                del out
                torch.cuda.empty_cache()

                seen[task] += 1
                C[task]["full"] += int(full_pred == gt)
                for r in args.rhos:
                    K = max(1, int(round(r * M)))
                    for s in SCORERS:
                        kl = keep_local(s, dec_scores, q_proj, M, K, rng)
                        pred = predict_pruned(model, base, pos, mask, video_idx, kl, letter_ids)
                        C[task][f"{s}@{r}"] += int(pred == gt)
                torch.cuda.empty_cache()
            except Exception as e:
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

    # ---- report ----
    print(f"\n{'task':26s} {'n':>4} " + " ".join(f"{c:>12s}" for c in settings))
    tot = defaultdict(int); ntot = 0
    for task in tasks:
        n = seen[task]
        if not n:
            continue
        ntot += n
        row = []
        for c in settings:
            tot[c] += C[task][c]
            row.append(f"{100.0 * C[task][c] / n:11.1f}%")
        print(f"{task:26s} {n:>4} " + " ".join(row))
    if ntot:
        print(f"{'OVERALL':26s} {ntot:>4} " +
              " ".join(f"{100.0 * tot[c] / ntot:11.1f}%" for c in settings))
        print()
        for r in args.rhos:
            d, e = tot[f"decoder@{r}"], tot[f"encquery@{r}"]
            rd, u = tot[f"random@{r}"], tot[f"uniform@{r}"]
            f = tot["full"]
            print(f"  rho={r}:  decoder {100.0*d/ntot:.1f}%  encquery {100.0*e/ntot:.1f}%  "
                  f"random {100.0*rd/ntot:.1f}%  uniform {100.0*u/ntot:.1f}%   (full {100.0*f/ntot:.1f}%)")
            print(f"          decoder-random {100.0*(d-rd)/ntot:+.1f}pts (exploitable?)   "
                  f"encquery-random {100.0*(e-rd)/ntot:+.1f}pts (proxy usable?)   "
                  f"decoder-encquery {100.0*(d-e)/ntot:+.1f}pts (proxy gap)")

    with open(args.out, "w") as fh:
        json.dump({"settings": settings, "seen": dict(seen),
                   "correct": {t: dict(C[t]) for t in C}}, fh, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
