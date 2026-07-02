"""
layerwise_attention_videomme.py -- is layer 12-16 averaging the right scorer input?
====================================================================================
Diagnostic companion to ``inference_only_videomme.py``. That script scores video
tokens by attention AVERAGED over a fixed layer band (default 12,13,14,15,16) and
keeps the top-K. The band is an unexamined constant. This script asks whether a
SINGLE layer -- or a different layer -- is a better selection signal, by sweeping
accuracy over every LLM layer at matched retention, for the two attention-driven
strategies:

  * attention_topk    : top-K by that layer's language->video attention
  * inverse_transform : per-frame sampling prop. to softmax(attn / temp)

Content-free strategies (random / stratified_uniform) do NOT read attention, so
they cannot vary by layer; ``uniform`` is reported once as a floor. Comparing the
two curves shows whether the top-k-vs-sampling gap is itself layer-dependent.

Motivation (Information-Horizon, arXiv 2512.07580): visual-token information is
not layer-stationary -- it becomes uniform and "vanishes" past an intermediate
horizon, and the horizon is deeper for stronger models (Qwen2.5-VL) and for
fine-grained tasks. So the layer you read attention from changes signal quality;
the best selection layer should sit BEFORE the horizon and may shift by task.

Scope: this probes the attention proxy's OWN suboptimality (wrong-layer signal),
NOT the answer-dependence ceiling from ``overlap_subset.py`` -- different gaps.

Run:
    python layerwise_attention_videomme.py \
        --data_root ~/Video-MME --durations short medium long \
        --rhos 0.05,0.10 --num_frames 16 --max_samples 300 \
        --layer_stride 1 --out results_layerwise_videomme.json
"""

import os
import json
import argparse
import numpy as np
import sys
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Reuse the shared, validated selection/positions helpers ...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oracle_check import build_full_positions, select_uniform, select_by_scores
# ... the OFFICIAL Video-MME protocol ...
from inference_videomme import (DURATIONS, load_questions, build_prompt,
                                make_video_prompt, load_subtitle, letter_token_ids)
# ... and the identical readout / dispersion / sampling from the sibling experiment.
from inference_only_videomme import score_answer, mean_dispersion, select_inverse_transform


# the two selection strategies that depend on the attention signal (hence on layer).
PER_LAYER_STRATEGIES = ("attention_topk", "inverse_transform")


def select_layer(strategy, scores_row, n_frames, k, device, rng, it_temp):
    """Select K tokens from a SINGLE layer's attention scores, per strategy."""
    if strategy == "attention_topk":
        return select_by_scores(scores_row, n_frames, k)
    if strategy == "inverse_transform":
        return select_inverse_transform(scores_row, n_frames, k, it_temp, rng).to(device)
    raise ValueError(strategy)


# --------------------------------------------------------------------------- #
# per-layer attention signal (language -> video), one forward, all layers kept
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_layer_attention_scores(model, base_embeds, video_positions, position_ids, attn):
    """Mean attention received by each video token from text queries, PER LAYER.
    Identical to inference_only_videomme.attention_scores but returns every layer
    instead of averaging a band. Returns (n_layers, n_video) and n_layers."""
    out = model(inputs_embeds=base_embeds, position_ids=position_ids,
                attention_mask=attn, use_cache=False, output_attentions=True)
    text_mask = torch.ones(base_embeds.shape[1], dtype=torch.bool, device=base_embeds.device)
    text_mask[video_positions] = False
    n_layers = len(out.attentions)
    scores = torch.zeros(n_layers, video_positions.numel(), device=base_embeds.device)
    for li in range(n_layers):
        a = out.attentions[li][0].mean(0)             # (S,S) mean over heads
        scores[li] = a[text_mask][:, video_positions].sum(dim=0)
    del out
    return scores.float(), n_layers


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    letter_ids = letter_token_ids(processor)
    baseline_layers = [int(x) for x in args.baseline_layers.split(",")]
    rhos = [float(x) for x in args.rhos.split(",")]

    durations = list(DURATIONS) if args.durations == ["all"] else args.durations
    unknown = [d for d in durations if d not in DURATIONS]
    if unknown:
        raise ValueError(f"unknown durations {unknown}; choices: {list(DURATIONS)}")

    # counters filled lazily once we learn n_layers from the first usable sample.
    correct_layer = None                 # {strat: {li: {rho: {dur: int}}}}
    disp_layer = None                    # {strat: {li: {rho: [floats]}}}
    layers_eval = None
    n_layers = None
    correct_uniform = {r: {d: 0 for d in durations} for r in rhos}
    correct_avg = {st: {r: {d: 0 for d in durations} for r in rhos} for st in PER_LAYER_STRATEGIES}
    full_correct = {d: 0 for d in durations}
    seen = {d: 0 for d in durations}
    n = 0
    rng = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    questions = [q for q in load_questions(
        os.path.join(os.path.expanduser(args.data_root), args.json_name))
        if q["duration"] in durations]
    if args.max_samples:
        questions = questions[: args.max_samples]

    video_dir = os.path.join(os.path.expanduser(args.data_root), args.video_subdir)
    sub_dir = os.path.join(os.path.expanduser(args.data_root), args.subtitle_subdir)

    for rec in tqdm(questions, desc="layer-sweep"):
        dur = rec["duration"]
        try:
            path = os.path.join(video_dir, f"{rec['videoID']}.mp4")
            subtitle = (load_subtitle(os.path.join(sub_dir, f"{rec['videoID']}.srt"))
                        if args.subtitles else None)
            text, gt_idx = build_prompt(rec, subtitle=subtitle)
            prompt = make_video_prompt(path, text, args.num_frames, args.max_pixels)
            chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(prompt)
            inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")
        except Exception as e:
            tqdm.write(f"skip [{dur}] {rec.get('videoID')}: {e}")
            continue

        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        vpos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if vpos.numel() < 50:
            continue
        n_video = vpos.numel()
        grid_thw = inputs["video_grid_thw"].to(device)
        n_frames = int(grid_thw[0][0].item())
        pix = inputs["pixel_values_videos"].to(device)

        with torch.no_grad():
            ve = model.get_video_features(pix, grid_thw).pooler_output
            ve = torch.cat(ve, dim=0).to(device)
            base = model.get_input_embeddings()(input_ids).clone()
            base[0, vpos] = ve.to(base.dtype)
        pos = build_full_positions(model, input_ids, vpos, grid_thw, attn)

        scores, L = per_layer_attention_scores(model, base, vpos, pos, attn)

        # first usable sample: fix the layer set and allocate counters.
        if correct_layer is None:
            n_layers = L
            bad = [b for b in baseline_layers if b >= L]
            if bad:
                raise ValueError(f"--baseline_layers {bad} >= n_layers {L}")
            layers_eval = list(range(0, L, max(1, args.layer_stride)))
            correct_layer = {st: {li: {r: {d: 0 for d in durations} for r in rhos}
                                  for li in layers_eval} for st in PER_LAYER_STRATEGIES}
            disp_layer = {st: {li: {r: [] for r in rhos} for li in layers_eval}
                          for st in PER_LAYER_STRATEGIES}

        # full-model reference (all video tokens kept)
        keep_all = torch.arange(n_video, device=device)
        lp_full = score_answer(model, base, pos, attn, vpos, keep_all, letter_ids)
        full_correct[dur] += int(lp_full.argmax().item() == gt_idx)

        avg_scores = scores[baseline_layers].mean(0)   # current 12-16 band signal

        for rho in rhos:
            k = max(n_frames, int(round(rho * n_video)))

            # uniform floor (content-free, layer-independent)
            keep_u = select_uniform(n_video, n_frames, k, device)
            lp_u = score_answer(model, base, pos, attn, vpos, keep_u, letter_ids)
            correct_uniform[rho][dur] += int(lp_u.argmax().item() == gt_idx)

            # band-average reference, per strategy
            for st in PER_LAYER_STRATEGIES:
                keep_a = select_layer(st, avg_scores, n_frames, k, device, rng, args.it_temp)
                lp_a = score_answer(model, base, pos, attn, vpos, keep_a, letter_ids)
                correct_avg[st][rho][dur] += int(lp_a.argmax().item() == gt_idx)

            # per-layer sweep, per strategy
            for li in layers_eval:
                for st in PER_LAYER_STRATEGIES:
                    keep = select_layer(st, scores[li], n_frames, k, device, rng, args.it_temp)
                    lp = score_answer(model, base, pos, attn, vpos, keep, letter_ids)
                    correct_layer[st][li][rho][dur] += int(lp.argmax().item() == gt_idx)
                    d = mean_dispersion(keep, n_video, n_frames)
                    if d == d:
                        disp_layer[st][li][rho].append(d)

        seen[dur] += 1
        n += 1
        del base, ve, scores

    if n == 0:
        print("no usable samples -- check --data_root layout (Video-MME.json and data/).")
        return

    valid = [d for d in durations if seen[d]]

    def per_bucket_mean(counts):
        return float(np.mean([counts[d] / seen[d] for d in valid]))

    def micro(counts):
        return sum(counts[d] for d in valid) / n

    full_mean, full_micro = per_bucket_mean(full_correct), micro(full_correct)

    out = {"experiment": "videomme_per_layer_topk_vs_inverse",
           "model_name": args.model_name, "data_root": args.data_root,
           "subtitles": args.subtitles, "num_frames": args.num_frames,
           "rhos": rhos, "baseline_layers": baseline_layers, "it_temp": args.it_temp,
           "strategies": list(PER_LAYER_STRATEGIES),
           "n_layers": n_layers, "layers_evaluated": layers_eval,
           "durations": valid, "n": n,
           "full_accuracy_mean": full_mean, "full_accuracy_micro": full_micro,
           "per_duration_seen": {d: seen[d] for d in valid},
           "reference": {}, "per_layer": {}, "best_layer": {}}

    print(f"\n==== per-layer attention_topk vs inverse_transform on Video-MME "
          f"({n} samples, {len(valid)} bucket(s), {n_layers} layers) ====")
    print(f"full-model accuracy: mean(per-bucket) {full_mean:.4f}  micro {full_micro:.4f}\n")

    for rho in rhos:
        u_mean = per_bucket_mean(correct_uniform[rho])
        avg_mean = {st: per_bucket_mean(correct_avg[st][rho]) for st in PER_LAYER_STRATEGIES}
        out["reference"][str(rho)] = {
            "uniform": {"accuracy_mean": u_mean, "accuracy_micro": micro(correct_uniform[rho])},
            "avg_baseline": {st: {"accuracy_mean": avg_mean[st],
                                  "accuracy_micro": micro(correct_avg[st][rho])}
                             for st in PER_LAYER_STRATEGIES}}

        rows = {st: [{"layer": li,
                      "accuracy_mean": per_bucket_mean(correct_layer[st][li][rho]),
                      "accuracy_micro": micro(correct_layer[st][li][rho]),
                      "dispersion": float(np.mean(disp_layer[st][li][rho])) if disp_layer[st][li][rho] else float("nan")}
                     for li in layers_eval] for st in PER_LAYER_STRATEGIES}
        out["per_layer"][str(rho)] = rows

        best = {st: max(rows[st], key=lambda r: r["accuracy_mean"]) for st in PER_LAYER_STRATEGIES}
        out["best_layer"][str(rho)] = {
            st: {"layer": best[st]["layer"], "accuracy_mean": best[st]["accuracy_mean"],
                 "delta_vs_avg_baseline": best[st]["accuracy_mean"] - avg_mean[st]}
            for st in PER_LAYER_STRATEGIES}

        tk, inv = "attention_topk", "inverse_transform"
        print(f"---- rho = {rho:.2f}  (uniform {u_mean:.4f} | "
              f"avg{baseline_layers}: topk {avg_mean[tk]:.4f}, inv {avg_mean[inv]:.4f}) ----")
        hdr = f"{'layer':>6}{'topk_acc':>10}{'inv_acc':>10}{'topk_disp':>11}{'inv_disp':>11}{'':>10}"
        print(hdr); print("-" * len(hdr))
        rt = {r["layer"]: r for r in rows[tk]}
        ri = {r["layer"]: r for r in rows[inv]}
        for li in layers_eval:
            mark = (" *topk" if li == best[tk]["layer"] else "") + (" *inv" if li == best[inv]["layer"] else "")
            print(f"{li:>6}{rt[li]['accuracy_mean']:>10.4f}{ri[li]['accuracy_mean']:>10.4f}"
                  f"{rt[li]['dispersion']:>11.3f}{ri[li]['dispersion']:>11.3f}{mark:>10}")
        print(f"best topk layer {best[tk]['layer']}: {best[tk]['accuracy_mean']:.4f} "
              f"(delta vs avg: {best[tk]['accuracy_mean'] - avg_mean[tk]:+.4f})")
        print(f"best inv  layer {best[inv]['layer']}: {best[inv]['accuracy_mean']:.4f} "
              f"(delta vs avg: {best[inv]['accuracy_mean'] - avg_mean[inv]:+.4f})\n")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")
    print("\nread: a single layer beating avg(12-16) => the band dilutes the signal;")
    print("      topk vs inv curves crossing => the sampling gap is layer-dependent;")
    print("      flat curves => layer choice is not the bottleneck (look elsewhere).")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-layer attention_topk vs inverse_transform sweep on Video-MME.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_root", required=True, help="Dir holding Video-MME.json, data/ and subtitle/.")
    p.add_argument("--json_name", default="Video-MME.json")
    p.add_argument("--durations", nargs="+", default=["all"], choices=["all", *DURATIONS])
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--subtitles", action="store_true")
    p.add_argument("--video_subdir", default="data")
    p.add_argument("--subtitle_subdir", default="subtitle")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=300,
                   help="cap questions (this sweep runs n_layers x n_rhos x 2 forwards/sample).")
    p.add_argument("--rhos", default="0.05,0.10")
    p.add_argument("--baseline_layers", default="12,13,14,15,16",
                   help="the current averaged band, reported as a reference row.")
    p.add_argument("--layer_stride", type=int, default=1,
                   help="evaluate every k-th layer (>1 to speed up the sweep).")
    p.add_argument("--it_temp", type=float, default=1.0,
                   help="inverse-transform temperature: ->0 top-k, ->inf uniform")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_layerwise_videomme.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())