"""Standalone gated evaluation on OpenGVLab/MVBench: ours vs uniform vs random.

The MVBench twin of ``evaluate_oracle.py``. It reuses that file's self-contained
gating path verbatim -- merged-token features (``model.get_video_features(...)
.pooler_output``, exactly what the scorer was trained on), stratified Pareto
selection, and the *physically shortened* answer forward (dropped video tokens
removed; survivors keep their original M-RoPE ids) -- and only swaps the data
layer for MVBench's frame sampling.

MVBench is 20 video-MC tasks. Annotations live in ``<data_root>/json/<task>.json``
and clips under ``<data_root>/video/<source_subdir>/``; each clip is fed through
``qwen_vl_utils.process_vision_info`` -- the exact sampler train_oracle.py uses --
so eval frame-selection and preprocessing match training. Temporally bounded tasks
pass ``video_start``/``video_end`` (the frame-folder task passes a bound-restricted
frame list, sampled at fps=3). The official task -> (json, subdir, data_type,
has_temporal_bound) mapping is reproduced in ``DATA_LIST`` below.

For each retention ratio rho we keep K = round(rho * n_video) video tokens with
three strategies and read the option-letter logits at the answer position:

  * ours    -- learned scorer + stratified Pareto selection (paper section 2.4)
  * uniform -- evenly spaced video tokens (matched retention, no importance)
  * random  -- random video tokens (matched-retention chance baseline)

The full model (keep all tokens) is the accuracy reference and the LLM-forward
speedup denominator. The MVBench headline number is the mean over the 20 per-task
accuracies; we also report per-sample (micro) accuracy and the timing/speedup.

Setup (run once on the remote machine):
    pip install -U "huggingface_hub[cli]" decord pillow
    hf download OpenGVLab/MVBench --repo-type dataset --local-dir ~/MVBench
    cd ~/MVBench/video && for z in $(find . -name '*.zip'); do unzip -n -q "$z"; done

Example:
    python evaluate_mvbench.py \
        --data_root ~/MVBench \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --scorer_ckpt checkpoints_oracle/oracle_scorer_best.pt \
        --hidden_dim 512 \
        --rhos 0.25 0.50 0.75 \
        --strategies ours uniform random
"""

import os
import json
import argparse
import warnings

import numpy as np
import torch
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

from train_oracle import build_full_positions
# Reuse evaluate_oracle's gating/selection helpers so eval matches it (and
# training) exactly. evaluate_oracle in turn reuses train_oracle's M-RoPE path.
from evaluate_oracle import (
    STRATEGIES,
    select_ours,
    select_uniform,
    select_random,
    k_from_rho,
    gated_answer_logits,
    load_scorer,
)

# task -> (json file, video subdir under <data_root>/video, data_type, has_temporal_bound)
DATA_LIST = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}

SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question."
)


# --------------------------------------------------------------------------- #
# MVBench data loading (identical to the reference loader / accuracy_mvbench.py)
# --------------------------------------------------------------------------- #
TEMPORAL_PATCH_SIZE = 2  # Qwen2.5-VL pairs adjacent frames; the sampled count must be even (FRAME_FACTOR).


def frame_indices(bound, fps, max_frame, num_segments, first_idx=0):
    """Frame indices for the frame-folder (tvqa) task, matching qwen_vl_utils.

    Mirrors ``qwen_vl_utils._read_video_decord`` (the sampler train_oracle.py
    uses): ``linspace(start_frame, end_frame, nframes).round()`` with the count
    forced even. Video-file tasks are sampled by ``process_vision_info`` itself;
    this is only needed for the frame folder, where qwen_vl_utils takes an
    explicit frame list and neither trims to the clip bound nor sub-samples it."""
    start, end = (bound[0], bound[1]) if bound else (-1e5, 1e5)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    n = max(TEMPORAL_PATCH_SIZE, round(num_segments / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    return np.linspace(start_idx, end_idx, n).round().astype(int)


def make_mvbench_prompt(path, data_type, has_bound, record, text, max_frames, max_pixels):
    """Qwen chat prompt whose video item drives ``process_vision_info`` -- the
    same sampler train_oracle.py uses, so eval frame-selection matches training.

    Video-file tasks pass the path plus ``nframes`` (and ``video_start`` /
    ``video_end`` seconds when the task is temporally bounded, so qwen_vl_utils
    trims before its ``linspace`` sampling). The frame-folder task (tvqa) passes
    an explicit, bound-restricted, uniformly sub-sampled list of frame paths,
    because qwen_vl_utils neither trims nor sub-samples a frame list."""
    if data_type == "frame":
        bound = (record["start"], record["end"]) if has_bound else None
        names = sorted(os.listdir(path))
        idxs = frame_indices(bound, 3, len(names), max_frames, first_idx=1)
        vid = {"type": "video",
               "video": [os.path.join(path, f"{i:05d}.jpg") for i in idxs]}
    else:
        vid = {"type": "video", "video": path, "nframes": max_frames}
        if has_bound:
            vid["video_start"] = record["start"]
            vid["video_end"] = record["end"]
    if max_pixels is not None:
        vid["max_pixels"] = max_pixels
    return [{"role": "user", "content": [vid, {"type": "text", "text": text}]}]


def build_prompt(record):
    """MVBench option block + the letters present, and the ground-truth index."""
    letters = [chr(ord("A") + i) for i in range(len(record["candidates"]))]
    opts = "".join(f"({L}) {c}\n" for L, c in zip(letters, record["candidates"]))
    text = (f"{SYSTEM_PROMPT}\nQuestion: {record['question']}\nOptions:\n{opts}"
            "Answer with the option's letter (A, B, C, ...) directly.")
    gt_idx = record["candidates"].index(record["answer"])
    return text, letters, gt_idx


def letter_token_ids(tokenizer, letters):
    """First-token id candidates per option letter (with/without leading space)."""
    out = []
    for L in letters:
        cands = set()
        for form in (L, f" {L}"):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if ids:
                cands.add(ids[0])
        out.append(sorted(cands))
    return out


# --------------------------------------------------------------------------- #
# Per-sample preparation (no_grad twin of train_oracle.compute_oracle, MVBench
# records instead of NExT-QA) and multiple-choice scoring.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prepare_sample(model, processor, path, data_type, has_bound, record,
                   text, letters, gt_idx, video_token_id, device, args):
    """Build everything a gated forward needs for one MVBench sample.

    Mirrors ``evaluate_oracle.prepare_sample``: the clip is fed through
    ``process_vision_info`` (the qwen_vl_utils sampler train_oracle.py uses), so
    frame-selection and preprocessing match training exactly. Returns the full
    ``input_embeds`` (video slots filled with the merged visual features), the
    M-RoPE ``position_ids``, ``video_pos``, the fp32 ``features`` (scorer input),
    ``n_frames`` (temporal bins), and the MC ``letter_ids`` / ``gt_idx``. Returns
    ``None`` if there are no video tokens.
    """
    prompt = make_mvbench_prompt(path, data_type, has_bound, record, text,
                                 args.max_frames, args.max_pixels)
    chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(text=[chat], images=image_inputs, videos=video_inputs, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention = inputs["attention_mask"].to(device)
    video_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    if video_pos.numel() == 0:
        warnings.warn("[Warning] found zero video tokens. Skipping")
        return None
    video_grid_thw = inputs["video_grid_thw"].to(device)
    pixels = inputs["pixel_values_videos"].to(device)

    # Merged visual tokens -- the exact scorer input used in training.
    visual_feats = model.get_video_features(pixels, video_grid_thw).pooler_output
    visual_feats = torch.cat(visual_feats, dim=0).to(device)
    features = visual_feats.float()

    input_embeds = model.get_input_embeddings()(input_ids).clone()
    input_embeds[0, video_pos] = visual_feats.to(input_embeds.dtype)
    position_ids = build_full_positions(model, input_ids, video_pos, video_grid_thw, attention)

    n_frames = int(video_grid_thw[0][0].item())  # temporal bins T (post temporal patch)
    return {
        "input_embeds": input_embeds,
        "position_ids": position_ids,
        "video_pos": video_pos,
        "features": features,
        "n_frames": n_frames,
        "letter_ids": letter_token_ids(processor.tokenizer, letters),
        "gt_idx": gt_idx,
    }


def mc_correct(logits, letter_ids, gt_idx):
    """True if the argmax over the option-letter logits is the gold option.

    ``letter_ids[i]`` is the set of first-token ids for option i (with/without a
    leading space); we score each option by the max logit over its variants."""
    opt = [max(logits[i].item() for i in ids) if ids else float("-inf") for ids in letter_ids]
    return int(np.argmax(opt)) == gt_idx


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
def run(args):
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)  # cpu generator for 'random' strategy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown_t = [t for t in tasks if t not in DATA_LIST]
    if unknown_t:
        raise ValueError(f"unknown tasks {unknown_t}; choices: {list(DATA_LIST)}")

    strategies = list(args.strategies)
    unknown_s = [s for s in strategies if s not in STRATEGIES]
    if unknown_s:
        raise ValueError(f"unknown strategies {unknown_s}; choices: {list(STRATEGIES)}")
    if "ours" in strategies and not args.scorer_ckpt:
        raise ValueError("strategy 'ours' requires --scorer_ckpt (trained via train_oracle.py).")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation=args.attn
    )
    model.eval()
    model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")
    print(f"Eval: tasks={len(tasks)} | strategies={strategies} | rhos={args.rhos}")

    scorer = None
    # per-task accuracy (MVBench headline = mean over per-task accuracies)
    correct = {t: {s: {r: 0 for r in args.rhos} for s in strategies} for t in tasks}
    full_correct = {t: 0 for t in tasks}
    seen = {t: 0 for t in tasks}
    # global LLM-forward timing
    llm_ms = {s: {r: 0.0 for r in args.rhos} for s in strategies}
    n_timed = {s: {r: 0 for r in args.rhos} for s in strategies}
    full_llm_ms = 0.0
    full_n_timed = 0
    evaluated = 0

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[: args.max_samples]

        for rec in tqdm(records, desc=task):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt_idx = build_prompt(rec)
                sample = prepare_sample(model, processor, path, data_type, has_bound, rec,
                                        text, letters, gt_idx, video_token_id, device, args)
            except Exception as e:  # missing/corrupt clip -> skip
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue
            if sample is None:
                continue

            features = sample["features"]
            n_video = features.shape[0]
            n_frames = sample["n_frames"]
            warm = evaluated >= args.warmup

            # Lazily build the scorer once we know the feature width.
            scores = None
            if "ours" in strategies:
                if scorer is None:
                    scorer = load_scorer(args.scorer_ckpt, features.shape[-1], args.hidden_dim, device)
                scores = scorer(features.unsqueeze(0))[0]

            # Full-model reference (keep all tokens).
            if not args.no_full:
                keep_all = torch.arange(n_video, device=device)
                logits, ms = gated_answer_logits(model, sample, keep_all, time_llm=True)
                full_correct[task] += int(mc_correct(logits, sample["letter_ids"], sample["gt_idx"]))
                if warm:
                    full_llm_ms += ms
                    full_n_timed += 1

            for rho in args.rhos:
                k = k_from_rho(n_video, rho)
                for strat in strategies:
                    if strat == "ours":
                        kept = select_ours(scores, k, n_frames, args)
                    elif strat == "uniform":
                        kept = select_uniform(n_video, k, device)
                    else:  # random
                        kept = select_random(n_video, k, device, rng)
                    logits, ms = gated_answer_logits(model, sample, kept, time_llm=True)
                    correct[task][strat][rho] += int(
                        mc_correct(logits, sample["letter_ids"], sample["gt_idx"])
                    )
                    if warm:
                        llm_ms[strat][rho] += ms
                        n_timed[strat][rho] += 1
            seen[task] += 1
            evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No samples evaluated -- check --data_root layout (json/ and video/).")

    _report(args, tasks, strategies, evaluated, seen, correct, full_correct,
            llm_ms, n_timed, full_llm_ms, full_n_timed)


def _report(args, tasks, strategies, evaluated, seen, correct, full_correct,
            llm_ms, n_timed, full_llm_ms, full_n_timed):
    valid = [t for t in tasks if seen[t]]
    full_llm = full_llm_ms / max(1, full_n_timed)

    # column specs: full first (if run), then strat-major x rho.
    col_specs = []
    if not args.no_full:
        col_specs.append(("full", None, None))
    for s in strategies:
        for r in args.rhos:
            col_specs.append((f"{s[:4]}@{r:g}", s, r))

    def task_acc(t, s, r):
        if s is None:
            return full_correct[t] / seen[t]
        return correct[t][s][r] / seen[t]

    # MVBench headline: mean over per-task accuracies (and per-sample micro).
    def mean_over_tasks(s, r):
        return float(np.mean([task_acc(t, s, r) for t in valid])) if valid else float("nan")

    def micro(s, r):
        tot = sum((full_correct[t] if s is None else correct[t][s][r]) for t in valid)
        return tot / evaluated

    full_mean = mean_over_tasks(None, None) if not args.no_full else float("nan")

    table = {}
    for s in strategies:
        table[s] = {}
        for r in args.rhos:
            nt = max(1, n_timed[s][r])
            acc = mean_over_tasks(s, r)
            s_llm = llm_ms[s][r] / nt
            table[s][str(r)] = {
                "accuracy_mean": acc,
                "accuracy_micro": micro(s, r),
                "accuracy_retention": (acc / full_mean) if (not args.no_full and full_mean > 0) else float("nan"),
                "compute_saving": 1.0 - r,
                "speedup_llm": (full_llm / s_llm) if (not args.no_full and s_llm > 0) else float("nan"),
                "llm_ms": s_llm,
            }

    result = {
        "experiment": "mvbench_oracle_eval",
        "model_name": args.model_name,
        "scorer_ckpt": args.scorer_ckpt,
        "data_root": args.data_root,
        "tasks": valid,
        "n_evaluated": evaluated,
        "rhos": args.rhos,
        "strategies": strategies,
        "k_min": args.k_min,
        "full_accuracy_mean": full_mean,
        "full_llm_ms": full_llm if not args.no_full else float("nan"),
        "per_task": {
            t: {"n": seen[t], **{name: task_acc(t, s, r) for name, s, r in col_specs}}
            for t in valid
        },
        "table": table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # ---- per-task accuracy table (one column per setting) ----
    cols = [name for name, _, _ in col_specs]
    print(f"\nEvaluated {evaluated} samples across {len(valid)} task(s)"
          + (f" | full(mean)={full_mean:.4f} | full_llm={full_llm:.2f}ms" if not args.no_full else "")
          + "\n")
    hdr = f"{'task':<26}{'n':>5}" + "".join(f"{c:>11}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for t in valid:
        accs = "".join(f"{task_acc(t, s, r):>11.4f}" for _, s, r in col_specs)
        print(f"{t:<26}{seen[t]:>5}{accs}")
    print("-" * len(hdr))
    mean_row = "".join(f"{mean_over_tasks(s, r):>11.4f}" for _, s, r in col_specs)
    micro_row = "".join(f"{micro(s, r):>11.4f}" for _, s, r in col_specs)
    print(f"{'mean (per-task)':<26}{'':>5}{mean_row}")
    print(f"{'micro (per-sample)':<26}{evaluated:>5}{micro_row}")

    # ---- retention / speedup summary (mirrors evaluate_oracle) ----
    hdr2 = f"\n{'strat':<9}{'rho':>6}{'acc':>9}{'ret%':>8}{'save%':>8}{'sp_llm':>9}"
    print(hdr2)
    print("-" * len(hdr2.strip("\n")))
    for s in strategies:
        for r in args.rhos:
            tt = table[s][str(r)]
            print(f"{s:<9}{r:>6.2f}{tt['accuracy_mean']:>9.4f}{tt['accuracy_retention'] * 100:>8.1f}"
                  f"{tt['compute_saving'] * 100:>8.1f}{tt['speedup_llm']:>9.2f}")
    print(f"\nSaved -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Oracle-scorer gated eval on MVBench: ours vs uniform vs random.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/ (see module docstring).")
    p.add_argument("--tasks", nargs="+", default=["all"], help="MVBench task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--scorer_ckpt", default="", help="Scorer checkpoint from train_oracle.py (oracle_scorer_*.pt).")
    p.add_argument("--hidden_dim", type=int, default=512, help="Must match the trained scorer (train_oracle default 512).")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.50, 0.75], help="retention ratios")
    p.add_argument("--strategies", nargs="+", default=["ours", "uniform", "random"],
                   help=f"gating strategies to sweep: {list(STRATEGIES)} (full is always the baseline)")
    p.add_argument("--max_samples", type=int, default=None, help="Cap records evaluated PER TASK (debug).")
    p.add_argument("--max_frames", type=int, default=16, help="frames sampled per clip (MVBench default 16).")
    p.add_argument("--max_pixels", type=int, default=None, help="Cap per-frame resolution (e.g. 100352) on small GPUs.")
    # stratified-Pareto selection knobs (must match how you intend to deploy 'ours')
    p.add_argument("--k_min", type=int, default=1, help="per-frame coverage floor for stratified selection")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--warmup", type=int, default=3, help="samples excluded from timing stats (CUDA warm-up)")
    p.add_argument("--no_full", action="store_true", help="Skip the full-model reference (no retention/speedup).")
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa/eager/flash_attention_2).")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_mvbench_eval.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
