import os
import re
import json
import random
import argparse
import warnings
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from efficient_vlm.scorer import Scorer
from efficient_vlm.utils import einmahlHaan

def make_prompt(record, video_root, max_frames, max_pixels=None):
    abs_path = os.path.normpath(os.path.join(video_root, record["video"]["path"]))
    content = []
    for msg in record["messages"]:
        if msg["role"] != "user":
            continue
        for item in msg["content"]:
            if item.get("type") == "video":
                vid = {"type": "video", "video": abs_path, "nframes": max_frames}
                if max_pixels is not None:
                    vid["max_pixels"] = max_pixels
                content.append(vid)
            else:
                content.append({"type": "text", "text": item["text"]})
    return [{"role": "user", "content": content}]

def options_and_answer(record):
    """Extract the option letters and the correct-answer index from a record.

    Handles both NeXTVideo jsonl layouts:
      * explicit fields -- top-level ``all_choices``/``index2ans`` + ``gt`` letter
        (the val format, user-only messages); and
      * chat-encoded -- options as 'A. ...', 'B. ...' lines in the user turn's
        text and the answer letter in the assistant turn (the train format).
    Returns ``(choices, correct_idx)`` where ``choices`` is the list of option
    letters and ``choices[correct_idx]`` is the answer letter, or ``(None, None)``
    if the record is unusable.
    """
    # Path 1: explicit top-level fields (val format).
    choices = record.get("all_choices") or sorted((record.get("index2ans") or {}).keys())
    gt = record.get("gt")
    if choices and gt is not None:
        gt = gt.strip().upper()
        gt = gt[0] if gt else ''
        if gt in choices:
            return choices, choices.index(gt)
        return None, None

    # Path 2: parse the chat messages (train format).
    user_text, answer = None, None
    for msg in record.get("messages", []):
        role = msg.get("role")
        for item in msg.get("content", []):
            if item.get("type") != "text" or not item.get("text"):
                continue
            if role == "user":
                user_text = item["text"]
            elif role == "assistant":
                answer = item["text"]
    if not user_text or not answer:
        return None, None
    # Option letters that appear as their own "A. ...", "B. ..." lines.
    choices = re.findall(r'^\s*([A-Z])\.\s', user_text, flags=re.MULTILINE)
    gt = answer.strip().upper()
    gt = gt[0] if gt else ''   # answer may be a bare letter ('D') or 'D. two'
    if not choices or gt not in choices:
        return None, None
    return choices, choices.index(gt)


def build_full_positions(model, input_ids, video_positions, video_grid_thw, attention_mask):
    mm = torch.zeros_like(input_ids)
    mm[0, video_positions] = 2  # video == 2 per get_rope_index
    pos, _ = model.model.get_rope_index(
        input_ids, mm_token_type_ids=mm,
        video_grid_thw=video_grid_thw, attention_mask=attention_mask)
    return pos  # (3,1,S)


def topk_overlap(pred, target, k):
    """fraction of target's top-k that also appear in pred's top-k (recall@k)."""
    k = min(k, pred.numel())
    tp = set(torch.topk(target, k).indices.tolist())
    pp = set(torch.topk(pred, k).indices.tolist())
    return len(tp & pp) / k


def compute_oracle(model, processor, rec, video_token_id, device, args, video_root):
    """Run the frozen VLM forward+backward to build the gradient-oracle target.

    Returns ``(features, oracle_target, oracle)`` where ``features`` are the
    detached merged visual tokens (fp32, the scorer's input), ``oracle_target``
    is the min-max normalised [0,1] saliency used as the BCE target, and
    ``oracle`` is the raw relu saliency (for the EVT tail diagnostic). Returns
    ``None`` when the sample is unusable (missing answer / no video tokens).

    Note: this needs a backward pass through the (frozen) VLM, so callers must
    NOT wrap it in ``torch.no_grad()`` -- only the scorer's own forward is.
    """
    choices, correctIdx = options_and_answer(rec)
    if choices is None:
        return None
    gt_token = processor.tokenizer(choices[correctIdx], add_special_tokens=False).input_ids[0]
    prompt = make_prompt(rec, video_root, args.max_frames, args.max_pixels)
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors='pt'
    )
    input_ids = inputs["input_ids"].to(device)
    attention = inputs['attention_mask'].to(device)
    video_pos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    if video_pos.numel() == 0:
        warnings.warn("[Warning] found zero video tokens. Skipping")
        return None
    video_grid_thw = inputs['video_grid_thw'].to(device)
    pixels = inputs['pixel_values_videos'].to(device)
    with torch.no_grad():
        visual_feats = model.get_video_features(pixels, video_grid_thw).pooler_output
        visual_feats = torch.cat(visual_feats, dim=0).to(device)
    visual_feats = visual_feats.detach().requires_grad_(True)
    input_embeds = model.get_input_embeddings()(input_ids).clone()
    input_embeds[0, video_pos] = visual_feats.to(input_embeds.dtype)
    position_ids = build_full_positions(model, input_ids, video_pos, video_grid_thw, attention)
    out = model(inputs_embeds=input_embeds, position_ids=position_ids, attention_mask=attention, use_cache=False)
    L_answer = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)[gt_token]
    if visual_feats.grad is not None:
        visual_feats.grad = None
    L_answer.backward()
    gradient = visual_feats.grad.float()
    oracle = F.relu((gradient * visual_feats.detach().float()).sum(dim=-1))
    oracle_min, oracle_max = oracle.min(), oracle.max()
    oracle_target = ((oracle - oracle_min) / (oracle_max - oracle_min + 1e-8)).detach()
    features = visual_feats.detach().float()
    del out, input_embeds, gradient, visual_feats, L_answer
    model.zero_grad(set_to_none=True)
    return features, oracle_target, oracle.detach()


def save_checkpoint(scorer, optimiser, scheduler, step, ckpt_dir, loss, tag=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"oracle_scorer_{tag}.pt" if tag else f"oracle_scorer_step_{step}.pt"
    path = os.path.join(ckpt_dir, name)
    torch.save({"step": step, "model_state": scorer.state_dict(), "loss": loss,
                "optimiser_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict()}, path)
    print(f"  saved checkpoint -> {path} (loss {loss:.4f})")


def run_validation(model, processor, scorer, val_records, args, device,
                   video_token_id, max_samples, val_ratios, val_root):
    """Evaluate the scorer against the gradient oracle on a held-out set.

    Reports mean BCE, Spearman rho, and top-k recall (k = round(ratio*n_video))
    at each retention ratio. The scorer is switched to eval() for the pass (so
    dropout is off) and back to train() after. The oracle target still needs a
    VLM backward per sample, so this is NOT a no_grad pass -- only the scorer's
    forward is wrapped. Returns ``(metrics_dict, n_evaluated)``.
    """
    was_training = scorer.training
    scorer.eval()
    losses, rhos, n = [], [], 0
    recalls = {r: [] for r in val_ratios}
    for rec in val_records:
        if n >= max_samples:
            break
        result = compute_oracle(model, processor, rec, video_token_id, device, args, val_root)
        if result is None:
            continue
        features, oracle_target, _ = result
        with torch.no_grad():
            logits = scorer(features.unsqueeze(0))[0]
        losses.append(F.binary_cross_entropy_with_logits(logits, oracle_target).item())
        rho = spearmanr(logits.cpu().numpy(), oracle_target.cpu().numpy()).statistic
        if rho == rho:  # skip NaN (constant inputs)
            rhos.append(rho)
        for r in val_ratios:
            k = max(1, int(round(r * logits.numel())))
            recalls[r].append(topk_overlap(logits, oracle_target, k))
        n += 1
    if was_training:
        scorer.train()

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    metrics = {"val/bce": _mean(losses), "val/spearman_rho": _mean(rhos)}
    for r in val_ratios:
        metrics[f"val/recall@{int(round(r * 100))}"] = _mean(recalls[r])
    return metrics, n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                         entity=args.wandb_entity, config=vars(args))
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }[args.dtype]
    # SDPA (PyTorch's fused attention) instead of eager: the gradient oracle only
    # needs a forward+backward on the answer log-prob, never the attention weights,
    # so we don't pay eager's O(S^2) attention-matrix materialisation. Much faster
    # and lower memory on long video-token sequences.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=dtype, device_map='auto', attn_implementation='sdpa')
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    # diagnostic print.
    print(f"Video token ID: {video_token_id}")
    with open(args.data_file) as filehead:
        records = [json.loads(l) for l in filehead if l.strip()]
    random.Random(args.seed).shuffle(records)

    # Held-out validation set (optional). Shuffled with a fixed seed so the first
    # --val_samples records form a stable subset across evaluations.
    val_records = None
    if args.val_file:
        with open(args.val_file) as fh:
            val_records = [json.loads(l) for l in fh if l.strip()]
        random.Random(args.seed).shuffle(val_records)
        print(f"Loaded {len(val_records)} val records from {args.val_file}")

    scorer = None
    optimiser = None
    scheduler = None

    # Optionally resume: the scorer is built lazily (we need the feature width
    # from the first sample), so the actual state load happens in that block.
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        print(f"Resuming from {args.resume} (step {resume_state['step']})")

    # `step` counts *optimiser* steps; with --grad_accum > 1 each step accumulates
    # gradients over that many samples (effective batch size) before updating.
    step = 0
    micro = 0           # samples accumulated toward the current optimiser step
    window_loss = 0.0   # un-scaled loss summed over the current accumulation window
    cumulativeLoss = 0.0
    last_loss = 0.0
    last_logits = last_target = None    # most recent sample, for the rho diagnostic
    best_val_metric = float('-inf')     # for best-checkpoint tracking (--best_metric)
    spearmanHist, recallHist, xiHist = [], [], []
    while step < args.max_steps:
        for rec in records:
            if step >= args.max_steps:
                break
            result = compute_oracle(model, processor, rec, video_token_id, device, args, args.video_root)
            if result is None:
                continue
            features, oracle_target, oracle = result

            if scorer is None:
                scorer = Scorer(input_dim=features.shape[-1], hidden_dim=args.hidden_dim).to(device)
                optimiser = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.max_steps, eta_min=args.learning_rate*0.1)
                scorer.train()
                print(f"Built scorer with input dim: {features.shape[-1]} and hidden dim: {args.hidden_dim}")
                if resume_state is not None:
                    scorer.load_state_dict(resume_state["model_state"])
                    optimiser.load_state_dict(resume_state["optimiser_state"])
                    scheduler.load_state_dict(resume_state["scheduler_state"])
                    step = int(resume_state["step"])
                    print(f"Restored scorer/optimiser/scheduler; continuing from step {step}")
                    resume_state = None

            logits = scorer(features.unsqueeze(0))[0]
            # Scale by grad_accum so the accumulated gradient is the *mean* over
            # the effective batch, matching a single larger-batch update.
            loss = F.binary_cross_entropy_with_logits(logits, oracle_target) / args.grad_accum
            loss.backward()
            micro += 1
            window_loss += loss.item() * args.grad_accum    # track un-scaled loss
            last_logits, last_target = logits.detach(), oracle_target.detach()
            if oracle.numel() >= 50:
                xiHist.append(einmahlHaan(oracle))

            if micro < args.grad_accum:
                continue    # keep accumulating before the optimiser step

            # A full effective batch is ready -> update.
            nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()
            micro = 0

            with torch.no_grad():
                k = max(1, int(0.25 * last_logits.numel()))
                recallHist.append(topk_overlap(last_logits, last_target, k))
                rho = spearmanr(last_logits.cpu().numpy(), last_target.cpu().numpy()).statistic
                spearmanHist.append(rho if rho == rho else 0.0)

            last_loss = window_loss / args.grad_accum
            window_loss = 0.0
            cumulativeLoss += last_loss
            step += 1
            if step % args.log_interval == 0:
                average_loss = cumulativeLoss / args.log_interval
                xi = np.median(xiHist) if xiHist else float('nan')
                message = (f"step {step}/{args.max_steps} | BCE {average_loss:.4f} | "
                   f"rho {np.mean(spearmanHist[-args.log_interval:]):.3f} | "
                   f"recall@25% {np.mean(recallHist[-args.log_interval:]):.3f} | "
                   f"oracle xi {xi:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")
                print(message)
                if run is not None:
                    run.log({"train/bce": average_loss, "train/spearman_rho": float(np.mean(spearmanHist[-args.log_interval:])), "train/recall@25": float(np.mean(recallHist[-args.log_interval:])), "train/lr": scheduler.get_last_lr()[0]}, step=step)
                cumulativeLoss = 0.0
            if step % args.save_every == 0:
                save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, last_loss)
            if val_records is not None and step % args.val_interval == 0:
                val_root = args.val_video_root or args.video_root
                val_metrics, n_val = run_validation(
                    model, processor, scorer, val_records, args, device,
                    video_token_id, args.val_samples, args.val_ratios, val_root,
                )
                rec_str = ", ".join(
                    f"R@{int(round(r*100))} {val_metrics[f'val/recall@{int(round(r*100))}']:.3f}"
                    for r in args.val_ratios
                )
                print(f"  [val] step {step}: bce {val_metrics['val/bce']:.4f}, "
                      f"rho {val_metrics['val/spearman_rho']:.4f}, {rec_str} over {n_val} samples")
                if run is not None:
                    run.log(val_metrics, step=step)
                # Save the best scorer by the chosen selection-aligned metric.
                mval = val_metrics.get(f"val/{args.best_metric}")
                if mval is not None and mval == mval and mval > best_val_metric:
                    best_val_metric = mval
                    save_checkpoint(scorer, optimiser, scheduler, step,
                                    args.checkpoint_dir, val_metrics["val/bce"], tag="best")
        random.Random(args.seed + step).shuffle(records)

    # Final checkpoint, unless the last step already triggered a periodic save.
    if scorer is not None and step % args.save_every != 0:
        save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, last_loss)
    print('Training complete.')
    if run is not None:
        run.finish()

def parse_args():
    p = argparse.ArgumentParser(description="Online gradient-oracle scorer training.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True, help="Training jsonl (NExT-QA format).")
    p.add_argument("--video_root", required=True)
    p.add_argument("--val_file", type=str, default=None, help="Held-out validation jsonl. When set, periodically evaluates BCE/rho/recall and saves a best-by-metric checkpoint.")
    p.add_argument("--val_video_root", type=str, default=None, help="Video root for the validation set. Defaults to --video_root.")
    p.add_argument("--val_interval", type=int, default=500, help="Run validation every this many steps.")
    p.add_argument("--val_samples", type=int, default=100, help="Number of held-out samples to evaluate each validation pass.")
    p.add_argument("--val_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="Retention ratios at which to report top-k recall during validation.")
    p.add_argument("--best_metric", type=str, default="recall@25", help="Validation metric for the best checkpoint, without the 'val/' prefix (e.g. recall@25, spearman_rho).")
    p.add_argument("--grad_accum", type=int, default=1, help="Accumulate gradients over this many samples per optimiser step (effective batch size).")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--checkpoint_dir", default="./checkpoints_oracle")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint (.pt) to resume scorer/optimiser/scheduler and step from.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="efficientvlm")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user); defaults to your default entity.")
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
