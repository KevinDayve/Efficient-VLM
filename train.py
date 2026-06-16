from efficient_vlm.attention_extractor import AttentionExtractor
from efficient_vlm.loss import listmle_loss
from efficient_vlm.scorer import Scorer
from efficient_vlm.utils import einmahlHaan, topk_recall, ndcg_at_k, select_pareto_stratified
import os
import json
import random
import warnings
import argparse
import torch
from scipy.stats import spearmanr
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


def get_patch_embeds(model: Qwen2_5_VLForConditionalGeneration, pixel_values: torch.Tensor, video_grid_thw: torch.Tensor, n_video: int = None) -> torch.Tensor:
    base = getattr(model, "model", model)
    visual = base.visual
    with torch.no_grad():
        out = visual(pixel_values, grid_thw=video_grid_thw)
    # Some HF versions wrap the output; unwrap to the feature tensor.
    feats = out if isinstance(out, torch.Tensor) else out.last_hidden_state
    if feats.dim() == 3:
        feats = feats.squeeze(0)
    # If the tower returned pre-merger tokens, apply the spatial merger so the
    # token count lines up 1:1 with the video placeholders (and the teacher scores).
    if n_video is not None and feats.shape[0] != n_video:
        if hasattr(visual, "merger"):
            feats = visual.merger(feats)
        else:
            ratio = feats.shape[0] // n_video
            feats = feats[: ratio * n_video].view(n_video, ratio, -1).mean(dim=1)
    return feats.unsqueeze(0)

def count_video_tokens(input_ids: torch.Tensor, video_token_id: int) -> int:
    return (input_ids == video_token_id).sum().item()


def save_checkpoint(scorer: nn.Module, optimiser: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, step: int, checkpoint_dir: str, loss: float, tag: str = None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    name = f"scorer_{tag}.pt" if tag else f"scorer_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, name)
    torch.save({
        "step": step,
        "model_state": scorer.state_dict(),
        "loss": loss,
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step} to path: {checkpoint_path} with loss: {loss}")


def make_conversation(sample, video_root, max_frames: int = 8, max_pixels: int = None):
    video_item = {
        "type": "video",
        "video": f"{video_root}/{sample['video']}",
        "nframes": max_frames,
    }
    if max_pixels is not None:
        video_item["max_pixels"] = max_pixels
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    video_item,
                    {
                        "type": "text",
                        "text": sample['question']
                    }
                ]
            },
            {
                "role": "assistant",
                "content": sample['answer']
            },
        ]
    }

def make_conversation_local(record: dict, video_root: str, max_frames: int = 8, max_pixels: int = None):
    """Adapt a rhymes-ai/NeXTVideo jsonl record to the prompt format the loop expects.

    Each record already carries a chat-style ``messages`` list (question + options
    in the user turn, answer letter in the assistant turn) and a separate
    ``video`` dict ``{"path": "./NExTVideo/<grp>/<id>.mp4", "num_frames": N}``.
    We resolve the relative video path against ``video_root`` and inline it into
    the video content item so ``process_vision_info`` can load the frames.

    ``max_pixels`` caps each frame's resolution (Qwen uses dynamic resolution, so
    this bounds the sequence length and the O(S^2) attention memory). It is read
    by ``qwen_vl_utils.process_vision_info``, which resizes the frames before the
    processor sees them.
    """
    rel_path = record["video"]["path"]
    # Override the per-record num_frames so the frame count is fixed by --max_frames
    # and matches evaluate.py (keeps the scorer's token counts consistent train/eval).
    nframes = max_frames
    abs_path = os.path.normpath(os.path.join(video_root, rel_path))
    prompt = []
    for msg in record["messages"]:
        content = []
        for item in msg["content"]:
            if item.get("type") == "video":
                vid_item = {"type": "video", "video": abs_path, "nframes": nframes}
                if max_pixels is not None:
                    vid_item["max_pixels"] = max_pixels
                content.append(vid_item)
            else:
                content.append({"type": "text", "text": item["text"]})
        prompt.append({"role": msg["role"], "content": content})
    return {"prompt": prompt}


def load_local_jsonl(data_file: str, video_root: str, seed: int, max_frames: int = 8, max_pixels: int = None):
    with open(data_file) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    random.Random(seed).shuffle(records)
    return [make_conversation_local(r, video_root, max_frames=max_frames, max_pixels=max_pixels) for r in records]


@torch.no_grad()
def run_validation(model, processor, scorer, attn_extractor, val_dataset, args,
                   device, special_ids, video_token_id, max_samples, val_ratios):
    """Evaluate the scorer on a held-out set.

    Reports mean ListMLE loss, full-ranking Spearman rho, and the
    selection-aligned metrics top-k recall / NDCG@k at each retention ratio in
    ``val_ratios`` (k = round(ratio * n_video)). Recall@k is the metric that
    actually predicts downstream quality: at retention r the scorer keeps the
    top-k tokens, so what matters is how many of the teacher's top tokens survive.

    The scorer is switched to eval() for the pass and back to train() after.
    Uses the same truncated forward as training (no gradients flow anywhere).
    Returns ``(metrics_dict, n_evaluated)``.
    """
    was_training = scorer.training
    scorer.eval()
    losses, rhos, n = [], [], 0
    recalls = {r: [] for r in val_ratios}
    ndcgs = {r: [] for r in val_ratios}
    # Selection-faithful recall: overlap between the tokens the deployed
    # stratified-Pareto selector would keep (from predicted scores) and the
    # teacher's top-k. Predicts deployed behaviour better than global recall@k.
    sel_recalls = {r: [] for r in val_ratios}
    for sample in val_dataset:
        if n >= max_samples:
            break
        prompt = sample['prompt']
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(prompt)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors='pt')
        pixel_values = inputs['pixel_values_videos'].to(device)
        input_ids = inputs['input_ids'].to(device)
        video_grid_thw = inputs['video_grid_thw'].to(device)
        video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
        if video_positions.numel() == 0:
            continue
        attn_extractor.set_sample(input_ids, video_token_id, special_ids)
        patch_embeds = get_patch_embeds(
            model, pixel_values, video_grid_thw, n_video=video_positions.numel()
        ).to(device)
        logits = scorer(patch_embeds.float())
        attentions = attn_extractor.truncated_forward(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'].to(device),
            pixel_values_videos=pixel_values,
            video_grid_thw=video_grid_thw,
            output_attentions=True,
            use_cache=False,
        )
        targets = attn_extractor.scores_from_attentions(attentions)
        del attentions
        if targets is None:
            continue
        targets = targets.to(device)
        losses.append(listmle_loss(logits, targets, top_m=args.top_m).item())
        pred, teach = logits[0].float(), targets[0].float()
        rho = spearmanr(pred.cpu().numpy(), teach.cpu().numpy()).statistic
        if rho == rho:  # skip NaN (constant inputs)
            rhos.append(rho)
        n_video = pred.numel()
        n_frames = int(video_grid_thw[0][0].item())
        for r in val_ratios:
            k = max(1, int(round(r * n_video)))
            recalls[r].append(topk_recall(pred, teach, k))
            nd = ndcg_at_k(pred, teach, k)
            if nd == nd:
                ndcgs[r].append(nd)
            # What the deployed selector actually keeps, vs the teacher's top-k.
            kept = set(select_pareto_stratified(pred, k, n_frames).tolist())
            teach_top = set(torch.topk(teach, min(k, n_video)).indices.tolist())
            sel_recalls[r].append(len(kept & teach_top) / max(1, len(teach_top)))
        n += 1
    if was_training:
        scorer.train()

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    metrics = {"val/loss": _mean(losses), "val/spearman_rho": _mean(rhos)}
    for r in val_ratios:
        pct = int(round(r * 100))
        metrics[f"val/recall@{pct}"] = _mean(recalls[r])
        metrics[f"val/ndcg@{pct}"] = _mean(ndcgs[r])
        metrics[f"val/sel_recall@{pct}"] = _mean(sel_recalls[r])
    return metrics, n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optional Weights & Biases logging (no-op unless --wandb is passed).
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
    # Resolve the compute dtype for the frozen VLM. bf16 is the default and the
    # right choice on Ampere+ (A10G/A100/H100): Qwen2.5-VL's activations and
    # attention logits routinely exceed float16's max (~65504), so with eager
    # attention the overflowing QK^T scores become inf -> softmax emits nan, which
    # poisons the captured teacher attention. bf16 shares float32's exponent range,
    # so it doesn't overflow. fp16 is only for pre-Ampere GPUs (T4/V100) lacking bf16.
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name, use_fast=True)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>") # Should return 151656
    # Special-token ids are excluded from the language query rows when reading
    # teacher attention, so tokens like <|im_end|>/padding don't add noise.
    special_ids = set(processor.tokenizer.all_special_ids)
    # Sanity check
    print(f"Video token ID: {video_token_id}")
    # Scorer/optimiser/scheduler are built lazily on the first batch, once we
    # know the real feature width. The scorer scores the *merged* visual tokens
    # (the ones the LLM attends to, dim = LLM token space e.g. 2048 for the 3B
    # model), so it aligns 1:1 with the language->video teacher scores. This is
    # NOT visual.config.hidden_size (1280, the pre-merger ViT width).
    scorer = None
    optimiser = None
    scheduler = None
    if args.data_file:
        print(f"Loading local jsonl: {args.data_file}")
        dataset = load_local_jsonl(args.data_file, args.video_root, args.seed, max_frames=args.max_frames, max_pixels=args.max_pixels)
        print(f"Loaded {len(dataset)} local samples.")
    else:
        dataset = load_dataset(args.dataset_name, split='train')
        dataset = dataset.map(lambda x: make_conversation(x, args.video_root, max_frames=args.max_frames, max_pixels=args.max_pixels))
        dataset = dataset.shuffle(seed=args.seed)

    # Held-out validation set (optional). Shuffled with a fixed seed so the first
    # --val_samples items form a stable subset across evaluations.
    val_dataset = None
    if args.val_file:
        val_root = args.val_video_root or args.video_root
        val_dataset = load_local_jsonl(args.val_file, val_root, args.seed, max_frames=args.max_frames, max_pixels=args.max_pixels)
        print(f"Loaded {len(val_dataset)} val samples from {args.val_file}")

    attn_extractor = AttentionExtractor(model, Layers=args.layers)

    # Optionally resume: the scorer is built lazily (we need the feature width from
    # the first batch), so the actual state load happens in the lazy-build block.
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        print(f"Resuming from {args.resume} (step {resume_state['step']})")

    # The training begins! (scorer is created + set to train() lazily on the first batch)
    # `step` counts *optimiser* steps; with --grad_accum > 1 each step accumulates
    # gradients over that many samples (effective batch size) before updating.
    step = 0
    micro = 0           # samples accumulated toward the current optimiser step
    window_loss = 0.0   # un-scaled loss summed over the current accumulation window
    cumulativeLoss = 0.0
    last_loss = 0.0
    last_logits = last_targets = None   # most recent sample, for the rho diagnostic
    best_val_metric = float('-inf')   # for best-checkpoint tracking (see --best_metric)
    # Raw (un-normalised) teacher scores accumulated over each log interval, used
    # to monitor the EVT Pareto tail index of visual-token importance.
    raw_score_history = []
    while step < args.max_steps:
        for sample in dataset:
            if step >= args.max_steps:
                break
            prompt = sample['prompt']
            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(prompt)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors='pt',
            )
            pixel_values = inputs['pixel_values_videos'].to(device)
            input_ids = inputs['input_ids'].to(device)
            video_grid_thw = inputs['video_grid_thw'].to(device)
            # Absolute indices of the video tokens in the sequence. The video
            # block is contiguous and ordered the same as the ViT patch embeds,
            # so these indices align the teacher scores with the scorer inputs.
            video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
            if video_positions.numel() == 0:
                continue
            # Sets attn_extractor.video_positions and the special-token-excluding
            # query mask used by scores_from_attentions.
            attn_extractor.set_sample(input_ids, video_token_id, special_ids)
            patch_embeds = get_patch_embeds(
                model, pixel_values, video_grid_thw, n_video=video_positions.numel()
            ).to(device)

            # Build the scorer from the real (merged) feature width on the first
            # batch -- 2048 for the 3B model, not visual.config.hidden_size.
            if scorer is None:
                scorer = Scorer(input_dim=patch_embeds.shape[-1], hidden_dim=args.hidden_dim).to(device)
                optimiser = torch.optim.AdamW(
                    scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=args.max_steps, eta_min=args.learning_rate * 0.1,
                )
                scorer.train()
                print(f"Scorer built: input_dim={patch_embeds.shape[-1]}, "
                      f"{sum(p.numel() for p in scorer.parameters()) / 1e3:.0f}K params")
                if resume_state is not None:
                    scorer.load_state_dict(resume_state["model_state"])
                    optimiser.load_state_dict(resume_state["optimiser_state"])
                    scheduler.load_state_dict(resume_state["scheduler_state"])
                    step = int(resume_state["step"])
                    print(f"Restored scorer/optimiser/scheduler; continuing from step {step}")
                    resume_state = None

            # forward for the scoring module; keep the scorer in fp32 (stable for
            # LayerNorm/AdamW) and cast the half-precision features up to match its weights.
            logits = scorer(patch_embeds.float())
            # Truncated forward: capture attention at the critical layers and abort
            # right after max(layers), so the VLM never computes the layers above.
            with torch.no_grad():
                attentions = attn_extractor.truncated_forward(
                    input_ids=input_ids,
                    attention_mask=inputs['attention_mask'].to(device),
                    pixel_values_videos=pixel_values,
                    video_grid_thw=video_grid_thw,
                    output_attentions=True,
                    use_cache=False,
                )
            targets = attn_extractor.scores_from_attentions(attentions)
            # Raw scores for the EVT tail-index diagnostic (min-max normalisation
            # would destroy the heavy tail the estimator reads).
            raw_targets = attn_extractor.scores_from_attentions(attentions, normalise=False)
            del attentions
            if targets is None:
                warnings.warn("No attention scores extracted. Thus, skipping this sample.")
                continue
            if raw_targets is not None:
                raw_score_history.append(raw_targets.flatten().cpu())
            targets = targets.to(device)
            # Scale by grad_accum so the accumulated gradient is the *mean* over
            # the effective batch, matching a single larger-batch update.
            loss = listmle_loss(logits, targets, top_m=args.top_m) / args.grad_accum
            loss.backward()
            micro += 1
            window_loss += loss.item() * args.grad_accum   # track un-scaled loss
            last_logits, last_targets = logits.detach(), targets.detach()

            if micro < args.grad_accum:
                continue   # keep accumulating before the optimiser step

            # A full effective batch is ready -> update.
            nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()
            micro = 0

            last_loss = window_loss / args.grad_accum
            window_loss = 0.0
            cumulativeLoss += last_loss
            step += 1
            if step % args.log_interval == 0:
                rho = spearmanr(last_logits[0].float().cpu().numpy(), last_targets[0].float().cpu().numpy()).statistic
                avg_loss = cumulativeLoss / args.log_interval
                msg = f"Step {step} / {args.max_steps}, Loss: {avg_loss:.4f}, Correlation (between target and predicted): {rho}"
                metrics = {"train/loss": avg_loss, "train/spearman_rho": rho,
                           "train/lr": scheduler.get_last_lr()[0]}
                if raw_score_history:
                    all_scores = torch.cat(raw_score_history)
                    tail_index = einmahlHaan(all_scores)
                    median_score = all_scores.median().item()
                    msg += f", EVT tail index (eps): {tail_index:.4f}, median raw score: {median_score:.4e}"
                    metrics["train/evt_tail_index"] = tail_index
                    metrics["train/median_raw_score"] = median_score
                    raw_score_history.clear()
                print(msg)
                if run is not None:
                    run.log(metrics, step=step)
                cumulativeLoss = 0.0
            if step % args.save_every == 0:
                save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, last_loss)
            if val_dataset is not None and step % args.val_interval == 0:
                val_metrics, n_val = run_validation(
                    model, processor, scorer, attn_extractor, val_dataset, args,
                    device, special_ids, video_token_id, args.val_samples, args.val_ratios,
                )
                rec_str = ", ".join(
                    f"R@{int(round(r*100))} {val_metrics[f'val/recall@{int(round(r*100))}']:.3f}"
                    for r in args.val_ratios
                )
                print(f"  [val] step {step}: loss {val_metrics['val/loss']:.4f}, "
                      f"rho {val_metrics['val/spearman_rho']:.4f}, {rec_str} over {n_val} samples")
                if run is not None:
                    run.log(val_metrics, step=step)
                # Save the best scorer by the chosen selection-aligned metric
                # (default val/recall@25 -- predicts the aggressive-retention
                # downstream accuracy better than full-ranking rho).
                mval = val_metrics.get(f"val/{args.best_metric}")
                if mval is not None and mval == mval and mval > best_val_metric:
                    best_val_metric = mval
                    save_checkpoint(scorer, optimiser, scheduler, step,
                                    args.checkpoint_dir, val_metrics["val/loss"], tag="best")

    # Final checkpoint so the fully-trained scorer is always saved, even when
    # max_steps isn't a multiple of save_every. Skip if the last step already
    # triggered a periodic save (avoids a redundant double-write).
    if scorer is not None and step % args.save_every != 0:
        save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, last_loss)

    if run is not None:
        run.finish()


def parse_args():
    arguments = argparse.ArgumentParser(description="Train the projector for efficient VLM token pruning.")
    arguments.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help='The name of the VLM backbone model. Currently supports only Qwen variants.')
    arguments.add_argument("--dataset_name", type=str, default='lmms-lab/NExTVideo', help='The name of the training dataset. Should be compatible with HF datasets library. Ignored when --data_file is set.')
    arguments.add_argument("--data_file", type=str, default=None, help='Path to a local jsonl (rhymes-ai/NeXTVideo format). When set, overrides --dataset_name.')
    arguments.add_argument("--video_root", type=str, required=True, help='The root directory where video files are stored. Use the snapshot downloaded as the path.')
    arguments.add_argument("--val_file", type=str, default=None, help='Path to a held-out validation jsonl. When set, periodically evaluates loss + Spearman rho and saves a best-by-rho checkpoint.')
    arguments.add_argument("--val_video_root", type=str, default=None, help='Video root for the validation set. Defaults to --video_root.')
    arguments.add_argument("--val_interval", type=int, default=500, help='Run validation every this many steps.')
    arguments.add_argument("--val_samples", type=int, default=100, help='Number of held-out samples to evaluate each validation pass.')
    arguments.add_argument("--val_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75], help='Retention ratios at which to report top-k recall / NDCG@k during validation.')
    arguments.add_argument("--best_metric", type=str, default="recall@25", help="Validation metric for the best checkpoint, without the 'val/' prefix (e.g. recall@25, sel_recall@25, spearman_rho). Must correspond to a logged metric.")
    arguments.add_argument("--grad_accum", type=int, default=1, help='Accumulate gradients over this many samples per optimiser step (effective batch size). Reduces gradient noise vs the default bs=1.')
    arguments.add_argument("--hidden_dim", type=int, default=256, help='The hidden dimension of the scorer module.')
    arguments.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16], help="The layers from which to extract the attention scores.")
    arguments.add_argument("--learning_rate", type=float, default=1e-4, help='The learning rate for the optimiser module.')
    arguments.add_argument("--weight_decay", type=float, default=1e-2, help='The weight decay for the optimiser module.')
    arguments.add_argument("--max_steps", type=int, default=10000, help='The number of steps to train the scorer module.')
    arguments.add_argument("--log_interval", type=int, default=500, help='The interval (in steps) at which to log the training loss.')
    arguments.add_argument("--save_every", type=int, default=1000, help='The interval (in steps) at which to save model checkpoints.')
    arguments.add_argument("--top_m", type=int, default=None, help="The listmle loss to run over top m tokens to avoid noisy gradients. Defaults to `None`.")
    arguments.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Compute dtype for the frozen VLM. bf16 (default) is recommended on Ampere+ (A10G/A100/H100); fp16 is only for pre-Ampere GPUs (T4/V100) and risks NaN attention on Qwen2.5-VL; fp32 for max precision at 2x memory.")
    arguments.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    arguments.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    arguments.add_argument("--resume", type=str, default=None, help="Path to a checkpoint (.pt) to resume scorer/optimiser/scheduler and step from.")
    arguments.add_argument("--max_frames", type=int, default=8, help="Number of frames sampled per video. Sets nframes directly (overriding any per-record num_frames) so training matches evaluate.py's --max_frames and the token counts line up.")
    arguments.add_argument("--max_pixels", type=int, default=None, help="Cap per-frame resolution (in pixels, e.g. 100352 = 128*28*28) to bound sequence length and attention memory. Lower this to fix OOM.")
    arguments.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    arguments.add_argument("--wandb_project", type=str, default="efficientvlm", help="W&B project name.")
    arguments.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to an auto-generated name).")
    arguments.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user); defaults to your default entity.")
    return arguments.parse_args()

if __name__ == "__main__":
    train(parse_args())