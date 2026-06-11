from efficient_vlm.attention_extractor import AttentionExtractor, StopForwardPass
from efficient_vlm.loss import listmle_loss
from efficient_vlm.scorer import Scorer
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


def save_checkpoint(scorer: nn.Module, optimiser: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, step: int, checkpoint_dir: str, loss: float):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"scorer_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state": scorer.state_dict(),
        "loss": loss,
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step} to path: {checkpoint_path} with loss: {loss}")


def make_conversation(sample, video_root, max_frames: int = 8):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_root}/{sample['video']}",
                        "nframes": max_frames
                    },
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

def make_conversation_local(record: dict, video_root: str, default_frames: int = 8):
    """Adapt a rhymes-ai/NeXTVideo jsonl record to the prompt format the loop expects.

    Each record already carries a chat-style ``messages`` list (question + options
    in the user turn, answer letter in the assistant turn) and a separate
    ``video`` dict ``{"path": "./NExTVideo/<grp>/<id>.mp4", "num_frames": N}``.
    We resolve the relative video path against ``video_root`` and inline it into
    the video content item so ``process_vision_info`` can load the frames.
    """
    rel_path = record["video"]["path"]
    nframes = int(record["video"].get("num_frames", default_frames))
    abs_path = os.path.normpath(os.path.join(video_root, rel_path))
    prompt = []
    for msg in record["messages"]:
        content = []
        for item in msg["content"]:
            if item.get("type") == "video":
                content.append({"type": "video", "video": abs_path, "nframes": nframes})
            else:
                content.append({"type": "text", "text": item["text"]})
        prompt.append({"role": msg["role"], "content": content})
    return {"prompt": prompt}


def load_local_jsonl(data_file: str, video_root: str, seed: int):
    with open(data_file) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    random.Random(seed).shuffle(records)
    return [make_conversation_local(r, video_root) for r in records]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name, use_fast=True)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>") # Should return 151656
    # Sanity check
    print(f"Video token ID: {video_token_id}")
    vit_dim = model.model.visual.config.hidden_size
    scorer = Scorer(input_dim=vit_dim, hidden_dim=args.hidden_dim).to(device)

    # optimisers
    optimiser = torch.optim.AdamW(
        scorer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.max_steps, eta_min=args.learning_rate * 0.1
    )
    if args.data_file:
        print(f"Loading local jsonl: {args.data_file}")
        dataset = load_local_jsonl(args.data_file, args.video_root, args.seed)
        print(f"Loaded {len(dataset)} local samples.")
    else:
        dataset = load_dataset(args.dataset_name, split='train')
        dataset = dataset.map(lambda x: make_conversation(x, args.video_root))
        dataset = dataset.shuffle(seed=args.seed)

    attn_extractor = AttentionExtractor(model, Layers=args.layers)
    
    # The training begins!
    scorer.train()
    step = 0
    cumulativeLoss = 0.0
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
            attn_extractor.video_positions = video_positions
            patch_embeds = get_patch_embeds(
                model, pixel_values, video_grid_thw, n_video=video_positions.numel()
            ).to(device)

            # forward for the scoring module;
            logits = scorer(patch_embeds)
            with attn_extractor:
                with torch.no_grad():
                    try:
                        model(
                            input_ids=input_ids,
                            attention_mask=inputs['attention_mask'].to(device),
                            pixel_values=pixel_values,
                            video_grid_thw=video_grid_thw,
                            output_attentions=True,
                        )
                    except StopForwardPass:
                        pass  # truncated at the last critical layer; attention already captured
            targets = attn_extractor.get_scores()
            if targets is None:
                warnings.warn("No attention scores extracted. Thus, skipping this sample.")
                attn_extractor._store.clear()
                continue
            targets = targets.to(device)
            loss = listmle_loss(logits, targets, top_m=args.top_m)
            optimiser.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()

            attn_extractor._store.clear()
            cumulativeLoss += loss.item()
            step += 1
            if step % args.log_interval == 0:
                rho = spearmanr(logits[0].detach().float().cpu().numpy(), targets[0].detach().float().cpu().numpy()).statistic
                print(f"Step {step} / {args.max_steps}, Loss: {cumulativeLoss / args.log_interval:.4f}, Correlation (between target and predicted): {rho}")
                cumulativeLoss = 0.0
            if step % args.save_every == 0:
                save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, loss.item())
            

def parse_args():
    arguments = argparse.ArgumentParser(description="Train the projector for efficient VLM token pruning.")
    arguments.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help='The name of the VLM backbone model. Currently supports only Qwen variants.')
    arguments.add_argument("--dataset_name", type=str, default='lmms-lab/NExTVideo', help='The name of the training dataset. Should be compatible with HF datasets library. Ignored when --data_file is set.')
    arguments.add_argument("--data_file", type=str, default=None, help='Path to a local jsonl (rhymes-ai/NeXTVideo format). When set, overrides --dataset_name.')
    arguments.add_argument("--video_root", type=str, required=True, help='The root directory where video files are stored. Use the snapshot downloaded as the path.')
    arguments.add_argument("--hidden_dim", type=int, default=256, help='The hidden dimension of the scorer module.')
    arguments.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16], help="The layers from which to extract the attention scores.")
    arguments.add_argument("--learning_rate", type=float, default=1e-4, help='The learning rate for the optimiser module.')
    arguments.add_argument("--weight_decay", type=float, default=1e-2, help='The weight decay for the optimiser module.')
    arguments.add_argument("--max_steps", type=int, default=10000, help='The number of steps to train the scorer module.')
    arguments.add_argument("--log_interval", type=int, default=500, help='The interval (in steps) at which to log the training loss.')
    arguments.add_argument("--save_every", type=int, default=1000, help='The interval (in steps) at which to save model checkpoints.')
    arguments.add_argument("--top_m", type=int, default=None, help="The listmle loss to run over top m tokens to avoid noisy gradients. Defaults to `None`.")
    arguments.add_argument("--fp16", action="store_true", help="Load the frozen VLM in float16 (recommended for fitting/speed).")
    arguments.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    arguments.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    return arguments.parse_args()

if __name__ == "__main__":
    train(parse_args())