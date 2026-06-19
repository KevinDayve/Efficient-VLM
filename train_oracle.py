import os
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
    choices = record.get("all_choices") or sorted(record.get("index2ans", {}).keys())
    gt = record.get("gt")
    if not choices or gt is None:
        return None, None
    gt = gt.strip().upper()
    if gt not in choices:
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
 
 
def save_checkpoint(scorer, optimiser, scheduler, step, ckpt_dir, loss):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"oracle_scorer_step_{step}.pt")
    torch.save({"step": step, "model_state": scorer.state_dict(), "loss": loss,
                "optimiser_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict()}, path)
    print(f"  saved checkpoint -> {path} (loss {loss:.4f})")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=dtype, device_map='auto', attn_implementation='eager')
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
    scorer = None
    optimiser = None
    scheduler = None
    step = 0
    cumulativeLoss = 0.0
    spearmanHist, recallHist, xiHist = [], [], []
    while step < args.max_steps:
        for rec in records:
            if step >= args.max_steps:
                break
            choices, correctIdx = options_and_answer(rec)
            if choices is None:
                continue
            gt_token = processor.tokenizer(choices[correctIdx], add_special_tokens=False).input_ids[0]
            prompt = make_prompt(rec, args.video_root, args.max_frames, args.max_pixels)
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
                continue
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
            gradient = visual_feats.float()
            oracle = F.relu((gradient * visual_feats.detach().float()).sum(dim=-1))
            oracle_min, oracle_max = oracle.min(), oracle.max()
            oracle_target = ((oracle_min - oracle_max) / (oracle_max - oracle_min + 1e-8)).detach()
            features = visual_feats.detach().float()
            del out, input_embeds, gradient, visual_feats, L_answer
            model.zero_grad(set_to_none=True)
            if scorer is None:
                scorer = Scorer(input_dim=visual_feats.shape[-1], hidden_dim=args.hidden_dim).to(device)
                scorer.train()
                optimiser = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.max_steps, eta_min=args.learning_rate*0.1)
                print(f"Built scorer with input dim: {features.shape[-1]} and hidden dim: {args.hidden_dim}")
                logits = scorer(features.unsqueeze(0))[0]
                loss = F.binary_cross_entropy_with_logits(logits, oracle_target)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                optimiser.step()
                scheduler.step()
                cumulativeLoss += loss.item()
                with torch.no_grad():
                    k = max(1, int(0.25 * logits.numel()))
                    recallHist.append(topk_overlap(logits, oracle_target, k))
                    rho = spearmanr(logits.detach().cpu().numpy(), oracle_target.detach().cpu().numpy()).statistic
                    spearmanHist.append(rho if rho == rho else 0.0)
                    if oracle.numel() >= 50:
                        xiHist.append(einmahlHaan(oracle.detach()))
                step += 1
                if step % args.log_interval == 0:
                    average_loss = cumulativeLoss / args.log_interval
                    message = (f"step {step}/{args.max_steps} | BCE {average_loss:.4f} | "
                       f"rho {np.mean(spearmanHist[-args.log_interval:]):.3f} | "
                       f"recall@25% {np.mean(recallHist[-args.log_interval:]):.3f} | "
                       f"oracle xi {np.median(xiHist):.3f} | lr {scheduler.get_last_lr()[0]:.2e}")
                    print(message)
                    if run is not None:
                        run.log({"train/bce": average_loss, "train/sperman": float(np.mean(spearmanHist[-args.log_interval:])), "train/recall@25": float(np.mean(recallHist[-args.log_interval:])), "train/lr": scheduler.get_last_lr()[0]}, step=step)
                    cumulativeLoss = 0.0
                if step % args.save_every == 0:
                    save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, loss.item())

            if scorer is not None:
                save_checkpoint(scorer, optimiser, scheduler, step, args.checkpoint_dir, loss.item())
            print('Training complete.')

def parse_args():
    p = argparse.ArgumentParser(description="Online gradient-oracle scorer training.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_file", required=True, help="Training jsonl (NExT-QA format).")
    p.add_argument("--video_root", required=True)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--checkpoint_dir", default="./checkpoints_oracle")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="efficient-vlm-oracle")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())