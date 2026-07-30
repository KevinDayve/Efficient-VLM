"""
frame_clip_select.py -- Exp 7: can a CONTRASTIVE image-text model (SigLIP / CLIP / X-CLIP)
pick the answerable frame PRE-LLM?

The gap this fills. Every pre-LLM frame scorer in Exp 5 read out of Qwen's NON-contrastive
projection (V . q_hat) and failed at the random floor. The one untested lever is an
externally-aligned image-text space -- the thing Qwen's encoder cannot give. This scores
each frame with a contrastive model against the question and asks whether cross-modal MATCH
localizes the frame the LLM can actually answer from.

Discipline (from the arc). Match != answerability: a frame matching the question's TOPIC is
necessary, not sufficient (the same gamma>0-necessary-not-sufficient trap as Exp 4). So the
FIRST, load-bearing number here is a CEILING diagnostic, not a deployable scorer:

  qgt_oracle  cosine(frame, "question + CORRECT option")   -- USES THE LABEL, not deployable.
              It answers the only question that matters first: is the answerable frame even
              VISIBLE in an aligned space? within-clip AUC >> 0.5 => yes, engineering a real
              selector is worthwhile; AUC ~ 0.5 => contrastive matching fundamentally cannot
              see it => kills the direction for ~an hour of CLIP compute (no Qwen, no prompts).

Deployable scorers (no label):
  q_match     cosine(frame, question)                       -- "which frame is on topic"
  qopt_margin per frame, cosine to "question + option_j" for every option; score = top1-top2
              across options -- the frame that most DECISIVELY favors one option.

All three are judged by the Exp-5 machinery: accuracy of the picked frame vs anchors, and
per-frame within-clip AUC (does a high score mark an individually-CORRECT frame?), reported
PER TASK -- the expected win is object/scene tasks (Scene Transition, Object Existence,
Counterfactual, where conf won), the expected loss is temporal/counting (Action Localization,
Moving Direction), CLIP's known weak spots.

Labels/anchors. Which frames are individually answerable is a Qwen property, so per-frame
correctness is taken from the SAME single-frame protocol as frame_sufficiency (each frame as
the [f,f] zero-motion clip, option-letter logit readout). Qwen runs sdpa (no attentions
needed); the contrastive model scores raw PIL frames (never touches Qwen's temporal merge --
so this is also the clean control for the "is it the [f,f] merge?" question).

Per-frame X-CLIP note. X-CLIP's native output is a VIDEO embedding; to score a single frame
it is passed as a static num_frames clip ([f]*T -- the [f,f] motif again). Its temporal module
is somewhat at odds with per-frame selection, so SigLIP (default) is usually the better frame
scorer; X-CLIP is offered to test whether video-text training helps.

Run:
    python frame_clip_select.py --data_root ../MVBench/ --num_frames 8 --max_samples 40 \
        --max_pixels 200704 --clip_model siglip
    # long-form:  --data_root ~/Experiments/EgoSchema --tasks EgoSchema
    # video-text: --clip_model xclip   (or --clip_model clip)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))

from inference import (
    DATA_LIST, TEMPORAL_PATCH_SIZE, build_prompt, letter_token_ids, official_frames, predict,
)
from frame_sufficiency_mvbench import frames_prompt, build_inputs
from frame_tailindex_select_mvbench import _auc, pick     # reuse Exp-5 AUC + argmax helpers

PICK_SCORERS = ["q_match", "qopt_margin", "qgt_oracle"]

# A-PRIORI task grouping (defined by task semantics, NOT fit to the measured AUCs, so the
# bucket summary is not circular): does the answer localize to a single decisive frame/moment
# (selection-shaped) or require integrating across frames -- existence / counting / tracking /
# relative order (accumulation)? The Exp-7 hypothesis is that contrastive matching localizes
# the frame on selection-shaped tasks and is at chance on accumulation ones (e.g. Object
# Existence needs the whole clip, so no single frame is discriminative -- Exp 3). Edit freely;
# tasks absent here (e.g. EgoSchema) are simply omitted from the bucket summary.
TASK_BUCKET = {
    "Action Sequence": "selection", "Action Prediction": "selection",
    "Action Antonym": "selection", "Fine-grained Action": "selection",
    "Unexpected Action": "selection", "Object Interaction": "selection",
    "Scene Transition": "selection", "Character Order": "selection",
    "Episodic Reasoning": "selection", "Counterfactual Inference": "selection",
    "Action Localization": "selection", "State Change": "selection",
    "Object Existence": "accumulation", "Object Shuffle": "accumulation",
    "Moving Direction": "accumulation", "Action Count": "accumulation",
    "Moving Count": "accumulation", "Moving Attribute": "accumulation",
    "Egocentric Navigation": "accumulation",
}

CLIP_IDS = {
    "siglip": "google/siglip-base-patch16-224",
    "clip":   "openai/clip-vit-base-patch32",
    "xclip":  "microsoft/xclip-base-patch32",
}


class ClipScorer:
    """A uniform contrastive image/video-text scorer over SigLIP / CLIP / X-CLIP. Every
    backend exposes normalized image (per-frame) and text embeddings, so scoring is a plain
    cosine and all downstream code is backend-agnostic."""

    def __init__(self, kind, device, dtype):
        self.kind = kind
        self.device = device
        self.proc = AutoProcessor.from_pretrained(CLIP_IDS[kind])
        if kind == "xclip":
            from transformers import XCLIPModel
            self.model = XCLIPModel.from_pretrained(CLIP_IDS[kind], torch_dtype=dtype).to(device).eval()
            self.num_frames = self.model.config.vision_config.num_frames
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(CLIP_IDS[kind], torch_dtype=dtype).to(device).eval()
        # SigLIP's text tower is trained at a fixed length; CLIP/X-CLIP pad dynamically.
        self._text_pad = "max_length" if kind == "siglip" else True

    @staticmethod
    def _pool(feat):
        """The pooled embedding tensor. Some transformers versions return a bare tensor from
        get_*_features; others return a BaseModelOutputWithPooling / ModelOutput -- unwrap
        either to the pooled vector (falls back to the CLS/first token of last_hidden_state)."""
        if torch.is_tensor(feat):
            return feat
        for attr in ("pooler_output", "image_embeds", "text_embeds", "video_embeds"):
            v = getattr(feat, attr, None)
            if v is not None:
                return v
        lhs = getattr(feat, "last_hidden_state", None)
        if lhs is not None:
            return lhs[:, 0]
        raise TypeError(f"cannot extract an embedding tensor from {type(feat).__name__}")

    @torch.no_grad()
    def image_feats(self, frames):
        """(N, d) L2-normalized per-frame embeddings for a list of N PIL frames."""
        if self.kind == "xclip":                          # each frame -> a static num_frames clip
            vids = [[fr] * self.num_frames for fr in frames]
            px = self.proc(videos=vids, return_tensors="pt").to(self.device)
            feat = self._pool(self.model.get_video_features(**px))
        else:
            px = self.proc(images=frames, return_tensors="pt").to(self.device)
            feat = self._pool(self.model.get_image_features(**px))
        return torch.nn.functional.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def text_feats(self, texts):
        """(M, d) L2-normalized embeddings for a list of M strings."""
        tx = self.proc(text=list(texts), return_tensors="pt",
                       padding=self._text_pad, truncation=True).to(self.device)
        feat = self._pool(self.model.get_text_features(**tx))
        return torch.nn.functional.normalize(feat.float(), dim=-1)


def clip_scores(scorer, frames, question, candidates, answer):
    """The three length-N score vectors. Cosine is monotone, so raw cosine is fine for both
    argmax-pick and AUC (both invariant to per-clip monotone rescaling)."""
    img = scorer.image_feats(frames)                                  # (N, d)
    q = scorer.text_feats([question])                                 # (1, d)
    q_match = (img @ q.T).squeeze(1).cpu().numpy()                    # (N,)

    opt = scorer.text_feats([f"{question} {c}" for c in candidates])  # (n_opt, d)
    sim = (img @ opt.T).cpu().numpy()                                 # (N, n_opt)
    srt = np.sort(sim, axis=1)[:, ::-1]                               # per frame, options desc
    qopt_margin = srt[:, 0] - srt[:, 1]                               # top1 - top2 decisiveness

    g = scorer.text_feats([f"{question} {answer}"])                  # (1, d)  -- USES LABEL
    qgt_oracle = (img @ g.T).squeeze(1).cpu().numpy()                 # (N,) ceiling diagnostic
    return {"q_match": q_match, "qopt_margin": qopt_margin, "qgt_oracle": qgt_oracle}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Can a contrastive image-text model pick the answerable frame?")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["all"], help="task names, or 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct", help="the VLM (for labels/anchors).")
    p.add_argument("--clip_model", default="siglip", choices=list(CLIP_IDS), help="contrastive scorer backend.")
    p.add_argument("--num_frames", type=int, default=8, help="candidate frames (rounded to even).")
    p.add_argument("--max_samples", type=int, default=40, help="cap samples PER TASK.")
    p.add_argument("--max_pixels", type=int, default=200704)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out", default="frame_clip_select.json")
    args = p.parse_args()

    N = max(TEMPORAL_PATCH_SIZE, round(args.num_frames / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE)
    tasks = list(DATA_LIST) if args.tasks == ["all"] else args.tasks
    unknown = [t for t in tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(     # sdpa: labels only, no attentions
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    scorer = ClipScorer(args.clip_model, model.device, dtype)

    cols = ["all", "single_best", "single_mean", "middle"] + PICK_SCORERS
    correct = defaultdict(lambda: defaultdict(float))
    count = defaultdict(lambda: defaultdict(int))
    picks_log = []
    disc_pos = {s: [] for s in PICK_SCORERS}
    disc_neg = {s: [] for s in PICK_SCORERS}
    within_auc = {s: [] for s in PICK_SCORERS}
    within_auc_task = {s: defaultdict(list) for s in PICK_SCORERS}
    within_top1 = {s: [] for s in PICK_SCORERS}
    n_mixed = 0
    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")

    for task in tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        try:                                    # a task json absent under this data_root
            with open(os.path.join(json_dir, fname)) as fh:   # (e.g. EgoSchema under MVBench/)
                records = json.load(fh)
        except (FileNotFoundError, OSError) as e:
            tqdm.write(f"skip task [{task}]: {e}")
            continue
        if args.max_samples:
            records = records[:args.max_samples]

        for rec in tqdm(records, desc=f"{task} (N={N}, {args.clip_model})"):
            try:
                path = os.path.join(video_dir, subdir, rec["video"])
                text, letters, gt = build_prompt(rec)
                letter_ids = letter_token_ids(processor, letters)
                frames = official_frames(path, data_type, has_bound, rec, N)   # N PIL frames

                # per-frame answerability (Qwen labels) + anchors, single-frame [f,f] protocol
                singles = np.array([int(predict(model, build_inputs(
                    processor, model, frames_prompt([f, f], text, args.max_pixels)), letter_ids) == gt)
                    for f in frames], dtype=float)
                all_ok = int(predict(model, build_inputs(
                    processor, model, frames_prompt(frames, text, args.max_pixels)), letter_ids) == gt)

                # contrastive frame scores (raw PIL frames -- no Qwen merge)
                scores = clip_scores(scorer, frames, rec["question"], rec["candidates"], rec["answer"])
                mid = len(frames) // 2

                cor = singles.astype(bool)
                mixed = bool(cor.any() and (~cor).any())
                n_mixed += int(mixed)
                for s in PICK_SCORERS:
                    sc = scores[s]
                    valid = ~np.isnan(sc)
                    disc_pos[s].extend(sc[valid & cor].tolist())
                    disc_neg[s].extend(sc[valid & ~cor].tolist())
                    if mixed:
                        vs, vc = sc[valid], cor[valid]
                        if vc.any() and (~vc).any():
                            w = _auc(vs[vc], vs[~vc])
                            within_auc[s].append(w)
                            within_auc_task[s][task].append(w)
                            within_top1[s].append(float(vc[int(np.argmax(vs))]))

                correct[task]["all"] += all_ok;                       count[task]["all"] += 1
                correct[task]["single_best"] += int(singles.any());    count[task]["single_best"] += 1
                correct[task]["single_mean"] += float(singles.mean()); count[task]["single_mean"] += 1
                correct[task]["middle"] += singles[mid];               count[task]["middle"] += 1
                rec_picks = {}
                for s in PICK_SCORERS:
                    fp = pick(scores[s])
                    if fp is None:
                        continue
                    correct[task][s] += singles[fp]
                    count[task][s] += 1
                    rec_picks[s] = fp
                picks_log.append({"task": task, "video": rec.get("video"),
                                  "gt_frames": [i for i, c in enumerate(singles) if c > 0],
                                  "picks": rec_picks})
                torch.cuda.empty_cache()
            except Exception as e:
                tqdm.write(f"skip [{task}] {rec.get('video')}: {e}")
                continue

    # ---- report ----
    def acc(task, c):
        n = count[task][c]
        return 100.0 * correct[task][c] / n if n else float("nan")

    print(f"\n[clip_model={args.clip_model}]  {'task':26s} {'n':>4} " + " ".join(f"{c:>12s}" for c in cols))
    ntot = 0
    tot_c = defaultdict(float); tot_n = defaultdict(int)
    for task in tasks:
        n = count[task]["all"]
        if not n:
            continue
        ntot += n
        for c in cols:
            tot_c[c] += correct[task][c]; tot_n[c] += count[task][c]
        print(f"{task:26s} {n:>4} " + " ".join(f"{acc(task, c):11.1f}%" for c in cols))
    if ntot:
        def oacc(c):
            return 100.0 * tot_c[c] / tot_n[c] if tot_n[c] else float("nan")
        print(f"{'OVERALL':26s} {ntot:>4} " + " ".join(f"{oacc(c):11.1f}%" for c in cols))
        sm, sb = oacc("single_mean"), oacc("single_best")
        print(f"\n  anchors:  single_mean {sm:.1f}% (random-frame floor)   "
              f"single_best {sb:.1f}% (oracle ceiling)   all {oacc('all'):.1f}%")
        span = sb - sm
        for s in PICK_SCORERS:
            a_ = oacc(s)
            frac = 100.0 * (a_ - sm) / span if span else float("nan")
            note = "  <- USES LABEL (ceiling)" if s == "qgt_oracle" else ""
            print(f"  {s:11s} {a_:5.1f}%   vs single_mean {a_ - sm:+.1f}pts   "
                  f"vs single_best {a_ - sb:+.1f}pts   headroom {frac:+.0f}%{note}")

    # ---- discrimination: does a high contrastive score mark a CORRECT frame? ----
    disc = {}
    print(f"\n  --- does a high score mark a CORRECT frame? (n_mixed={n_mixed}) ---")
    for s in PICK_SCORERS:
        pos, neg = np.asarray(disc_pos[s]), np.asarray(disc_neg[s])
        p_auc = _auc(pos, neg)
        dmean = float(pos.mean() - neg.mean()) if pos.size and neg.size else float("nan")
        wa, wt = np.asarray(within_auc[s]), np.asarray(within_top1[s])
        w_auc = float(np.nanmean(wa)) if wa.size else float("nan")
        w_top1 = float(np.nanmean(wt)) if wt.size else float("nan")
        disc[s] = {"pooled_auc": p_auc, "mean_score_correct_minus_incorrect": dmean,
                   "within_clip_auc": w_auc, "top1_hit_on_mixed": w_top1, "n_mixed_scored": int(wa.size)}
        note = "  <- CEILING" if s == "qgt_oracle" else ""
        print(f"  {s:11s} pooled AUC {p_auc:.3f}  (Δmean {dmean:+.4f})   "
              f"within-clip AUC {w_auc:.3f}   top1-on-mixed {100 * w_top1:.1f}%  (n={wa.size}){note}")

    # ---- per-task within-clip AUC: the selection-vs-accumulation split ----
    per_task_auc = {}
    seen_tasks = [t for t in tasks if count[t]["all"]]
    if seen_tasks:
        print("\n  --- per-task within-clip AUC (mixed clips) ---")
        print(f"    {'task':26s} {'nmix':>5} " + " ".join(f"{s:>11s}" for s in PICK_SCORERS))
        for task in seen_tasks:
            row, nmix, cells = {}, 0, []
            for s in PICK_SCORERS:
                wa = np.asarray(within_auc_task[s].get(task, []))
                nmix = max(nmix, wa.size)
                v = float(np.nanmean(wa)) if wa.size else float("nan")
                row[s] = {"within_clip_auc": v, "n_mixed": int(wa.size)}
                cells.append(f"{v:11.3f}")
            per_task_auc[task] = row
            print(f"    {task:26s} {nmix:>5} " + " ".join(cells))

    # a-priori bucket summary: does contrastive matching pick the frame on SELECTION-shaped
    # tasks (a single decisive frame) but not ACCUMULATION ones? Pools per-clip AUCs over each
    # bucket's tasks (so it is weighted by mixed-clip count, matching the pooled metric).
    bucket_auc = {}
    buckets = sorted({TASK_BUCKET[t] for t in seen_tasks if t in TASK_BUCKET})
    if buckets:
        print("\n  --- within-clip AUC by task type (a-priori grouping, not fit to AUCs) ---")
        print(f"    {'bucket':14s} {'ntask':>5} {'nmix':>5} " + " ".join(f"{s:>11s}" for s in PICK_SCORERS))
        for b in buckets:
            btasks = [t for t in seen_tasks if TASK_BUCKET.get(t) == b]
            row, nmix, cells = {}, 0, []
            for s in PICK_SCORERS:
                vals = [w for t in btasks for w in within_auc_task[s].get(t, [])]
                nmix = max(nmix, len(vals))
                v = float(np.mean(vals)) if vals else float("nan")
                row[s] = {"within_clip_auc": v, "n_mixed": len(vals)}
                cells.append(f"{v:11.3f}")
            bucket_auc[b] = {"tasks": btasks, "scorers": row}
            print(f"    {b:14s} {len(btasks):>5} {nmix:>5} " + " ".join(cells))

    with open(args.out, "w") as fh:
        json.dump({"clip_model": args.clip_model, "num_frames": N, "cols": cols, "n_mixed": n_mixed,
                   "correct": {t: dict(correct[t]) for t in correct},
                   "count": {t: dict(count[t]) for t in count},
                   "discrimination": disc, "per_task_within_auc": per_task_auc,
                   "task_bucket_auc": bucket_auc, "picks": picks_log}, fh, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
