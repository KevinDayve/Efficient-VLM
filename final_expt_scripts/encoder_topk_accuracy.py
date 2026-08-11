"""
encoder_topk_accuracy.py -- is the ENCODER's structure exploitable? Accuracy under
top-K visual-token pruning where the ranking is read from the vision tower, against
random and evenly-spaced keeps, swept over the keep rate.

This is the encoder half of Claim 2/3 ("repeat (2) and (3) for the encoder").
tail_vs_layer.py --stack encoder says the encoder's importance is NOT concentrated
(gamma <= 0 at most layers, unlike the decoder's), and Exp 2 says no encoder-stage
signal predicts what the decoder ends up using. Both are statements about
representations. This script is the accuracy statement: if the encoder carried
rankable structure, an encoder-derived score would beat an unranked keep at matched
budget. Every selector prunes through the SAME mechanism at the SAME budget, so any
gap is the score's and nothing else's.

Why this is the deployable direction
------------------------------------
The decoder scorers of stage_topk_accuracy.py need a full-context forward before they
can rank anything, so pruning on them saves decode but not prefill or encoder compute.
An encoder-side score is available BEFORE the language model runs, which is the entire
point of a "pre-LLM pruner" -- it is the variant that would actually save compute. So a
negative result here is the stronger one: the cheap, deployable ranking fails too.

Selectors  (--selectors)
------------------------
enc_attn   top-K by mean-incoming ViT attention -- how much the rest of the image
           attends to this patch. Read over --enc_band, aggregated by --enc_agg
           (default: the band's HEAVIEST-TAILED layer per clip, i.e. the layer at
           which this clip's encoder importance is most concentrated, which is the
           best case top-K could ask for). This is tail_vs_layer.py's `--stack
           encoder` score exactly, so the gamma curves and these accuracies describe
           the same quantity.
enc_norm   top-K by ||v||, the L2 norm of the token's post-projector feature: the
           standard query-free saliency proxy, and the control that says whether
           enc_attn is doing anything a norm could not.
enc_query  top-K by v . q_hat, the token's post-projector feature projected on the
           unit mean question embedding -- the input-space "visual x query" score.
           This is the cheap pre-LLM proxy that Exp 2 found DOES moderately predict
           decoder importance at the token level (Spearman ~ +0.36, up to +0.43 with
           the real question). It is therefore the sharpest test in this file: a score
           with a real, positive correlation to the decoder's own importance, used as
           a selector. If a correlated-but-cheap score still does not beat random,
           correlation with the decoder was never the thing that mattered.
dec_mid    top-K by the DECODER's attention over --dec_band (stage_topk_accuracy.py's
           attn_mid), carried here as the in-LLM reference so the encoder scorers,
           the decoder scorer and the unranked floors all appear in ONE table on ONE
           set of clips. It costs no extra capture -- the dense forward runs anyway.
random     K visual tokens drawn uniformly at random (the distribution-free floor).
uniform    K evenly spaced tokens, midpoint of each M/K block (structured, content-free).
uniform_stagger  evenly spaced in time, rotated in space so frames do not share a
           spatial offset -- the coverage-maximising floor.
bot_*      (--include_bottom) BOTTOM-K by the same score. The sharpest control: if
           bot_x ~ enc_x, the score carries no usable order at all, not even a
           reversed one. (On the decoder side bottom-K is catastrophically bad, which
           is what says its score is real; whether the encoder's is too is exactly
           what this file measures.)

Where the pruning happens, and why not inside the ViT
-----------------------------------------------------
Tokens are dropped from the DECODER's visual block, at the input, exactly as in
stage_topk_accuracy.py -- only the SCORE comes from the encoder. Dropping patches
inside the vision tower instead would change the grid the projector pools over and the
positions the merger merges, so a drop in accuracy could not be attributed to the
ranking rather than to the broken geometry. Keeping one prune mechanism across both
files also means the encoder and decoder numbers are directly comparable, which is the
comparison the claim needs.

Mapping encoder patches onto decoder tokens
-------------------------------------------
The encoder scores RAW patches; the decoder consumes MERGED tokens. The map is per
backbone, and each one mirrors the model's own pooling operator rather than
approximating it:

    qwen      The ViT runs on a window-permuted sequence, in groups of
              spatial_merge_unit (= 4) patches that the merger turns into one decoder
              token; the model restores canonical order after merging. Window-order
              patch p therefore lands in decoder token window_index[p // 4], which is
              recovered from the tower's own window index. Under eager, the vision
              attention splits the sequence by cu_seqlens and concatenates the chunk
              outputs IN ORDER, so the captured score vector is in window order --
              the run asserts its length against the grid before trusting this.
    llava_ov  SigLIP's 27x27 patches per frame are pooled to 14x14 by a bilinear
              F.interpolate. The score map is pooled by the IDENTICAL call, so the
              score of a decoder token is the same combination of patch scores as its
              feature is. The clip's trailing newline token has no encoder counterpart
              and is given the clip's median score -- neutral, so it is neither
              preferentially kept nor preferentially dropped.
    llava     CLIP's 24x24 patches map one-to-one onto the 576 decoder tokens (no
              pooling); only the CLS column is dropped, as it never reaches the LM.

Several patches per decoder token are combined by --enc_pool (mean, the default, or max).

The text-only floor  (--text_only, on by default)
-------------------------------------------------
One extra forward per clip with EVERY visual token dropped: the rho -> 0 limit of the
same prune, i.e. what the model scores from the question and options alone. No keep
rate is worth reporting unless it sits above this, and it is the same operation (and so
the same number) as stage_topk_accuracy.py's floor on the same clips.

Accuracy and testing
--------------------
One forward per (clip, config), no generation; the prompt ends with "Best option:(" and
the prediction is the argmax over the option-letter token ids at the last position --
the same protocol as every other script in this family. Each selector is McNemar-tested
(exact, two-sided, paired per clip) against every unranked floor and against the
text-only floor.

Cost
----
n_clips x (1 + n_selectors x n_rhos + text_only) forwards. The dense one is eager on
BOTH stacks (the encoder scores need the ViT's attention weights materialised, which is
the memory-hungry part at 16 frames); the pruned ones are short. Defaults are 7
selectors x 5 rhos + 2 references = 37 forwards/clip, so use --max_samples first.

Run
---
    # Qwen, MVBench
    python encoder_topk_accuracy.py --data_root ~/Experiments/MVBench --tasks mvbench \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --num_segments 16 --max_pixels 200704 \
        --out enc_topk_mvb_qwen.json --plot enc_topk_mvb_qwen.png

    # LLaVA-OneVision, EgoSchema, with the bottom-K controls
    python encoder_topk_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --num_segments 16 \
        --include_bottom --out enc_topk_ego_llavaov.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from lv_knockout_accuracy import gold_index, option_token_ids
from stage_topk_accuracy import (attach_capture, is_baseline, keep_abs_idx, keep_local,
                                 mcnemar, parse_band, predict_pruned)
from tail_vs_layer import (ANSWER_PREFIX, DATA_LIST, DEFAULT_SEGMENTS, LLAVA_FAMILY,
                           MVBENCH_TASKS, attach_encoder_capture, build_inputs,
                           context_limit, encoder_attn_modules, infer_backbone, iter_clips,
                           load_model, moment_tail_index, sample_frames, text_model_of,
                           vision_tower_of, visual_query_masks)

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

ENCODER_SELECTORS = ["enc_attn", "enc_norm", "enc_query"]
DECODER_SELECTORS = ["dec_mid"]
BASELINE_SELECTORS = ["random", "uniform", "uniform_stagger"]
DEFAULT_SELECTORS = ENCODER_SELECTORS + DECODER_SELECTORS + BASELINE_SELECTORS
SCORED = ENCODER_SELECTORS + DECODER_SELECTORS          # the ones a bot_* control exists for


# --------------------------------------------------------------------------- #
# 1. Encoder patch scores -> decoder token scores
# --------------------------------------------------------------------------- #
def _pool_to(target: np.ndarray, idx: np.ndarray, src: np.ndarray, how: str) -> np.ndarray:
    """Combine every patch score that lands on the same decoder token."""
    if how == "max":
        out = np.full(target.shape, -np.inf)
        np.maximum.at(out, idx, src)
        return out
    out = np.zeros(target.shape)
    cnt = np.zeros(target.shape)
    np.add.at(out, idx, src)
    np.add.at(cnt, idx, 1.0)
    return out / np.maximum(cnt, 1.0)


def qwen_patch_map(model, grid_thw, n_patches: int) -> np.ndarray:
    """Window-order patch index -> canonical decoder token index, for Qwen.

    The tower reshapes the patch sequence into groups of spatial_merge_unit and permutes
    the GROUPS by window_index before the blocks run; the merger turns each group into
    one decoder token, and the model then un-permutes with argsort(window_index). So the
    decoder token that patch p feeds is window_index[p // spatial_merge_unit]. The window
    index is a pure function of the grid, so it is recomputed here rather than captured."""
    tower = vision_tower_of(model)
    unit = int(tower.spatial_merge_unit)
    try:
        from transformers.vision_utils import get_vision_window_index
        window_index, _ = get_vision_window_index(
            grid_thw, spatial_merge_size=int(tower.spatial_merge_size),
            window_size=int(tower.window_size), patch_size=int(tower.patch_size))
    except ImportError:                       # older transformers: the method on the tower
        window_index, _ = tower.get_window_index(grid_thw)

    window_index = np.asarray(window_index.detach().cpu(), dtype=np.int64)
    if n_patches % unit or window_index.size != n_patches // unit:
        raise RuntimeError(f"window index has {window_index.size} groups but the captured "
                           f"encoder score has {n_patches} patches (unit {unit}) -- the "
                           f"chunk order assumption does not hold on this model version")
    return np.repeat(window_index, unit)


def encoder_to_decoder(scores: np.ndarray, model, args, inputs, M: int, how: str) -> np.ndarray:
    """(M,) decoder-token scores from a (P,) raw-patch encoder score vector."""
    if args.backbone == "qwen":
        idx = qwen_patch_map(model, inputs["video_grid_thw"], scores.size)
        if idx.max(initial=-1) >= M:
            raise RuntimeError(f"patch map points at token {idx.max()} but the decoder has {M}")
        return _pool_to(np.zeros(M), idx, scores, how)

    cfg = model.config.vision_config
    side = int(cfg.image_size) // int(cfg.patch_size)
    if scores.size % (side * side):
        raise RuntimeError(f"{scores.size} patch scores is not a multiple of {side}x{side}")
    n_frames = scores.size // (side * side)

    if args.backbone == "llava":
        # CLIP: 24x24 patches -> 576 decoder tokens per frame, one to one.
        if scores.size != M:
            raise RuntimeError(f"{scores.size} patch scores vs {M} decoder tokens")
        return scores.astype(float)

    # OneVision: the same bilinear F.interpolate the features are pooled with, applied to
    # the score map, so a decoder token's score is the same mixture of patch scores as its
    # feature is a mixture of patch features.
    import math

    import torch.nn.functional as Fn
    grid = torch.from_numpy(scores.astype(np.float32)).view(n_frames, 1, side, side)
    out = int(math.ceil(side / 2))
    pooled = Fn.interpolate(grid, size=[out, out], mode="bilinear").flatten().numpy().astype(float)
    if pooled.size == M:
        return pooled
    if M - pooled.size == 1:
        # The clip's trailing newline is a visual position with no patch behind it.
        # Median = neutral: neither top-K nor bottom-K is biased toward taking it.
        return np.append(pooled, float(np.median(pooled)))
    raise RuntimeError(f"pooled {pooled.size} scores but the decoder block holds {M} tokens")


def band_reduce(per_layer: list[np.ndarray], band: list[int], agg: str, k_frac: float):
    """(scores, layer used or None, gamma there or None) over a band of encoder layers.

    heaviest: the band's most heavy-tailed layer for THIS clip -- the layer at which its
              importance is most concentrated, i.e. the most generous read for top-K.
    mean:     the band average of L1-normalised per-layer scores (raw means would just
              report whichever layer is loudest)."""
    if agg == "heaviest":
        gammas = [moment_tail_index(torch.from_numpy(per_layer[l]), k_frac) for l in band]
        best = int(max(range(len(band)),
                       key=lambda i: gammas[i] if np.isfinite(gammas[i]) else -np.inf))
        return per_layer[band[best]], band[best], gammas[best]
    stack = np.stack([per_layer[l] / max(per_layer[l].sum(), 1e-12) for l in band])
    return stack.mean(axis=0), None, None


def decoder_band_score(store: dict, band: list[int]) -> np.ndarray:
    """stage_topk_accuracy.py's attn_mid: the mean of L1-normalised per-layer scores."""
    normed = torch.stack([store[l] / store[l].sum().clamp_min(1e-12) for l in band])
    return normed.mean(dim=0).numpy()


# --------------------------------------------------------------------------- #
# 2. Sweep
# --------------------------------------------------------------------------- #
def resolve_args(args):
    if args.backbone == "auto":
        args.backbone = infer_backbone(args.model_name)
    if args.num_segments is None:
        args.num_segments = DEFAULT_SEGMENTS[args.backbone]

    if args.tasks == ["all"]:
        args.tasks = list(DATA_LIST)
    elif args.tasks == ["mvbench"]:
        args.tasks = MVBENCH_TASKS
    unknown = [t for t in args.tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    if args.backbone in LLAVA_FAMILY:
        if args.max_pixels is not None or args.min_pixels is not None:
            fixed = "576" if args.backbone == "llava" else "196"
            print(f"[warn] --max_pixels/--min_pixels are ignored on {args.backbone} "
                  f"(fixed {fixed} decoder tokens/frame).")
        args.max_pixels = args.min_pixels = None
    else:
        if args.min_pixels is None:
            args.min_pixels = args.max_pixels
        if args.min_pixels is not None and args.min_pixels <= 0:
            args.min_pixels = None

    selectors = list(args.selectors)
    if args.include_bottom:
        selectors += [f"bot_{s}" for s in selectors if s in SCORED]
    known = set(SCORED + BASELINE_SELECTORS + [f"bot_{s}" for s in SCORED])
    bad = [s for s in selectors if s not in known]
    if bad:
        raise ValueError(f"unknown selectors {bad}; choices: {sorted(known)}")
    return selectors


def main(args):
    selectors = resolve_args(args)
    rhos = sorted(set(args.rhos))
    configs = [f"{s}@{r:g}" for r in rhos for s in selectors]
    refs = ["full"] + (["text_only"] if args.text_only else [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[model] {args.model_name}  backbone={args.backbone}  dtype={args.dtype}  device={device}")
    model, processor, visual_token_id = load_model(args, dtype)     # eager on both stacks
    max_positions = context_limit(model)
    n_layers = len(text_model_of(model).layers)
    n_enc_layers = len(encoder_attn_modules(model, args.backbone))
    enc_band = parse_band(args.enc_band, n_enc_layers)
    dec_band = parse_band(args.dec_band, n_layers)

    need_enc_attn = any(s.endswith("enc_attn") for s in selectors)
    need_dec = any(s.endswith("dec_mid") for s in selectors)
    print(f"[model] {n_layers} decoder layers, {n_enc_layers} encoder layers, "
          f"{max_positions} LM positions")
    print(f"[stage] encoder band {enc_band[0]}-{enc_band[-1]} ({len(enc_band)}), "
          f"agg={args.enc_agg}, patches pooled by {args.enc_pool}")
    if need_dec:
        print(f"[stage] decoder reference band {dec_band[0]}-{dec_band[-1]} ({len(dec_band)})")
    print(f"[prune] input-space drop of DECODER visual tokens, original position ids kept, "
          f"budgets rho={rhos}")
    print(f"[prune] selectors: {selectors}  ->  {len(refs) + len(configs)} forwards/clip")
    if args.text_only:
        print("[floor] text-only: one forward per clip with every visual token dropped")
    print(f"[data] tasks={args.tasks}  {args.num_segments} frames/clip")

    dec_store, ctx = {}, {"capture": False}
    enc_store: dict = {}
    uninstall_dec = attach_capture(model, dec_store, ctx)
    uninstall_enc = attach_encoder_capture(model, args.backbone, enc_store)

    letter_cache = {}
    hits = {c: [] for c in refs + configs}
    per_task_hits, seen_by_task, per_sample, skipped = {}, {}, [], []
    heaviest_layers: dict = {}
    n_seen = 0
    try:
        for idx, (task, rec, path, data_type, bound) in enumerate(
                tqdm(list(iter_clips(args)), unit="clip")):
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": rec["video"], "reason": "missing file"})
                continue
            try:
                gold = gold_index(rec)
                n_opt = len(rec["candidates"])
                if n_opt not in letter_cache:
                    letter_cache[n_opt] = torch.tensor(
                        option_token_ids(processor.tokenizer, n_opt), device=device)
                letter_ids = letter_cache[n_opt]

                frames = sample_frames(path, data_type, bound, args.num_segments)
                inputs = build_inputs(processor, frames, rec, args, device, dtype)
                ids = inputs["input_ids"][0]
                S = ids.numel()
                if max_positions is not None and S > max_positions:
                    raise RuntimeError(f"sequence is {S} tokens but the LM holds "
                                       f"{max_positions} -- lower --num_segments")
                visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
                M = int(visual_idx.numel())
                if M < 50:
                    raise RuntimeError(f"too few visual tokens ({M} < 50)")

                # ---- the one dense forward: answer, encoder scores, decoder scores,
                #      and the sequence the pruned forwards slice ----
                dec_store.clear()
                enc_store.clear()
                ctx.update({"capture": True, "visual_idx": visual_idx, "text_q": text_q,
                            "embeds": None, "position_ids": None})
                with torch.no_grad():
                    dense = model(**inputs, use_cache=False)
                full_pred = int(torch.argmax(dense.logits[0, -1][letter_ids]).item())
                ctx["capture"] = False
                del dense

                embeds = ctx["embeds"]
                if embeds is None or embeds.shape[1] != S:
                    raise RuntimeError("did not capture the merged input embeddings")
                pos = ctx["position_ids"]
                if pos is None:
                    pos = torch.arange(S, device=embeds.device)[None]

                # Post-projector features of the visual block, in the LM's own input
                # space -- what enc_norm and enc_query are computed from.
                V = embeds[0][visual_idx.to(embeds.device)].float()
                q = embeds[0][text_q.to(embeds.device)].float().mean(0)
                q_hat = q / q.norm().clamp_min(1e-12)

                score = {"enc_norm": V.norm(dim=1).cpu().numpy(),
                         "enc_query": (V @ q_hat).cpu().numpy()}

                if need_enc_attn:
                    missing = [l for l in range(n_enc_layers) if l not in enc_store]
                    if missing:
                        raise RuntimeError(f"no encoder attention for layer(s) {missing[:5]}")
                    per_layer = [torch.cat(enc_store[l]).numpy() for l in range(n_enc_layers)]
                    raw, layer, _ = band_reduce(per_layer, enc_band, args.enc_agg, args.k_frac)
                    score["enc_attn"] = encoder_to_decoder(raw, model, args, inputs, M,
                                                           args.enc_pool)
                    if layer is not None:
                        heaviest_layers[layer] = heaviest_layers.get(layer, 0) + 1
                if need_dec:
                    missing = [l for l in dec_band if l not in dec_store]
                    if missing:
                        raise RuntimeError(f"no decoder attention for layer(s) {missing[:5]}")
                    score["dec_mid"] = decoder_band_score(dec_store, dec_band)

                for k, v in score.items():
                    if v.size != M:
                        raise RuntimeError(f"{k} produced {v.size} scores for {M} tokens")

                # ---- the floor, then one pruned forward per (rho, selector) ----
                preds, kept = {}, {}
                if args.text_only:
                    preds["text_only"] = predict_pruned(
                        model, embeds, pos,
                        keep_abs_idx(S, visual_idx, np.empty(0, dtype=np.int64)), letter_ids)

                rng = np.random.default_rng([args.seed, idx])
                for r in rhos:
                    K = int(min(M, max(1, round(r * M))))
                    for sel in selectors:
                        base = sel[4:] if sel.startswith("bot_") else sel
                        s = score.get(base)
                        # keep_local reads the "attn_"/"bot_" prefix to decide the sort
                        # direction; encoder selectors take the same top-K path.
                        name = sel if is_baseline(sel) else ("bot_x" if sel.startswith("bot_")
                                                             else "attn_x")
                        local = keep_local(name, s, M, K, rng, n_frames=args.num_segments)
                        preds[f"{sel}@{r:g}"] = predict_pruned(
                            model, embeds, pos, keep_abs_idx(S, visual_idx, local), letter_ids)
                    kept[f"{r:g}"] = K
            except Exception as e:
                ctx.update({"capture": False, "embeds": None, "position_ids": None})
                dec_store.clear()
                enc_store.clear()
                skipped.append({"task": task, "video": rec["video"],
                                "reason": f"{type(e).__name__}: {e}"})
                tqdm.write(f"skip [{task}] {rec['video']}: {type(e).__name__}: {e}")
                continue

            n_seen += 1
            seen_by_task[task] = seen_by_task.get(task, 0) + 1
            tc = per_task_hits.setdefault(task, {c: 0 for c in refs + configs})
            preds["full"] = full_pred
            for c in refs + configs:
                hit = int(preds[c] == gold)
                hits[c].append(hit)
                tc[c] += hit
            per_sample.append({"task": task, "question_idx": rec.get("question_idx"),
                               "video": rec["video"], "gold": gold, "n_options": n_opt,
                               "n_visual_tokens": M, "keep_budget": kept,
                               "pred_by_config": preds})

            del inputs, embeds, V
            ctx.update({"embeds": None, "position_ids": None})
            dec_store.clear()
            enc_store.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        uninstall_dec()
        uninstall_enc()

    if not n_seen:
        print("no usable clips -- check --data_root layout (json/ and video/).")
        return

    acc = {c: sum(hits[c]) / n_seen for c in refs + configs}
    tests = [dict(rho=r, selector=a, baseline=b,
                  delta_pts=100 * (acc[f"{a}@{r:g}"] - acc[f"{b}@{r:g}"]),
                  **mcnemar(hits[f"{a}@{r:g}"], hits[f"{b}@{r:g}"]))
             for r in rhos
             for a in selectors if not is_baseline(a)
             for b in selectors if is_baseline(b)]
    floor_tests = [dict(rho=r, selector=a, baseline="text_only",
                        delta_pts=100 * (acc[f"{a}@{r:g}"] - acc["text_only"]),
                        **mcnemar(hits[f"{a}@{r:g}"], hits["text_only"]))
                   for r in rhos for a in selectors] if args.text_only else []

    out = {"experiment": "encoder_topk_accuracy", "prune": "input_drop_visual_tokens",
           "score_source": "vision_encoder", "backbone": args.backbone,
           "model_name": args.model_name, "data_root": args.data_root, "tasks": args.tasks,
           "num_segments": args.num_segments, "max_pixels": args.max_pixels,
           "min_pixels": args.min_pixels, "n_layers": n_layers,
           "n_encoder_layers": n_enc_layers, "enc_band": enc_band, "enc_agg": args.enc_agg,
           "enc_pool": args.enc_pool, "dec_band": dec_band, "queries": args.queries,
           "estimator": "moment", "k_frac": args.k_frac, "seed": args.seed,
           "rhos": rhos, "selectors": selectors,
           "scoring": "argmax over option-letter tokens", "answer_prefix": ANSWER_PREFIX,
           "n_clips": n_seen, "full_accuracy": acc["full"],
           "text_only": args.text_only, "text_only_accuracy": acc.get("text_only"),
           "chance_accuracy": sum(1.0 / s["n_options"] for s in per_sample) / n_seen,
           "accuracy_by_config": acc,
           "accuracy_by_rho": {f"{r:g}": {s: acc[f"{s}@{r:g}"] for s in selectors} for r in rhos},
           "retention_by_rho": {f"{r:g}": {s: (acc[f"{s}@{r:g}"] / acc["full"]
                                               if acc["full"] > 0 else float("nan"))
                                           for s in selectors} for r in rhos},
           "above_floor_by_rho": ({f"{r:g}": {s: ((acc[f"{s}@{r:g}"] - acc["text_only"])
                                                  / (acc["full"] - acc["text_only"])
                                                  if acc["full"] != acc["text_only"]
                                                  else float("nan"))
                                              for s in selectors} for r in rhos}
                                  if args.text_only else None),
           "mcnemar": tests, "mcnemar_vs_text_only": floor_tests,
           "heaviest_encoder_layer_histogram": {str(k): v
                                                for k, v in sorted(heaviest_layers.items())},
           "per_task_seen": seen_by_task,
           "accuracy_by_task": {t: {c: h[c] / seen_by_task[t] for c in refs + configs}
                                for t, h in sorted(per_task_hits.items())},
           "skipped": skipped}

    print(f"\n==== encoder top-K vs. random/uniform ({n_seen} clips) ====")
    print(f"{args.backbone}: {args.model_name}, {args.num_segments} frames, "
          f"{len(seen_by_task)} task(s)")
    print(f"full (no pruning)   accuracy = {acc['full']:.4f}   <- ceiling")
    if args.text_only:
        print(f"text only (0 tokens) accuracy = {acc['text_only']:.4f}   <- floor")
    print(f"chance (1/n_options)         = {out['chance_accuracy']:.4f}\n")
    print(f"{'rho':>6} " + " ".join(f"{s:>16s}" for s in selectors))
    for r in rhos:
        print(f"{r:>6g} " + " ".join(f"{100 * acc[f'{s}@{r:g}']:15.1f}%" for s in selectors))
    print("\nMcNemar (exact, two-sided) against the unranked floors:")
    for t in tests:
        flag = "significant" if t["p_value"] < 0.05 else "n.s."
        print(f"  rho={t['rho']:<5g} {t['selector']:>14s} - {t['baseline']:<15s} "
              f"{t['delta_pts']:+6.2f} pts  (b={t['b']:>4d} c={t['c']:>4d}, "
              f"p={t['p_value']:.3g}, {flag})")
    if floor_tests:
        print("\nMcNemar against the text-only floor (does this budget beat seeing nothing?):")
        for t in floor_tests:
            flag = "significant" if t["p_value"] < 0.05 else "n.s."
            print(f"  rho={t['rho']:<5g} {t['selector']:>14s} - text_only "
                  f"{t['delta_pts']:+6.2f} pts  (p={t['p_value']:.3g}, {flag})")
    if heaviest_layers:
        print("\nheaviest-tailed encoder layer, over clips: "
              + ", ".join(f"L{k}x{v}" for k, v in sorted(heaviest_layers.items())))
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s)")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")

    if args.plot and rhos and selectors:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        xs = 100 * np.array(rhos, dtype=float)
        for i, s in enumerate(selectors):
            ys = np.array([100 * acc[f"{s}@{r:g}"] for r in rhos])
            plt.plot(xs, ys, "--" if is_baseline(s) else "-", marker="o", ms=4,
                     color=f"C{i}", label=s)
        plt.axhline(100 * acc["full"], color="gray", lw=0.8, ls=":", label="no pruning")
        if args.text_only:
            plt.axhline(100 * acc["text_only"], color="black", lw=0.8, ls="-.",
                        label="text only (0 visual tokens)")
        plt.xscale("log")
        plt.xlabel("visual tokens kept (%)")
        plt.ylabel("accuracy (%)")
        plt.title(f"{os.path.basename(args.model_name)} encoder top-K vs. floors "
                  f"({n_seen} clips, {args.num_segments}f)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Accuracy under top-K visual-token pruning ranked by ENCODER-side "
                    "scores, vs. random / evenly-spaced keeps, on MVBench / EgoSchema.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' (all 20 MVBench tasks) / 'all' (+ EgoSchema). "
                        "MVBench and EgoSchema live under different roots -- don't mix in one run.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_ov", "llava"], default="auto",
                   help="auto infers from --model_name (onevision/-ov- -> llava_ov, else "
                        "'llava' -> llava-1.5, else qwen).")
    p.add_argument("--rhos", type=float, nargs="*", default=[0.01, 0.05, 0.1, 0.25, 0.5],
                   help="keep rates: K = round(rho * M) visual tokens survive.")
    p.add_argument("--selectors", nargs="*", default=DEFAULT_SELECTORS,
                   help=f"any of {SCORED + BASELINE_SELECTORS}.")
    p.add_argument("--include_bottom", action="store_true",
                   help="also run bottom-K by each score -- the control that says whether "
                        "the ranking carries any usable order at all.")
    p.add_argument("--text_only", action=argparse.BooleanOptionalAction, default=True,
                   help="one extra forward per clip with EVERY visual token dropped: the "
                        "rho->0 floor of the same prune. --no-text_only skips it.")
    p.add_argument("--enc_band", default="0:-1",
                   help="0-indexed vision-tower layers the enc_attn score is read over "
                        "(default the whole stack). 'a:b' inclusive, negatives count from "
                        "the end, comma-separated; a leading minus needs the equals form, "
                        "--enc_band=-8:-1.")
    p.add_argument("--enc_agg", choices=["heaviest", "mean"], default="heaviest",
                   help="how the encoder band is reduced to one score (default heaviest: "
                        "the band's most heavy-tailed layer for this clip, i.e. top-K's "
                        "best case). mean = average of L1-normalised per-layer scores.")
    p.add_argument("--enc_pool", choices=["mean", "max"], default="mean",
                   help="how several patch scores landing on one decoder token combine.")
    p.add_argument("--dec_band", default="11:15",
                   help="0-indexed decoder layers for the dec_mid reference (default 11:15, "
                        "stage_topk_accuracy.py's attn_mid band).")
    p.add_argument("--queries", choices=["post", "all", "last"], default="post",
                   help="text query rows the decoder score and the question embedding use.")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="upper-tail fraction for the gamma used by '--enc_agg heaviest'.")
    p.add_argument("--num_segments", type=int, default=None,
                   help=f"frames sampled per clip. Default per backbone: {DEFAULT_SEGMENTS}.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="qwen only: per-frame pixel cap, e.g. 200704. Both LLaVA backbones "
                        "have a fixed frame size and ignore this.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="qwen only: floor on per-frame pixels. Defaults to --max_pixels; 0 opts out.")
    p.add_argument("--max_samples", type=int, default=None, help="cap on records PER TASK.")
    p.add_argument("--seed", type=int, default=0, help="seeds the random selector, per clip.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="encoder_topk_accuracy.json")
    p.add_argument("--per_sample_out", default="", help="default: <--out stem>_per_sample.json")
    p.add_argument("--plot", default=None, help="optional PNG of accuracy vs. keep rate.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
