"""
two_stage_accuracy.py -- the method end to end: encoder-side diversity selection with
merging (stage one), then decoder-side importance pruning at layer 14 from a single
query row (stage two), against the controls that say which stage is doing the work.

    stage one   group the visual tokens into windows of W adjacent frames, select
                within each group by greedy max-min in feature space, and fold every
                unselected token into its nearest survivor. Input-side, so the saving
                applies to every decoder layer. No class token, no text tower, no
                question -- it is a pure function of the clip.
    stage two   score the survivors by text-to-visual attention at layer k, rebuilt
                from that layer's W_Q and W_K on ONE query row, and prune there.

Retention composes as rho_1 * (k + (N - k) * rho_2) / N. At rho_1 = 0.5, rho_2 = 0.2,
k = 14, N = 28 that is 30% average visual retention across the decoder, plus whatever
stage one saves inside the vision tower if it is run there rather than after it.

Configs  (--configs), written "<stage one>+<stage two>"
------------------------------------------------------
stage one   none            every visual token reaches the decoder
            maxmin          the method: greedy max-min, unselected tokens merged in
            maxmin_nomerge  the same selection, unselected tokens DROPPED
            random          K drawn uniformly per group, merged the same way
            random_nomerge  K drawn uniformly per group, dropped
stage two   none            no decoder-side prune
            attn            the method: top-K by the rebuilt single-row score at layer k
            random          K of the survivors drawn uniformly, pruned at the same layer

The default set is chosen so every claim the method makes has its own control:

    maxmin+attn           the method
    maxmin_nomerge+attn   does merging pay for itself? (the 93%-vs-97% hypothesis)
    random+attn           is max-min better than an unranked keep at stage one?
    maxmin+random         is the layer-k score better than an unranked keep at stage two?
    random+random         both stages unranked -- the floor through the same mechanism
    maxmin+none           stage one in isolation
    none+attn             stage two in isolation

Every config prunes through the same code at the same budget, so a difference between
any two of them is attributable to the thing their names differ in and nothing else.
Stage isolation is reported at matched TOTAL retention rather than matched rho, because
the two stages do not buy the same thing per token: stage one saves in all N layers and
stage two only in the N - k above the prune point.

The floors
----------
full        every visual token, no pruning -- the ceiling.
text_only   every visual token dropped: what the model scores from the question and the
            option strings alone. No retention rate is worth reporting unless it sits
            above this. Same prune mechanism at K = 0, not a re-prompt, so it is the
            rho -> 0 limit of the sweep rather than a differently-built input.

Self-tests  (--self_test, on by default; run once on the first usable clip)
--------------------------------------------------------------------------
1. identity   rho_1 = rho_2 = 1.0 through the whole path must reproduce the dense
              forward's logits. If the plumbing corrupts the sequence, every number
              below is describing a broken model rather than a pruned one.
2. scorer     the hand-rebuilt attention row at layer k must match the eager attention
              weights at that layer and row, to fp tolerance and in ranking. THIS IS
              THE LOAD-BEARING GATE. FlashAttention never materialises the attention
              matrix, so the deployable scorer has to rebuild the row from W_Q and W_K;
              if the rebuild is not the same quantity the analysis measured, the whole
              result describes something the paper does not claim.

Both are hard failures by default -- the run stops rather than writing numbers nobody
should trust. --no-self_test skips them and is there for reruns, not for first runs.

Frame sampling, prompts, the answer-letter readout and the exact McNemar all come from
tail_vs_layer.py / lv_knockout_accuracy.py, unchanged, so these accuracies sit on the
same scale as every number already committed in final_expt_results/.

Cost
----
n_clips x (1 dense eager forward + 2 forwards per config). The dense one is eager (the
self-test needs the weights materialised once); the scoring pass is abandoned at layer k
by exception, so it costs half a prefill rather than a whole one. Use --max_samples for
a first look.

Run
---
    # the method and its controls, Qwen + EgoSchema
    python two_stage_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --num_segments 16 --max_pixels 200704 \
        --rho1 0.5 --rho2 0.2 --prune_layer 14 --out two_stage_ego_qwen.json

    # smoke test: two clips, both gates, nothing else
    python two_stage_accuracy.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --max_samples 2 --out /tmp/smoke.json

    # budget split: which (rho1, rho2) pair spends a 30% budget best
    python two_stage_accuracy.py --data_root ~/Experiments/MVBench --tasks mvbench \
        --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf --max_samples 50 \
        --rho1 1.0 0.5 0.3 --rho2 1.0 0.2 0.6 --configs maxmin+attn
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
import zlib

import numpy as np
import torch
from tqdm import tqdm

from lv_knockout_accuracy import gold_index, option_token_ids
from stage1_diversity import stage1_reduce
from stage2_importance import (average_retention, capture_layer_input, score_from_row,
                               stage2_keep, verify_against_eager)
from stage_topk_accuracy import keep_abs_idx, mcnemar, predict_midforward, predict_pruned
from tail_vs_layer import (ANSWER_PREFIX, DATA_LIST, DEFAULT_SEGMENTS, LLAVA_FAMILY,
                           MVBENCH_TASKS, build_inputs, context_limit, infer_backbone,
                           iter_clips, load_model, sample_frames, text_model_of)

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

STAGE1_MODES = ("none", "maxmin", "maxmin_nomerge", "random", "random_nomerge")
STAGE2_MODES = ("none", "attn", "random")
DEFAULT_CONFIGS = ["maxmin+attn", "maxmin_nomerge+attn", "random+attn",
                   "maxmin+random", "random+random", "maxmin+none", "none+attn"]


class SelfTestFailed(RuntimeError):
    """A self-test that did not merely fail its gate but could not be run at all.

    Carried as its own type so the per-clip handler lets it through: the gates run once,
    on the first usable clip, and a swallowed failure there leaves self_test_report None,
    so every remaining clip re-runs the same broken gate and is skipped for the same
    reason. That turns one fatal fault into a whole dataset of identical skip lines and a
    run that ends with no numbers and no summary of why.
    """


def parse_config(spec: str) -> tuple[str, str]:
    if spec.count("+") != 1:
        raise ValueError(f"config {spec!r} must be '<stage1>+<stage2>'")
    s1, s2 = spec.split("+")
    if s1 not in STAGE1_MODES:
        raise ValueError(f"unknown stage one {s1!r}; choose from {STAGE1_MODES}")
    if s2 not in STAGE2_MODES:
        raise ValueError(f"unknown stage two {s2!r}; choose from {STAGE2_MODES}")
    return s1, s2


# --------------------------------------------------------------------------- #
# 1. Capture -- one dense forward gives the ceiling, the sequence, and both gates
# --------------------------------------------------------------------------- #
def attach_dense_capture(model, prune_layer: int, query_row: int, ctx: dict):
    """Hooks for the dense forward. Returns an uninstaller.

    Gated on ctx["capture"] so the pruned forwards, which run on shorter sequences,
    never re-enter them.

        layer[0] pre-hook       the merged input embeddings -- visual features already
                                scattered into their placeholder positions, i.e. what
                                stage one consumes.
        decoder pre-hook        the position ids the model built (3D mRoPE on Qwen).
        layer[k] pre-hook       that layer's input and its (cos, sin), which is what the
                                rebuilt score row is computed from.
        layer[k] attn hook      the eager attention weights at the prune layer, for the
                                ONE query row the score is read from, kept only so the
                                self-test has something to check the rebuild against.
                                The method never reads this.

    The attention hook slices its row out before storing anything: the full weight
    tensor is (H, S, S), which at 28 heads and a 3.4k-token clip is roughly 650 MB, and
    the gate needs one row of it. Same reason the sibling scripts slice inside the hook.
    """
    text_model = text_model_of(model)
    handles = []

    def stack_pre(module, args, kwargs):
        if ctx.get("capture"):
            ctx["position_ids"] = kwargs.get("position_ids")
        return None

    def embed_pre(module, args, kwargs):
        if ctx.get("capture"):
            h = kwargs.get("hidden_states")
            ctx["embeds"] = (args[0] if h is None else h).detach()
        return None

    def prune_layer_pre(module, args, kwargs):
        if ctx.get("capture"):
            h = args[0] if args else kwargs.get("hidden_states")
            ctx["h_k"] = h.detach()
            ctx["pe_k"] = kwargs.get("position_embeddings")
        return None

    def prune_layer_attn(module, inputs, output):
        if not ctx.get("capture") or not ctx.get("want_eager"):
            return None
        if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
            raise RuntimeError(
                "self_attn returned no attention weights -- load the model with "
                "attn_implementation='eager' (sdpa/flash never materialize them).")
        a = output[1][0]                                   # (H, S, S)
        qrow = query_row + a.shape[-2] if query_row < 0 else query_row
        ctx["attn_k"] = a[:, qrow, :].detach().clone()     # (H, S) -- one row, not 650 MB

    handles.append(text_model.register_forward_pre_hook(stack_pre, with_kwargs=True))
    handles.append(text_model.layers[0].register_forward_pre_hook(embed_pre, with_kwargs=True))
    handles.append(text_model.layers[prune_layer].register_forward_pre_hook(
        prune_layer_pre, with_kwargs=True))
    handles.append(text_model.layers[prune_layer].self_attn.register_forward_hook(
        prune_layer_attn))
    return lambda: [h.remove() for h in handles]


# --------------------------------------------------------------------------- #
# 2. Stage one applied to a real sequence
# --------------------------------------------------------------------------- #
def apply_stage1(embeds, pos, visual_idx, S, keep_local, new_vis):
    """Build the shortened sequence stage one hands to the decoder.

    Returns (embeds_1, pos_1, visual_idx_1, S1). Text tokens are never dropped, every
    survivor keeps its ORIGINAL position id, and the merged values are written into the
    positions their representatives occupy -- so the decoder sees a shorter visual block
    in the same order, at the same mRoPE coordinates, carrying cluster averages.
    """
    keep_abs = keep_abs_idx(S, visual_idx, keep_local)
    embeds_1 = embeds[:, keep_abs].clone()
    pos_1 = pos[..., keep_abs]
    kept_visual_abs = visual_idx[torch.as_tensor(np.ascontiguousarray(keep_local),
                                                 device=visual_idx.device, dtype=torch.long)]
    visual_idx_1 = torch.searchsorted(keep_abs, kept_visual_abs)
    if not torch.equal(keep_abs[visual_idx_1], kept_visual_abs):
        raise RuntimeError("stage one lost track of where its survivors landed")
    embeds_1[0, visual_idx_1] = new_vis.to(embeds_1.dtype)
    return embeds_1, pos_1, visual_idx_1, int(keep_abs.numel())


# --------------------------------------------------------------------------- #
# 3. One config on one clip
# --------------------------------------------------------------------------- #
def run_config(model, text_model, s1, s2, embeds, pos, visual_idx, S, letter_ids,
               args, rho1, rho2, rng, n_frames):
    """(prediction, info) for one config on one clip."""
    M = int(visual_idx.numel())

    # ---- stage one ----
    if s1 == "none":
        embeds_1, pos_1, visual_idx_1, S1 = embeds, pos, visual_idx, S
    else:
        keep_local, new_vis = stage1_reduce(
            embeds[0, visual_idx], n_frames, rho1,
            window=args.window, group_by=args.group_by,
            merge=not s1.endswith("_nomerge"), center=args.center,
            seed=args.maxmin_seed, select=s1.split("_")[0], rng=rng)
        embeds_1, pos_1, visual_idx_1, S1 = apply_stage1(
            embeds, pos, visual_idx, S, keep_local, new_vis)
    M1 = int(visual_idx_1.numel())

    # ---- stage two ----
    if s2 == "none":
        pred = predict_pruned(model, embeds_1, pos_1,
                              torch.arange(S1, device=embeds_1.device), letter_ids)
        return pred, {"n_after_stage1": M1, "n_after_stage2": M1}

    if s2 == "attn":
        attn_mask = torch.ones(1, S1, dtype=torch.long, device=embeds_1.device)
        cap = capture_layer_input(
            text_model, args.prune_layer,
            lambda: model(inputs_embeds=embeds_1, position_ids=pos_1,
                          attention_mask=attn_mask, use_cache=False))
        score = score_from_row(text_model.layers[args.prune_layer],
                               cap["hidden_states"], cap["position_embeddings"],
                               visual_idx_1, args.query_row, args.score_norm)
        keep2 = stage2_keep(score, rho2)
    else:
        K2 = int(max(1, min(M1, round(rho2 * M1))))
        keep2 = np.sort(rng.choice(M1, size=K2, replace=False))

    keep_abs_2 = keep_abs_idx(S1, visual_idx_1, keep2)
    pred = predict_midforward(model, embeds_1, pos_1, keep_abs_2, letter_ids,
                              args.prune_layer)
    return pred, {"n_after_stage1": M1, "n_after_stage2": int(keep2.size),
                  "n_dense": M}


# --------------------------------------------------------------------------- #
# 4. Self-tests -- run once, on the first usable clip
# --------------------------------------------------------------------------- #
def self_test(model, text_model, ctx, embeds, pos, visual_idx, S, letter_ids,
              dense_logits, args, n_frames) -> dict:
    """The two gates that need real weights. Returns a report; raises on failure unless
    --no-strict_self_test."""
    report = {}

    # ---- gate 1: the whole path at rho = 1 must be the dense forward ----
    pred, _ = run_config(model, text_model, "maxmin", "attn", embeds, pos, visual_idx,
                         S, letter_ids, args, 1.0, 1.0, np.random.default_rng(0), n_frames)
    dense_pred = int(torch.argmax(dense_logits[letter_ids.to(dense_logits.device)]).item())
    report["identity"] = {"pruned_pred": pred, "dense_pred": dense_pred,
                          "ok": pred == dense_pred}

    # ---- gate 2: the rebuilt row must be the row the model computed ----
    if ctx.get("attn_k") is None:
        raise RuntimeError("no eager attention captured at the prune layer")
    rebuilt = score_from_row(text_model.layers[args.prune_layer], ctx["h_k"], ctx["pe_k"],
                             visual_idx, args.query_row, "all")
    report["scorer"] = verify_against_eager(rebuilt, ctx["attn_k"], visual_idx,
                                            args.scorer_atol)

    # The cheap form is a genuine approximation (per-head denominators change), so its
    # agreement with the exact one is measured rather than assumed.
    cheap = score_from_row(text_model.layers[args.prune_layer], ctx["h_k"], ctx["pe_k"],
                           visual_idx, args.query_row, "visual")
    k = max(1, rebuilt.numel() // 10)
    overlap = len(set(torch.argsort(-rebuilt)[:k].tolist())
                  & set(torch.argsort(-cheap)[:k].tolist())) / k
    report["visual_norm_shortcut"] = {"top10pct_overlap": overlap}

    print("\n---- self-test ----")
    g1, g2 = report["identity"], report["scorer"]
    print(f"  identity  rho=1 through both stages -> pred {g1['pruned_pred']} "
          f"vs dense {g1['dense_pred']}   {'PASS' if g1['ok'] else 'FAIL'}")
    print(f"  scorer    rebuilt row vs eager attention at layer {args.prune_layer}: "
          f"max|d|={g2['max_abs_diff']:.2e}, top-10% overlap={g2['top10pct_overlap']:.3f} "
          f"over {g2['n_visual']} tokens   {'PASS' if g2['ok'] else 'FAIL'}")
    print(f"  shortcut  norm='visual' agrees with norm='all' on "
          f"{overlap:.3f} of the top 10%")
    print("-------------------\n")

    bad = [n for n in ("identity", "scorer") if not report[n]["ok"]]
    if bad and args.strict_self_test:
        raise RuntimeError(
            f"self-test failed: {bad}. Numbers from this configuration would not mean "
            f"what the paper claims. Re-run with --no-strict_self_test to override.")
    return report


# --------------------------------------------------------------------------- #
# 5. Sweep
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
    for spec in args.configs:
        parse_config(spec)
    return [parse_config(s) for s in args.configs]


def main(args):
    parsed = resolve_args(args)
    budgets = [(r1, r2) for r1 in sorted(set(args.rho1), reverse=True)
               for r2 in sorted(set(args.rho2), reverse=True)]
    labels = [f"{spec}@{r1:g}/{r2:g}" for r1, r2 in budgets for spec in args.configs]
    refs = ["full"] + (["text_only"] if args.text_only else [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[model] {args.model_name}  backbone={args.backbone}  dtype={args.dtype}  device={device}")
    model, processor, visual_token_id = load_model(args, dtype)
    max_positions = context_limit(model)
    text_model = text_model_of(model)
    n_layers = len(text_model.layers)
    if not 0 <= args.prune_layer < n_layers:
        raise ValueError(f"--prune_layer {args.prune_layer} outside a {n_layers}-layer stack")

    print(f"[model] {n_layers} decoder layers, prune at {args.prune_layer}, "
          f"{max_positions} LM positions")
    print(f"[stage1] window={args.window} frames, group_by={args.group_by}, "
          f"{'centred' if args.center else 'UNCENTRED'}, seed={args.maxmin_seed}")
    print(f"[stage2] single query row ({args.query_row}), softmax over "
          f"{'all keys' if args.score_norm == 'all' else 'visual keys only'}, "
          f"rebuilt from W_Q/W_K")
    print(f"[budgets] " + ", ".join(f"rho1={r1:g} rho2={r2:g} -> "
                                    f"{100 * average_retention(r1, r2, args.prune_layer, n_layers)['average_visual_retention']:.1f}% avg retention"
                                    for r1, r2 in budgets))
    print(f"[configs] {args.configs}  ->  {len(refs) + 2 * len(labels)} forwards/clip")
    print(f"[data] tasks={args.tasks}  {args.num_segments} frames/clip")

    ctx = {"capture": False, "want_eager": args.self_test}
    uninstall = attach_dense_capture(model, args.prune_layer, args.query_row, ctx)

    letter_cache = {}
    hits = {c: [] for c in refs + labels}
    per_task_hits, seen_by_task, per_sample, skipped = {}, {}, [], []
    token_counts = {c: [] for c in labels}
    self_test_report, n_seen = None, 0
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
                visual_idx = (ids == visual_token_id).nonzero(as_tuple=False).flatten()
                M = int(visual_idx.numel())
                if M < 50:
                    raise RuntimeError(f"too few visual tokens ({M} < 50)")

                # ---- the one dense forward ----
                ctx.update({"capture": True, "embeds": None, "position_ids": None,
                            "h_k": None, "pe_k": None, "attn_k": None})
                with torch.no_grad():
                    dense = model(**inputs, use_cache=False)
                dense_logits = dense.logits[0, -1].detach()
                full_pred = int(torch.argmax(
                    dense_logits[letter_ids.to(dense_logits.device)]).item())
                ctx["capture"] = False
                del dense

                embeds = ctx["embeds"]
                if embeds is None or embeds.shape[1] != S:
                    raise RuntimeError("did not capture the merged input embeddings")
                pos = ctx["position_ids"]
                if pos is None:
                    pos = torch.arange(S, device=embeds.device)[None]
                if pos.shape[-1] != S:
                    raise RuntimeError(f"captured position ids are {tuple(pos.shape)}, "
                                       f"expected last dim {S}")

                # Under device_map="auto" the decoder stack need not sit on the GPU the
                # inputs were built on -- with a 7B LM and a vision tower to place, the
                # whole stack can land on cuda:1 while input_ids stay on cuda:0. Every
                # selector below slices `embeds` with indices derived from `visual_idx`,
                # so the indices and the position ids follow the embeddings here, once,
                # rather than at each of the dozen sites that consume them.
                visual_idx = visual_idx.to(embeds.device)
                pos = pos.to(embeds.device)

                if args.self_test and self_test_report is None:
                    try:
                        self_test_report = self_test(model, text_model, ctx, embeds, pos,
                                                     visual_idx, S, letter_ids,
                                                     dense_logits, args, args.num_segments)
                    except Exception as e:
                        raise SelfTestFailed(
                            f"the self-test could not be run on the first usable clip -- "
                            f"{type(e).__name__}: {e}") from e
                    ctx["want_eager"] = False       # the gate is done; stop paying for it

                preds, info = {}, {}
                if args.text_only:
                    preds["text_only"] = predict_pruned(
                        model, embeds, pos,
                        keep_abs_idx(S, visual_idx, np.empty(0, dtype=np.int64)),
                        letter_ids)

                for r1, r2 in budgets:
                    for spec, (s1, s2) in zip(args.configs, parsed):
                        # Seeded per (clip, config, budget) so the unranked arms are
                        # reproducible across runs. crc32 rather than hash(): Python
                        # randomises string hashes per process, so hash(spec) would give
                        # a different random keep on every run and the "same seed, same
                        # draw" pairing this family relies on would silently not hold.
                        rng = np.random.default_rng([args.seed, idx,
                                                     zlib.crc32(spec.encode()),
                                                     int(1000 * r1), int(1000 * r2)])
                        lab = f"{spec}@{r1:g}/{r2:g}"
                        preds[lab], info[lab] = run_config(
                            model, text_model, s1, s2, embeds, pos, visual_idx, S,
                            letter_ids, args, r1, r2, rng, args.num_segments)
            except SelfTestFailed:
                raise
            except Exception as e:
                ctx.update({"capture": False, "embeds": None, "position_ids": None,
                            "h_k": None, "pe_k": None, "attn_k": None})
                skipped.append({"task": task, "video": rec["video"],
                                "reason": f"{type(e).__name__}: {e}"})
                tqdm.write(f"skip [{task}] {rec['video']}: {type(e).__name__}: {e}")
                continue

            n_seen += 1
            seen_by_task[task] = seen_by_task.get(task, 0) + 1
            tc = per_task_hits.setdefault(task, {c: 0 for c in refs + labels})
            preds["full"] = full_pred
            for c in refs + labels:
                hit = int(preds[c] == gold)
                hits[c].append(hit)
                tc[c] += hit
            for c in labels:
                token_counts[c].append(info[c]["n_after_stage2"] / M)
            per_sample.append({"task": task, "question_idx": rec.get("question_idx"),
                               "video": rec["video"], "gold": gold, "n_options": n_opt,
                               "n_visual_tokens": M, "pred_by_config": preds,
                               "tokens_by_config": info})

            del inputs, embeds, dense_logits
            ctx.update({"embeds": None, "position_ids": None, "h_k": None,
                        "pe_k": None, "attn_k": None})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        uninstall()

    if not n_seen:
        print("no usable clips -- check --data_root layout (json/ and video/).")
        return

    acc = {c: sum(hits[c]) / n_seen for c in refs + labels}
    # Realised retention, measured rather than requested: per-group budgets are rounded
    # and floored at one token, so the achieved rate drifts above rho at tight budgets.
    realised = {c: float(np.mean(token_counts[c])) for c in labels}
    avg_ret = {f"{r1:g}/{r2:g}": average_retention(r1, r2, args.prune_layer, n_layers)
               for r1, r2 in budgets}

    def tests_against(base_spec, note):
        out = []
        for r1, r2 in budgets:
            b = f"{base_spec}@{r1:g}/{r2:g}"
            if b not in acc:
                continue
            for spec in args.configs:
                a = f"{spec}@{r1:g}/{r2:g}"
                if a == b:
                    continue
                out.append(dict(rho1=r1, rho2=r2, selector=spec, baseline=base_spec,
                                note=note,
                                delta_pts=100 * (acc[a] - acc[b]),
                                **mcnemar(hits[a], hits[b])))
        return out

    # Each control answers one question, so each gets its own paired test against the
    # method rather than one omnibus comparison.
    ablations = (tests_against("maxmin+attn", "vs the method")
                 + [dict(rho1=r1, rho2=r2, selector="maxmin+attn",
                         baseline="text_only", note="vs the blind floor",
                         delta_pts=100 * (acc[f"maxmin+attn@{r1:g}/{r2:g}"] - acc["text_only"]),
                         **mcnemar(hits[f"maxmin+attn@{r1:g}/{r2:g}"], hits["text_only"]))
                    for r1, r2 in budgets
                    if args.text_only and f"maxmin+attn@{r1:g}/{r2:g}" in acc])

    out = {"experiment": "two_stage_accuracy",
           "method": "stage1 maxmin+merge (input-side) -> stage2 single-row attention prune",
           "backbone": args.backbone, "model_name": args.model_name,
           "data_root": args.data_root, "tasks": args.tasks,
           "num_segments": args.num_segments, "max_pixels": args.max_pixels,
           "min_pixels": args.min_pixels, "n_layers": n_layers,
           "prune_layer": args.prune_layer, "query_row": args.query_row,
           "score_norm": args.score_norm, "window": args.window,
           "group_by": args.group_by, "center": args.center,
           "maxmin_seed": args.maxmin_seed, "seed": args.seed,
           "configs": args.configs, "budgets": [list(b) for b in budgets],
           "variants": labels,
           "scoring": "argmax over option-letter tokens", "answer_prefix": ANSWER_PREFIX,
           "n_clips": n_seen, "full_accuracy": acc["full"],
           "text_only": args.text_only, "text_only_accuracy": acc.get("text_only"),
           "chance_accuracy": (sum(1.0 / s["n_options"] for s in per_sample) / n_seen),
           "self_test": self_test_report,
           "accuracy_by_config": acc,
           "retention_vs_dense": {c: (acc[c] / acc["full"] if acc["full"] > 0 else float("nan"))
                                  for c in labels},
           "realised_visual_keep_rate": realised,
           "average_retention_accounting": avg_ret,
           "above_floor": ({c: ((acc[c] - acc["text_only"]) / (acc["full"] - acc["text_only"])
                                if acc["full"] != acc["text_only"] else float("nan"))
                            for c in labels} if args.text_only else None),
           "mcnemar_ablations": ablations,
           "per_task_seen": seen_by_task,
           "accuracy_by_task": {t: {c: h[c] / seen_by_task[t] for c in refs + labels}
                                for t, h in sorted(per_task_hits.items())},
           "skipped": skipped}

    # ---- report ----
    print(f"\n==== two-stage: diversity at the encoder, importance at layer "
          f"{args.prune_layer} ({n_seen} clips) ====")
    print(f"{args.backbone}: {args.model_name}, {args.num_segments} frames, "
          f"{len(seen_by_task)} task(s)")
    print(f"full (no pruning)      = {acc['full']:.4f}   <- ceiling")
    if args.text_only:
        print(f"text only (0 tokens)   = {acc['text_only']:.4f}   <- floor")
    print(f"chance (1/n_options)   = {out['chance_accuracy']:.4f}\n")
    w = max(20, max(len(s) for s in args.configs))
    for r1, r2 in budgets:
        a = avg_ret[f"{r1:g}/{r2:g}"]
        print(f"rho1={r1:g} rho2={r2:g}  avg visual retention "
              f"{100 * a['average_visual_retention']:.1f}%  "
              f"(decoder layers {100 * a['decoder_layer_fraction']:.1f}%)")
        for spec in args.configs:
            lab = f"{spec}@{r1:g}/{r2:g}"
            print(f"  {spec:<{w}s} {100 * acc[lab]:6.2f}%   "
                  f"{100 * out['retention_vs_dense'][lab]:6.2f}% of dense   "
                  f"keeps {100 * realised[lab]:5.2f}% of visual tokens")
        print()
    print("McNemar (exact, two-sided):")
    for t in ablations:
        flag = "significant" if t["p_value"] < 0.05 else "n.s."
        print(f"  rho1={t['rho1']:<4g} rho2={t['rho2']:<4g} {t['selector']:>20s} - "
              f"{t['baseline']:<20s} {t['delta_pts']:+6.2f} pts  "
              f"(b={t['b']:>4d} c={t['c']:>4d}, p={t['p_value']:.3g}, {flag})")
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s)")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage token reduction: encoder-side diversity with merging, "
                    "then decoder-side single-row importance pruning at layer 14.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' (all 20 tasks) / 'all'.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava_ov", "llava"], default="auto")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help=f"'<stage1>+<stage2>'. stage1 in {STAGE1_MODES}, "
                        f"stage2 in {STAGE2_MODES}. Default runs the method and every control.")
    p.add_argument("--rho1", type=float, nargs="+", default=[0.5],
                   help="stage-one keep fraction within each group (default 0.5, ~2x).")
    p.add_argument("--rho2", type=float, nargs="+", default=[0.2],
                   help="stage-two keep fraction of the survivors (default 0.2).")
    p.add_argument("--prune_layer", type=int, default=14,
                   help="0-indexed decoder layer the stage-two score is read at and the "
                        "prune happens at (default 14, VScan's published k for "
                        "Qwen2.5-VL-7B; both target backbones have 28 layers).")
    p.add_argument("--window", type=int, default=2,
                   help="stage one: how many adjacent frames a group spans (default 2).")
    p.add_argument("--group_by", choices=["window", "cell"], default="window",
                   help="stage one: 'window' groups every spatial cell of W adjacent "
                        "frames (default, a few hundred tokens/group); 'cell' groups one "
                        "spatial cell across W frames (|g| = W).")
    p.add_argument("--center", action=argparse.BooleanOptionalAction, default=True,
                   help="stage one: centre the visual embeddings on the clip mean before "
                        "cosine. On by default -- uncentred LM embeddings are anisotropic "
                        "enough that the max-min traversal loses its grip.")
    p.add_argument("--maxmin_seed", choices=["centroid", "first"], default="centroid",
                   help="stage one: which token seeds the traversal (default: the group "
                        "medoid, which keeps randomness out of the method).")
    p.add_argument("--query_row", type=int, default=-1,
                   help="stage two: which row the score is read from (default -1, the "
                        "answer slot -- the only row a prefill has finished at the prune "
                        "point). Must sit after every visual token.")
    p.add_argument("--score_norm", choices=["all", "visual"], default="all",
                   help="stage two: softmax over every key (default, and the quantity the "
                        "analysis measured) or over the visual keys alone (the strict "
                        "O(N*d) form -- an approximation, not a free simplification).")
    p.add_argument("--text_only", action=argparse.BooleanOptionalAction, default=True,
                   help="one extra forward per clip with every visual token dropped.")
    p.add_argument("--self_test", action=argparse.BooleanOptionalAction, default=True,
                   help="run the identity and scorer gates on the first usable clip.")
    p.add_argument("--strict_self_test", action=argparse.BooleanOptionalAction, default=True,
                   help="stop the run if a gate fails (default). --no-strict_self_test "
                        "reports and continues, which is for debugging, not for results.")
    p.add_argument("--scorer_atol", type=float, default=2e-3,
                   help="tolerance for the rebuilt-row gate (bf16 hidden states make an "
                        "exact match impossible).")
    p.add_argument("--num_segments", type=int, default=None,
                   help=f"frames/clip. Default per backbone: {DEFAULT_SEGMENTS}.")
    p.add_argument("--max_pixels", type=int, default=None, help="qwen only, e.g. 200704.")
    p.add_argument("--min_pixels", type=int, default=None, help="qwen only; defaults to --max_pixels.")
    p.add_argument("--max_samples", type=int, default=None, help="cap on records PER TASK.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="two_stage_accuracy.json")
    p.add_argument("--per_sample_out", default="", help="default: <--out stem>_per_sample.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
