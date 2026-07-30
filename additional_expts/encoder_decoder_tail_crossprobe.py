"""
encoder_decoder_tail_crossprobe.py -- does the DECODER carry the concentration the
ENCODER geometry lacks?

Companion to encoder_redundancy_tailindex.py. That probe established (26 MVBench
clips, 3B) that the encoder's temporal-redundancy signal is UNIFORM at every
granularity -- lag-1 keep/copy and track keep/copy tails are all gamma <= 0, so
there is nothing to rank/select on in the encoder patch geometry. The open
question it left: the encoder signal is query-FREE geometry; the rankable
concentration, if it exists, should be query-DEPENDENT and live in the decoder's
attention. This script tests exactly that, on the SAME clips, with ONE model:

  encoder side  -- temporal redundancy tails (gamma[keep], gamma[copy]) from the
                   post-merge vision features (query-free), reusing the probe's math.
  decoder side  -- anchor_layer_prune.plan()'s per-layer debiased attention-importance
                   tail gamma over a candidate band; we take gamma at L* (the
                   heaviest-tailed layer) as the clip's decoder concentration.

Verdict logic:
  decoder gamma* > 0  while encoder gamma <= 0  => concentration is query-dependent,
     lives in the decoder attention -> a rankable budget belongs THERE, and the
     encoder signal is only good for a uniform temporal-compression prior.
  decoder gamma* also <= 0  => the "concentrated importance" premise itself is in
     question; pivot efficiency to recoverability, not selection.

Same generic query on both sides (controlled comparison; using the real MVBench
question per clip is a follow-up). Matched frame count on both sides so the two
gammas describe the same sampled clip.

MEMORY NOTE: plan() runs the decoder with output_attentions=True -> O(seq^2) per
layer, materialized for all layers. That is why --max_frames defaults to 8, not
the probe's 16-32; bump it only with headroom, and clips that OOM are skipped.

Run:
    # base cross-probe (4-stage tail table + token corr + generic query overlap)
    python encoder_decoder_tail_crossprobe.py --max_frames 8 --band 2,3,4,5,6,7,8
    # (#hardening 1) k_frac sensitivity -- are the stage gamma SIGNS stable?
    python encoder_decoder_tail_crossprobe.py --max_frames 8 --k_frac_sweep 0.05 0.10 0.20
    # (#hardening 2) query overlap with REAL MVBench questions (own + mismatched-task)
    python encoder_decoder_tail_crossprobe.py --max_frames 8 --real_queries \
        --data_root ../MVBench/ --max_per_task 3 --n_real_queries 3
"""
import os
import sys
import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from efficient_vlm.utils import einmahlHaan
from anchor_layer_prune import plan, QWEN_MODEL_ID
from decoder_tail_index import _qwen_inputs        # builds merged embeds/positions for plan()
from vision_encoder_tail_index import encoder_tail_index_by_layer   # pre-projector ViT attn tail

QUERY = "Describe what happens in the video."

# (#1) Deliberately DIVERGENT queries (action / objects / background). If the decoder's
# top-importance token SET is query-driven, these should select different tokens; if it
# is a fixed attention-sink set, the selection barely moves regardless of query.
QUERIES = [
    QUERY,
    "What objects and people are present in the video?",
    "Describe the background scene and setting of the video.",
]


# ----------------------------------------------------------------------------- #
# Three encoder-stage concentration signals, ordered by pipeline stage:
#   (1) pre-projector,  query-FREE  : ViT self-attention importance tail @ E*
#   (2) post-projector, query-FREE  : temporal-redundancy geometry tail
#   (3) post-projector, query-GUIDED: visual .  query-direction projection tail
# A query-guided PRE-projector score does not exist in Qwen: the ViT never sees
# the text, so visual & text share a space only at/after the merger (pooler_output).
# ----------------------------------------------------------------------------- #
def tail_gamma(x, k_frac):
    return einmahlHaan(torch.as_tensor(x, dtype=torch.float32).flatten(), k_frac)


def query_direction(processor, model, query=QUERY):
    """Unit mean embedding of the query tokens, in LLM embedding space."""
    ids = processor.tokenizer(query, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        e = model.get_input_embeddings()(ids)[0].float().mean(0)
    return F.normalize(e, dim=-1)


def encoder_gammas(model, processor, path, query, max_frames, fps, k_frac):
    """(1) pre-projector ViT attention tail @ E*  and  (2) post-projector temporal
    redundancy tail -- both query-FREE, from the same vision forward inputs."""
    from qwen_vl_utils import process_vision_info
    merge = model.config.vision_config.spatial_merge_size

    messages = [{"role": "user", "content": [
        {"type": "video", "video": path, "fps": fps, "max_frames": max_frames},
        {"type": "text", "text": query}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, vid = process_vision_info(messages)
    inputs = processor(text=[chat], videos=vid, return_tensors="pt").to(model.device)
    pix, grid = inputs["pixel_values_videos"], inputs["video_grid_thw"]
    t, h, w = [int(x) for x in grid[0]]
    T, H, W = t, h // merge, w // merge

    # (2) post-projector geometry
    with torch.no_grad():
        feats = model.model.visual(pix, grid_thw=grid).pooler_output.float()
    assert feats.shape[0] == T * H * W, f"{feats.shape[0]} != T*H*W {T*H*W}"
    x3 = F.normalize(feats, dim=-1).view(T, H, W, -1)
    temp = (x3[1:] * x3[:-1]).sum(-1).flatten().cpu().numpy() if T > 1 else np.array([])
    red_mean = float(temp.mean()) if temp.size else float("nan")

    # (1) pre-projector ViT self-attention importance tail; E* = heaviest ViT layer
    vit = encoder_tail_index_by_layer(model, pix, grid, k_frac=k_frac, estimator="moment")
    vit_valid = {l: g for l, g in vit.items() if g == g}
    Estar = max(vit_valid, key=vit_valid.get) if vit_valid else None
    g_vit = vit_valid[Estar] if Estar is not None else float("nan")

    return dict(T=T, red_mean=red_mean,
                g_keep=tail_gamma(1 - temp, k_frac), g_copy=tail_gamma(temp, k_frac),
                E_star=Estar, g_vit=g_vit)


# --------------------------- decoder side (query-dependent) ------------------ #
def decoder_gamma(model, processor, path, query, max_frames, max_pixels, fps, band, k_frac, q_hat):
    """gamma at L* from plan()'s per-band decoder attention-importance tails, PLUS
    (3) the post-projector query-guided tail: projection of each visual token onto
    the query direction, computed on the SAME base_embeds fed to the decoder."""
    ns = SimpleNamespace(video=path, query=query, max_frames=max_frames,
                         max_pixels=max_pixels, fps=fps)
    base_embeds, position_ids, attn_mask, video_idx = _qwen_inputs(model, processor, ns, model.device)

    v = base_embeds[0, video_idx].float()                 # (M,d) visual tokens, LLM space
    q_scores = (v @ q_hat).cpu().numpy()                  # (M,) query-guided per-token score
    vnorm = v.norm(dim=-1).cpu().numpy()                  # (M,) residual-norm baseline
    g_query = tail_gamma(q_scores, k_frac)

    out = plan(model, base_embeds, position_ids, attn_mask, video_idx, band,
               k_frac=k_frac, estimator="moment")
    L = out["L_star"]
    taus = out["tail_indices"]
    dec_scores = out["scores"].float().cpu().numpy()      # (M,) debiased decoder importance @L*, video_idx order

    # (#2) TOKEN-LEVEL correlation (M ~ 10^3 per clip -> far more powerful than the
    # 26-clip tail-index correlation): does a token's encoder-side score predict its
    # decoder importance? sp_query = query-guided; sp_norm = residual-norm confound check.
    return dict(M=int(video_idx.numel()), L_star=L, g_star=float(taus[L]),
                g_query=float(g_query),
                sp_query=spearman(q_scores, dec_scores),
                sp_norm=spearman(vnorm, dec_scores))


# --------------------------------- correlation ------------------------------- #
def spearman(x, y):
    """Rank correlation over finite pairs; ties broken by argsort order (fine here)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    def rank(v):
        r = np.empty(len(v)); r[v.argsort()] = np.arange(len(v)); return r
    rx, ry = rank(x) - (len(x) - 1) / 2, rank(y) - (len(y) - 1) / 2
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def query_overlap(model, processor, path, queries, max_frames, max_pixels, fps, band, k_frac):
    """(#1) Run plan() under several DIFFERENT queries on the same clip and ask whether
    the top-importance token SET moves. Returns pairwise top-K Jaccard and full-vector
    Spearman, plus the random-subset Jaccard baseline. High overlap => fixed/sink set
    (bad: not query-driven); overlap near random => fully query-driven."""
    per_q, M_ref = [], None
    for q in queries:
        ns = SimpleNamespace(video=path, query=q, max_frames=max_frames,
                             max_pixels=max_pixels, fps=fps)
        be, pos, am, vidx = _qwen_inputs(model, processor, ns, model.device)
        out = plan(model, be, pos, am, vidx, band, k_frac=k_frac, estimator="moment")
        s = out["scores"].float().cpu().numpy()
        if M_ref is None:
            M_ref = s.shape[0]
        if s.shape[0] != M_ref:               # frame-sampling drift -> token sets not comparable
            return None
        per_q.append((s, out["L_star"]))
        torch.cuda.empty_cache()

    M = M_ref
    K = max(1, int(round(k_frac * M)))
    tops = [set(np.argsort(s)[::-1][:K].tolist()) for s, _ in per_q]
    jac, spr = [], []
    for a in range(len(tops)):
        for b in range(a + 1, len(tops)):
            u = len(tops[a] | tops[b])
            jac.append(len(tops[a] & tops[b]) / u if u else float("nan"))
            spr.append(spearman(per_q[a][0], per_q[b][0]))
    return dict(M=M, K=K, Lstars=[l for _, l in per_q],
                jaccard=float(np.nanmean(jac)),
                rand_jaccard=K / (2 * M - K),          # E[Jaccard] of two random K-subsets
                spearman=float(np.nanmean(spr)))


def run(clips, model, processor, args, queries=None):
    """queries: optional per-clip real question (parallel to clips) for the query-guided
    stages (3)/(4) + the token proxy; None => the generic QUERY for every clip. Stages
    (1)/(2) are query-free either way."""
    generic_q_hat = query_direction(processor, model)
    rows = []
    for i, path in enumerate(clips):
        try:
            q = queries[i] if queries else QUERY
            q_hat = query_direction(processor, model, q) if queries else generic_q_hat
            enc = encoder_gammas(model, processor, path, q, args.max_frames, args.fps, args.k_frac)
            dec = decoder_gamma(model, processor, path, q, args.max_frames,
                                args.max_pixels, args.fps, args.band, args.k_frac, q_hat)
            qtag = f'  q="{q[:48]}"' if queries else ""
            print(f"\n[{i}] {os.path.basename(path)}  encT={enc['T']} decM={dec['M']}{qtag}")
            print(f"  (1) pre-proj  ViT attn   E*={str(enc['E_star']):>3}  gamma@E* {enc['g_vit']:+.3f}   [query-free]")
            print(f"  (2) post-proj geometry   redmean {enc['red_mean']:+.3f}  gamma[keep] {enc['g_keep']:+.3f}   [query-free]")
            print(f"  (3) post-proj xquery                          gamma {dec['g_query']:+.3f}   [query-guided]")
            print(f"  (4) in-LLM   decoder     L*={dec['L_star']:<2d}  gamma* {dec['g_star']:+.3f}   [query-guided]")
            print(f"  (#2) token-corr  enc.query->dec {dec['sp_query']:+.3f}   ||v||->dec {dec['sp_norm']:+.3f}   [per-token Spearman, M={dec['M']}]")
            ov = None
            if args.query_overlap and (not args.overlap_max_clips or i < args.overlap_max_clips):
                ov = query_overlap(model, processor, path, QUERIES, args.max_frames,
                                   args.max_pixels, args.fps, args.band, args.k_frac)
                if ov:
                    print(f"  (#1) query-overlap  top-{ov['K']} Jaccard {ov['jaccard']:.3f} "
                          f"(random {ov['rand_jaccard']:.3f})   score Spearman {ov['spearman']:+.3f}   L*={ov['Lstars']}")
            rows.append((enc, dec, ov))
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[{i}] {os.path.basename(path)}  OOM -> skipped (lower --max_frames/--max_pixels)")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{i}] {os.path.basename(path)}  skipped: {e}")

    if not rows:
        print("\nno clips processed."); return
    cols = {
        "(1) pre-proj  ViT attn @E* [qfree]": [r[0]["g_vit"] for r in rows],
        "(2) post-proj geometry     [qfree]": [r[0]["g_keep"] for r in rows],
        "(3) post-proj x-query      [qguid]": [r[1]["g_query"] for r in rows],
        "(4) in-LLM    decoder @L*  [qguid]": [r[1]["g_star"] for r in rows],
    }
    dec_g = cols["(4) in-LLM    decoder @L*  [qguid]"]
    fin = lambda v: [x for x in v if np.isfinite(x)]
    print(f"\n=== POOLED / CROSS ===   clips used {len(rows)}")
    print(f"  {'stage':36s}  mean_gamma   frac(gamma>0)   spearman_vs_decoder")
    stage_summary = {}
    for name, vals in cols.items():
        f = fin(vals)
        pos = f"{sum(x > 0 for x in f)}/{len(f)}"
        sp = "   --  " if name.startswith("(4)") else f"{spearman(vals, dec_g):+.3f}"
        print(f"  {name:36s}  {np.mean(f):+7.3f}     {pos:>7}          {sp}")
        stage_summary[name] = (float(np.mean(f)) if f else float("nan"),
                               sum(x > 0 for x in f), len(f))
    print("\n  READ: the EARLIEST stage whose gamma>0 AND that correlates with the decoder is where a")
    print("        cheap prune can live. query-free stages >0 => concentration exists pre-query;")
    print("        only (3)/(4) >0 => it needs the query; (3) tracking (4) => a pre-LLM query prune works.")

    # (#2) token-level correlation summary (per-clip Spearman averaged; scale-free, so pooling is fair)
    sp_q = fin([r[1]["sp_query"] for r in rows])
    sp_n = fin([r[1]["sp_norm"] for r in rows])
    print(f"\n=== (#2) TOKEN-LEVEL: does an encoder score predict decoder importance? ===")
    print(f"  enc.query -> decoder   per-clip Spearman  mean {np.mean(sp_q):+.3f}  "
          f"(>0.1 in {sum(x > 0.1 for x in sp_q)}/{len(sp_q)})")
    print(f"  ||v|| (norm) -> decoder per-clip Spearman  mean {np.mean(sp_n):+.3f}  "
          f"(>0.1 in {sum(x > 0.1 for x in sp_n)}/{len(sp_n)})")
    print("  READ: |mean| near 0 => encoder token score does NOT predict which tokens the decoder keeps")
    print("        (selection must be made in-LLM). A large ||v||->dec would flag a residual-norm confound.")

    # (#1) query-dependence summary
    ovs = [r[2] for r in rows if r[2]]
    if ovs:
        jac = np.mean([o["jaccard"] for o in ovs]); rnd = np.mean([o["rand_jaccard"] for o in ovs])
        spr = np.mean([o["spearman"] for o in ovs])
        print(f"\n=== (#1) QUERY-DEPENDENCE: does the decoder's kept-set move with the query? ===")
        print(f"  clips {len(ovs)}   top-K Jaccard {jac:.3f}  vs random {rnd:.3f}   score Spearman {spr:+.3f}")
        print("  READ: Jaccard>>random AND Spearman>>0 => a FIXED (sink-like) set, NOT query-driven "
              "(threatens the 'query-dependent' claim).")
        print("        Jaccard near random / low Spearman => the kept-set genuinely tracks the query.")

    return {"clips": len(rows), "stages": stage_summary,
            "sp_query": float(np.mean(sp_q)) if sp_q else float("nan"),
            "sp_norm": float(np.mean(sp_n)) if sp_n else float("nan")}


# --------------------------------------------------------------------------- #
# (#hardening 1) k_frac sensitivity, and (#hardening 2) real-query overlap.
# --------------------------------------------------------------------------- #
def print_kfrac_stability(results, kfracs):
    """Pooled mean gamma (and frac>0) per stage at each k_frac -- is each stage's SIGN
    stable? A flip at 0.05 or 0.20 flags a k_frac-sensitive conclusion."""
    stages = next((list(results[kf]["stages"]) for kf in kfracs if results.get(kf)), None)
    if stages is None:
        print("\nno sweep results."); return
    print("\n=== k_frac SENSITIVITY: pooled mean gamma (frac>0) per stage; signs stable? ===")
    print(f"  {'k_frac':>7}  " + " ".join(f"{s[:20]:>22s}" for s in stages))
    for kf in kfracs:
        r = results.get(kf)
        if not r:
            print(f"  {kf:>7.3f}  (no data)"); continue
        cells = [f"{r['stages'][s][0]:+.3f}({r['stages'][s][1]}/{r['stages'][s][2]})" for s in stages]
        print(f"  {kf:>7.3f}  " + " ".join(f"{c:>22s}" for c in cells))
    print("  READ: same sign of mean-gamma AND of frac>0 across all k_frac => robust;")
    print("        a stage that flips sign at 0.05/0.20 is a k_frac-sensitive conclusion.")


def load_mvbench_records(data_root, tasks, max_per_task):
    """(task, path, question) records from the MVBench json, for --real_queries."""
    import json
    from inference import DATA_LIST
    json_dir = os.path.join(os.path.expanduser(data_root), "json")
    video_dir = os.path.join(os.path.expanduser(data_root), "video")
    chosen = list(DATA_LIST) if tasks == ["all"] else tasks
    unknown = [t for t in chosen if t not in DATA_LIST]
    if unknown:
        raise SystemExit(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")
    recs = []
    for task in chosen:
        fname, subdir, _data_type, _has_bound = DATA_LIST[task]
        with open(os.path.join(json_dir, fname)) as fh:
            data = json.load(fh)
        if max_per_task:
            data = data[:max_per_task]
        for rec in data:
            recs.append({"task": task,
                         "path": os.path.join(video_dir, subdir, rec["video"]),
                         "question": rec["question"]})
    return recs


def run_real_query_overlap(records, model, processor, args):
    """(#hardening 2) The Exp-2b query-dependence test with REAL MVBench questions:
    per clip, the query set is its OWN question plus (n_real_queries-1) mismatched
    questions drawn from OTHER tasks -- a sharper divergent set than the 3 generic
    prompts. High Jaccard/Spearman here => the kept-set is query-INVARIANT even to
    real, task-relevant questions (strengthens the finding)."""
    rng = np.random.default_rng(args.seed)
    n = len(records)
    jac, rnd, spr = [], [], []
    used = 0
    print(f"\n=== REAL-QUERY OVERLAP: {n} clips, {args.n_real_queries} real questions/clip "
          f"(own + mismatched-task) ===")
    for i, rec in enumerate(records):
        if args.overlap_max_clips and used >= args.overlap_max_clips:
            break
        others = [k for k in range(n) if records[k]["task"] != rec["task"]] or \
                 [k for k in range(n) if k != i]
        if not others:
            continue
        m = min(args.n_real_queries - 1, len(others))
        pick = rng.choice(others, size=m, replace=False) if m > 0 else []
        queries = [rec["question"]] + [records[int(k)]["question"] for k in pick]
        if len(queries) < 2:
            continue
        try:
            ov = query_overlap(model, processor, rec["path"], queries, args.max_frames,
                               args.max_pixels, args.fps, args.band, args.k_frac)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{i}] {os.path.basename(rec['path'])}  OOM -> skipped"); continue
        except Exception as e:
            print(f"[{i}] {os.path.basename(rec['path'])}  skipped: {e}"); continue
        if not ov:                                    # frame-sampling drift -> not comparable
            print(f"[{i}] {os.path.basename(rec['path'])}  frame-drift -> skipped"); continue
        used += 1
        jac.append(ov["jaccard"]); rnd.append(ov["rand_jaccard"]); spr.append(ov["spearman"])
        q0 = rec["question"][:44].replace("\n", " ")
        print(f"[{i}] {rec['task'][:18]:18s} {os.path.basename(rec['path'])[:20]:20s} "
              f"top-{ov['K']} Jac {ov['jaccard']:.3f} (rand {ov['rand_jaccard']:.3f}) "
              f"spr {ov['spearman']:+.3f}  q='{q0}'")
        torch.cuda.empty_cache()
    if not jac:
        print("no clips processed for real-query overlap."); return
    print(f"\n=== (#1 REAL) QUERY-DEPENDENCE on {len(jac)} clips ===")
    print(f"  top-K Jaccard {np.mean(jac):.3f}  vs random {np.mean(rnd):.3f}   "
          f"score Spearman {np.mean(spr):+.3f}")
    print("  READ: Jaccard>>random AND Spearman>>0 => FIXED/sink set, query-INVARIANT even to")
    print("        REAL task questions (strengthens the query-invariance finding).")
    print("        Jaccard~random / low Spearman => the kept-set genuinely tracks the real question.")


def main():
    ap = argparse.ArgumentParser(description="Encoder-geometry vs decoder-attention tail-index cross-probe.")
    ap.add_argument("--model", default=QWEN_MODEL_ID)
    ap.add_argument("--max_frames", type=int, default=8, help="matched frames both sides (O(seq^2) decoder mem).")
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--band", default="2,3,4,5,6,7,8", help="decoder anchor-candidate layer band.")
    ap.add_argument("--k_frac", type=float, default=0.10)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--query_overlap", action="store_true",
                    help="(#1) also run plan() under 3 divergent queries per clip and measure "
                         "top-K kept-set overlap (fixed/sink vs query-driven). ~3x decoder cost.")
    ap.add_argument("--overlap_max_clips", type=int, default=0,
                    help="cap clips for the (#1) overlap test (0 = all).")
    ap.add_argument("--k_frac_sweep", type=float, nargs="+", default=None,
                    help="(#hardening 1) re-run the cross-probe at each k_frac and print a "
                         "sign-stability table, e.g. --k_frac_sweep 0.05 0.10 0.20.")
    ap.add_argument("--real_queries", action="store_true",
                    help="(#hardening 2) run the query-overlap test with REAL MVBench questions "
                         "(own + mismatched-task) instead of the 3 generic prompts; needs --data_root.")
    ap.add_argument("--data_root", default=None,
                    help="Dataset root (json/ + video/). With --real_queries: MVBench root. "
                         "Without it: base cross-probe clips are drawn from --tasks (e.g. EgoSchema) "
                         "instead of the hardcoded MVBench list.")
    ap.add_argument("--tasks", nargs="+", default=["all"], help="task name(s) from inference.DATA_LIST.")
    ap.add_argument("--max_clips", type=int, default=26,
                    help="cap on base-probe clips when --data_root is set (0 = all); matches the 26-clip probe.")
    ap.add_argument("--real_query", action="store_true",
                    help="base cross-probe: use each clip's OWN official question (query-guided stages "
                         "3/4 + token proxy) instead of the generic prompt; needs --data_root. "
                         "Distinct from --real_queries (the query-overlap test).")
    ap.add_argument("--max_per_task", type=int, default=3, help="clips per task for --real_queries.")
    ap.add_argument("--n_real_queries", type=int, default=3, help="real questions per clip for --real_queries.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.band = [int(x) for x in args.band.split(",")]

    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model)

    if args.real_queries:
        if not args.data_root:
            raise SystemExit("--real_queries requires --data_root (MVBench json/ + video/).")
        records = load_mvbench_records(args.data_root, args.tasks, args.max_per_task)
        run_real_query_overlap(records, model, processor, args)
        return

    # base cross-probe clip source: hardcoded MVBench probes, or --data_root/--tasks (e.g. EgoSchema)
    queries = None
    if args.data_root:
        recs = load_mvbench_records(args.data_root, args.tasks, max_per_task=None)
        seen, pairs = set(), []
        for r in recs:                                   # de-dup videos (keep first question per clip)
            if r["path"] not in seen:
                seen.add(r["path"]); pairs.append((r["path"], r["question"]))
        np.random.default_rng(args.seed).shuffle(pairs)
        pairs = pairs[:args.max_clips] if args.max_clips else pairs
        clips = [p for p, _ in pairs]
        if args.real_query:
            queries = [q for _, q in pairs]
        mode = "OWN official question" if args.real_query else "generic prompt"
        print(f"loaded {len(clips)} clip(s) from tasks {args.tasks} under {args.data_root} ({mode})")
    else:
        if args.real_query:
            raise SystemExit("--real_query needs --data_root (to read each clip's question).")
        clips = CLIPS

    if args.k_frac_sweep:
        results = {}
        for kf in args.k_frac_sweep:
            args.k_frac = kf
            print(f"\n{'#' * 22} k_frac = {kf} {'#' * 22}")
            results[kf] = run(clips, model, processor, args, queries=queries)
        print_kfrac_stability(results, args.k_frac_sweep)
    else:
        run(clips, model, processor, args, queries=queries)


CLIPS = [
    "/home/ubuntu/Experiments/MVBench/video/FunQA_test/test/test_humor/H_T_1227_0000_0191.mp4",
    "/home/ubuntu/Experiments/MVBench/video/FunQA_test/test/test_humor/H_T_568_0000_0500.mp4",
    "/home/ubuntu/Experiments/MVBench/video/FunQA_test/test/test_creative/C_KT_14_7911_8000.mp4",
    "/home/ubuntu/Experiments/MVBench/video/Moments_in_Time_Raw/videos/validation/giving/getty-pantages-theater-marquee-for-the-30th-annual-academy-awards-honoring-video-id136343623_28.mp4",
    "/home/ubuntu/Experiments/MVBench/video/Moments_in_Time_Raw/videos/validation/raining/getty-close-up-woman-looking-out-window-at-rain-with-worried-expression-video-id587-33_6.mp4",
    "/home/ubuntu/Experiments/MVBench/video/Moments_in_Time_Raw/videos/validation/punting/yt-NQy2ohW9OaI_94.mp4",
    "/home/ubuntu/Experiments/MVBench/video/clevrer/video_validation/video_12178.mp4",
    "/home/ubuntu/Experiments/MVBench/video/clevrer/video_validation/video_14309.mp4",
    "/home/ubuntu/Experiments/MVBench/video/clevrer/video_validation/video_14661.mp4",
    "/home/ubuntu/Experiments/MVBench/video/perception/videos/video_3740.mp4",
    "/home/ubuntu/Experiments/MVBench/video/perception/videos/video_6303.mp4",
    "/home/ubuntu/Experiments/MVBench/video/perception/videos/video_4415.mp4",
    "/home/ubuntu/Experiments/MVBench/video/scene_qa/video/Top079_02905.mp4",
    "/home/ubuntu/Experiments/MVBench/video/scene_qa/video/Top019_04990.mp4",
    "/home/ubuntu/Experiments/MVBench/video/scene_qa/video/Top069_05960.mp4",
    "/home/ubuntu/Experiments/MVBench/video/ssv2_video/131887.webm",
    "/home/ubuntu/Experiments/MVBench/video/ssv2_video/175245.webm",
    "/home/ubuntu/Experiments/MVBench/video/ssv2_video/92623.webm",
    "/home/ubuntu/Experiments/MVBench/video/sta/sta_video/4BIMI.mp4",
    "/home/ubuntu/Experiments/MVBench/video/sta/sta_video/TU9K1.mp4",
    "/home/ubuntu/Experiments/MVBench/video/sta/sta_video/DLOS7.mp4",
    "/home/ubuntu/Experiments/MVBench/video/star/Charades_v1_480/M3S4D.mp4",
    "/home/ubuntu/Experiments/MVBench/video/star/Charades_v1_480/UDGP2.mp4",
    "/home/ubuntu/Experiments/MVBench/video/vlnqa/stop/2141_frame54.mp4",
    "/home/ubuntu/Experiments/MVBench/video/vlnqa/forward/5644_frame30.mp4",
    "/home/ubuntu/Experiments/MVBench/video/vlnqa/left/4668_frame54.mp4",
]


if __name__ == "__main__":
    main()
