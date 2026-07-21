"""
layerwise_attn_tail_ai2d.py -- per-layer EVT tail index (xi) of the language->image
attention distribution.  Does the tail collapse toward exponential at some layers?
======================================================================================
AI2D sibling of ``layerwise_attn_tail_mvbench.py`` (and the image-based
``layerwise_attn_tail_mmbench.py``).  Companion to the layerwise-attention scripts:
those ask which LLM layer is the best *selection* signal; this one asks about the
SHAPE of the signal, layer by layer, with a single principled instrument: the Extreme
Value Index xi (gamma).

The scorer used in ``lcds_ai2d.py`` keeps the top-K image tokens by attention mass
over a fixed band (layers 12..16).  We have seen some layers look heavy-tailed -- a
handful of image tokens soak up most of the attention.  The clean question is not
"how concentrated" but "which extreme-value domain is the upper tail in":

  xi > 0   Frechet    -> genuinely heavy-tailed (Pareto-like)
  xi = 0   Gumbel     -> the tail IS exponential  ("collapsed")
  xi < 0   Weibull    -> bounded / light tail

So a single xi(layer) curve, read against the xi = 0 line, says exactly which layers
collapse toward exponential.  We estimate xi with the Dekkers-Einmahl-de Haan moment
estimator (sign-aware, unlike Hill which is pinned to xi >= 0), computed on the upper
order statistics of each diagram's per-token attention-mass distribution.

For every LLM layer we build ONE distribution per diagram:

  p_i = attention mass image-token i receives, summed over all text queries,
        renormalized so sum_i p_i = 1        (xi is scale-invariant, so the
                                              normalization is only for the plot).

and report mean xi +/- SEM across diagrams, per layer.  Because the scorer averages
the 12-16 band before ranking, we ALSO report the xi of that exact band-averaged
distribution -- the tail the scorer actually sees.  A mean sorted-decay curve per
layer is kept for the visual companion panel (straight on a semilog-rank axis <=> xi=0).

Uniformization probe (Information-Horizon).  A distribution that flattens toward
UNIFORM does not merely leave Frechet -- it enters the Weibull domain (bounded
support), with xi -> -1 for a perfectly uniform law.  So "does the attention
uniformize with depth?" is the EVT question "does xi descend from Frechet (xi>0)
through exponential (xi=0) toward uniform (xi=-1)?".  A handful of attention SINK
tokens can hold the UPPER tail in Frechet even if the body flattens, so we ALSO
report a DE-SINKED xi -- the top --sink_frac fraction dropped before estimating --
which turns toward 0/negative if the bulk uniformizes while the sinks stay sharp.
Per-layer we classify each diagram into Frechet / Gumbel / Weibull (|xi| vs
--domain_eps), so the horizon is visible as a trajectory, not inferred.

AI2D is the interesting stress case for the tail question: on a science diagram the
evidence is labels and arrows scattered over mostly-blank canvas, so the language may
concentrate hard on a few label tokens (heavier tail) rather than diffuse over a
natural image.  Differences from the MVBench/MMBench siblings are all in the data
layer, imported from ``overlap_subset_ai2d`` (single diagram, positional options,
parquet PNG bytes).  AI2D has no category column, so the per-task machinery collapses
to a single "ai2d" bucket -- the overall line is the result.

Two backbones (--backbone, auto-detected from --model_name).  The EVT analysis is
identical for both; only the model class, the image-token id, and how inputs are built
differ (see the backbone-plumbing section):
  * qwen  : Qwen2.5-VL-3B, dynamic resolution + mRoPE + 2x2 merge (default).
  * llava : LLaVA-1.5-7B, fixed 336x336 -> a flat 24x24 = 576-token grid, 1D RoPE.
            --max_pixels/--min_pixels are ignored (LLaVA hard-resizes).

Run:
    # Qwen2.5-VL (default)
    python layerwise_attn_tail_ai2d.py \
        --data_root ~/Experiments/AI2D \
        --max_samples 300 --out results_layerwise_attn_tail_ai2d.json

    # LLaVA-1.5
    python layerwise_attn_tail_ai2d.py --backbone llava \
        --data_root ~/Experiments/AI2D \
        --max_samples 300 --out results_layerwise_attn_tail_ai2d_llava.json
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import (Qwen2_5_VLForConditionalGeneration,
                          LlavaForConditionalGeneration, AutoProcessor)
from qwen_vl_utils import process_vision_info

# AI2D data layer / prompt (single diagram, positional options, parquet PNG bytes).
# Same import the AI2D LCDS/overlap scripts use; overlap_subset_ai2d sits alongside
# this file, so the script's own directory (sys.path[0]) resolves it. build_ai2d_prompt
# emits Qwen-style message dicts; the LLaVA backbone builds its flat prompt below.
from overlap_subset_ai2d import build_ai2d_prompt, decode_image, load_items

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


# --------------------------------------------------------------------------- #
# EVT tail index (Dekkers-Einmahl-de Haan moment estimator), implemented inline
# --------------------------------------------------------------------------- #
def evt_xi(scores, k_frac=0.10):
    """Sign-aware moment estimator of the extreme-value index xi from a 1-D array.

    On the sorted positive values x_(1) <= ... <= x_(n), with k top order stats and
    excesses  L_i = log x_(n-i) - log x_(n-k)  (i = 0..k-1):

        M1 = mean(L_i),   M2 = mean(L_i^2)
        xi = M1 + 1 - 0.5 / (1 - M1^2 / M2)

    The M1 term alone is the Hill estimator (xi >= 0 only); the correction makes xi
    sign-aware.  xi > 0 heavy (Frechet), xi = 0 exponential (Gumbel), xi < 0 light
    (Weibull).  Returns NaN when there aren't enough positive samples to estimate.
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    x = np.sort(x[x > 0])
    n = x.size
    if n < 12:
        return float("nan")
    k = min(max(10, int(k_frac * n)), n - 1)
    log_top = np.log(x[n - k:])                 # x_(n-k+1) .. x_(n)
    L = log_top - np.log(x[n - k - 1])          # excess over the (n-k)-th order stat
    M1 = L.mean()
    M2 = (L ** 2).mean()
    if M2 <= 0 or (1.0 - M1 * M1 / M2) == 0:
        return float("nan")
    return float(M1 + 1.0 - 0.5 / (1.0 - M1 * M1 / M2))


def evt_xi_desinked(scores, k_frac=0.10, sink_frac=0.01):
    """xi of the distribution with its top `sink_frac` fraction of tokens removed.

    Attention SINK tokens (a few positions hogging mass) can hold the upper tail in
    the Frechet domain even when the rest of the distribution flattens.  Dropping the
    largest `sink_frac` values before estimating isolates the BULK: if the body has
    uniformized, this de-sinked xi drifts toward 0 (exponential) and negative
    (Weibull / bounded), while the full xi can stay positive on the sinks alone."""
    x = np.asarray(scores, dtype=np.float64).ravel()
    x = np.sort(x[x > 0])
    if sink_frac > 0 and x.size:
        drop = int(round(sink_frac * x.size))
        if drop > 0:
            x = x[:-drop]                           # remove the largest `drop` values
    return evt_xi(x, k_frac)


def decay_on_grid(p, grid):
    """Sorted-descending p interpolated onto a fixed normalized-rank grid in [0,1]."""
    s = np.sort(p)[::-1]
    u = np.linspace(0.0, 1.0, s.size)
    return np.interp(grid, u, s)


# --------------------------------------------------------------------------- #
# per-layer language -> image attention distribution, one forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_layer_image_attention(model, inputs, image_token_id):
    """For each LLM layer return the normalized distribution of attention mass the
    image tokens receive from the text queries.  Uses the standard image forward
    (pixel_values + image_grid_thw, mRoPE computed internally), unlike the MVBench
    sibling's inputs_embeds path.  Returns (n_layers, M) numpy and n_layers."""
    out = model(**inputs, output_attentions=True, use_cache=False)
    attn_mask = inputs["attention_mask"][0].bool()
    img_mask = (inputs["input_ids"][0] == image_token_id) & attn_mask
    img_pos = img_mask.nonzero(as_tuple=False).flatten()
    text_mask = attn_mask & ~img_mask                      # every non-image query position
    n_layers = len(out.attentions)
    dists = np.zeros((n_layers, img_pos.numel()), dtype=np.float64)
    for li in range(n_layers):
        a = out.attentions[li][0].float().mean(0)          # (S,S), mean over heads
        recv = a[text_mask][:, img_pos].sum(dim=0)         # (M,) mass per image token
        tot = recv.sum()
        if tot > 0:
            recv = recv / tot
        dists[li] = recv.cpu().numpy()
    del out
    return dists, n_layers


# --------------------------------------------------------------------------- #
# backbone plumbing (the ONLY backbone-specific code -- the EVT analysis below is
# identical for both).  Qwen2.5-VL: dynamic resolution, mRoPE, 2x2 merge, message-dict
# prompt via process_vision_info.  LLaVA-1.5: fixed 336x336 -> 576-token grid, plain
# prompt string + PIL image straight to the processor, image_token_index not _id.
# --------------------------------------------------------------------------- #
def resolve_backbone(args):
    """'auto' -> infer from the model name ('llava' substring => llava, else qwen).
    With no --model_name given, 'auto' falls back to qwen (the default backbone)."""
    if args.backbone != "auto":
        return args.backbone
    return "llava" if "llava" in (args.model_name or "").lower() else "qwen"


def default_model_id(backbone):
    return LLAVA_MODEL_ID if backbone == "llava" else QWEN_MODEL_ID


def load_backbone(backbone, model_name, dtype):
    """Load model + processor + image-token id for the chosen backbone."""
    cls = LlavaForConditionalGeneration if backbone == "llava" else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(model_name, torch_dtype=dtype, device_map="auto",
                                attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_name)
    if backbone == "llava":
        # LLaVA names the placeholder image_token_index; newer transformers also expose
        # image_token_id. Take whichever exists rather than betting on the version.
        image_token_id = getattr(model.config, "image_token_index",
                                 getattr(model.config, "image_token_id", None))
        if image_token_id is None:
            raise RuntimeError("could not find the image token id on the LLaVA config "
                               "(looked for image_token_index / image_token_id).")
    else:
        image_token_id = model.config.image_token_id
    return model, processor, image_token_id


def build_llava_prompt(processor, text):
    """LLaVA-1.5's "USER: <image>\\n{text} ASSISTANT:", ending exactly where the model's
    next token is the answer letter. Uses the processor's shipped chat template when
    there is one, and falls back to the literal v1.5 format otherwise."""
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    try:
        return processor.apply_chat_template(conv, add_generation_prompt=True)
    except (AttributeError, ValueError, TypeError):
        return f"USER: <image>\n{text} ASSISTANT:"


def build_inputs(backbone, processor, item, args, min_pixels, device, dtype):
    """Processor inputs (on device) ready for `model(**inputs, output_attentions=True)`.
    Qwen: message dicts + process_vision_info, --max_pixels/--min_pixels honoured.
    LLaVA: flat prompt + PIL straight to the processor (hard-resized to 336x336, so the
    pixel-budget args are ignored); pixel_values cast to the model dtype (the processor
    hands back fp32, the CLIP patch conv is in `dtype`)."""
    image = decode_image(item["image_cell"])
    if backbone == "llava":
        prompt = build_llava_prompt(processor, item["text"])
        proc = processor(images=image, text=prompt, return_tensors="pt")
        return {"input_ids": proc["input_ids"].to(device),
                "attention_mask": proc["attention_mask"].to(device),
                "pixel_values": proc["pixel_values"].to(device=device, dtype=dtype)}
    messages = build_ai2d_prompt(image, item["text"], args.max_pixels, min_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    proc = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in proc.items()}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resolve_backbone(args)
    # --dtype auto: LLaVA-1.5 ships fp16, Qwen2.5-VL bf16.
    dtype_name = args.dtype if args.dtype != "auto" else ("fp16" if backbone == "llava" else "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model_name = args.model_name or default_model_id(backbone)
    print(f"[backbone] {backbone}  model={model_name}  dtype={dtype_name}")
    model, processor, image_token_id = load_backbone(backbone, model_name, dtype)
    band = [int(x) for x in args.baseline_band.split(",")] if args.baseline_band else []

    # min_pixels: default mirrors max_pixels (fixed per-diagram token budget across
    # diagrams); <=0 disables the floor (variable budget, capped only by max_pixels).
    # Qwen only -- LLaVA hard-resizes to 336x336, so both pixel-budget args are ignored.
    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    items = load_items(args.data_root, args.split, args.max_samples)
    tasks = sorted({it["task"] for it in items})  # AI2D has no category column -> ["ai2d"]

    grid = np.linspace(0.0, 1.0, args.grid_points)
    xi_per_layer = None        # list of per-sample xi, one list per layer
    xi_desink_per_layer = None  # same, but with the top --sink_frac tokens removed
    xi_task_layer = None       # {task: [ [xi per sample] per layer ]}  (per-task horizon)
    decay_sum = None           # (n_layers, grid_points)
    band_xi_all = []           # xi of the band-averaged distribution (the scorer input)
    band_xi_task = {t: [] for t in tasks}
    n_layers = None
    seen = {t: 0 for t in tasks}
    n = 0

    for it in tqdm(items, desc="scoring", unit="ex"):
        task = it["task"]
        try:
            inputs = build_inputs(backbone, processor, it, args, min_pixels, device, dtype)
        except Exception as e:  # missing/corrupt diagram -> skip
            tqdm.write(f"skip idx={it.get('index')}: {e}")
            continue

        ipos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=False).flatten()
        if ipos.numel() < args.min_tokens:             # too few image tokens for a tail estimate
            continue

        try:
            dists, L = per_layer_image_attention(model, inputs, image_token_id)
        except Exception as e:
            tqdm.write(f"skip idx={it.get('index')}: {e}")
            continue

        if xi_per_layer is None:
            n_layers = L
            bad = [b for b in band if b >= L]
            if bad:
                raise ValueError(f"--baseline_band {bad} >= n_layers {L}")
            xi_per_layer = [[] for _ in range(L)]
            xi_desink_per_layer = [[] for _ in range(L)]
            xi_task_layer = {t: [[] for _ in range(L)] for t in tasks}
            decay_sum = np.zeros((L, args.grid_points))

        for li in range(n_layers):
            p = dists[li]
            x = evt_xi(p, args.k_frac)
            if x == x:                                 # skip NaNs
                xi_per_layer[li].append(x)
                xi_task_layer[task][li].append(x)
            xd = evt_xi_desinked(p, args.k_frac, args.sink_frac)
            if xd == xd:
                xi_desink_per_layer[li].append(xd)
            decay_sum[li] += decay_on_grid(p, grid)

        # xi of the exact distribution the scorer ranks on: the band average.
        if band:
            xb = evt_xi(dists[band].mean(0), args.k_frac)
            if xb == xb:
                band_xi_all.append(xb)
                band_xi_task[task].append(xb)

        seen[task] += 1
        n += 1
        del dists
        torch.cuda.empty_cache()

    if n == 0:
        print("no usable samples -- check --data_root layout (data/<split>-*.parquet).")
        return

    eps = args.domain_eps
    layers = list(range(n_layers))
    per_layer = []
    for li in layers:
        v = np.array(xi_per_layer[li], dtype=np.float64)
        vd = np.array(xi_desink_per_layer[li], dtype=np.float64)
        mean = float(v.mean()) if v.size else float("nan")
        std = float(v.std()) if v.size else float("nan")
        sem = float(std / np.sqrt(v.size)) if v.size else float("nan")
        # sign-aware domain split: Frechet (xi>eps) / Gumbel (|xi|<=eps) / Weibull (xi<-eps).
        frac_frechet = float((v > eps).mean()) if v.size else float("nan")
        frac_gumbel = float((np.abs(v) <= eps).mean()) if v.size else float("nan")
        frac_weibull = float((v < -eps).mean()) if v.size else float("nan")
        dmean = float(vd.mean()) if vd.size else float("nan")
        dsem = float(vd.std() / np.sqrt(vd.size)) if vd.size else float("nan")
        per_layer.append({"layer": li, "xi_mean": mean, "xi_std": std, "xi_sem": sem,
                          "xi_desink_mean": dmean, "xi_desink_sem": dsem,
                          "frac_heavy_tailed": frac_frechet,  # kept for back-compat
                          "frac_frechet": frac_frechet, "frac_gumbel": frac_gumbel,
                          "frac_weibull": frac_weibull, "n_est": int(v.size)})

    def _stats(vals):
        a = np.array(vals, dtype=np.float64)
        if not a.size:
            return {"xi_mean": float("nan"), "xi_sem": float("nan"), "n_est": 0}
        return {"xi_mean": float(a.mean()),
                "xi_sem": float(a.std() / np.sqrt(a.size)), "n_est": int(a.size)}

    valid = [t for t in tasks if seen[t]]
    band_summary = {**_stats(band_xi_all),
                    "per_task": {t: _stats(band_xi_task[t]) for t in valid}}
    # per-task xi(layer): mean xi at each layer, per task -> the horizon-by-task curve.
    # (For AI2D this is a single "ai2d" bucket equal to the overall curve.)
    per_task_per_layer_xi = {
        t: [float(np.mean(xi_task_layer[t][li])) if xi_task_layer[t][li] else float("nan")
            for li in layers] for t in valid}
    mean_decay = decay_sum / n

    out = {"experiment": "ai2d_layerwise_attention_evt_xi",
           "backbone": backbone, "model_name": model_name, "data_root": args.data_root,
           "split": args.split, "max_pixels": args.max_pixels, "min_pixels": min_pixels,
           "tasks": valid, "n": n, "n_layers": n_layers,
           "per_task_seen": {t: seen[t] for t in valid},
           "baseline_band": band, "k_frac": args.k_frac,
           "sink_frac": args.sink_frac, "domain_eps": eps,
           "estimator": "dekkers_einmahl_dehaan_moment",
           "grid": grid.tolist(), "layers": layers, "per_layer": per_layer,
           "band_xi": band_summary, "per_task_per_layer_xi": per_task_per_layer_xi,
           "mean_decay": mean_decay.tolist()}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {args.out}")

    # ---- console summary ----
    print(f"\n==== per-layer EVT tail index xi on AI2D ({n} samples, "
          f"{len(valid)} task(s), {n_layers} layers) ====")
    print("  xi > 0 Frechet (concentrated) | xi = 0 Gumbel (exponential) | "
          "xi < 0 Weibull (bounded -> toward UNIFORM, xi=-1)")
    print(f"  xi_ds = de-sinked xi (top {args.sink_frac:.0%} dropped): body drifting <0 "
          "while xi stays >0 => sinks sharp, bulk uniformizing")
    hdr = (f"{'layer':>6}{'xi':>9}{'xi_ds':>9}{'%Fre':>7}{'%Gum':>7}{'%Wei':>7}{'domain':>14}")
    print(hdr); print("-" * len(hdr))
    for r in per_layer:
        dom = ("heavy/Frechet" if r["xi_mean"] > eps else
               ("uniform/Weibull" if r["xi_mean"] < -eps else "~exponential"))
        mark = " <band" if r["layer"] in band else ""
        print(f"{r['layer']:>6}{r['xi_mean']:>9.3f}{r['xi_desink_mean']:>9.3f}"
              f"{r['frac_frechet']*100:>6.0f}%{r['frac_gumbel']*100:>6.0f}%"
              f"{r['frac_weibull']*100:>6.0f}%{dom:>14}{mark}")

    if band:
        print(f"\nband{band} averaged (the tail the scorer actually ranks on): "
              f"xi = {band_summary['xi_mean']:.3f} +/- {band_summary['xi_sem']:.3f} "
              f"(n={band_summary['n_est']})")
        print(f"  {'task':>22}{'band_xi':>10}{'sem':>8}{'n':>6}")
        for t in valid:
            s = band_summary["per_task"][t]
            print(f"  {t:>22}{s['xi_mean']:>10.3f}{s['xi_sem']:>8.3f}{s['n_est']:>6}")

    print("\nread: xi stays >0 at all depths => attention never uniformizes (stays Frechet);")
    print("      xi (or xi_ds) descending 0 -> -1 with depth => drift toward UNIFORM (Info-Horizon);")
    print("      xi_ds < 0 while xi > 0 => a few sinks stay sharp but the BULK has flattened.")

    if not args.no_plot:
        plot(out, os.path.splitext(args.out)[0] + ".png")


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def plot(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE = "#2a78d6"; RED = "#e34948"
    INK = "#0b0b0b"; INK_SECOND = "#52514e"; MUTED = "#898781"
    GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECOND, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})

    ORANGE = "#e08a1e"
    layers = np.array(out["layers"])
    band = out["baseline_band"]
    xi = np.array([r["xi_mean"] for r in out["per_layer"]])
    sem = np.array([r["xi_sem"] for r in out["per_layer"]])
    xid = np.array([r.get("xi_desink_mean", np.nan) for r in out["per_layer"]])
    grid = np.array(out["grid"]); mean_decay = np.array(out["mean_decay"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: xi vs layer, between the xi=0 (exponential) and xi=-1 (uniform) lines ----
    if band:
        axA.axvspan(min(band) - 0.5, max(band) + 0.5, color=GRID, alpha=0.6, zorder=0,
                    label=f"band {min(band)}-{max(band)}")
    axA.axhline(0, color=INK, lw=1.4, ls="-")
    axA.text(layers.max(), 0, "  xi = 0  (exponential / Gumbel)", color=INK,
             va="bottom", ha="right", fontsize=9)
    axA.axhline(-1, color=MUTED, lw=1.2, ls="--")
    axA.text(layers.max(), -1, "  xi = -1  (uniform / Weibull)", color=MUTED,
             va="bottom", ha="right", fontsize=9)
    axA.fill_between(layers, xi - sem, xi + sem, color=BLUE, alpha=0.18, zorder=1)
    pos = xi > 0
    axA.plot(layers, xi, "-", color=INK_SECOND, lw=1.4, zorder=2, label="xi (full)")
    axA.scatter(layers[pos], xi[pos], s=26, color=RED, zorder=3)
    axA.scatter(layers[~pos], xi[~pos], s=26, color=BLUE, zorder=3)
    axA.plot(layers, xid, "--o", color=ORANGE, lw=1.3, ms=4, zorder=2,
             label=f"xi de-sinked (top {out.get('sink_frac', 0.01):.0%} dropped)")
    lo = float(np.nanmin([np.nanmin(xi), np.nanmin(xid), -1.0])) - 0.1
    hi = float(np.nanmax([np.nanmax(xi), np.nanmax(xid)])) + 0.12
    axA.set_ylim(lo, hi)
    axA.set_xlabel("LLM layer"); axA.set_ylabel("EVT tail index  xi")
    axA.set_title("Per-layer tail index: Frechet -> exponential -> uniform", loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=9, loc="best")

    # ---- Panel B: mean sorted-decay curves for every layer (visual companion) ----
    norm = matplotlib.colors.Normalize(vmin=layers.min(), vmax=layers.max())
    cmap = plt.cm.viridis
    for li in layers:
        axB.semilogy(grid, np.clip(mean_decay[li], 1e-8, None),
                     color=cmap(norm(li)), lw=1.3, alpha=0.85)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axB, pad=0.01)
    cbar.set_label("LLM layer", color=INK_SECOND); cbar.outline.set_edgecolor(MUTED)
    axB.set_xlabel("normalized rank  (0 = most-attended image token)")
    axB.set_ylabel("mean attention mass  (log)")
    axB.set_title("Sorted decay, all layers  (straight = exponential)", loc="left", weight="bold")
    axB.grid(color=GRID, lw=0.7, which="both")

    fig.suptitle(f"AI2D ({out['split']}), n={out['n']}, {out['model_name'].split('/')[-1]} — "
                 f"does the diagram-attention distribution drift toward uniform with depth?",
                 x=0.01, ha="left", weight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    print(f"wrote {png_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-layer EVT tail index (xi) of language->image attention on AI2D.")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava"], default="auto",
                   help="VLM backbone. 'auto' infers from --model_name ('llava' substring "
                        "=> LLaVA-1.5, else Qwen2.5-VL). qwen: dynamic resolution + mRoPE; "
                        "llava: fixed 336x336 -> 576-token grid, 1D RoPE.")
    p.add_argument("--model_name", default=None,
                   help=f"HF id. Default per backbone: qwen={QWEN_MODEL_ID}, llava={LLAVA_MODEL_ID}.")
    p.add_argument("--data_root", required=True,
                   help="AI2D dir holding data/ with <split>-*.parquet (lmms-lab/ai2d).")
    p.add_argument("--split", default="test", choices=["test"],
                   help="AI2D ships answers with `test`; there is no other split to use.")
    p.add_argument("--max_samples", type=int, default=300, help="cap total samples.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="per-diagram pixel ceiling (downscales large diagrams).")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="per-diagram pixel floor. Default: mirror --max_pixels for a FIXED "
                        "per-diagram token budget across diagrams; pass <=0 to disable the floor.")
    p.add_argument("--min_tokens", type=int, default=16,
                   help="skip diagrams with fewer than this many image tokens (too few for a tail).")
    p.add_argument("--baseline_band", default="12,13,14,15,16",
                   help="scorer band: shaded in the plot AND used for the band-averaged xi ('' to disable).")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="top fraction of order statistics used by the moment estimator.")
    p.add_argument("--sink_frac", type=float, default=0.01,
                   help="fraction of largest-mass tokens dropped for the DE-SINKED xi "
                        "(isolates the bulk from attention sinks). 0 disables.")
    p.add_argument("--domain_eps", type=float, default=0.05,
                   help="|xi|<=eps counts as Gumbel/exponential in the per-layer domain split.")
    p.add_argument("--grid_points", type=int, default=200,
                   help="resolution of the stored mean sorted-decay curve.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 for qwen, fp16 for llava (its shipped weights).")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out", default="results_layerwise_attn_tail_ai2d.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
