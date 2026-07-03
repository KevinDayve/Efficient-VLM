"""
layerwise_attn_tail_mmbench.py -- per-layer EVT tail index (xi) of the language->image
attention distribution.  Does the tail collapse toward exponential at some layers?
======================================================================================
Companion to ``layerwise_attention_mvbench.py``.  That script asks which LLM layer is
the best *selection* signal; this one asks about the SHAPE of the signal, layer by
layer, with a single principled instrument: the Extreme Value Index xi (gamma).

The scorer used in ``inference_only*.py`` keeps the top-K image tokens by attention
mass over a fixed band (layers 12..16).  We have seen some layers look heavy-tailed --
a handful of image tokens soak up most of the attention.  The clean question is not
"how concentrated" but "which extreme-value domain is the upper tail in":

  xi > 0   Frechet    -> genuinely heavy-tailed (Pareto-like)
  xi = 0   Gumbel     -> the tail IS exponential  ("collapsed")
  xi < 0   Weibull    -> bounded / light tail

So a single xi(layer) curve, read against the xi = 0 line, says exactly which layers
collapse toward exponential.  We estimate xi with the Dekkers-Einmahl-de Haan moment
estimator (sign-aware, unlike Hill which is pinned to xi >= 0), computed on the upper
order statistics of each image's per-token attention-mass distribution.

For every LLM layer we build ONE distribution per image:

  p_i = attention mass image-token i receives, summed over all text queries,
        renormalized so sum_i p_i = 1        (xi is scale-invariant, so the
                                              normalization is only for the plot).

and report mean xi +/- SEM across images, per layer.  A mean sorted-decay curve per
layer is kept for the visual companion panel (straight on a semilog-rank axis <=> xi=0).

Self-contained: no imports from the rest of the repo.

Run:
    python layerwise_attn_tail_mmbench.py \
        --data_root ~/datasets/MMBench --split dev \
        --max_samples 200 --out results_layerwise_attn_tail_mmbench.json
"""

import argparse
import glob
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

SYSTEM_PROMPT = (
    "Carefully look at the image and pay attention to the objects, their "
    "attributes, spatial relations, and any text present. Based on your "
    "observations, select the best option that accurately addresses the question."
)
OPTION_LETTERS = ["A", "B", "C", "D"]


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
    image tokens receive from the text queries.  Returns (n_layers, M) numpy."""
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
# MMBench loading + prompt (single image, multiple choice), self-contained
# --------------------------------------------------------------------------- #
def _is_present(v):
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s.lower() != "nan"


def build_prompt_text(record):
    letters, texts = [], []
    for L in OPTION_LETTERS:
        if _is_present(record.get(L)):
            letters.append(L)
            texts.append(str(record[L]).strip())
    lines = SYSTEM_PROMPT + "\n"
    if _is_present(record.get("hint")):
        lines += f"Hint: {str(record['hint']).strip()}\n"
    lines += f"Question: {str(record['question']).strip()}\nOptions:\n"
    lines += "".join(f"({L}) {t}\n" for L, t in zip(letters, texts))
    lines += "Answer with the option's letter (A, B, C, ...) directly."
    return lines, letters


def build_messages(image, text, max_pixels=None):
    img = {"type": "image", "image": image}
    if max_pixels is not None:
        img["max_pixels"] = max_pixels
    return [{"role": "user", "content": [img, {"type": "text", "text": text}]}]


def load_items(data_root, split, l2_categories, max_samples):
    import pandas as pd
    data_dir = os.path.join(os.path.expanduser(data_root), "data")
    matches = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not matches:
        raise FileNotFoundError(f"no {split}-*.parquet under {data_dir}")
    df = pd.read_parquet(matches[0])
    if l2_categories != ["all"]:
        df = df[df["l2-category"].isin(set(l2_categories))]
    if max_samples:
        df = df.iloc[:max_samples]

    def _decode(cell):
        b = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(b)).convert("RGB")

    items = []
    for _, rec in tqdm(df.iterrows(), total=len(df), desc=f"Loading MMBench/{split}"):
        text, letters = build_prompt_text(rec)
        if len(letters) < 2:
            continue
        items.append({"index": rec.get("index"),
                      "task": rec.get("l2-category", "unknown"),
                      "image": _decode(rec["image"]), "text": text})
    return items


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="cuda", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    image_token_id = model.config.image_token_id
    device = model.device
    band = [int(x) for x in args.baseline_band.split(",")] if args.baseline_band else []

    items = load_items(args.data_root, args.split, args.l2_categories, args.max_samples)

    grid = np.linspace(0.0, 1.0, args.grid_points)
    xi_per_layer = None        # list of per-sample xi, one list per layer
    decay_sum = None           # (n_layers, grid_points)
    n_layers = None
    n = 0

    for item in tqdm(items, desc="scoring", unit="ex"):
        try:
            messages = build_messages(item["image"], item["text"], args.max_pixels)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(messages)
            inputs = processor(text=[text], images=img_in, videos=vid_in,
                               padding=True, return_tensors="pt").to(device)
            dists, L = per_layer_image_attention(model, inputs, image_token_id)
        except Exception as e:
            tqdm.write(f"skip idx={item.get('index')}: {e}")
            continue
        if dists.shape[1] < 16:                # too few image tokens for a tail estimate
            continue

        if xi_per_layer is None:
            n_layers = L
            xi_per_layer = [[] for _ in range(L)]
            decay_sum = np.zeros((L, args.grid_points))

        for li in range(n_layers):
            p = dists[li]
            x = evt_xi(p, args.k_frac)
            if x == x:                         # skip NaNs
                xi_per_layer[li].append(x)
            decay_sum[li] += decay_on_grid(p, grid)
        n += 1
        torch.cuda.empty_cache()

    if n == 0:
        print("no usable samples -- check --data_root layout (data/<split>-*.parquet).")
        return

    layers = list(range(n_layers))
    per_layer = []
    for li in layers:
        v = np.array(xi_per_layer[li], dtype=np.float64)
        mean = float(v.mean()) if v.size else float("nan")
        std = float(v.std()) if v.size else float("nan")
        sem = float(std / np.sqrt(v.size)) if v.size else float("nan")
        frac_heavy = float((v > 0).mean()) if v.size else float("nan")
        per_layer.append({"layer": li, "xi_mean": mean, "xi_std": std, "xi_sem": sem,
                          "frac_heavy_tailed": frac_heavy, "n_est": int(v.size)})

    mean_decay = decay_sum / n

    out = {"experiment": "mmbench_layerwise_attention_evt_xi",
           "model_name": args.model_name, "data_root": args.data_root, "split": args.split,
           "l2_categories": args.l2_categories, "n": n, "n_layers": n_layers,
           "baseline_band": band, "k_frac": args.k_frac, "estimator": "dekkers_einmahl_dehaan_moment",
           "grid": grid.tolist(), "layers": layers, "per_layer": per_layer,
           "mean_decay": mean_decay.tolist()}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {args.out}")

    # ---- console summary ----
    print(f"\n==== per-layer EVT tail index xi on MMBench ({n} samples, {n_layers} layers) ====")
    print("  xi > 0 heavy (Frechet) | xi = 0 exponential (Gumbel) | xi < 0 light (Weibull)")
    hdr = f"{'layer':>6}{'xi_mean':>10}{'xi_sem':>9}{'%heavy':>9}{'domain':>13}"
    print(hdr); print("-" * len(hdr))
    for r in per_layer:
        dom = "heavy/Frechet" if r["xi_mean"] > 0.05 else ("light/Weibull" if r["xi_mean"] < -0.05 else "~exponential")
        mark = " <band" if r["layer"] in band else ""
        print(f"{r['layer']:>6}{r['xi_mean']:>10.3f}{r['xi_sem']:>9.3f}"
              f"{r['frac_heavy_tailed']*100:>8.0f}%{dom:>13}{mark}")
    print("\nread: xi rising above 0 in a band => heavier tail there; xi near 0 => the")
    print("      attention distribution has collapsed toward an exponential law.")

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

    layers = np.array(out["layers"])
    band = out["baseline_band"]
    xi = np.array([r["xi_mean"] for r in out["per_layer"]])
    sem = np.array([r["xi_sem"] for r in out["per_layer"]])
    grid = np.array(out["grid"]); mean_decay = np.array(out["mean_decay"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: xi vs layer, with the xi = 0 exponential boundary ----
    if band:
        axA.axvspan(min(band) - 0.5, max(band) + 0.5, color=GRID, alpha=0.6, zorder=0,
                    label=f"band {min(band)}-{max(band)}")
    axA.axhline(0, color=INK, lw=1.4, ls="-")
    axA.text(layers.max(), 0, "  xi = 0  (exponential / Gumbel)", color=INK,
             va="bottom", ha="right", fontsize=9)
    axA.fill_between(layers, xi - sem, xi + sem, color=BLUE, alpha=0.18, zorder=1)
    pos = xi > 0
    axA.plot(layers, xi, "-", color=INK_SECOND, lw=1.4, zorder=2)
    axA.scatter(layers[pos], xi[pos], s=26, color=RED, zorder=3, label="heavy (xi>0)")
    axA.scatter(layers[~pos], xi[~pos], s=26, color=BLUE, zorder=3, label="light (xi<=0)")
    axA.set_xlabel("LLM layer"); axA.set_ylabel("EVT tail index  xi")
    axA.set_title("Per-layer tail index of image attention", loc="left", weight="bold")
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

    fig.suptitle(f"MMBench ({out['split']}), n={out['n']}, {out['model_name'].split('/')[-1]} — "
                 f"does the image-attention tail collapse to exponential?",
                 x=0.01, ha="left", weight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    print(f"wrote {png_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-layer EVT tail index (xi) of language->image attention on MMBench.")
    p.add_argument("--model_name", default=MODEL_ID)
    p.add_argument("--data_root", required=True,
                   help="MMBench dir holding data/ with <split>-*.parquet (lmms-lab/MMBench).")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--l2_categories", nargs="+", default=["all"])
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--baseline_band", default="12,13,14,15,16",
                   help="scorer band, only shaded for reference in the plot/table ('' to disable).")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="top fraction of order statistics used by the moment estimator.")
    p.add_argument("--grid_points", type=int, default=200,
                   help="resolution of the stored mean sorted-decay curve.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out", default="results_layerwise_attn_tail_mmbench.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
