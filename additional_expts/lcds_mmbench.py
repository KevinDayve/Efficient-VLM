"""
lcds_mmbench.py -- Layer-Consensus Diverse Selection, inference-only, on MMBench.
================================================================================
A training-free visual-token selection method, tested as a SELECTION strategy
(no in-LLM pruning yet): compute the selection from one dense forward, then score
the multiple-choice answer through the frozen model on the kept token subset, and
compare accuracy + spatial dispersion against uniform and vanilla attention-top-K.

The method (your diagram, made concrete):

  dense image tokens
        |
        v   at every prune layer m  (progressive, monotone)
  [debiased attention top-K]   keep top-rho_m of the SURVIVING tokens by
        |                      r_m = A_m - proj_B(A_m), where B is the positional
        |                      attention template captured from the EARLY layers.
        |                      (subtracting the early-layer positional bias is the
        |                       whole reason this can beat uniform; raw attention
        |                       top-K does not -- see inference_only.py.)
        v   at the final layer
  [maximal-distance selection] farthest-point sampling on the final-layer features
        |                      of the survivors -> k diverse anchors (drops
        v                      geometric/semantic redundancy; the World-2 hedge).
  kept subset -> score answer

Strategies reported (so one run attributes every gain):
  * uniform         : evenly spaced tokens over the merged grid          (floor)
  * attention_topk  : band-averaged raw attention, global top-K          (FastV-ish)
  * lcds            : debiased cascade + farthest-point diversity        (ours)
  * lcds_nodebias   : ours but with RAW attention in the cascade         (ablate debias)
  * lcds_nodiv      : ours but final = top-K by salience, no diversity   (ablate diversity)

Run:
    python lcds_mmbench.py \
        --data_root ~/datasets/MMBench --split dev \
        --max_samples 300 --rhos 0.05,0.10,0.25 \
        --out results_lcds_mmbench.json
"""

import os
import sys
import glob
import io
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# shared, validated helpers ------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# score_answer is visual-modality-agnostic: it masks all visual positions off, turns
# the kept ones back on, slices the (absolute) position ids, and reads the first
# answer-token log-probs. Works for image tokens exactly as for video tokens.
from oracle_check import score_answer
# MMBench data layer / prompt (single image, l2-category grouping, parquet PNG bytes).
from overlap_subset_mmbench import build_mmbench_prompt, build_prompt_text

STRATEGIES = ("uniform", "attention_topk", "lcds", "lcds_nodebias", "lcds_nodiv")


# --------------------------------------------------------------------------- #
# positions (mirror oracle_check.build_full_positions, image variant)
# --------------------------------------------------------------------------- #
def build_image_positions(model, input_ids, image_positions, image_grid_thw, attention_mask):
    """mRoPE position ids for a single-image sequence. Mirrors build_full_positions
    but marks image tokens (mm type 1; video used 2) and passes image_grid_thw.
    Falls back to the mm_token_type_ids-free signature if the fork doesn't take it."""
    try:
        mm = torch.zeros_like(input_ids)
        mm[0, image_positions] = 1
        pos, _ = model.model.get_rope_index(
            input_ids, mm_token_type_ids=mm,
            image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    except TypeError:
        pos, _ = model.model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    return pos  # (3,1,S)


# --------------------------------------------------------------------------- #
# per-layer language->image attention (received attention per image token)
# --------------------------------------------------------------------------- #
def received_attention(attentions, image_positions, layers):
    """{layer -> (M,)}: attention each image token RECEIVES from text queries,
    summed over text rows, head-averaged. Computed only for `layers` (a set)."""
    S = attentions[0].shape[-1]
    text_mask = torch.ones(S, dtype=torch.bool, device=attentions[0].device)
    text_mask[image_positions] = False
    out = {}
    for L in layers:
        a = attentions[L][0].mean(0)                     # (S,S) head-mean
        out[L] = a[text_mask][:, image_positions].sum(dim=0).float()  # (M,)
    return out


# --------------------------------------------------------------------------- #
# debias: remove the early-layer positional template (least squares residual)
# --------------------------------------------------------------------------- #
def build_bias_basis(received, bias_layers, M, device):
    """Design matrix Phi = [1 | B_l0 | B_l1 | ...] over image tokens; the early
    layers are ~content-free positional bias, so projecting onto their span and
    taking the residual removes the positional component."""
    cols = [torch.ones(M, device=device)]
    for L in bias_layers:
        cols.append(received[L])
    return torch.stack(cols, dim=1)                       # (M, 1+n_bias)


def debias_vec(a, Phi):
    """residual of a after least-squares projection onto columns of Phi. Using a
    regression residual (not raw subtraction) auto-scales the basis -- essential
    for the null-prompt basis, whose magnitude differs (different #text tokens)."""
    sol = torch.linalg.lstsq(Phi.float(), a.float().unsqueeze(1)).solution
    return a.float() - (Phi.float() @ sol).squeeze(1)


@torch.no_grad()
def null_received(model, processor, image, null_text, max_pixels, layers,
                  image_token_id, device):
    """Null-prompt positional/agnostic template: run the SAME image with a generic
    query and return {layer -> received attention (M,)} plus M. Subtracting this
    (via debias_vec) removes positional bias AND question-agnostic saliency,
    isolating the query-specific signal. Costs one extra forward per sample."""
    messages = build_mmbench_prompt(image, null_text, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    ipos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    pos = build_image_positions(model, input_ids, ipos, image_grid_thw, attn)
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                use_cache=False, output_attentions=True)
    recv = received_attention(out.attentions, ipos, set(layers))
    del out
    return recv, ipos.numel()


# --------------------------------------------------------------------------- #
# selection primitives
# --------------------------------------------------------------------------- #
def select_uniform_image(M, k, device):
    """Evenly spaced tokens over the flattened merged grid (content-free floor)."""
    if k >= M:
        return torch.arange(M, device=device)
    step = max(1, M // k)
    idx = list(range(0, M, step))[:k]
    i = 0
    while len(idx) < k and i < M:
        if i not in idx:
            idx.append(i)
        i += 1
    return torch.tensor(sorted(idx[:k]), device=device, dtype=torch.long)


def select_topk(scores, k):
    """Global top-k indices, sorted."""
    k = min(k, scores.numel())
    return torch.topk(scores, k).indices.sort().values


def farthest_point(feats, seed_scores, k):
    """Greedy maximal-distance selection in cosine space. Seeded at the highest-
    salience point, then repeatedly add the point farthest (max-min cosine
    distance) from the chosen set. Returns LOCAL indices into `feats`."""
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    X = F.normalize(feats.float(), dim=1)
    first = int(seed_scores.argmax())
    sel = [first]
    mind = 1.0 - (X @ X[first])                           # cosine distance to seed
    mind[first] = -1.0
    for _ in range(k - 1):
        nxt = int(mind.argmax())
        sel.append(nxt)
        mind = torch.minimum(mind, 1.0 - (X @ X[nxt]))
        mind[torch.tensor(sel, device=feats.device)] = -1.0
    return torch.tensor(sorted(sel), device=feats.device)


# --------------------------------------------------------------------------- #
# the cascade: progressive (debiased) attention top-K over the surviving set
# --------------------------------------------------------------------------- #
def cascade(received, prune_layers, phi_by_layer, M, cand, device, debias_on):
    """Monotone shrink M -> cand across prune_layers. Returns (surviving_idx,
    aggregated_salience over ALL M). Aggregated salience is the per-layer signal
    averaged over prune layers -- used for the FPS seed and the no-diversity ablation.
    phi_by_layer[L] is the debias basis for layer L (shared for 'project', per-layer
    null map for 'null_prompt')."""
    surviving = torch.arange(M, device=device)
    n = len(prune_layers)
    sal = torch.zeros(M, device=device)
    for i, L in enumerate(prune_layers):
        a = debias_vec(received[L], phi_by_layer[L]) if debias_on else received[L]
        sal += a
        # geometric schedule M -> cand; last step lands exactly on cand
        target = round(M * (cand / M) ** ((i + 1) / n))
        target = cand if i == n - 1 else max(cand, target)
        target = min(target, surviving.numel())
        keep_local = torch.topk(a[surviving], target).indices
        surviving = surviving[keep_local]
    return surviving.sort().values, sal / n


def lcds_select(strategy, received, feats, prune_layers, phi_by_layer, M, k, cand_mult, device):
    """Full method (and its ablations) -> kept image-token indices (0..M-1)."""
    if k >= M:
        return torch.arange(M, device=device)
    cand = min(M, max(k, int(round(cand_mult * k))))
    debias_on = (strategy != "lcds_nodebias")
    surviving, sal = cascade(received, prune_layers, phi_by_layer, M, cand, device, debias_on)
    if strategy == "lcds_nodiv":                          # final = top-K by salience
        local = select_topk(sal[surviving], k)
    else:                                                 # final = maximal-distance
        local = farthest_point(feats[surviving], sal[surviving], k)
    return surviving[local].sort().values


# --------------------------------------------------------------------------- #
# spatial dispersion (single merged grid): 0 = clustered, 1 = spread
# --------------------------------------------------------------------------- #
def grid_dims(M, image_grid_thw, merge):
    """Merged-grid (Hm, Wm) from the patch grid; fall back to a square if it
    doesn't factor cleanly."""
    try:
        _, h, w = [int(x) for x in image_grid_thw[0].tolist()]
        Hm, Wm = h // merge, w // merge
        if Hm * Wm == M:
            return Hm, Wm
    except Exception:
        pass
    Wm = int(round(M ** 0.5))
    return max(1, M // max(1, Wm)), max(1, Wm)


def mean_dispersion(keep_idx, Hm, Wm):
    idx = keep_idx.detach().cpu().numpy()
    if idx.size < 2:
        return float("nan")
    ys, xs = idx // Wm, idx % Wm
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    d = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
    iu = np.triu_indices(coords.shape[0], k=1)
    max_d = max(1, max(Hm - 1, Wm - 1))
    return float(d[iu].mean() / max_d) if iu[0].size else float("nan")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_items(data_root, split, l2_categories, max_samples):
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

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
    for _, rec in tqdm(df.iterrows(), total=len(df), desc=f"MMBench/{split}"):
        text, letters, gt_idx = build_prompt_text(rec)
        if len(letters) < 2 or gt_idx < 0:
            continue
        items.append({"task": rec.get("l2-category", "unknown"), "image": _decode(rec["image"]),
                      "text": text, "letters": letters, "gt_idx": gt_idx})
    return items


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    image_token_id = model.config.image_token_id
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)

    rhos = [float(x) for x in args.rhos.split(",")]
    items = load_items(args.data_root, args.split, args.l2_categories, args.max_samples)

    # layer plan resolved lazily once we know n_layers.
    bias_layers = prune_layers = feature_hs_idx = None

    correct = {s: {r: 0 for r in rhos} for s in STRATEGIES}
    disp = {s: {r: [] for r in rhos} for s in STRATEGIES}
    full_correct = 0
    n = 0

    from tqdm import tqdm
    for it in tqdm(items, desc="eval"):
        letter_ids = [processor.tokenizer.encode(L, add_special_tokens=False)[0] for L in it["letters"]]
        messages = build_mmbench_prompt(it["image"], it["text"], args.max_pixels)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        ipos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
        if ipos.numel() < 16:
            continue
        M = ipos.numel()
        pos = build_image_positions(model, input_ids, ipos, image_grid_thw, attn)

        # single dense forward: attentions (all layers) + merged embeds + features.
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                        use_cache=False, output_attentions=True, output_hidden_states=True)
        attentions = out.attentions
        base = out.hidden_states[0].detach()              # merged inputs_embeds (1,S,d)

        if bias_layers is None:                           # resolve the layer plan once
            L = len(attentions)
            bias_layers = [x for x in ([int(v) for v in args.bias_layers.split(",")]
                                       if args.bias_layers else [0, 1, 2]) if x < L]
            if args.prune_stride > 0:                      # prune at every k-th layer
                prune_layers = list(range(args.prune_stride, L, args.prune_stride))
            elif args.prune_layers:
                prune_layers = [int(v) for v in args.prune_layers.split(",")]
            else:
                prune_layers = [round(L * f) for f in (0.4, 0.6, 0.8)]
            prune_layers = sorted({min(x, L - 1) for x in prune_layers})
            fl = args.feature_layer if args.feature_layer >= 0 else prune_layers[-1]
            feature_hs_idx = min(fl + 1, len(out.hidden_states) - 1)  # hidden_states[l+1] = layer l out
            print(f"[plan] n_layers={L} debias={args.debias} bias={bias_layers} "
                  f"prune={prune_layers} feature_hs_idx={feature_hs_idx}")

        feats = out.hidden_states[feature_hs_idx][0].detach()        # (S,d)
        feats_img = feats[ipos]                                       # (M,d)
        # vanilla attention_topk uses the SAME layers as the cascade, band-averaged
        # and un-debiased -- the fair "raw attention" foil.
        need = set(prune_layers) | (set(bias_layers) if args.debias == "project" else set())
        recv = received_attention(attentions, ipos, need)
        band = torch.stack([recv[L] for L in prune_layers]).mean(0)
        del out, attentions

        # debias basis per prune layer (see --debias).
        if args.debias == "project":                      # early-layer positional template (no extra fwd)
            Phi = build_bias_basis(recv, bias_layers, M, device)
            phi_by_layer = {L: Phi for L in prune_layers}
        else:                                             # null_prompt: same image, generic query
            null_recv, M_null = null_received(model, processor, it["image"], args.null_text,
                                              args.max_pixels, prune_layers, image_token_id, device)
            if M_null != M:
                print(f"skip: null-prompt M {M_null} != {M} (grid mismatch)"); continue
            ones = torch.ones(M, 1, device=device)
            phi_by_layer = {L: torch.cat([ones, null_recv[L].to(device).unsqueeze(1)], dim=1)
                            for L in prune_layers}

        Hm, Wm = grid_dims(M, image_grid_thw, merge)

        # full-model reference (all image tokens kept)
        keep_all = torch.arange(M, device=device)
        lp_full = score_answer(model, base, pos, attn, ipos, keep_all, letter_ids)
        full_correct += int(lp_full.argmax().item() == it["gt_idx"])

        for rho in rhos:
            k = max(1, int(round(rho * M)))
            for s in STRATEGIES:
                if s == "uniform":
                    keep = select_uniform_image(M, k, device)
                elif s == "attention_topk":
                    keep = select_topk(band, k)
                else:
                    keep = lcds_select(s, recv, feats_img, prune_layers, phi_by_layer, M, k,
                                       args.cand_mult, device)
                lp = score_answer(model, base, pos, attn, ipos, keep, letter_ids)
                correct[s][rho] += int(lp.argmax().item() == it["gt_idx"])
                d = mean_dispersion(keep, Hm, Wm)
                if d == d:
                    disp[s][rho].append(d)
        n += 1
        del base, feats, feats_img, recv
        if n % 25 == 0:
            r = rhos[-1]
            msg = " | ".join(f"{s[:8]} {correct[s][r]/n:.3f}" for s in STRATEGIES)
            print(f"[{n}] rho={r}: {msg}  (full {full_correct/n:.3f})")

    if n == 0:
        print("no usable samples."); return

    out_json = {"n": n, "full_accuracy": full_correct / n, "rhos": rhos,
                "debias": args.debias, "bias_layers": bias_layers, "prune_layers": prune_layers,
                "feature_hs_idx": feature_hs_idx, "cand_mult": args.cand_mult, "table": {}}
    print(f"\n==== LCDS on MMBench ({n} samples) ====")
    print(f"full-model accuracy: {full_correct/n:.4f}\n")
    hdr = f"{'strategy':<16}{'rho':>6}{'acc':>9}{'dispersion':>12}"
    print(hdr); print("-" * len(hdr))
    for s in STRATEGIES:
        out_json["table"][s] = {}
        for rho in rhos:
            acc = correct[s][rho] / n
            dsp = float(np.mean(disp[s][rho])) if disp[s][rho] else float("nan")
            out_json["table"][s][str(rho)] = {"accuracy": acc, "dispersion": dsp}
            print(f"{s:<16}{rho:>6.2f}{acc:>9.4f}{dsp:>12.3f}")
    with open(args.out, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nread: lcds > uniform => the method beats the floor attention_topk can't.")
    print("      lcds > lcds_nodebias => de-biasing the early-layer positional prior earns its keep.")
    print("      lcds > lcds_nodiv    => the maximal-distance diversity head earns its keep.")


def parse_args():
    p = argparse.ArgumentParser(description="Layer-Consensus Diverse Selection on MMBench (inference-only).")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data_root", required=True, help="MMBench dir with data/<split>-*.parquet.")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--l2_categories", nargs="+", default=["all"])
    p.add_argument("--rhos", default="0.05,0.10,0.25")
    p.add_argument("--debias", choices=["project", "null_prompt"], default="project",
                   help="'project': residual off early-layer template (no extra fwd). "
                        "'null_prompt': residual off a same-image generic-query attention map "
                        "(one extra fwd/sample; removes positional + question-agnostic saliency).")
    p.add_argument("--null_text", default="Describe the image.",
                   help="generic query for the --debias null_prompt template.")
    p.add_argument("--bias_layers", default="", help="early positional-template layers (default 0,1,2).")
    p.add_argument("--prune_layers", default="", help="cascade prune layers (default ~0.4,0.6,0.8*depth).")
    p.add_argument("--prune_stride", type=int, default=0,
                   help="if >0, cluster/prune at every k-th layer (overrides --prune_layers). e.g. 4.")
    p.add_argument("--feature_layer", type=int, default=-1,
                   help="decoder layer whose hidden states drive the final maximal-distance "
                        "selection (default = last prune layer; pass 35 for 3B's final layer). "
                        "Values >= depth clamp to the final layer.")
    p.add_argument("--cand_mult", type=float, default=2.0,
                   help="candidate superset size before the final diversity selection = cand_mult*k.")
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="results_lcds_mmbench.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())