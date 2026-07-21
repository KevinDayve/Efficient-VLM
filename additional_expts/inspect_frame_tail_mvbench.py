"""
inspect_frame_tail_mvbench.py -- ONE clip, laid out for the eye: every frame the model
saw, that frame's EVT tail index, and the prompt it was asked.
======================================================================================
Single-clip inspector for ``framewise_attn_tail_mvbench.py``.  That script aggregates
xi(layer, frame) over many clips and asks whether the tail is frame-specific ON AVERAGE.
This one refuses to aggregate: it runs a single clip and prints the picture, so you can
read the question, look at the frames, decide yourself which frame answers it, and check
whether THAT frame's xi stands out.

There is no ground-truth "correct frame" label in MVBench to key on -- the 5 bounded
tasks carry a coarse [start, end] segment, and the other 15 carry nothing at all.  So
the correct frame here is the one YOU identify from the prompt.  That is the whole
point of putting the frames, the xi values, and the question in one figure: this is a
qualitative instrument for forming a hypothesis, not a measurement.  If a frame does
look different, ``framewise_attn_tail_mvbench.py`` over many clips is what tests it.

Same forward, same head-averaged text->video attention read-out, same moment estimator,
same official protocol as the siblings -- all imported, not reimplemented.

xi is estimated WITHIN a frame, over that frame's own spatial tokens:
    xi > 0  a few spatial tokens hold the frame   |  xi = 0  exponential
    xi < 0  flat / bounded over the frame (xi = -1 uniform)
and is reported next to that frame's SHARE of the clip's attention mass, because the two
come apart: a frame can hold very little mass and still be internally heavy-tailed.

Frames come in PAIRS (Qwen2.5-VL temporal_patch_size=2), so one xi labels two frames --
they are drawn together under a single value.

Run:
    python inspect_frame_tail_mvbench.py \
        --data_root ~/Experiments/MVBench \
        --task "Action Sequence" --index 0 \
        --official_sampling --num_segments 16 \
        --out inspect_action_sequence_0.png
"""

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oracle_check import build_full_positions
from inference import (DATA_LIST, ANSWER_PREFIX, make_mvbench_prompt, build_prompt,
                       letter_token_ids)
from layerwise_attn_tail_mvbench import evt_xi, per_layer_video_attention

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def frames_as_numpy(vid_in):
    """The actual pixels handed to the vision tower, as a list of (H, W, 3) uint8.

    Read back from process_vision_info's output rather than re-decoding the clip, so what
    is drawn is exactly what the model saw (post-resize), for either sampling mode."""
    v = vid_in[0]
    if isinstance(v, torch.Tensor):                 # (n_frames, 3, H, W)
        a = v.permute(0, 2, 3, 1).cpu().numpy()
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        return [a[i] for i in range(a.shape[0])]
    return [np.asarray(f.convert("RGB")) for f in v]   # list of PIL


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    model.eval(); model.requires_grad_(False)
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    band = [int(x) for x in args.baseline_band.split(",")] if args.baseline_band else []
    merge = model.config.vision_config.spatial_merge_size

    min_pixels = args.min_pixels if args.min_pixels is not None else args.max_pixels
    if min_pixels is not None and min_pixels <= 0:
        min_pixels = None

    if args.task not in DATA_LIST:
        raise ValueError(f"unknown task {args.task!r}; choices: {list(DATA_LIST)}")
    fname, subdir, data_type, has_bound = DATA_LIST[args.task]
    json_dir = os.path.join(os.path.expanduser(args.data_root), "json")
    video_dir = os.path.join(os.path.expanduser(args.data_root), "video")
    with open(os.path.join(json_dir, fname)) as fh:
        records = json.load(fh)

    # pick the clip: by video filename if given, else by position in the task's json
    if args.video:
        hits = [r for r in records if r.get("video") == args.video]
        if not hits:
            raise ValueError(f"video {args.video!r} not in {fname} "
                             f"({len(records)} records; e.g. {records[0].get('video')!r})")
        rec = hits[0]
    else:
        if not 0 <= args.index < len(records):
            raise ValueError(f"--index {args.index} out of range for {fname} "
                             f"({len(records)} records)")
        rec = records[args.index]

    path = os.path.join(video_dir, subdir, rec["video"])
    text, letters, gt = build_prompt(rec)
    prompt = make_mvbench_prompt(path, data_type, has_bound, rec, text,
                                 args.max_frames, args.max_pixels, args.fps,
                                 official=args.official_sampling,
                                 num_segments=args.num_segments, min_pixels=min_pixels)
    chat = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    chat += ANSWER_PREFIX
    img_in, vid_in = process_vision_info(prompt)
    inputs = processor(text=[chat], images=img_in, videos=vid_in, return_tensors="pt")
    pixels = frames_as_numpy(vid_in)

    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    vpos = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten()
    grid_thw = inputs["video_grid_thw"].to(device)
    pix = inputs["pixel_values_videos"].to(device)

    T = int(grid_thw[0, 0])
    S = (int(grid_thw[0, 1]) // merge) * (int(grid_thw[0, 2]) // merge)
    if T * S != vpos.numel():
        raise RuntimeError(f"grid mismatch: T*S={T*S} vs {vpos.numel()} video tokens")
    if S < 12:
        raise RuntimeError(f"only {S} tokens per frame -- too few for a tail estimate; "
                           f"raise --max_pixels")

    with torch.no_grad():
        ve = model.get_video_features(pix, grid_thw).pooler_output
        ve = torch.cat(ve, dim=0).to(device)
        base = model.get_input_embeddings()(input_ids).clone()
        base[0, vpos] = ve.to(base.dtype)
    pos = build_full_positions(model, input_ids, vpos, grid_thw, attn)
    dists, L = per_layer_video_attention(model, base, vpos, pos, attn)
    bad = [b for b in band if b >= L]
    if bad:
        raise ValueError(f"--baseline_band {bad} >= n_layers {L}")

    # the model's forced-choice answer, from the same prompt (ANSWER_PREFIX -> option letter)
    with torch.no_grad():
        last = model(inputs_embeds=base, position_ids=pos, attention_mask=attn,
                     use_cache=False).logits[0, -1].float()
    lids = letter_token_ids(processor, letters)
    pred = int(np.argmax([max(last[i].item() for i in ids) if ids else float("-inf")
                          for ids in lids]))

    g = dists.reshape(L, T, S)                       # (layer, frame, within-frame token)
    xi_lt = np.array([[evt_xi(g[li, t], args.k_frac) for t in range(T)] for li in range(L)])
    src = g[band].mean(0) if band else g.mean(0)     # the distribution the scorer ranks on
    xi_f = np.array([evt_xi(src[t], args.k_frac) for t in range(T)])
    share = src.sum(1) / src.sum()

    # ---- console ----
    print(f"\n==== {args.task} | {rec['video']} | {args.model_name.split('/')[-1]} ====")
    print(f"Q: {rec['question']}")
    for i, (Lt, c) in enumerate(zip(letters, rec["candidates"])):
        mark = "  <- ground truth" if i == gt else ""
        mark += "  <- model" if i == pred else ""
        print(f"   ({Lt}) {c}{mark}")
    print(f"model {'CORRECT' if pred == gt else 'WRONG'}")
    if has_bound:
        print(f"note: this task has a temporal bound [{rec['start']:.1f}, {rec['end']:.1f}]s "
              f"and the official sampler draws all frames INSIDE it -- every frame below is "
              f"already in the ground-truth segment.")
    print(f"\nxi within each frame ({T} groups x 2 frames, {S} tokens per frame, "
          f"band {band or 'all layers'}):")
    hdr = f"{'group':>6}{'frames':>10}{'xi':>9}{'share':>9}{'domain':>16}"
    print(hdr); print("-" * len(hdr))
    for t in range(T):
        dom = ("heavy/Frechet" if xi_f[t] > args.domain_eps else
               ("uniform/Weibull" if xi_f[t] < -args.domain_eps else "~exponential"))
        print(f"{t:>6}{f'{2*t}-{2*t+1}':>10}{xi_f[t]:>9.3f}{share[t]:>9.3f}{dom:>16}")
    print(f"\nspread across frames: std_t(xi) = {np.nanstd(xi_f):.3f}, "
          f"range {np.nanmin(xi_f):.3f} .. {np.nanmax(xi_f):.3f}")
    print("one clip -- eyeball only. A frame standing out here is a hypothesis; "
          "framewise_attn_tail_mvbench.py over many clips is what tests it.")

    out_json = os.path.splitext(args.out)[0] + ".json"
    with open(out_json, "w") as fh:
        json.dump({"experiment": "mvbench_single_clip_framewise_xi",
                   "model_name": args.model_name, "task": args.task,
                   "video": rec["video"], "question": rec["question"],
                   "candidates": rec["candidates"], "gt": gt, "pred": pred,
                   "correct": bool(pred == gt), "has_bound": bool(has_bound),
                   "bound": [rec["start"], rec["end"]] if has_bound else None,
                   "sampling": ("official" if args.official_sampling else "fps"),
                   "n_layers": L, "n_frame_groups": T, "tokens_per_frame": S,
                   "baseline_band": band, "k_frac": args.k_frac,
                   "xi_per_frame": xi_f.tolist(), "share_per_frame": share.tolist(),
                   "xi_layer_frame": xi_lt.tolist()}, fh, indent=2)
    print(f"saved -> {out_json}")

    plot(args, rec, letters, gt, pred, has_bound, pixels, xi_f, share, xi_lt, band, T, L)


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def plot(args, rec, letters, gt, pred, has_bound, pixels, xi_f, share, xi_lt, band, T, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BLUE = "#2a78d6"; RED = "#e34948"; GREEN = "#3f9142"
    INK = "#0b0b0b"; INK_SECOND = "#52514e"; MUTED = "#898781"
    GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECOND, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})

    frames = np.arange(T)
    ncol = min(T, args.per_row)
    nrow = int(np.ceil(T / ncol))
    fig = plt.figure(figsize=(3.1 * ncol, 2.9 * nrow + 5.4))
    gs = GridSpec(nrow + 1, ncol, figure=fig,
                  height_ratios=[1.0] * nrow + [1.45], hspace=0.42, wspace=0.12)

    # ---- the frames themselves: each pair drawn together under its single xi ----
    lim = float(np.nanmax(np.abs(xi_f))) or 1.0
    cmap = plt.cm.RdBu_r
    for t in range(T):
        ax = fig.add_subplot(gs[t // ncol, t % ncol])
        pair = pixels[2 * t: 2 * t + 2]
        if len(pair) == 2:                       # side by side, thin divider between them
            h = min(p.shape[0] for p in pair)
            pair = [p[:h] for p in pair]
            sep = np.full((h, 4, 3), 255, dtype=np.uint8)
            img = np.concatenate([pair[0], sep, pair[1]], axis=1)
        else:
            img = pair[0]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        col = cmap(0.5 + 0.5 * float(np.clip(xi_f[t] / lim, -1, 1)))
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(3.5)
        ax.set_title(f"g{t}  (frames {2*t}-{2*t+1})\nxi = {xi_f[t]:.3f}   "
                     f"share = {share[t]:.3f}", loc="left", fontsize=9.5, weight="bold")

    # ---- the prompt, so the "correct frame" can be judged from it ----
    axP = fig.add_subplot(gs[nrow, 0:max(1, ncol // 3)])
    axP.axis("off")
    lines = [f"Q: {rec['question']}", ""]
    for i, (Lt, c) in enumerate(zip(letters, rec["candidates"])):
        tag = ("  <- GT" if i == gt else "") + ("  <- model" if i == pred else "")
        lines += textwrap.wrap(f"({Lt}) {c}{tag}", 46, subsequent_indent="     ")
    lines += ["", f"model {'CORRECT' if pred == gt else 'WRONG'}"]
    if has_bound:
        lines += textwrap.wrap(f"bound [{rec['start']:.1f},{rec['end']:.1f}]s -- official "
                               f"sampler draws every frame INSIDE it", 46)
    axP.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9.5, family="monospace",
             color=INK, linespacing=1.45, transform=axP.transAxes)
    axP.set_title(f"{args.task} — {rec['video']}", loc="left", weight="bold", fontsize=11)

    # ---- xi per frame ----
    axA = fig.add_subplot(gs[nrow, max(1, ncol // 3):max(2, 2 * ncol // 3)])
    axA.axhline(0, color=INK, lw=1.3)
    axA.text(frames.max(), 0, " xi = 0 (exponential)", color=INK, va="bottom",
             ha="right", fontsize=8.5)
    axA.plot(frames, xi_f, "-", color=INK_SECOND, lw=1.4, zorder=2)
    pos = xi_f > 0
    axA.scatter(frames[pos], xi_f[pos], s=34, color=RED, zorder=3)
    axA.scatter(frames[~pos], xi_f[~pos], s=34, color=BLUE, zorder=3)
    ax2 = axA.twinx()
    ax2.bar(frames, share, color=MUTED, alpha=0.25, zorder=0, width=0.7)
    ax2.set_ylabel("share of clip attention", color=MUTED, fontsize=9)
    ax2.tick_params(axis="y", colors=MUTED); ax2.spines["top"].set_visible(False)
    axA.set_zorder(ax2.get_zorder() + 1); axA.patch.set_visible(False)
    axA.set_xlabel("frame group"); axA.set_ylabel("xi within frame")
    axA.set_title(f"xi per frame  (band {min(band)}-{max(band)})" if band else "xi per frame",
                  loc="left", weight="bold", fontsize=11)

    # ---- xi(layer, frame) for this clip ----
    axB = fig.add_subplot(gs[nrow, max(2, 2 * ncol // 3):])
    m = float(np.nanmax(np.abs(xi_lt))) or 1.0
    im = axB.imshow(xi_lt, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-m, vmax=m,
                    extent=[-0.5, T - 0.5, -0.5, L - 0.5])
    if band:
        for y in (min(band) - 0.5, max(band) + 0.5):
            axB.axhline(y, color=INK, lw=1.0, ls="--")
    axB.set_xlabel("frame group"); axB.set_ylabel("LLM layer")
    axB.set_title("xi per (layer, frame)   red = heavy", loc="left", weight="bold", fontsize=11)
    cb = fig.colorbar(im, ax=axB, pad=0.02); cb.set_label("xi", color=INK_SECOND)
    cb.outline.set_edgecolor(MUTED)

    ok = "correct" if pred == gt else "WRONG"
    fig.suptitle(f"{args.task} — {rec['video']} — {args.model_name.split('/')[-1]}, model {ok} "
                 f"— does the frame that answers the question have a different tail?",
                 x=0.01, ha="left", weight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="One MVBench clip: every frame, its within-frame EVT tail index, and "
                    "the prompt, in one figure.")
    p.add_argument("--model_name", default=MODEL_ID)
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--task", required=True, help=f"one MVBench task, e.g. 'Action Sequence'.")
    p.add_argument("--index", type=int, default=0, help="record position in the task's json.")
    p.add_argument("--video", default=None,
                   help="pick by video filename instead of --index.")
    p.add_argument("--max_frames", type=int, default=8, help="upper cap on frames (fps sampling).")
    p.add_argument("--fps", type=float, default=2.0, help="frames-per-second for fps sampling.")
    p.add_argument("--official_sampling", action="store_true",
                   help="reference mvbench.ipynb sampler (fixed --num_segments frames at segment "
                        "midpoints), matching the rest of the MVBench experiments.")
    p.add_argument("--num_segments", type=int, default=16, help="frames for --official_sampling.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="per-frame pixel ceiling. Also sets the tokens per frame, i.e. how many "
                        "order statistics each xi rests on.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="per-frame pixel floor. Default: mirror --max_pixels; <=0 disables.")
    p.add_argument("--baseline_band", default="12,13,14,15,16",
                   help="scorer band used for the per-frame xi ('' = average all layers).")
    p.add_argument("--k_frac", type=float, default=0.10,
                   help="top fraction of order statistics used by the moment estimator, over the "
                        "tokens of ONE frame.")
    p.add_argument("--domain_eps", type=float, default=0.05,
                   help="|xi|<=eps is labelled ~exponential in the console table.")
    p.add_argument("--per_row", type=int, default=8, help="frame groups per row in the figure.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="inspect_frame_tail_mvbench.png")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
