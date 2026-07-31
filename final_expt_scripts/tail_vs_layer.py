"""
tail_vs_layer.py -- tail index (gamma) of visual-token importance vs. layer, for
Qwen2.5-VL or LLaVA-1.5, on EgoSchema or MVBench, on the DECODER or the ENCODER.

Standalone by design: nothing here imports from the rest of the repo. Frame
sampling, prompt building, the importance score and the EVT estimator are all
inlined below, so this file can be copied out and run on its own.

What it measures
----------------
One forward per clip. At every layer l of the chosen stack we build a per-visual-
token importance vector s^(l) in R^M, then fit the EVT tail index gamma of its
upper tail (Dekkers-Einmahl-de Haan moment estimator; --estimator hill for plain
Hill). gamma > 0 => heavy tail => importance concentrates in a few tokens at that
layer (top-k pruning has something to grab). gamma <= 0 => light/bounded tail =>
importance is spread out, and selecting a small elite is not supported by the
distribution.

    --stack decoder    s^(l)_t = mean_q mean_h A^(l,h)_{q,t} * ||v^(l,h)_t||
                       q = the TEXT query positions (--queries), t = visual tokens
                       in the LLM sequence (post-projector; for Qwen, post 2x2
                       merge). This is text->vision attention: how much answer-
                       bearing query mass each visual token draws.

    --stack encoder    s^(l)_t = mean_q mean_h A^(l,h)_{q,t} * ||v^(l,h)_t||
                       q = ALL patch positions. Neither ViT has a text signal, so
                       this is the "mean-incoming" variant: how much the rest of
                       the image attends to this patch. Scored on raw patches
                       (pre-merge, pre-projector) -- a different token population
                       from the decoder, so the two curves are NOT interchangeable.

--score attn drops the ||v|| factor on either stack. The value-norm is what makes
the score "debiased": attention sinks receive huge A but carry almost no value
mass, so multiplying by ||v|| demotes them without any threshold or detection.

Backbones
---------
Inferred from --model_name.

    qwen   Qwen/Qwen2.5-VL-{3B,7B}-Instruct -- native video. --max_pixels sets
           tokens/frame; 16 frames at 200704 px is 2048 decoder visual tokens.
    llava  llava-hf/llava-1.5-7b-hf -- an IMAGE model, so frames go in as N
           separate <image> placeholders. Every frame costs exactly 576 tokens
           (336x336 CLIP grid; --max_pixels/--min_pixels do NOTHING here) and the
           LM tops out at 4096 positions, so only ~6 frames fit. It was never
           trained on multi-image input; treat its decoder curve as "what does a
           single-image model do when handed a filmstrip", not a video baseline.

Data layout
-----------
    <data_root>/json/<task json>        MVBench's own files, or egoschema.json
    <data_root>/video/<subdir>/<video>  clips (EgoSchema: subdir is empty)

EgoSchema json is built by make_egoschema_json.py. MVBench and EgoSchema live
under different roots, so --tasks should not mix them in one run.

Run
---
    # decoder, Qwen, EgoSchema
    python tail_vs_layer.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --num_segments 16 \
        --max_pixels 200704 --max_samples 100 --out tail_dec_ego_qwen.json

    # encoder, same clips
    python tail_vs_layer.py --data_root ~/Experiments/EgoSchema --tasks EgoSchema \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --stack encoder \
        --num_segments 16 --max_pixels 200704 --max_samples 100 \
        --out tail_enc_ego_qwen.json

    # MVBench, a few tasks, LLaVA-1.5
    python tail_vs_layer.py --data_root ~/Experiments/MVBench \
        --tasks "Action Sequence" "Scene Transition" \
        --model_name llava-hf/llava-1.5-7b-hf --num_segments 6 \
        --max_samples 40 --out tail_dec_mvb_llava.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision.*")

SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question."
)
# Forcing the answer with an assistant-turn prefix makes the LAST position the slot the
# option letter is read from, so --queries last/post are anchored on the decision token.
ANSWER_PREFIX = "Best option:("

# LLaVA-1.5's v1 conversation template, hardcoded rather than taken from the hub's
# chat_template so the prompt cannot drift between checkpoint revisions.
LLAVA_TEMPLATE = "USER: {images}{question} ASSISTANT: "

# Per-backbone frame default: Qwen decodes video natively, LLaVA-1.5 spends 576 tokens
# per frame against a 4096-position LM, so 6 frames (3456 tokens) is the practical cap.
DEFAULT_SEGMENTS = {"qwen": 16, "llava": 6}

# task -> (json file, video subdir under <data_root>/video, data_type, has_temporal_bound)
DATA_LIST = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
    # EgoSchema (long-form egocentric, 500-question Subset): built by make_egoschema_json.py.
    # subdir="" -> path = <data_root>/video/<uuid>.mp4 (symlink <root>/video -> videos/videos).
    "EgoSchema": ("egoschema.json", "", "video", False),
}
MVBENCH_TASKS = [t for t in DATA_LIST if t != "EgoSchema"]


# --------------------------------------------------------------------------- #
# 1. EVT tail index
# --------------------------------------------------------------------------- #
def moment_tail_index(x: torch.Tensor, k_frac: float = 0.10) -> float:
    """Dekkers-Einmahl-de Haan moment estimator of the EVT index gamma.

    Sign-aware, unlike Hill: it can return gamma <= 0 to say the upper tail is
    light/bounded. Its first term M1 is exactly the Hill estimator. NaN when
    there are too few positive samples to estimate."""
    x = x.flatten().float()
    x = x[x > 0].sort().values
    n = x.numel()
    if n < 12:
        return float("nan")
    k = min(max(10, int(k_frac * n)), n - 1)
    logs = torch.log(x[n - k:]) - torch.log(x[n - k - 1])
    M1 = logs.mean()
    M2 = (logs ** 2).mean()
    return (M1 + 1.0 - 0.5 / (1.0 - M1 ** 2 / M2)).item()


def hill_tail_index(x: torch.Tensor, k_frac: float = 0.10) -> float:
    """Hill estimator gamma = (1/k) sum_i log( x_(n-i+1) / x_(n-k) ). Always >= 0,
    so it can never signal a non-heavy tail -- that is why moment is the default."""
    x = x.flatten().float()
    x = x[x > 0].sort().values
    n = x.numel()
    if n < 12:
        return float("nan")
    k = min(max(10, int(k_frac * n)), n - 1)
    logs = torch.log(x[n - k:]) - torch.log(x[n - k - 1])
    return logs.mean().item()


# --------------------------------------------------------------------------- #
# 2. Clips -> model inputs
# --------------------------------------------------------------------------- #
def segment_midpoints(bound, fps, max_frame, num_segments, first_idx=0):
    """The reference MVBench sampler: the midpoint index of each of num_segments
    equal temporal segments within [start, end] seconds, or the whole clip."""
    start, end = bound if bound else (-1e5, 1e5)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + seg / 2 + np.round(seg * i)) for i in range(num_segments)])


def sample_frames(path, data_type, bound, num_segments):
    """num_segments PIL frames. Video clips decode with decord at their average
    fps; the frame-folder task (tvqa) indexes 1-based <idx>.jpg names at fps 3."""
    if data_type == "frame":
        names = sorted(os.listdir(path))
        idxs = segment_midpoints(bound, 3.0, len(names), num_segments, first_idx=1)
        return [Image.open(os.path.join(path, f"{i:05d}.jpg")).convert("RGB") for i in idxs]
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idxs = segment_midpoints(bound, float(vr.get_avg_fps()), len(vr) - 1, num_segments)
    return [Image.fromarray(f) for f in vr.get_batch(idxs).asnumpy()]


def iter_clips(args):
    """(task, record, path, data_type, bound) for every record of every --task."""
    root = os.path.expanduser(args.data_root)
    for task in args.tasks:
        fname, subdir, data_type, has_bound = DATA_LIST[task]
        json_path = os.path.join(root, "json", fname)
        if not os.path.isfile(json_path):
            print(f"[skip task] {task!r}: no {json_path}")
            continue
        with open(json_path) as fh:
            records = json.load(fh)
        if args.max_samples:
            records = records[: args.max_samples]
        for rec in records:
            bound = (rec["start"], rec["end"]) if has_bound else None
            yield task, rec, os.path.join(root, "video", subdir, rec["video"]), data_type, bound


def build_question(record: dict) -> str:
    """MVBench-style user turn: system preamble, question, lettered option block."""
    letters = [chr(ord("A") + i) for i in range(len(record["candidates"]))]
    opts = "".join(f"({L}) {c}\n" for L, c in zip(letters, record["candidates"]))
    return (f"{SYSTEM_PROMPT}\nQuestion: {record['question']}\nOptions:\n{opts.rstrip()}"
            "\nOnly give the best option.")


def build_inputs(processor, frames, record, args, device, dtype):
    """Processor inputs for one clip, with the answer-forcing assistant prefix appended."""
    question = build_question(record)

    if args.backbone == "llava":
        # One <image> placeholder per frame; the processor expands each to 576 tokens.
        # No pixel budget exists here -- LLaVA-1.5 always resizes to 336x336.
        prompt = LLAVA_TEMPLATE.format(images="<image>\n" * len(frames), question=question)
        inputs = processor(text=[prompt + ANSWER_PREFIX], images=frames, return_tensors="pt")
    else:
        from qwen_vl_utils import process_vision_info
        vid = {"type": "video", "video": frames}
        if args.max_pixels is not None:
            vid["max_pixels"] = args.max_pixels
        if args.min_pixels is not None:
            vid["min_pixels"] = args.min_pixels
        messages = [{"role": "user",
                     "content": [vid, {"type": "text", "text": question}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The pixel budget lives on the message dict, so the frames MUST go through
        # process_vision_info -- handing raw PIL frames to the processor would silently
        # ignore max_pixels/min_pixels (and skip the pad-to-even-frames Qwen needs).
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[chat + ANSWER_PREFIX], images=img_in, videos=vid_in,
                           return_tensors="pt")

    out = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k.startswith("pixel_values"):
            out[k] = v.to(device=device, dtype=dtype)     # pixels must match model dtype
        else:
            out[k] = v.to(device)
    return out


# --------------------------------------------------------------------------- #
# 3. Model plumbing
# --------------------------------------------------------------------------- #
def load_model(args, dtype):
    """(model, processor, visual_token_id) for whichever backbone --model_name names.

    Both stacks load eager: every capture below reads the attention weight tensor,
    which only the eager path ever materializes (sdpa/flash never build it)."""
    from transformers import AutoProcessor

    if args.backbone == "llava":
        from transformers import LlavaForConditionalGeneration as ModelCls
        visual_token = "<image>"
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        visual_token = "<|video_pad|>"

    model = ModelCls.from_pretrained(args.model_name, torch_dtype=dtype,
                                     device_map="auto", attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_name)
    return model, processor, processor.tokenizer.convert_tokens_to_ids(visual_token)


def text_model_of(model):
    """The decoder stack. Both wrappers nest it identically: <Model>ForConditional
    Generation.model.language_model (Qwen2_5_VLTextModel / LlamaModel)."""
    base = model.model if hasattr(model, "model") and hasattr(model.model, "language_model") else model
    return base.language_model


def vision_tower_of(model):
    """The vision tower, across the layout that moved it into an inner wrapper
    (model.model.visual / .vision_tower) and the older flat one (model.visual)."""
    attr = "visual" if hasattr(model, "visual") or hasattr(getattr(model, "model", None), "visual") \
        else "vision_tower"
    base = model.model if hasattr(model, "model") and hasattr(model.model, attr) else model
    return getattr(base, attr)


def encoder_attn_modules(model, backbone):
    """The vision tower's per-layer attention modules, in depth order."""
    tower = vision_tower_of(model)
    if backbone == "llava":
        # Newer transformers puts .encoder straight on CLIPVisionModel; older versions
        # nest a CLIPVisionTransformer under .vision_model first.
        tower = getattr(tower, "vision_model", tower)
        return [layer.self_attn for layer in tower.encoder.layers]
    return [block.attn for block in tower.blocks]


def context_limit(model):
    """The LM's position budget -- LLaVA-1.5's 4096 is low enough that a few extra
    frames silently overrun it, so clips are checked against this before the forward."""
    cfg = getattr(model.config, "text_config", model.config)
    return getattr(cfg, "max_position_embeddings", None)


def repeat_kv(v: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, kv_heads, S, hd) -> (B, kv_heads*n_rep, S, hd), matching GQA broadcasting."""
    if n_rep == 1:
        return v
    b, kv, s, hd = v.shape
    return v[:, :, None].expand(b, kv, n_rep, s, hd).reshape(b, kv * n_rep, s, hd)


# --------------------------------------------------------------------------- #
# 4. Decoder capture: forward hooks on each self_attn (peak memory = ONE layer)
# --------------------------------------------------------------------------- #
def attach_decoder_capture(model, store: dict, ctx: dict, use_value_norm: bool):
    """One forward hook per decoder layer's self_attn.

    Under eager the attention module returns its (1,H,Sq,Sk) weight tensor whether
    or not output_attentions was requested, so we read it here, reduce it to an
    (M,) importance vector immediately, and let the tensor die with the hook --
    never asking the model to accumulate all L of them. `ctx` carries the current
    clip's masks; `store` collects layer -> [scores]."""
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    "self_attn returned no attention weights -- load the model with "
                    "attn_implementation='eager' (sdpa/flash never materialize them).")
            # Slice the text-query rows BEFORE upcasting: the full (H,Sq,Sk) tensor is
            # hundreds of MB at these sequence lengths, and we only ever want a few rows.
            recv = output[1][0][:, ctx["text_q"], :].float().mean(dim=1)   # (H, Sk)

            if use_value_norm:
                # This module's input is already post-input_layernorm (the decoder layer
                # normalizes before calling self_attn), so v_proj applies directly.
                h = kwargs.get("hidden_states", args[0] if args else None)
                v_flat = module.v_proj(h)                       # (1, S, kv_heads*hd)
                hd = module.head_dim
                v = v_flat.view(1, -1, v_flat.shape[-1] // hd, hd).transpose(1, 2)
                v = repeat_kv(v, module.num_key_value_groups)   # GQA -> query heads
                recv = recv * v[0].float().norm(dim=-1)         # (H, Sk)

            store[layer_idx] = [recv.mean(dim=0)[ctx["visual_idx"]].detach().cpu()]
            return output
        return hook

    for i, layer in enumerate(text_model_of(model).layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True))
    return lambda: [h.remove() for h in handles]


# --------------------------------------------------------------------------- #
# 5. Encoder capture: mean-incoming score, per backbone
# --------------------------------------------------------------------------- #
def _incoming_score(A, value, use_value_norm):
    """(.., H, Sq, Sk) attention + (.., H, Sk, hd) values -> flat (Sk*,) scores:
    mean over queries, value-norm weighted, mean over heads."""
    recv = A.float().mean(dim=-2)                            # (.., H, Sk)
    if use_value_norm:
        recv = recv * value.float().norm(dim=-1)             # (.., H, Sk)
    return recv.mean(dim=-2).flatten().detach().cpu()        # mean over heads


def attach_encoder_capture(model, backbone, store: dict, use_value_norm: bool):
    """Layer -> [scores] for the vision tower. Returns an uninstall callable.

    The two towers need different mechanisms. CLIP's attention returns
    (attn_output, attn_weights) like the decoder's, so a plain forward hook works.
    Qwen2_5_VLVisionAttention does NOT: its forward keeps only attn_output and
    discards the weights, so the sole place they exist is the shared module-level
    eager_attention_forward -- which the TEXT decoder calls too, hence the
    instance filter. It is called once per cu_seqlens chunk (each window, or each
    frame-group on full-attention layers), so scores are pooled across chunks;
    gamma is permutation-invariant over tokens, so chunk order never matters."""
    if backbone == "llava":
        handles = []

        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
                    raise RuntimeError(
                        "CLIP attention returned no weights -- load with "
                        "attn_implementation='eager' (sdpa/flash never materialize them).")
                A = output[1]                                # (N, H, S, S)
                v = None
                if use_value_norm:
                    h = kwargs.get("hidden_states", args[0] if args else None)
                    n, s, _ = h.shape
                    v = module.v_proj(h).view(n, s, -1, module.head_dim).transpose(1, 2)
                # Drop the CLS column: LLaVA feeds only the 576 patch tokens to the LM,
                # so scoring CLS would put a token the decoder never sees in the tail.
                score = _incoming_score(A[..., 1:], v[..., 1:, :] if v is not None else None,
                                        use_value_norm)
                store[layer_idx] = [score]
                return output
            return hook

        for i, attn in enumerate(encoder_attn_modules(model, backbone)):
            handles.append(attn.register_forward_hook(make_hook(i), with_kwargs=True))
        return lambda: [h.remove() for h in handles]

    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m
    index_of = {id(attn): i for i, attn in enumerate(encoder_attn_modules(model, backbone))}
    orig = m.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out = orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        layer_idx = index_of.get(id(module))
        if layer_idx is not None:                # a vision block, not the text decoder
            store.setdefault(layer_idx, []).append(
                _incoming_score(out[1], value, use_value_norm))
        return out

    m.eager_attention_forward = wrapped
    return lambda: setattr(m, "eager_attention_forward", orig)


# --------------------------------------------------------------------------- #
# 6. One clip -> gamma per layer
# --------------------------------------------------------------------------- #
def visual_query_masks(ids, visual_token_id, queries):
    """(visual_idx, text_q) for the decoder score: which positions are scored, and
    which positions' attention rows are averaged to score them."""
    visual_idx = (ids == visual_token_id).nonzero(as_tuple=False).flatten()
    S = ids.numel()
    is_visual = torch.zeros(S, dtype=torch.bool, device=ids.device)
    is_visual[visual_idx] = True

    if queries == "last":
        text_q = torch.zeros(S, dtype=torch.bool, device=ids.device)
        text_q[-1] = True
    elif queries == "post":                      # non-visual positions AFTER the visual block
        text_q = ~is_visual
        text_q[: int(visual_idx[-1]) + 1] = False
    else:                                        # "all" non-visual positions
        text_q = ~is_visual
    return visual_idx, text_q


@torch.no_grad()
def gammas_for_clip(model, inputs, visual_token_id, args, store, ctx, n_layers, max_positions):
    """gamma at every layer of the chosen stack. Returns (list[float], n_scored)."""
    store.clear()

    if args.stack == "encoder":
        # Vision tower only -- the LLM contributes nothing to this score, and skipping
        # it avoids an O(S^2) eager decoder pass per clip.
        if args.backbone == "llava":
            vision_tower_of(model)(inputs["pixel_values"])
        else:
            model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])
    else:
        ids = inputs["input_ids"][0]
        S = ids.numel()
        if max_positions is not None and S > max_positions:
            raise RuntimeError(f"sequence is {S} tokens but the LM holds {max_positions} -- "
                               f"lower --num_segments")
        visual_idx, text_q = visual_query_masks(ids, visual_token_id, args.queries)
        if visual_idx.numel() < 50:
            raise RuntimeError(f"too few visual tokens ({visual_idx.numel()} < 50)")
        ctx["visual_idx"], ctx["text_q"] = visual_idx, text_q
        model(**inputs, use_cache=False)          # hooks do all the work; logits ignored

    missing = [l for l in range(n_layers) if l not in store]
    if missing:
        raise RuntimeError(f"no attention captured for layer(s) {missing[:5]} -- "
                           f"is attn_implementation='eager' in effect?")

    estimator = moment_tail_index if args.estimator == "moment" else hill_tail_index
    scores = [torch.cat(store[l]) for l in range(n_layers)]
    return [estimator(s, args.k_frac) for s in scores], int(scores[0].numel())


# --------------------------------------------------------------------------- #
# 7. Dataset sweep
# --------------------------------------------------------------------------- #
def main(args):
    if args.backbone == "auto":
        args.backbone = "llava" if "llava" in args.model_name.lower() else "qwen"
    if args.num_segments is None:
        args.num_segments = DEFAULT_SEGMENTS[args.backbone]

    if args.tasks == ["all"]:
        args.tasks = list(DATA_LIST)
    elif args.tasks == ["mvbench"]:
        args.tasks = MVBENCH_TASKS
    unknown = [t for t in args.tasks if t not in DATA_LIST]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choices: {list(DATA_LIST)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[model] {args.model_name}  backbone={args.backbone}  stack={args.stack}  "
          f"dtype={args.dtype}  device={device}")
    model, processor, visual_token_id = load_model(args, dtype)
    max_positions = context_limit(model)

    if args.backbone == "llava":
        # 576 tokens per frame is fixed by the 336x336 CLIP grid, so the pixel knobs are
        # inert here -- say so rather than let them look like they took effect.
        if args.max_pixels is not None or args.min_pixels is not None:
            print("[warn] --max_pixels/--min_pixels are ignored on llava (fixed 576 tokens/frame).")
        args.max_pixels = args.min_pixels = None
    else:
        # Pin the frame resolution: with min_pixels left at the qwen_vl_utils default, only
        # the CAP is set and each clip lands wherever its native resolution falls, so M (the
        # sample gamma is estimated from) varies clip to clip. Tying min to max makes every
        # frame exactly max_pixels -> constant tokens/frame, so the curve is comparable across
        # clips. --min_pixels 0 opts out and restores the library default.
        if args.min_pixels is None:
            args.min_pixels = args.max_pixels
        if args.min_pixels is not None and args.min_pixels <= 0:
            args.min_pixels = None

    store, ctx = {}, {}
    if args.stack == "encoder":
        n_layers = len(encoder_attn_modules(model, args.backbone))
        uninstall = attach_encoder_capture(model, args.backbone, store,
                                           use_value_norm=(args.score == "debiased"))
    else:
        n_layers = len(text_model_of(model).layers)
        uninstall = attach_decoder_capture(model, store, ctx,
                                           use_value_norm=(args.score == "debiased"))
    print(f"[model] {n_layers} {args.stack} layers, {max_positions} LM positions")
    print(f"[data] tasks={args.tasks}  {args.num_segments} frames/clip")

    curves, tasks_seen, per_sample, skipped = [], [], [], []
    try:
        for task, rec, path, data_type, bound in tqdm(list(iter_clips(args)), unit="clip"):
            exists = os.path.isdir(path) if data_type == "frame" else os.path.isfile(path)
            if not exists:
                skipped.append({"task": task, "video": rec["video"], "reason": "missing file"})
                continue
            try:
                frames = sample_frames(path, data_type, bound, args.num_segments)
                inputs = build_inputs(processor, frames, rec, args, device, dtype)
                gammas, n_scored = gammas_for_clip(model, inputs, visual_token_id, args,
                                                   store, ctx, n_layers, max_positions)
            except Exception as e:
                skipped.append({"task": task, "video": rec["video"],
                                "reason": f"{type(e).__name__}: {e}"})
                tqdm.write(f"skip [{task}] {rec['video']}: {type(e).__name__}: {e}")
                continue

            curves.append(gammas)
            tasks_seen.append(task)
            per_sample.append({"task": task, "question_idx": rec.get("question_idx"),
                               "video": rec["video"], "n_scored_tokens": n_scored,
                               "gammas": gammas})
            del inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        uninstall()

    if not curves:
        print("no usable clips -- check --data_root layout (json/ and video/).")
        return

    G = np.array(curves, dtype=float)                       # (n_clips, n_layers)
    mean, std = np.nanmean(G, axis=0), np.nanstd(G, axis=0)
    valid = np.where(np.isfinite(mean))[0]
    L_star = int(valid[np.argmax(mean[valid])]) if valid.size else None
    per_clip_lstar = [int(np.nanargmax(g)) for g in G if np.isfinite(g).any()]

    tasks_arr = np.array(tasks_seen)
    per_task = {t: {int(l): float(v) for l, v in
                    enumerate(np.nanmean(G[tasks_arr == t], axis=0))}
                for t in sorted(set(tasks_seen))}

    out = {"experiment": "tail_vs_layer", "stack": args.stack, "backbone": args.backbone,
           "model_name": args.model_name, "data_root": args.data_root, "tasks": args.tasks,
           "num_segments": args.num_segments, "max_pixels": args.max_pixels,
           "min_pixels": args.min_pixels, "score": args.score,
           "queries": args.queries if args.stack == "decoder" else "all-patches",
           "estimator": args.estimator, "k_frac": args.k_frac,
           "n_clips": len(curves), "n_layers": n_layers,
           "per_task_seen": {t: int((tasks_arr == t).sum()) for t in sorted(set(tasks_seen))},
           "gamma_mean_by_layer": {int(l): float(mean[l]) for l in range(n_layers)},
           "gamma_std_by_layer": {int(l): float(std[l]) for l in range(n_layers)},
           "gamma_mean_by_task": per_task,
           "L_star_pooled": L_star,
           "L_star_histogram": {int(l): per_clip_lstar.count(l) for l in sorted(set(per_clip_lstar))},
           "skipped": skipped}

    print(f"\n==== {args.stack} tail index vs. layer ({len(curves)} clips) ====")
    print(f"{args.backbone}: {args.model_name}, {args.num_segments} frames, "
          f"{len(set(tasks_seen))} task(s)")
    print(f"score={args.score}  estimator={args.estimator}  k_frac={args.k_frac}"
          + (f"  queries={args.queries}" if args.stack == "decoder" else ""))
    lo, hi = float(np.nanmin(mean)), float(np.nanmax(mean))
    for l in range(n_layers):
        mark = "  <- L*" if l == L_star else ""
        if not np.isfinite(mean[l]):                 # a layer with too few positive scores
            print(f"  layer {l:>3}: gamma = NaN{mark}")
            continue
        bar = "#" * int(round(40 * (mean[l] - lo) / max(1e-9, hi - lo)))
        print(f"  layer {l:>3}: gamma = {mean[l]:+.4f} +/- {std[l]:.4f}  {bar}{mark}")
    if L_star is None:
        print("\nno layer produced a usable tail-index estimate.")
    else:
        print(f"\nheaviest-tailed layer pooled over clips: L* = {L_star} (gamma = {mean[L_star]:+.4f})")
    print(f"per-clip L* histogram: {out['L_star_histogram']}")
    if len(per_task) > 1:
        print("\nper-task L*:")
        for t, curve in per_task.items():
            finite = {l: g for l, g in curve.items() if np.isfinite(g)}
            best = max(finite, key=finite.get) if finite else None
            print(f"  {t:<28} n={out['per_task_seen'][t]:<5} L*={best}"
                  + (f"  gamma={finite[best]:+.4f}" if best is not None else ""))
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s)")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    per_sample_out = args.per_sample_out or (os.path.splitext(args.out)[0] + "_per_sample.json")
    with open(per_sample_out, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"wrote {per_sample_out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.arange(n_layers)
        plt.figure(figsize=(7, 4))
        plt.plot(xs, mean, marker="o", ms=3, color="#1f77b4", label="mean gamma")
        plt.fill_between(xs, mean - std, mean + std, color="#1f77b4", alpha=0.15, label="+/- 1 sd")
        plt.axhline(0.0, color="gray", lw=0.8, ls=":")
        if L_star is not None:
            plt.axvline(L_star, color="crimson", ls="--", lw=1, label=f"L* = {L_star}")
        plt.xlabel(f"{args.stack} layer")
        plt.ylabel("tail index gamma")
        plt.title(f"{os.path.basename(args.model_name)} {args.stack} "
                  f"({len(curves)} clips, {args.num_segments}f, {args.score})")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"saved plot -> {args.plot}")


def parse_args():
    p = argparse.ArgumentParser(description="Tail index of visual-token importance vs. layer, "
                                            "encoder or decoder, on MVBench / EgoSchema.")
    p.add_argument("--data_root", required=True, help="Dir holding json/ and video/.")
    p.add_argument("--tasks", nargs="+", default=["EgoSchema"],
                   help="task names, or 'mvbench' (all 20 MVBench tasks) / 'all' (+ EgoSchema). "
                        "MVBench and EgoSchema live under different roots -- don't mix in one run.")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--backbone", choices=["auto", "qwen", "llava"], default="auto",
                   help="auto infers from --model_name ('llava' in the name -> llava).")
    p.add_argument("--stack", choices=["decoder", "encoder"], default="decoder",
                   help="decoder = text->vision attention in the LLM; encoder = mean-incoming "
                        "patch attention in the vision tower. Different token populations; the "
                        "two curves are not comparable point-for-point.")
    p.add_argument("--num_segments", type=int, default=None,
                   help=f"frames sampled per clip. Default per backbone: {DEFAULT_SEGMENTS}.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="qwen only: per-frame pixel cap, e.g. 200704. LLaVA-1.5 is fixed at "
                        "576 tokens/frame and ignores this.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="qwen only: floor on per-frame pixels. Defaults to --max_pixels, which "
                        "pins every frame to exactly that size (constant tokens/frame); 0 opts out.")
    p.add_argument("--max_samples", type=int, default=None, help="cap on records PER TASK.")
    p.add_argument("--score", choices=["debiased", "attn"], default="debiased",
                   help="debiased = attention x value-norm (demotes sinks); attn = raw attention.")
    p.add_argument("--queries", choices=["post", "all", "last"], default="post",
                   help="decoder only: text query rows averaged over. post = non-visual positions "
                        "after the visual block, all = every non-visual position, last = the "
                        "answer slot only. The encoder always averages over all patch queries.")
    p.add_argument("--estimator", choices=["moment", "hill"], default="moment")
    p.add_argument("--k_frac", type=float, default=0.10, help="upper-tail fraction for the estimator.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", default="tail_vs_layer.json")
    p.add_argument("--per_sample_out", default="", help="default: <--out stem>_per_sample.json")
    p.add_argument("--plot", default=None, help="optional PNG of the gamma-vs-layer curve.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
