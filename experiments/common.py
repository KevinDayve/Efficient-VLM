"""Shared infrastructure for the Phase 0 validation experiments.

Everything here is *training-free*. The pieces are deliberately small and
composable so that experiments A-D all share one model load, one MC evaluator,
and one attention-masking primitive.

Design choices worth knowing before you read on
------------------------------------------------
* **Eager attention is required.** We both *read* attention weights (for the
  teacher/oracle scores) and *write* an additive attention bias (for layer
  knockout and top-k token keeping). Flash/SDPA do not expose the additive mask
  we manipulate, so the model is always loaded with
  ``attn_implementation="eager"``.
* **Single forward pass for MC.** We build the prompt with the answer options
  and ``add_generation_prompt=True`` and read the next-token logits at the final
  position, restricted to the option-letter tokens. No autoregressive decoding,
  so query length == key length == sequence length in every forward pass. That
  keeps the attention-mask surgery simple.
* **"language->video attention"** means attention whose *query* row is a text
  token positioned *after* the video block (rows before the video cannot attend
  to it under the causal mask anyway) and whose *key* column is a video token.
* **Dropping a token == masking it as a key everywhere.** For the oracle ceiling
  (Exp B) we never re-plumb the model's position ids/embeddings. Instead we add
  ``-inf`` to the additive attention mask at the dropped video *key* columns for
  every query at every layer. The token still occupies a position id but
  contributes nothing downstream -- which is exactly the information ceiling we
  want to measure. Wall-clock savings are a separate, later measurement.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

try:  # qwen-vl-utils is the official helper for packing video frames
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover - surfaced clearly at runtime
    process_vision_info = None


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model / processor
# --------------------------------------------------------------------------- #
def load_model_and_processor(
    model_name: str,
    fp16: bool = True,
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """Load a frozen Qwen2.5-VL model + processor in eval mode.

    ``attn_implementation="eager"`` is mandatory for these experiments; see the
    module docstring. We do not change ``requires_grad`` because nothing here
    calls ``.backward()``, but the model is put in ``eval()`` so dropout/LN are
    deterministic.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Return the language-model decoder layers (works across HF layouts)."""
    base = getattr(model, "model", model)
    # Newer Qwen2.5-VL nests the text stack under ``model.language_model``.
    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        return base.language_model.layers
    return base.layers


def video_token_id(processor) -> int:
    """Token id Qwen2.5-VL uses as the per-patch video placeholder."""
    return processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")


# --------------------------------------------------------------------------- #
# Dataset adapter (NExT-QA multiple choice)
# --------------------------------------------------------------------------- #
@dataclass
class MCSample:
    video_id: str
    video_path: str
    question: str
    options: List[str]
    answer_idx: int
    qid: str = ""
    qtype: str = ""


# Known field layouts for the various NExT-QA mirrors on the Hub. The first one
# whose keys are all present wins. If none match we raise with the real keys so
# the user can add their schema in one line.
_OPTION_KEY_SETS = [
    ["a0", "a1", "a2", "a3", "a4"],
    ["option_0", "option_1", "option_2", "option_3", "option_4"],
    ["choice_0", "choice_1", "choice_2", "choice_3", "choice_4"],
]
_QUESTION_KEYS = ["question", "Question", "q"]
_ANSWER_KEYS = ["answer", "answer_idx", "label", "correct", "gt"]
_VIDEO_KEYS = ["video", "video_id", "videoId", "vid", "video_name"]


def _first_present(row: dict, keys: Sequence[str]):
    for k in keys:
        if k in row and row[k] is not None:
            return k
    return None


def _coerce_answer_idx(value, options: List[str]) -> int:
    """Answer may be an int index or the literal option string."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        if len(s) == 1 and s.upper() in "ABCDE":
            return "ABCDE".index(s.upper())
        if s in options:
            return options.index(s)
    raise ValueError(f"Cannot interpret answer value {value!r}")


def load_nextqa_dev(
    dataset_name: str,
    split: str,
    video_root: str,
    max_pairs: Optional[int] = 500,
    video_ext: str = "mp4",
    seed: int = 42,
) -> List[MCSample]:
    """Load a small NExT-QA multiple-choice dev slice as ``MCSample`` objects.

    ``max_pairs`` keeps the slice small enough to iterate in hours (the plan asks
    for ~300-500 pairs). Set to ``None`` to take everything.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if len(ds) == 0:
        raise ValueError(f"{dataset_name}:{split} is empty")

    row0 = ds[0]
    opt_keys = next((ks for ks in _OPTION_KEY_SETS if all(k in row0 for k in ks)), None)
    q_key = _first_present(row0, _QUESTION_KEYS)
    a_key = _first_present(row0, _ANSWER_KEYS)
    v_key = _first_present(row0, _VIDEO_KEYS)
    if not (opt_keys and q_key and a_key and v_key):
        raise KeyError(
            "Could not map NExT-QA fields. Available keys: "
            f"{list(row0.keys())}. Edit experiments/common.py "
            "(_OPTION_KEY_SETS / _QUESTION_KEYS / _ANSWER_KEYS / _VIDEO_KEYS)."
        )

    indices = list(range(len(ds)))
    if max_pairs is not None and max_pairs < len(ds):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = sorted(indices[:max_pairs])

    samples: List[MCSample] = []
    for i in indices:
        row = ds[i]
        options = [str(row[k]) for k in opt_keys]
        try:
            ans = _coerce_answer_idx(row[a_key], options)
        except ValueError:
            continue
        vid = str(row[v_key])
        path = vid if vid.endswith((".mp4", ".avi", ".mkv")) else f"{video_root}/{vid}.{video_ext}"
        samples.append(
            MCSample(
                video_id=vid,
                video_path=path,
                question=str(row[q_key]),
                options=options,
                answer_idx=ans,
                qid=str(row.get("qid", row.get("question_id", i))),
                qtype=str(row.get("type", row.get("qtype", ""))),
            )
        )
    return samples


# --------------------------------------------------------------------------- #
# Prompt building + MC evaluation
# --------------------------------------------------------------------------- #
LETTERS = "ABCDE"


@dataclass
class PreparedInputs:
    """Tokenized model inputs plus the metadata the experiments need."""

    inputs: dict                      # kwargs ready for model(**inputs)
    video_positions: torch.Tensor     # 1-D long tensor of video-token indices
    lang_positions: torch.Tensor      # 1-D long tensor of post-video text indices
    seq_len: int
    n_video: int
    video_grid_thw: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None


def _options_block(options: List[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def build_mc_messages(sample: MCSample, max_frames: int) -> List[dict]:
    prompt = (
        f"{sample.question}\n{_options_block(sample.options)}\n"
        "Answer with the letter of the correct option."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": sample.video_path, "nframes": max_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs(model, processor, sample: MCSample, max_frames: int) -> PreparedInputs:
    if process_vision_info is None:
        raise ImportError("qwen-vl-utils is required (pip install qwen-vl-utils).")

    messages = build_mc_messages(sample, max_frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    enc = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    device = next(model.parameters()).device
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

    input_ids = enc["input_ids"][0]
    vid_id = video_token_id(processor)
    is_video = input_ids == vid_id
    video_positions = torch.nonzero(is_video, as_tuple=False).flatten()
    if video_positions.numel() == 0:
        raise RuntimeError("No video tokens found in the prompt.")
    first_video = int(video_positions[0].item())
    # language queries: text tokens after the video block (causal mask makes
    # earlier rows' attention to video structurally zero, so we exclude them).
    all_idx = torch.arange(input_ids.numel(), device=input_ids.device)
    lang_positions = all_idx[(~is_video) & (all_idx > first_video)]

    return PreparedInputs(
        inputs=enc,
        video_positions=video_positions,
        lang_positions=lang_positions,
        seq_len=int(input_ids.numel()),
        n_video=int(video_positions.numel()),
        video_grid_thw=enc.get("video_grid_thw"),
        pixel_values=enc.get("pixel_values_videos"),
    )


def _letter_token_ids(processor, n_options: int) -> List[List[int]]:
    """Candidate token ids for each answer letter (with and without a leading
    space) so we are robust to how the tokenizer splits the first output token."""
    tok = processor.tokenizer
    out = []
    for i in range(n_options):
        letter = LETTERS[i]
        cands = set()
        for form in (letter, f" {letter}"):
            ids = tok.encode(form, add_special_tokens=False)
            if ids:
                cands.add(ids[0])
        out.append(sorted(cands))
    return out


@dataclass
class MCResult:
    pred_idx: int
    correct: bool
    gold_logit: float                 # logit of the gold answer letter
    option_logits: List[float]        # per-option (max over letter variants)


def mc_evaluate(
    model,
    processor,
    prepared: PreparedInputs,
    answer_idx: int,
    n_options: int = 5,
    output_attentions: bool = False,
):
    """One forward pass -> multiple-choice prediction from final-position logits.

    Returns ``(MCResult, outputs)``. ``outputs.attentions`` is populated only
    when ``output_attentions=True`` (needed by the teacher/oracle path).
    """
    with torch.no_grad():
        outputs = model(**prepared.inputs, output_attentions=output_attentions, use_cache=False)

    last_logits = outputs.logits[0, -1, :]  # (vocab,)
    letter_ids = _letter_token_ids(processor, n_options)
    option_logits = [
        max(last_logits[i].item() for i in ids) if ids else float("-inf")
        for ids in letter_ids
    ]
    pred_idx = int(np.argmax(option_logits))
    res = MCResult(
        pred_idx=pred_idx,
        correct=(pred_idx == answer_idx),
        gold_logit=option_logits[answer_idx],
        option_logits=option_logits,
    )
    return res, outputs


# --------------------------------------------------------------------------- #
# Language -> video attention (teacher / oracle scores)
# --------------------------------------------------------------------------- #
def language_to_video_scores(
    outputs,
    prepared: PreparedInputs,
    layers: Sequence[int],
    normalize: bool = True,
) -> torch.Tensor:
    """Aggregate attention from language rows to video columns over ``layers``.

    ``outputs.attentions`` is a tuple of ``(B, heads, q, k)`` tensors, one per
    decoder layer. We average over heads and over the language query rows, then
    over the requested layers, giving one score per video token.
    """
    if outputs.attentions is None:
        raise RuntimeError(
            "outputs.attentions is None -- call mc_evaluate(..., output_attentions=True) "
            "and load the model with attn_implementation='eager'."
        )
    lang = prepared.lang_positions
    vid = prepared.video_positions
    per_layer = []
    for L in layers:
        attn = outputs.attentions[L][0]                 # (heads, q, k)
        block = attn[:, lang][:, :, vid]                # (heads, |lang|, |vid|)
        per_layer.append(block.mean(dim=0).mean(dim=0))  # (|vid|,)
    scores = torch.stack(per_layer, dim=0).mean(dim=0).float()
    if normalize:
        lo = scores.min()
        scores = (scores - lo) / (scores.max() - lo + 1e-8)
    return scores  # (n_video,)


# --------------------------------------------------------------------------- #
# Attention-mask surgery: block specific key columns
# --------------------------------------------------------------------------- #
def _find_attention_mask(args, kwargs):
    """Locate the additive 4-D attention mask among a decoder layer's inputs."""
    if "attention_mask" in kwargs and torch.is_tensor(kwargs["attention_mask"]):
        return "attention_mask", None, kwargs["attention_mask"]
    for i, a in enumerate(args):
        if torch.is_tensor(a) and a.dim() == 4:
            return None, i, a
    return None, None, None


def _make_block_hook(key_positions: torch.Tensor, query_positions: Optional[torch.Tensor]):
    """forward_pre_hook (with_kwargs) that adds -inf to the additive mask at the
    [query_positions, key_positions] entries, blocking those keys pre-softmax."""

    def hook(module, args, kwargs):
        kw_key, arg_idx, mask = _find_attention_mask(args, kwargs)
        if mask is None or mask.dim() != 4:
            # No usable additive mask (e.g. flash path). Fail loud rather than
            # silently running an un-knocked-out forward pass.
            raise RuntimeError(
                "Expected a 4-D additive attention mask on the decoder layer. "
                "Ensure attn_implementation='eager'."
            )
        neg = torch.finfo(mask.dtype).min
        mask = mask.clone()
        kpos = key_positions.to(mask.device)
        if query_positions is None:
            mask[:, :, :, kpos] = neg
        else:
            qpos = query_positions.to(mask.device)
            # advanced indexing over the (q, k) plane
            mask[:, :, qpos.unsqueeze(1), kpos.unsqueeze(0)] = neg
        if kw_key is not None:
            kwargs[kw_key] = mask
        else:
            args = list(args)
            args[arg_idx] = mask
            args = tuple(args)
        return args, kwargs

    return hook


@contextmanager
def block_keys(
    model,
    key_positions: torch.Tensor,
    query_positions: Optional[torch.Tensor] = None,
    layers: Optional[Sequence[int]] = None,
):
    """Context manager that blocks attention to ``key_positions``.

    * ``query_positions=None`` blocks the keys for *every* query row -- this is
      "drop these tokens" (used by the oracle ceiling, Exp B).
    * Passing ``query_positions`` blocks only those rows -- this is the
      language->video knockout (Exp A) when query=language, key=video.
    * ``layers=None`` applies to all decoder layers; otherwise just the listed
      ones (Exp A's sliding window).
    """
    decoder_layers = get_decoder_layers(model)
    target = range(len(decoder_layers)) if layers is None else layers
    handles = []
    if key_positions.numel() > 0:
        hook = _make_block_hook(key_positions, query_positions)
        for L in target:
            handles.append(
                decoder_layers[L].register_forward_pre_hook(hook, with_kwargs=True)
            )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Token-selection strategies (operate on n_video tokens -> kept indices)
# --------------------------------------------------------------------------- #
def k_from_rho(n_video: int, rho: float) -> int:
    return max(1, int(round(n_video * rho)))


def select_topk_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k highest-scoring tokens (sorted ascending)."""
    idx = torch.topk(scores, k=min(k, scores.numel()), dim=-1).indices
    return torch.sort(idx).values


def select_uniform(n_video: int, k: int, device=None) -> torch.Tensor:
    """k evenly spaced indices across the token axis."""
    k = min(k, n_video)
    idx = torch.linspace(0, n_video - 1, steps=k).round().long().unique()
    return idx.to(device) if device is not None else idx


def select_l2norm(features: torch.Tensor, k: int, largest: bool = False) -> torch.Tensor:
    """Top-k (``largest=True``) or bottom-k by L2 norm of the ViT feature.

    Phase 1's norm-asymmetry experiment uses ``largest=False`` (keep the
    high-norm tokens, i.e. discard the low-norm ones). Exposed here so Exp B can
    offer a norm baseline too.
    """
    norms = features.float().norm(dim=-1)            # (n_video,)
    idx = torch.topk(norms, k=min(k, norms.numel()), largest=largest).indices
    return torch.sort(idx).values


def select_kitoke(features: torch.Tensor, k: int) -> torch.Tensor:
    """Best-effort KiToke-style "key information token" selector.

    NOTE: this is an *approximation* of KiToke for a matched-retention baseline,
    not the authors' reference implementation. The intuition we encode is
    "keep the tokens that are least redundant with the rest": score each token by
    its distance to the mean feature (1 - cosine similarity to the centroid) and
    keep the top-k most distinctive. Replace with the official code before
    publishing a head-to-head number.
    """
    f = features.float()
    centroid = f.mean(dim=0, keepdim=True)
    f_n = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
    c_n = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-8)
    distinctiveness = 1.0 - (f_n * c_n).sum(dim=-1)   # (n_video,)
    idx = torch.topk(distinctiveness, k=min(k, f.shape[0])).indices
    return torch.sort(idx).values


def dropped_positions(
    video_positions: torch.Tensor, kept_local_idx: torch.Tensor
) -> torch.Tensor:
    """Map kept *local* video indices (0..n_video-1) to the dropped *absolute*
    sequence positions, ready for ``block_keys``."""
    n = video_positions.numel()
    keep_mask = torch.zeros(n, dtype=torch.bool, device=video_positions.device)
    keep_mask[kept_local_idx.to(video_positions.device)] = True
    return video_positions[~keep_mask]


# --------------------------------------------------------------------------- #
# Visual features (for norm / KiToke / scorer inputs)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_video_features(model, prepared: PreparedInputs) -> torch.Tensor:
    """Run the frozen ViT to get per-patch embeddings, shape (n_video, D)."""
    if prepared.pixel_values is None or prepared.video_grid_thw is None:
        raise RuntimeError("Prepared inputs are missing pixel_values_videos / video_grid_thw.")
    base = getattr(model, "model", model)
    visual = base.visual if hasattr(base, "visual") else model.visual
    feats = visual(prepared.pixel_values, grid_thw=prepared.video_grid_thw)
    return feats  # (n_video, D)


# --------------------------------------------------------------------------- #
# Small metric helpers
# --------------------------------------------------------------------------- #
def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation without a scipy dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties shared), matching scipy.rankdata."""
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
