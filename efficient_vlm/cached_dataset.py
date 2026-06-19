"""Dataset over a feature cache written by cache_features.py.

Each item is one pre-extracted sample: the merged visual features (the scorer
input), the raw teacher attention scores (the ListMLE target), and the temporal
layout (``t`` frames, ``n_per_frame`` tokens). Token counts vary per video, so the
collate keeps a *list* of samples rather than stacking into one padded tensor --
ListMLE runs per sample anyway, and the upcoming contrastive term pools each sample
to a single vector before batching, so ragged storage costs nothing.
"""
import os
import json

import torch
from torch.utils.data import Dataset


def load_meta(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "meta.json")) as fh:
        return json.load(fh)


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.meta = load_meta(cache_dir)
        with open(os.path.join(cache_dir, "manifest.jsonl")) as fh:
            self.index = [json.loads(line) for line in fh if line.strip()]
        if not self.index:
            raise ValueError(f"Empty cache manifest in {cache_dir}")

    def __len__(self):
        return len(self.index)

    @property
    def feature_dim(self) -> int:
        """D of the cached vision features (the scorer's input_dim)."""
        return int(self[0]["vision_feats"].shape[-1])

    @property
    def has_lang(self) -> bool:
        """Whether the cache carries language features (needed for the contrastive
        term). Reads the meta flag; defaults to False for schema-1 caches."""
        return bool(self.meta.get("has_lang", False))

    def __getitem__(self, i: int) -> dict:
        rec = self.index[i]
        blob = torch.load(os.path.join(self.cache_dir, rec["path"]), map_location="cpu")
        lang = blob.get("lang_feat")
        return {
            # detach(): cached vision feats can carry requires_grad (the VLM merger
            # ran outside no_grad at cache time), which is a non-leaf tensor that
            # can't be pickled across DataLoader workers ("Cowardly refusing to
            # serialize non-leaf tensor which requires_grad"). The scorer's frozen
            # input must not require grad anyway. Cast to fp32 to match train.py,
            # which casts patch_embeds.float() before scoring.
            "vision_feats": blob["vision_feats"].detach().float(),   # (n_video, D)
            "teacher_raw": blob["teacher_raw"].detach().float(),      # (n_video,)
            "lang_feat": lang.detach().float() if lang is not None else None,  # (n_layers, D) or None
            "t": int(blob["t"]),
            "n_per_frame": int(blob["n_per_frame"]),
            "video": rec.get("video"),
            "question": rec.get("question"),
        }


def ragged_collate(batch):
    """Collate that preserves per-sample variable token counts -- returns the list
    of sample dicts unchanged. The training loop iterates it for the per-sample
    ListMLE and gathers the pooled vectors for the batched contrastive loss."""
    return batch
