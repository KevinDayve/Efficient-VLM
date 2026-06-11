import torch
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
from transformers import Qwen2_5VLForConditionalGeneration

class AttentionExtractor(nn.Module):
    def __init__(self, model: Qwen2_5VLForConditionalGeneration, Layers: List[int] = [12, 13, 14, 15, 16], num_video_tokens: int = None,
                 video_start: int = None, query_mask: torch.Tensor = None):
        super().__init__()
        self.model = model
        self.Layers = Layers
        self.num_video_tokens = num_video_tokens
        self.video_start = video_start
        self.query_mask = query_mask
        self._store: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def set_sample(self, input_ids: torch.Tensor, video_token_id: int, special_ids: set):
        """
        Helper function to be called once per sample, before the VLM forward pass.
        """
        ids = input_ids[0]
        video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
        self.video_start = int(video_pos[0])
        self.num_video_tokens = int(video_pos.numel())
        vid_end = self.video_start + self.num_video_tokens
        positions = torch.arange(ids.numel(), device=ids.device)
        is_special = torch.isin(
            ids, torch.tensor(list(special_ids), device=ids.device)
        )
        self.query_mask = (positions >= vid_end) & (~is_special)
    
    def __enter__(self):
        self._register()
        return self
    
    def __exit__(self, *args):
        self._remove_hooks()
    
    # We haave to register our hook function.
    def _get_hook(self, index: int):
        def hook(module, input, output):
            if output[1] is not None:
                self._store[index] = output[1].detach()
        return hook
    
    def _register(self):
        layers = self.model.model.layers
        for index in self.Layers:
            hook = layers[index].self_attn.register_forward_hook(self._get_hook(index))
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._store.clear()

    def get_scores(self, normalise: bool = True) -> Optional[torch.Tensor]:
        if len(self._store) == 0:
            return None
        v0, v1 = self.video_start, self.video_start + self.num_video_tokens
        scores = []
        for index in self.Layers:
            if index in self._store:
                attn = self._store[index] # you'll obtain (B, H, S, S)
                lv = attn[:, :, :, v0:v1]
                lv = lv[:, :, self.query_mask, :]
                scores.append(lv.mean(dim=1).mean(dim=1)) #(B, n_video)
        scores = torch.stack(scores, dim=0).mean(dim=0)
        if not normalise:
            return scores.detach()
        scores = (scores - scores.min(dim=-1, keepdim=True).values) / \
                 (scores.max(dim=-1, keepdim=True).values -
                  scores.min(dim=-1, keepdim=True).values + 1e-8)
        return scores.detach()