# transformers patches

Local modifications to the `transformers` library that are **not** vendored into
this repo. Apply them on top of a clean install of the pinned version.

## qwen2_5_vl_token_gating.patch

Adds the learned video-token scorer + pre-LLM token gating to Qwen2.5-VL
(`Qwen2_5_VLTokenScorer`, Pareto/stratified budget selection, `load_token_scorer`
/ `disable_token_gating`, and `_gate_video_tokens`).

- **Base:** transformers `b4b5244c9c7cdb80d0aaafdb8f35244612788532` (v4.56.1 preview)
- **Touches:** `src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py` and the
  generated `modeling_qwen2_5_vl.py`.

### Apply

```bash
# clone transformers at the exact base commit
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout b4b5244c9c7cdb80d0aaafdb8f35244612788532

# apply the patch (run from the transformers/ root)
git apply /path/to/Efficient-VLM/patches/qwen2_5_vl_token_gating.patch

# install editable into your venv
pip install -e .
```

### Regenerate the patch after editing transformers

```bash
cd transformers
git diff > /path/to/Efficient-VLM/patches/qwen2_5_vl_token_gating.patch
```
