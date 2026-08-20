# Training runs on Unsloth on a rented GPU, not locally on MLX

The obvious choice for a personal experiment on a 24 GB Mac is local `mlx-lm` LoRA — no
upload, no VM, no cost. We are not doing that. As of 2026-08-20 `mlx-lm` 0.31.3 carries
four open defects that hit this exact configuration: LoRA target discovery misses Gemma 4's
`per_layer_model_projection` so the per-layer-embedding path is never trained (#1363); NaN
gradients with `mask_prompt=True` on chat-format data, which is the completion-only loss we
need (#1355); `mlx_lm.fuse` silently drops the adapter on a quantised base (#1172); and the
trainer has no gradient clipping or non-finite guard (#1658/#1659). The default
`--max-seq-length` of 2048 would also silently truncate our p99 rows at ~4,095 tokens.

Unsloth publishes a working Gemma 4 E2B path — `unsloth/gemma-4-E2B-it`, LoRA in 8-10 GB,
free Colab notebooks, `train_on_completions` for response-only loss — with the Gemma 4
traps already fixed (notably that `use_cache=False` corrupts attention on E2B/E4B).
Unsloth's own MLX support is "coming", not shipped.

The training data is synthetic fake-bank data, so there is no reason it cannot leave the
machine. Corroborating evidence: a prior prototype committed to local MLX in its spec and
its readout records that the run actually happened on Colab.

This becomes wrong if no GPU is reachable. Then go local and carry four mitigations: pin
`--max-seq-length 4096`, use `completion` rather than `chat` dataset format, confirm the
loss is not NaN at iteration 1, and never merge the adapter.
