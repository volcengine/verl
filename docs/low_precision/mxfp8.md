# MXFP8 RL on Blackwell

Last updated: 10/21/2026

MXFP8 is the OCP microscaling FP8 format: E4M3 elements with one E8M0 scale per 32-element block,
decoded natively by Blackwell tensor cores. This recipe trains with Transformer Engine's MXFP8 recipe and
serves the rollout in MXFP8, with the rollout's weights quantized on the trainer side with the learner's
block rule.

For FP8 on Hopper (the `blockwise` recipe), see [FP8 RL in verl](fp8.md). The rollout quantization
interface shared by all formats, and the vLLM and SGLang specifics, are described in
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Requirements

- Blackwell GPUs (SM100+)
- Megatron-Core >= 0.13 and Transformer Engine >= 2.1, on the Megatron-Bridge model path

## Configuration

```yaml
# MXFP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"                # element format; "hybrid" (e4m3 fwd + e5m2 bwd) also supported
  fp8_recipe: "mxfp8"        # 32-element block scaling

# MXFP8 rollout
actor_rollout_ref.rollout.quantization:
  linear: MXFP8
  moe: MXFP8
```

Notes:

- Weights stay in BF16 (`fp8_param` is not supported); only the GEMM inputs are cast to MXFP8, so
  checkpoints are unchanged.
- With `fp8_recipe: "mxfp8"`, verl pads packed sequences to the 32-token blocks MXFP8 needs.
- QAT and Transformer Engine FP8 training are mutually exclusive; the engine refuses the combination.
- `lm_head`, the token embedding and MoE routers stay in BF16 on the rollout, as in training. Further
  layers can be kept in BF16 with `rollout.quantization.ignore`.
- Given the same BF16 weight, the trainer-side exporter (compressed-tensors) and Transformer Engine pick
  the same E8M0 scale for every block except those whose amax is exactly `448 * 2**k`, where
  compressed-tensors picks the next one up; in such a block only elements below `amax / 14336`
  dequantize differently, by at most about `4.4e-6 * amax`. This compares the weight quantizers only, not
  activations, reduction order or the KV cache.
