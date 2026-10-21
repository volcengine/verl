# MXFP8 RL on Blackwell

Last updated: 10/21/2026

MXFP8 is the OCP microscaling FP8 format: E4M3 elements with one E8M0 scale per 32-element block,
decoded natively by Blackwell tensor cores. This recipe trains with Transformer Engine's MXFP8 recipe and
serves the rollout in MXFP8 on vLLM or SGLang. The rollout's weights are quantized on the trainer side with
the learner's block rule, and three checks report the failures a quantized rollout can otherwise hit
silently.

For FP8 on Hopper (the `blockwise` recipe), see [FP8 RL in verl](fp8.md). The rollout quantization
interface shared by all formats is described in
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Requirements

- Blackwell GPUs (SM100+)
- Training: Megatron-Core >= 0.13 and Transformer Engine >= 2.1, on the Megatron-Bridge model path
- Rollout: vLLM 0.29 or SGLang 0.5.20 (the versions verl locks)

## Configuration

```yaml
# MXFP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"                # element format; "hybrid" (e4m3 fwd + e5m2 bwd) also supported
  fp8_recipe: "mxfp8"        # 32-element block scaling

# MXFP8 rollout (actor_rollout_ref.rollout.name: vllm or sglang)
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

## Rollout on vLLM

vLLM serves the preset through compressed-tensors MXFP8. When vLLM 0.29 selects the backends itself, the
kernels depend on whether FlashInfer is available to its worker processes (`flashinfer-cubin` installed, or
`nvcc` on their `PATH`): with it, FlashInfer CuTe-DSL or CUTLASS for the dense GEMMs and FlashInfer TRT-LLM
for the experts; without it, Marlin for the dense GEMMs (MXFP8 weights, BF16 activations) and DeepGEMM for
the experts. `VLLM_LOGGING_LEVEL=INFO` logs the choice.

MoE layers kept in BF16 through `rollout.quantization.ignore` are supported: they run vLLM's unquantized
MoE method, whose FlashInfer kernels re-lay out the experts at load, and verl re-derives that layout after
every weight sync, as it does for the quantized layers.

## Rollout on SGLang

```yaml
actor_rollout_ref.rollout:
  name: sglang
  quantization:
    linear: MXFP8
    moe: MXFP8
```

SGLang serves the preset with its own MXFP8 method (`quant_method: mxfp8`): verl translates the
configuration, exports the scales under SGLang's names and re-derives SGLang's kernel layouts after every
weight sync (see [qat.md](qat.md)). verl pins SGLang's MXFP8 dense and MoE kernels to DeepGEMM;
`VERL_ROLLOUT_USE_DEEP_GEMM=0` leaves the choice to SGLang.

Two configurations are refused before anything is written:

- MoE layers kept in BF16 (`rollout.quantization.ignore`) whose experts the MoE runner re-laid out at load:
  SGLang's FlashInfer TRT-LLM runner stores BF16 experts in a block layout a weight sync cannot write.
  DeepGEMM keeps them in checkpoint layout.
- The delta checkpoint engine with any quantized rollout (`rollout.quantization` or the checkpoint's own
  `quantization_config`): it applies its payloads without the trainer-side weight quantization.

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for an SGLang rollout: verl starts SGLang
with `enable_memory_saver=True`, which refuses that allocator mode.

## Checks

| Check | Runs | Reports | Switch |
|---|---|---|---|
| Quantized-rollout sentinel | every training step | NaN log-probs, KL above 1.0, every response at the length cap | `VERL_QUANT_SENTINEL` |
| Post-sync self-check | after every weight sync | kernel scale layouts that did not follow the sync | `VERL_MXFP8_REFIT_CHECK` |
| Quantized-layer audit | first weight sync after a training step | layers that training and the rollout quantize differently | `VERL_QUANT_LAYER_AUDIT` |

Each switch takes `warn` (the default: log the finding and keep going), `raise` (stop the run or the sync)
or `0` (off). The sentinel and the audit also apply to FP8 rollouts.

### Quantized-rollout sentinel

When the rollout is quantized, both trainers check each step's metrics:

- `training/rollout_probs_diff_mean` is non-finite while the batch had valid tokens;
- `rollout_corr/kl` is non-finite or above `VERL_QUANT_SENTINEL_KL_MAX` (default 1.0; healthy quantized
  rollouts read about 0.001–0.03);
- every response hits the length cap (`response_length/clip_ratio` close to 1) on
  `VERL_QUANT_SENTINEL_CLIP_STEPS` consecutive steps (default 2; `0` turns this rule off, and it is off
  under `rollout.ignore_eos`, where every response runs to the cap by design).

The rollout counts as quantized when `rollout.quantization` sets `linear`, `moe` or `config_file`, or the
model's own `config.json` (local paths only) carries a `quantization_config`. The sentinel catches gross
failures only: one layer type left unquantized (KL 0.007–0.010) stays within the healthy range.

### Post-sync self-check

After loading, both engines derive kernel-specific copies of the MXFP8 weights and scales (swizzled
scales, shuffled or packed experts). If a sync skips that step, the kernel pairs fresh weights with stale
scales while every shape and dtype still matches. After every sync, verl therefore runs the smallest MXFP8
linear layer and one expert of the smallest MXFP8 MoE layer through their own kernels on a small random
input and compares the output with a BF16 reference on the dequantized canonical weights. Healthy layers
read a relative error of about 0.03 (dense) and 0.05 (experts); a stale layout reads O(1).

- `VERL_MXFP8_REFIT_CHECK_TOL` (default 0.25) and `VERL_MXFP8_REFIT_CHECK_MOE_TOL` set the thresholds.
- On SGLang, the refit loader also reports experts whose scales the sync did not write.
- The expert probe is skipped, with a warning, where its made-up routing cannot send every row to one
  expert at weight 1. On vLLM: expert or data parallelism, non-SiLU gates, and, on monolithic MoE kernels, a
  routed scaling factor, grouped top-k, a score correction bias or custom routing. On SGLang: expert
  parallelism, TP without `reduce_results`, and non-softmax, grouped, biased or custom routing.

What it does not show:

- It samples one dense layer and one expert per sync; a layer outside the sample can still be stale.
- It runs the kernels eagerly. A re-derivation into new storage, instead of the storage a CUDA graph
  captured, passes the probe while graph replays keep reading the old values. That the captured storage is
  updated rests on indirect evidence (the re-derived layouts are copied back in place, and the KL stays in
  its healthy range), not on a graph replay.

### Quantized-layer audit

Training decides implicitly which layers run FP8 (TE linear modules inside `fp8_autocast`); the rollout
decides by name (`rollout.quantization`, read by both the trainer-side exporter and the engine). When
training runs a TE FP8 recipe and the rollout is quantized, verl compares them at the first weight sync
after a training step:

- **Training vs export**: each decoder layer's FP8 trace (TE's FP8 weight workspaces) against the format
  the sync sends for that layer's GEMM projections (attention, MLP, experts, linear-attention and SSM in /
  out projections). Weights the exporter keeps in BF16 by design are skipped. Other decoder-layer weights,
  such as a convolution, are not judged, and the audit reports how many it left out.
- **Export vs engine**: on vLLM, the format of every synced weight against the quant method of the module
  that receives it. On SGLang, the refit loader compares each incoming weight's dtype with the parameter
  that receives it before writing it (switched by `VERL_MXFP8_REFIT_CHECK`).

`VERL_QUANT_LAYER_AUDIT_TRACE=1` logs each step of the audit itself. The training-side trace is per layer,
so a TE module kept in BF16 inside an FP8 layer would go unnoticed (Megatron has no such per-module option
today). With `disable_parameter_transpose_cache=True` TE keeps no FP8 workspace, and the audit warns once
that it cannot run.
