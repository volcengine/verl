# Low-Precision Rollout and Quantization-Aware Training

Last updated: 10/15/2026

In RL training, rollout usually dominates the step time, and for large MoE models it is bound by memory capacity and bandwidth. Running rollout in low precision:

- shrinks weight memory, leaving more room for KV cache and higher concurrency;
- reduces memory traffic during decode, improving throughput;
- uses FP8/FP4 Tensor Cores on Hopper/Blackwell to speed up prefill.

The cost is a numerical gap between the high-precision trainer and the low-precision rollout (training-inference mismatch). It makes importance sampling and off-policy correction less accurate and, in bad cases, destabilizes training. Quantization-Aware Training (QAT) narrows this gap by training the model against the same quantization the rollout uses.

## Background: model weight formats

HuggingFace checkpoints come in two flavors:

1. **High precision**: BF16 weights, no `quantization_config` in `config.json`.
2. **Low precision**: `config.json` carries a `quantization_config` describing how each layer is quantized. Models often use different formats for linear layers (attention, dense MLP, shared experts) and MoE (routed experts).

| Model | Architecture | `quant_method` | Linear | MoE (routed experts) | Not quantized |
|---|---|---|---|---|---|
| [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json) | 92 layers, 512 experts, top-10 | none | BF16 | BF16 | - |
| [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) | 78 layers, 256 experts, top-8, first 3 dense | none | BF16 | BF16 | - |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) | same as GLM-5.2 | `fp8` | FP8 E4M3, 128x128 block, FP32 scale (`weight_scale_inv`), dynamic activation | same as linear | router `mlp.gate`, `indexers_proj`, `embed_tokens`, `lm_head`, MTP `eh_proj`, norms (`modules_to_not_convert`) |
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) | 93 layers, 896 experts, first layer dense | `compressed-tensors` | BF16 | MXFP4 W4A16: FP4 E2M1, 1x32 group, UE8M0 scale (`weight_packed` + `weight_scale`) | `self_attn`, `shared_experts`, dense `mlp`, `lm_head`, `vision_tower`, `mm_projector` (`ignore` regex) |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) | 40 layers, 384 experts, top-6 | `fp8` | FP8 E4M3, 32x32 block (`wo_a` may be 128x128), UE8M0 scale (`scale_fmt: ue8m0`) | FP4 E2M1, 1x32 group, UE8M0 scale (`expert_dtype: fp4`), same layout as MXFP4 | router `ffn.gate`, indexer `wk`/`weights_proj`, `compressor` |

- GLM-5.3 uses the standard HF FineGrainedFP8 format (same as DeepSeek-V3), which both vLLM and SGLang parse.
- Kimi-K3 uses compressed-tensors: a `Linear`-targeted config group plus an `ignore` list that excludes non-expert layers yields "linear BF16 + MoE MXFP4".
- DeepSeek-V4.1-Flash's `fp8` config has private fields (`scale_fmt`, `expert_dtype`) and needs dedicated engine support.

The trainer always trains in high precision; low-precision checkpoints are dequantized before training. The rollout uses the model's own `quantization_config` if it has one; a BF16 model can run rollout in BF16 or, for higher throughput, in low precision through `rollout.quantization`.

## How it works

- **Trainer**: always trains in high precision (BF16). With QAT enabled, the forward pass uses fake-quantized weights (quantize then dequantize) and gradients flow through a straight-through estimator.
- **Rollout**: the inference engine starts with dummy weights and an overridden `quantization_config`, so it allocates quantized parameters as if it were loading a quantized checkpoint (e.g. FP8 `weight` + `weight_scale`). On every weight sync, the trainer quantizes its BF16 weights to that checkpoint format, and the engine's weight loader handles fusion, tensor-parallel sharding and kernel-specific layouts.

Your HuggingFace checkpoint stays in BF16; no offline quantization or calibration step is needed. The rollout format is described in the open [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) format, which both vLLM and SGLang understand and which can express different schemes for linear and MoE layers.

## Support matrix

The trainer must be Megatron (FSDP is not yet supported for this feature; see [NVFP4 QAT](nvfp4_qat.md) for FSDP). The rollout engine can be vLLM or SGLang; TRT-LLM raises an error if `rollout.quantization` is set. The engine columns list the compressed-tensors scheme each engine implements.

| Precision | Preset | QAT mode | GPU | vLLM Linear | vLLM MoE | SGLang Linear | SGLang MoE | Status in verl |
|---|---|---|---|---|---|---|---|---|
| FP8 block (E4M3, 128x128 weight blocks, dynamic 1x128 activations) | `FP8_BLOCK` | `fp8` | Hopper / Blackwell | `W8A8Fp8` | `W8A8Fp8MoE` | `W8A8Fp8` | `W8A8Fp8MoE` | ✅ Hopper (vLLM, SGLang); Blackwell not yet verified |
| MXFP8 | - | - | Blackwell | `W8A8Mxfp8` | `W8A8Mxfp8MoE` | ❌ (only whole-model `quant_method: mxfp8`) | ❌ | Planned |
| MXFP4 | - | - | Blackwell | `W4A4Mxfp4` | `W4A4Mxfp4MoE` | ❌ | `Mxfp4MoEMethod` | Planned |
| NVFP4 | - | - | Blackwell | W4A4 / W4A16 | W4A4 / W4A16 | W4A4 only | W4A4 | Planned |

MXFP8, MXFP4 and NVFP4 need Blackwell (SM100). On Hopper, vLLM falls back to Marlin weight-only kernels for these formats, whose numerics differ from Blackwell W8A8 / W4A4; verl does not support that path.

## Configuration

### Rollout quantization

```yaml
actor_rollout_ref:
  rollout:
    quantization:
      linear: FP8_BLOCK   # attention, dense MLP, shared experts
      moe: FP8_BLOCK      # routed experts
      ignore: []          # extra layers kept in BF16: exact names or "re:<regex>"
      config_file: null   # path to a compressed-tensors quantization_config JSON
```

- When `linear`, `moe` and `config_file` are all `null` (the default), the rollout runs in the model's native precision.
- `linear` and `moe` can be set independently. Setting one of them to `null` keeps those layers in BF16, e.g. `linear: null, moe: FP8_BLOCK` quantizes only the routed experts.
- The following layers are always kept in BF16: `lm_head`, embeddings, MoE routers (`mlp.gate`, `shared_expert_gate`), vision towers and MTP layers. Layers whose shapes are not divisible by the block size (e.g. Qwen3.5 `linear_attn.in_proj_a/b`) are skipped automatically.
- `ignore` adds more layers on top of these, using HuggingFace layer names.

From the command line:

```bash
actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK
```

### Custom `quantization_config`

For full control, point `config_file` at a compressed-tensors `quantization_config`. It can be either the `quantization_config` object itself or a HuggingFace `config.json` that contains one. `config_file` cannot be combined with `linear`, `moe` or `ignore`, and it is used as-is (the default ignore list is not added). Every config group must numerically match a supported preset.

Example equivalent to `linear: FP8_BLOCK, moe: FP8_BLOCK` for Qwen3.5:

```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": {"num_bits": 8, "type": "float", "symmetric": true,
                  "strategy": "block", "block_structure": [128, 128], "dynamic": false},
      "input_activations": {"num_bits": 8, "type": "float", "symmetric": true,
                            "strategy": "group", "group_size": 128, "dynamic": true},
      "format": "float-quantized"
    }
  },
  "ignore": [
    "re:.*lm_head$", "re:.*embed_tokens$", "re:.*visual\\..*", "re:^mtp\\..*",
    "re:.*mlp\\.gate$", "re:.*shared_expert_gate$",
    "re:.*linear_attn\\.in_proj_a$", "re:.*linear_attn\\.in_proj_b$"
  ]
}
```

### QAT (Megatron)

QAT is optional. Enable it on the actor and pick the mode that matches the rollout preset:

```yaml
actor_rollout_ref:
  actor:
    megatron:
      qat:
        enable: true
        mode: fp8   # must match rollout.quantization (FP8_BLOCK -> fp8)
```

QAT requires `rollout.quantization` to be set, and verl checks at startup that the QAT mode matches the rollout preset. Megatron QAT uses [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) for fake quantization.

### Engine notes

- **vLLM**: `VLLM_USE_DEEP_GEMM_E8M0` defaults to `0` when rollout quantization is enabled, so block scales stay in FP32.
- **SGLang**: the first launch of a quantized MoE model can take several minutes while Triton compiles FP8 MoE kernels during CUDA graph capture. Later launches use the cache.

## Example

Qwen3.5-35B-A3B on 8 GPUs with FP8 block rollout and FP8 QAT:

```bash
python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-35B-A3B \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.qat.enable=True \
    actor_rollout_ref.actor.megatron.qat.mode=fp8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
    actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK \
    ...
```

Use `actor_rollout_ref.rollout.name=sglang` for SGLang. For complete scripts, see the [low-precision recipe directory](https://github.com/verl-project/verl-recipe/tree/main/low_precision).

## Measuring the mismatch

`scripts/rollout_logprob_diff.py` runs one rollout, recomputes the log-probs with the Megatron actor, and reports the gap (`kl`, `k3_kl`, absolute log-prob difference, etc.). Use it to check a configuration before a long run:

```bash
python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \
    --quant-linear FP8_BLOCK --quant-moe FP8_BLOCK --qat-mode fp8
```

Pass `--engine sglang` for SGLang, and any extra Hydra overrides as trailing arguments. Reference `k3_kl` on Qwen3.5-35B-A3B (EP=8, rollout TP=2, H-series GPUs):

| Rollout | vLLM | SGLang |
|---|---|---|
| BF16 | ~0.0014 | - |
| FP8_BLOCK | 0.0039 | 0.0036 |
| FP8_BLOCK + FP8 QAT | 0.0033 | 0.0033 |

## Limitations

- Only the `FP8_BLOCK` preset on Hopper is supported today.
- Checkpoints that already ship a `quantization_config` (e.g. FP8 or compressed-tensors checkpoints) are not supported as a training target together with low-precision rollout yet; use a BF16 checkpoint.
- Megatron QAT and the rollout use separate ignore lists. ModelOpt may fake-quantize a few layers that the rollout keeps in BF16 (e.g. Qwen3.5 `in_proj_a/b`); this slightly overestimates the quantization error for those layers but does not affect correctness.
