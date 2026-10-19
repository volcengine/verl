# Low-Precision Rollout and Quantization-Aware Training

Last updated: 10/16/2026

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
- **Rollout**: the inference engine starts with dummy weights and either the model's native
  `quantization_config` or a `rollout.quantization` override. It allocates parameters as if
  it were loading that low-precision checkpoint (for example FP8 `weight` +
  `weight_scale_inv`).
- **Weight refit export**: the training backend first exports weights under canonical
  HuggingFace names. A trainer-side `WeightQuantizer`, selected from the effective rollout
  `quantization_config`, converts each sync bucket into exactly the checkpoint tensors
  expected by the rollout engine. Standard native FP8 is quantized to the `Fp8Config`
  layout; native and overridden compressed-tensors configs use the same
  compressed-tensors exporter. DeepSeek-V4 is already quantized by Megatron-Bridge, so its
  exporter only normalizes the packed byte dtype. The rollout worker therefore loads
  pre-quantized tensors and is not responsible for quantization.

![weight_quant](https://github.com/verl-project/verl-data/blob/main/images/weight_quantizer_refit.png?raw=true)

This exporter/loader boundary is intended to decouple the two sides. The target architecture
is the Cartesian product of training backends `[Megatron, VeOmni, TorchTitan]` and rollout
engines `[vLLM, SGLang]`: each trainer exports canonical named tensors, each
`WeightQuantizer` defines a checkpoint wire format, and each rollout engine only implements
loading/refit for that format. The support matrix below records the combinations currently
validated; it does not imply that the full target matrix is implemented yet.

For an override-based rollout, the HuggingFace checkpoint stays in BF16; no offline
quantization or calibration step is needed. The override format is described using the open
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors) schema, which can
express different schemes for linear and routed MoE layers.

## Support matrix

The trainer must be Megatron (FSDP is not yet supported for this feature). The rollout engine can be vLLM or SGLang; TRT-LLM raises an error if `rollout.quantization` is set. The engine columns list the compressed-tensors scheme each engine implements.

| Precision | Preset | QAT mode | GPU | vLLM Linear | vLLM MoE | SGLang Linear | SGLang MoE | vLLM status in verl | SGLang status in verl |
|---|---|---|---|---|---|---|---|---|---|
| FP8 block (E4M3, 128x128 weight blocks, dynamic 1x128 activations) | `FP8_BLOCK` | `fp8` | Hopper / Blackwell | `W8A8Fp8` | `W8A8Fp8MoE` | `W8A8Fp8` | `W8A8Fp8MoE` | ✅ Hopper | ✅ Hopper |
| MXFP8 | `MXFP8` | `mxfp8` | Blackwell | `W8A8Mxfp8` | `W8A8Mxfp8MoE` | ❌ (only whole-model `quant_method: mxfp8`) | ❌ | ✅ Blackwell | Blocked by engine support |
| MXFP4 (MoE only) | `MXFP4` | `mxfp4` / `mxfp4_mxfp8` | Blackwell | ❌ | MXFP4×MXFP8<br>`(moe_backend=flashinfer_cutlass)` | ❌ | ❌ | ✅ Blackwell (MoE only) | Blocked by engine support |
| NVFP4 (MoE only) | `NVFP4` | `nvfp4` | Blackwell | ❌ | W4A4<br>`(moe_backend=flashinfer_cutlass)` | ❌ | ❌ | ✅ Blackwell (MoE only) | Blocked by engine support |

MXFP8, MXFP4 and NVFP4 need Blackwell (SM100). On Hopper, vLLM falls back to Marlin weight-only kernels for these formats, whose numerics differ from Blackwell W8A8 / W4A4; verl does not support that path.

MXFP4 and NVFP4 support is scoped to routed experts. The supported combinations are BF16 linear + FP4 MoE or MXFP8 linear + FP4 MoE; FP4 linear QAT, export and refit are out of scope. verl defines its `MXFP4` preset as MXFP4 expert weights with dynamic MXFP8 expert activations (group size 32 and E8M0 scales for both). This is intentionally different from compressed-tensors' `MXFP4` preset, which is W4A4, and from vLLM's native `Mxfp4Config`, whose checkpoint format is nominally W4A16.

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
- `MXFP4` and `NVFP4` are valid only for `moe`. Use `linear: null` for BF16 dense layers or `linear: MXFP8` for MXFP8 dense layers.
- The following layers are always kept in BF16: `lm_head`, embeddings, MoE routers (`mlp.gate`, `shared_expert_gate`), vision towers and MTP layers.
- `ignore` adds more layers on top of these, using HuggingFace layer names. Layers incompatible with the selected scheme are not detected automatically: list them explicitly. For example, MXFP8 on Qwen3.5 requires `["re:.*linear_attn\\.in_proj_[ab]$"]` because these projections are too small for the vLLM kernel.

From the command line:

```bash
actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK
```

For Qwen3.5 MXFP8:

```bash
actor_rollout_ref.rollout.quantization.linear=MXFP8 \
actor_rollout_ref.rollout.quantization.moe=MXFP8 \
'actor_rollout_ref.rollout.quantization.ignore=["re:.*linear_attn\\.in_proj_[ab]$"]'
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
        mode: fp8   # FP8_BLOCK -> fp8, MXFP8 -> mxfp8, MXFP4 -> mxfp4, NVFP4 -> nvfp4
```

QAT requires `rollout.quantization` to be set, and verl checks at startup that the QAT mode matches the rollout preset. Megatron QAT uses [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) for fake quantization. MXFP8 uses ModelOpt's native `MXFP8_DEFAULT_CFG`; FP8_BLOCK uses a verl backend because ModelOpt's `FP8_DEFAULT_CFG` is per-tensor and its 2D-block preset is weight-only. MXFP4 and NVFP4 use compressed-tensors-compatible QDQ backends so fake quantization is bit-exact with the exported checkpoint format. Both FP4 modes apply only to routed experts. Use `mxfp4_mxfp8` when dense linear layers also use MXFP8.
Its `ignore_patterns` use Megatron module names and `fnmatch` syntax. Keep them aligned with the rollout ignore list; for Qwen3.5 MXFP8, add `*linear_attn.in_proj_a` and `*linear_attn.in_proj_b`.

### Engine notes

- **vLLM**: `VLLM_USE_DEEP_GEMM_E8M0` defaults to `0` for `FP8_BLOCK`, so its block scales stay in FP32. MX formats retain the E8M0 path.
- **vLLM MXFP4**: vLLM 0.29's compressed-tensors path otherwise ignores the MXFP8 input scheme and selects W4A4 CUTLASS. verl routes this explicit W4A8(MX) scheme through vLLM's MXFP4 backend oracle with `kMxfp8Dynamic` and defaults to FlashInfer CUTLASS. The TRTLLM MXFP4×MXFP8 backend is not used because it produced a large numerical mismatch on Qwen3.5-35B-A3B.
- **vLLM NVFP4**: verl supports only W4A4 routed experts and forces FlashInfer CUTLASS. W4A16, dense NVFP4 and other MoE backends are rejected.
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
    --quant-linear FP8_BLOCK --quant-moe FP8_BLOCK \
    --quant-ignore 're:.*linear_attn\.in_proj_[ab]$' \
    --qat-mode fp8 \
    --qat-ignore '*linear_attn.in_proj_a' --qat-ignore '*linear_attn.in_proj_b' \
    --r3-router-replay
```

Pass `--engine sglang` for SGLang, and any extra Hydra overrides as trailing arguments. Reference `k3_kl` on Qwen3.5-35B-A3B (EP=8, rollout TP=2):

| GPU | Engine | R3 | Rollout | MoE backend | k3_kl |
|---|---|---|---|---|---|
| Hopper | vLLM | ✅ | BF16 | - | 0.00031 |
| Hopper | SGLang | ✅ | BF16 | - | 0.00031 |
| Hopper | vLLM | ✅ | FP8_BLOCK | - | 0.00225 |
| Hopper | SGLang | ✅ | FP8_BLOCK | - | 0.00231 |
| Hopper | vLLM | ✅ | FP8_BLOCK + FP8 QAT | - | 0.00187 |
| Hopper | SGLang | ✅ | FP8_BLOCK + FP8 QAT | - | 0.00188 |
| Blackwell | vLLM | ❌ | MXFP8 linear + MoE | auto | 0.00148 |
| Blackwell | vLLM | ❌ | MXFP8 linear + MoE + MXFP8 QAT | auto | 0.00315 |
| Blackwell | vLLM | ❌ | MXFP4 MoE | FlashInfer CUTLASS | 0.00858 |
| Blackwell | vLLM | ❌ | MXFP4 MoE + MXFP4 QAT | FlashInfer CUTLASS | 0.00210 |
| Blackwell | vLLM | ✅ | NVFP4 MoE | FlashInfer CUTLASS | 0.00813 |
| Blackwell | vLLM | ✅ | NVFP4 MoE + NVFP4 QAT | FlashInfer CUTLASS | 0.00574 |

## Limitations

- `FP8_BLOCK` is supported on Hopper. `MXFP8` with vLLM is supported on Blackwell; SGLang has no per-layer MXFP8 support.
- Native vLLM `Fp8Config` rollout is supported without a rollout override. Standard
  128x128 block-FP8 models are quantized by the trainer-side exporter; DeepSeek-V4
  keeps the private mixed FP8/FP4 layout already exported by Megatron-Bridge.
- Native compressed-tensors configs are also supported without an override. The
  same trainer-side dispatcher handles native FP8, native compressed-tensors and
  `rollout.quantization` compressed-tensors overrides.
- Megatron QAT and the rollout use separate ignore lists and naming conventions. Users must keep them aligned.
