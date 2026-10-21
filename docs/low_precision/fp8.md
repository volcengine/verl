# FP8 RL in verl

Last updated: 10/20/2026

verl supports two FP8 modes for accelerating RL training:

| Mode | Training Precision | Rollout Precision |
|------|-------------------|-------------------|
| **FP8 Rollout Only** | BF16 | FP8 |
| **FP8 End-to-End** | FP8 (Megatron) | FP8 (vLLM) |

> [!TIP]
> For ready-to-run scripts, see the [low-precision recipe directory](https://github.com/verl-project/verl-recipe/tree/main/low_precision).

---

## FP8 Rollout Only

FP8 rollout-only mode keeps training in BF16 and quantizes rollout inference to FP8. This reduces GPU memory during generation and speeds up rollout without affecting training precision.

### Implementation

The rollout engine starts from a compressed-tensors `quantization_config` with dummy weights, and the trainer quantizes every weight sync to that config with the compressed-tensors compressors (`verl/utils/qat/compressed_tensors.py`). On the vLLM side we monkey patch `process_weights_after_loading` of the compressed-tensors FP8 linear/MoE methods so refits are written back into the live parameters (`verl/utils/vllm/vllm_fp8_utils.py`).

### Support Matrix

- FP8 blockwise quantization for rollout (compressed-tensors preset `FP8_BLOCK`)
  - Used in Deepseek, which is 1x128 quantization for activations and 128x128 quantization for model weights
- Dense models and MoE models
- Async rollout interfaces
- vLLM or SGLang rollout (SGLang validated on Hopper only) with the Megatron training backend; TRT-LLM and FSDP are not supported

### Usage

Enable in config file:

```yaml
rollout:
  quantization:
    linear: FP8_BLOCK  # dense linear layers (attention, dense MLP, shared experts)
    moe: FP8_BLOCK     # routed experts
    ignore: []         # extra "re:" patterns kept in BF16
```

Or via command line:

```bash
actor_rollout_ref.rollout.quantization.linear=FP8_BLOCK \
actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK
```

Routers, `lm_head`, embeddings, vision towers and MTP layers stay in BF16 by default, as do layers whose weight shape is not a multiple of the 128x128 block. To provide a complete compressed-tensors `quantization_config` instead, set `actor_rollout_ref.rollout.quantization.config_file=/path/to/quantization_config.json` (exclusive with `linear` / `moe` / `ignore`).

### Experiments and Outcomes

#### Qwen3-8B-Base Dense Model

**Configuration**
- DAPO recipe. AIME24 online validation.
- vLLM(FP8 spmd rollout) + FSDP
  - Note that SPMD rollout has been deprecated, so we removed the FP8 SPMD rollout.
- Prompt batch size 32, n=16.
- Rollout batch size: 32\*3*16
- Train_batch_size & ppo_mini_batch_size 32
- Max response length 20K
- Token-level TIS, C=2
- 8*H100
- vLLM 0.10.0+CUDA 12.6 vs vLLM 0.11.0+CUDA 12.9

**Accuracy**
![Qwen3-8b-base_fp8_acc](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-8b-base_fp8_acc.png?raw=true)
*dark green: BF16, orange: FP8 rollout + token-level TIS, light green: FP8 rollout without TIS*

Results and observations:
- With TIS, FP8 rollout aligns with BF16
- Obvious accuracy drop when TIS is not enabled
- Higher mismatch kl but within acceptable range throughout the training


**Performance**

![Qwen3-8b-base_fp8_rollout_perf](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-8b-base_fp8_rollout_perf.png?raw=true)
*green: BF16, orange: FP8 rollout + CUDA12.6 + DeepGemm, purple: FP8 rollout + CUDA 12.9 + DeepGemm*

Results and observations:
- FP8 rollout leads to around ~12% rollout speedup with CUDA 12.6 + DeepGemm
- When upgrading to CUDA 12.9, speedup can be up to ~18%

#### Qwen3-30B-A3B-Base MoE Model

**Configuration**
- DAPO recipe. AIME24 online validation.
- FP8 async rollout, vLLM+FSDP
- Prompt batch size 32
- Rollout batch size: 32\*3*16
- Train_batch_size & ppo_mini_batch_size 32
- Max response length 20K
- Token-level TIS, C=2
- 2\*8*H100
- vLLM 0.10.0+CUDA 12.6

**Accuracy**
![Qwen3-30b-a3b_fp8_acc](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-30b-a3b_fp8_acc.png?raw=true)
*grey: BF16 + token-level TIS, red: FP8 rollout + token-level TIS*

Results and observations:
- Rollout & training distribution mismatch is in general higher for MoE
- Rollout correction required even for BF16
- FP8 rollout with token-level TIS aligns with BF16


**Performance**

![Qwen3-30b-a3b_fp8_perf](
https://github.com/Agoniii/verl/blob/xueh/fp8_pr_images/docs/advance/images/Qwen3-30b-a3b_fp8_perf.png?raw=true)
*grey: BF16 + token-level TIS, red: FP8 rollout + token-level TIS​*

Results and observations:
- FP8 rollout : over 35% rollout speedup
- Expecting more perf gain with CUDA 12.9

---

## FP8 End-to-End (Training + Rollout)

FP8 E2E applies FP8 to the entire RL pipeline: forward/backward passes via Transformer Engine, FP8 optimizer states, and FP8 rollout inference via vLLM. This maximizes memory savings and throughput.

### Requirements

- **CUDA 12.9+** (required for block-wise FP8 scaling)
- **Transformer Engine** with block-wise FP8 support
- Environment variable: `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`, on **Hopper only**. Do not set it on Blackwell (SM100+): there Transformer Engine runs block-wise scaling through MXFP8 GEMMs, whose E8M0 scales hold only an exponent, so it requires power-of-two block scales and stops at the first FP8 quantization with `Assertion failed: pow2_scale ... requires using power of two scaling factors` when the variable is set (see [#6172](https://github.com/verl-project/verl/issues/6172)). Leaving it unset gives power-of-two scales, the Transformer Engine default.

### Key Configuration

```yaml
# FP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "hybrid"              # FP8 forward + backward; also supports "e4m3"
  fp8_recipe: "blockwise"    # block-wise scaling

# FP8 optimizer
actor_rollout_ref.actor.optim.override_optimizer_config:
  fp8_recipe: "blockwise"

# FP8 rollout inference (vLLM)
actor_rollout_ref.rollout.quantization:
  linear: FP8_BLOCK
  moe: FP8_BLOCK
```

### Support Matrix

- Megatron training backend (via Megatron-Bridge)
- Verified on Qwen3-30B-A3B and Qwen3-8B
- Block-wise FP8 scaling (`fp8_recipe: "blockwise"`)

### Experiments and Results

#### Qwen3-30B-A3B MoE Model

**Configuration**
- DAPO recipe. AIME24 online validation.
- Megatron + Megatron-Bridge, FP8 async rollout with vLLM
- MoE router in BF16 for both vLLM and Megatron-Core
- Prompt batch size 128, n=16
- Max response length 20K
- Token-level TIS, C=2
- 2\*8*H100, CUDA 12.9

![Qwen3-30b-a3b_fp8_e2e](https://github.com/user-attachments/assets/70fb1396-ec73-40d7-9a43-1d48553c0ad9)
*Orange: BF16, Green: FP8 E2E, Red: FP8 rollout + BF16 training*

Results and observations:
- FP8 E2E achieves comparable accuracy to the BF16 baseline, with the two curves closely aligned throughout training.
- The training/inference precision mismatch (measured by KL divergence) follows the ordering: FP8 rollout-only > FP8 E2E > BF16 E2E. This is expected, as FP8 E2E maintains consistent precision across both training and inference, resulting in lower distribution mismatch than the FP8 rollout-only setting where training remains in BF16.

---

## MXFP8 Training (Blackwell)

MXFP8 is the OCP microscaling FP8 format: E4M3 elements with one E8M0 scale per 32-element block,
decoded natively by Blackwell tensor cores. On Hopper, use the `blockwise` recipe above.

### Requirements

- Blackwell GPUs (SM100+)
- Megatron-Core >= 0.13 and Transformer Engine >= 2.1

### Key Configuration

```yaml
# MXFP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"                # element format; "hybrid" (e4m3 fwd + e5m2 bwd) also supported
  fp8_recipe: "mxfp8"        # 32-element block scaling

# MXFP8 rollout (vLLM)
actor_rollout_ref.rollout.quantization:
  linear: MXFP8
  moe: MXFP8
```

Notes:

- Weights stay in BF16 (`fp8_param` is not supported); only the GEMM inputs are cast to MXFP8, so
  checkpoints are unchanged.
- With `fp8_recipe: "mxfp8"`, verl pads packed sequences to the 32-token blocks MXFP8 needs.
- QAT and Transformer Engine FP8 training are mutually exclusive; the engine refuses the combination.
- The rollout's MXFP8 weights are quantized on the trainer side (the compressed-tensors exporter of
  `rollout.quantization`). Given the same BF16 weight, compressed-tensors and Transformer Engine pick the
  same E8M0 scale for every block except those whose amax is exactly `448 * 2**k`, where compressed-tensors
  picks the next one up; in such a block only elements below `amax / 14336` dequantize differently, by at
  most about `4.4e-6 * amax`. This compares the weight quantizers only, not activations, reduction order or
  the KV cache.
- `lm_head`, the token embedding and MoE routers stay in BF16 on the rollout, as in training.

### MXFP8 Rollout on SGLang

```yaml
actor_rollout_ref.rollout:
  name: sglang
  quantization:
    linear: MXFP8
    moe: MXFP8
```

SGLang serves this preset with its own MXFP8 method (`quant_method: mxfp8`): verl translates the
configuration, exports the scales under SGLang's names and re-derives SGLang's kernel layouts after every
weight sync; see [Low-Precision Rollout and Quantization-Aware Training](qat.md). verl pins SGLang's MXFP8
dense and MoE kernels to DeepGEMM; `VERL_ROLLOUT_USE_DEEP_GEMM=0` leaves the choice to SGLang.

Two configurations are refused before anything is written:

- MoE layers kept in BF16 (`rollout.quantization.ignore`) whose experts the MoE runner re-laid out at load:
  SGLang's FlashInfer TRT-LLM runner stores BF16 experts in a block layout a weight sync cannot write.
  DeepGEMM keeps them in checkpoint layout.
- The delta checkpoint engine with any quantized rollout (`rollout.quantization` or the checkpoint's own
  `quantization_config`): it applies its payloads without the trainer-side weight quantization.

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for an SGLang rollout: verl starts SGLang
with `enable_memory_saver=True`, which refuses that allocator mode.

---

## Citation

For more extensive experiments, ablation studies, and analysis on FP8 reinforcement learning, please refer to our technical report:

```bibtex
@article{qiu2026fp8rl,
  title={FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning},
  author={Qiu, Zhaopeng and Yu, Shuang and Zhang, Jingqi and Zhang, Shuai and Huang, Xue and Yang, Jingyi and Lai, Junjie},
  journal={arXiv preprint arXiv:2601.18150},
  year={2026},
  url={https://arxiv.org/abs/2601.18150}
}
```
