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
- Training: Megatron-Core >= 0.13 and Transformer Engine >= 2.1, on the Megatron-Bridge model path
- Rollout: vLLM 0.29 or SGLang 0.5.20 (the versions verl locks)

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

## Rollout kernels

- **vLLM** serves the preset through compressed-tensors MXFP8. When vLLM 0.29 selects the backends itself, the
  kernels depend on whether FlashInfer is available to its worker processes (`flashinfer-cubin` installed, or
  `nvcc` on their `PATH`): with it, FlashInfer TRT-LLM for the experts and, on B200, FlashInfer CuTe-DSL for
  the dense GEMMs (MXFP8 weights and activations); without it, DeepGEMM for the experts and Marlin for the dense
  GEMMs (MXFP8 weights, BF16 activations). vLLM logs its dense choice (`Using ... for MXFP8 GEMM`) at
  `INFO`; verl starts the workers with `VLLM_LOGGING_LEVEL=WARN`, so add
  `+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO` to see it.
- **SGLang** serves the preset with its own MXFP8 method (`quant_method: mxfp8`), and verl pins its MXFP8 dense
  and MoE kernels to DeepGEMM; `VERL_ROLLOUT_USE_DEEP_GEMM=0` leaves the choice to SGLang.

## Example: Qwen3-8B on GSM8K

[`examples/mxfp8/run_qwen3_8b_megatron.sh`](https://github.com/verl-project/verl/blob/main/examples/mxfp8/run_qwen3_8b_megatron.sh)
runs GRPO on Qwen3-8B with GSM8K, MXFP8 end to end; `PRECISION=bf16` runs the same job in BF16. Its defaults
fit one B200: the whole optimizer stays on the CPU, and parameters are offloaded while the rollout runs.

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
PRECISION=mxfp8 bash examples/mxfp8/run_qwen3_8b_megatron.sh
PRECISION=bf16 bash examples/mxfp8/run_qwen3_8b_megatron.sh
```

### Results

**Configuration**

- One B200; verl's locked environment (vLLM 0.29.0, Transformer Engine 2.16.1, Megatron-Core 0.19.2,
  Megatron-Bridge 0.6.2)
- The script's defaults: 64 prompts × 8 responses per step, prompts and responses up to 1024 and 2048 tokens,
  learning rate 1e-6; 10 steps (`TOTAL_TRAINING_STEPS=10`)
- Validation on the GSM8K test set (1,319 problems, greedy decoding) at steps 0, 5 and 10
- vLLM rollout with FlashInfer available to the workers (`nvcc` on their `PATH`), where vLLM picks FlashInfer
  CuTe-DSL for the MXFP8 dense GEMMs

![Qwen3-8B on GSM8K, BF16 vs MXFP8](FIGURE_URL)
*Blue: BF16, vermillion: MXFP8 end to end. wandb: [BF16](WANDB_BF16_URL), [MXFP8](WANDB_MXFP8_URL)*

Results and observations:

- GSM8K test accuracy at steps 0, 5 and 10 is 0.372, 0.616 and 0.803 with MXFP8, against 0.329, 0.610 and
  0.819 with BF16, and the training reward follows the BF16 curve. Both are single runs of 10 steps, too few
  to resolve the 1.6-point gap at step 10 (21 of 1,319 problems).
- The mismatch between the rollout and the trainer (`rollout_corr/kl`) is 0.0063–0.0071 with MXFP8, against
  0.0006–0.0008 with BF16, and stays flat over the run.
