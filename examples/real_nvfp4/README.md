# Native NVFP4 training and rollout

This example trains Qwen3-30B-A3B-Base with Megatron/Transformer Engine NVFP4
and runs vLLM rollout in the matching NVFP4 precision. Training parameters stay
BF16; each refit exports the routed experts as NVFP4 on the trainer side
(`rollout.quantization.moe=NVFP4_PER_TOKEN`), and vLLM reloads them into the
storage its CUDA graphs captured.

## Precision and supported configuration

`actor_rollout_ref.actor.megatron.real_nvfp4.enable=True` runs the routed-expert
MLPs in NVFP4 with per-token activation scaling, dequantized backward, and
adaptive 4-over-6 disabled. The `NVFP4_PER_TOKEN` rollout preset uses the same
precision: NVFP4 expert weights and activations quantized per token by
FlashInfer's TRT-LLM MoE kernel. Attention, router, norms, embeddings and the
LM head remain BF16 on both sides.

`num_layers_at_start_in_bf16` and `num_layers_at_end_in_bf16` keep whole decoder
layers in BF16 during training; `rollout.quantization.ignore` must list the same
routed-expert layers, which the trainer checks at startup. This example keeps
the first two and last four of 48 layers in BF16.

The current integration supports the all-MoE `Qwen3MoeForCausalLM` layout,
training PP=1 without virtual pipeline parallelism, and rollout TP/PP/EP=1/1/1
with the colocated `naive` checkpoint engine. It uses BF16 KV cache,
`FULL_DECODE_ONLY` CUDA graphs and no speculative decoding. Mixed dense/MoE
layouts and shared experts are not supported.

## Installation

The runtime uses PyTorch 2.13.0/CUDA 13, Transformer Engine 2.18.0,
Megatron-Core 0.19.2, Megatron-Bridge 0.6.2, vLLM 0.29.0, Transformers 5.12.1
and FlashInfer 0.6.18:

```bash
uv sync --frozen --extra megatron --extra vllm
```

Keep the FlashInfer Python, cubin and CUDA JIT-cache packages aligned. For a
non-Docker launch, use the environment's NCCL library:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
```

Transformer Engine 2.18.0 uses its official PyPI distribution; its PyTorch
extension selects a matching release wheel when available and otherwise builds
against the selected Torch runtime. Use a persistent uv cache to reuse the build.

## Run

Use preprocessed datasets in verl's standard `RLHFDataset` format. Each row
contains a `prompt` chat-message list, a `data_source` identifier and a
`reward_model` mapping with `style: rule` and the answer in `ground_truth`.
The built-in reward dispatcher selects scoring from `data_source`; use
`math_dapo` for DAPO math or an `aime`-prefixed identifier for AIME. Optional
`extra_info` is passed through by the dataset. Dataset preprocessing belongs
outside this example; see the existing [preprocessing scripts](../data_preprocess).

Start Ray on the allocated nodes, then provide shared model and parquet paths:

```bash
MODEL_PATH=/shared/models/Qwen3-30B-A3B-Base \
TRAIN_FILE=/shared/data/train.parquet \
TEST_FILE=/shared/data/validation.parquet \
CKPTS_DIR=/shared/checkpoints/my-run \
PRECISION_MODE=real_nvfp4 EXP_NAME=my-run NNODES=8 \
  bash examples/real_nvfp4/run_qwen3_30b_megatron.sh
```

Use `PRECISION_MODE=bf16` for a control with the same example settings. The
launcher uses verl's standard PPO entry point with the built-in DAPO reward
manager and the default rule-based scorer for each `data_source`. The default
configuration is 8 nodes × 4 GPUs, EP=4, DAPO (GRPO advantages, clip-higher,
token-level loss, dynamic sampling on accuracy and a 512-token overlong
penalty), R3 routing replay and token-level TIS, training on 32 prompts × 16
responses per update. Adam uses learning rate 1e-6 and betas (0.9, 0.999).

Rollout memory utilization defaults to 0.8, with 256 sequences and 32768 batched
tokens. Set `TOTAL_TRAINING_STEPS`, `RESUME_MODE` and checkpoint paths for the
run. Checkpoints include optimizer, scheduler and dataloader progress.
Logging uses verl's standard console and W&B integration.

Transformer Engine reads its NVFP4 settings from `NVTE_*` environment variables
in each worker; `runtime_env.yaml` sets them and the trainer rejects a recipe
that differs. Keep these variables when passing another file via `RUNTIME_ENV`.

R3 uses vLLM's native routing capture inside the fused MoE path. Runtime checks
validate the exported expert tensors, the NVFP4 layer selection and the scale
values after each reload.

## Tests

Run GPU tests in a Blackwell allocation using the prepared runtime:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1 TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1
python -m pytest -q tests/utils/real_nvfp4
python -m pytest -q tests/utils/test_bucketed_weight_transfer.py
```

Coverage includes recipe compatibility, the rollout preset, refit checks, and
bit-exact agreement of the trainer-side export with vLLM's online NVFP4 quantization.
