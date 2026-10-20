# Optimal Token Baseline (OTB)

OTB uses a token-wise optimal variance-reduction baseline derived from the per-prompt sample group. See `tir_optimal_token_baseline` variant for a TIR-specific form.

## Canonical Scripts

| Script                               | Infer | Train | Platform |
|--------------------------------------|-------|-------|----------|
| `run_qwen3_8b_fsdp.sh`          | vLLM  | FSDP  | NVIDIA   |
| `run_qwen2_5_7b_megatron.sh`    | vLLM-Ascend | Megatron | Ascend 910A3 |

Set `ADV_ESTIMATOR=tir_optimal_token_baseline` for the TIR variant.

## Key Flags

- `algorithm.adv_estimator=optimal_token_baseline`
- `actor_rollout_ref.actor.calculate_sum_pi_squared=True`
- `actor_rollout_ref.model.use_fused_kernels=False`

These flags follow the [OTB algorithm guide](../../docs/algo/otb.md).

## Ascend 910A3

Follow the [Ascend installation guide](../../docs/ascend_tutorial/zh/get_start/install_guidance.rst)
and the [data preparation instructions](../../docs/ascend_tutorial/en/model_support/examples/otb_ascend_practice.md#training).
Launch from the repository root:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B \
ADV_ESTIMATOR=optimal_token_baseline \
DATA_ROOT=/path/to/data \
OUTPUT_DIR=/path/to/persistent/output \
bash examples/otb_trainer/run_qwen2_5_7b_megatron.sh
```

The eight-device recipe uses Megatron actor tensor/pipeline parallel sizes 4/2,
rollout tensor parallel size 2, and rollout memory fraction 0.6.

The [practice report](../../docs/ascend_tutorial/en/model_support/examples/otb_ascend_practice.md)
provides the environment, full configuration, 100-step results, training curves,
and original metric records. Final validation accuracy was 91.2813% on GSM8K and
72.0488% on MATH.
