# FP8 for verl

Last updated: 09/18/2025.

This module is still in development. Currently we support FP8 rollout, using FP8 blockwise scaling (used in Deepseek,
which is 1x128 quantization for activations and 128x128 quantization for model weights).

We monkey patches several vLLM functions to enable FP8 rollout for reinforcement learning.
1. **load_weights**: A custom `load_weights` function to quantize the on-the-fly model weights from a higher-precision format to FP8.
2. **process weights after loading**: Replace `vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading`
function to handle model weights loading after quantization.

**Notes**: 
- Currently, we only support VLLM rollout with Megatron training. SGLang rollout with Megatron training is on the roadmap.
- Only support Batch generate sequences.
- We also support FP8 per tensor quantization, but after preliminary testing, there were issues with the accuracy and it is not recommended to use it.

## Usage

FP8 can be enabled in the config file `verl/trainer/config/ppo_megatron_trainer.yaml`:

```
  rollout:
    quantization: True

    use_block_quant_rollout: True
```

Or it can be enabled by command line:
- `actor_rollout_ref.rollout.quantization=True`
- `actor_rollout_ref.rollout.use_block_quant_rollout=True`

## Plans

- add performance of FP8 rollout
- add accuracy curves of FP8 rollout
- support SGLang rollout with FP8 quantization
- enable FP8 training in megatron