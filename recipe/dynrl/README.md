# DynRL

This recipe aims to dynamically scale up tensor parallelism during the rollout phase to reduce long-tail latency.

This recipe is under development. 

## Change of vLLM

This recipe leverages vLLM's sleep and wake-up functions to switch between engines with different tp_size. Consequently, modifications to vLLM are required to properly adapt its sleep mode for this purpose.

For simple modification, you can apply the [vllm-v0_8_5.patch](patch/vllm-v0_8_5.patch) provided in this recipe.

