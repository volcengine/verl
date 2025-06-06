RL(HF) algorithms with LoRA Support
===========================================

We support LoRA (Low-Rank Adaptation) for reinforcement learning algorithms such as PPO, GRPO, and others.

LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into pre-trained weights (typically linear layers). This reduces memory footprint and compute cost, making it possible to fine-tune large models with limited hardware.

The benefits this brings include:

- reinforcement learning with very large models (e.g. 70B+) with modest hardware (e.g. 8x80G GPUs),
- enable larger batch sizes due to reduced memory usage,
- simplify model transfer and deployment, as only LoRA adapters need to be saved,
- Combine with techniques like `SLoRA <https://arxiv.org/abs/2311.03285>`_ or `CCoE <https://arxiv.org/abs/2407.11686>`_ to serve multiple LoRA adapters efficiently

This guide explains how to enable LoRA in RL training and configure related parameters.

Usage Guide
------------------------
1. Lora is available in the `verl.trainer.ppo.ray_trainer.RayPPOTrainer`. Examples are provided via the `verl.trainer.main_ppo` entry point.

2. Currently, LoRA is supported via huggingface peft, only with fsdp/fsdp2 and vllm backend (sglang support coming soon).

- `strategy=fsdp` or `strategy=fsdp2`
- `rollout.name=vllm`

3. Required configurations for LoRA:

- `actor_rollout_ref.model.lora_rank`: int, set to a reasonable value greater than 0 (e.g., 8, 16, 32, 64)
- `actor_rollout_ref.model.lora_alpha`: float, the alpha term in LoRA
- `actor_rollout_ref.rollout.load_format="safetensors"`: required. This enables vLLM to load the base model.
- `actor_rollout_ref.model.target_modules`: the target modules for LoRA. Typically set to "all-linear".

4. Recommend options:

- `actor_rollout_ref.model.use_shm=True`: preload the model into `/dev/shm` to improve model loading speed.
- `actor_rollout_ref.rollout.layered_summon=True`: this enables the actor-model to gather the FSDP shards per layers when synchronizing the LoRA Adapter to vLLM, thereby reducing GPU peak memory. Recommended if the model is very large (70B+) or the GPU memory is limited (< 48GB)


Best Practices and Notes
-------------------------

1. **Learning rate**: it is recommended to increase the value of learning rate by an order of magnitude.

2. **LoRA Rank**:

- Too small a rank can hurt convergence.
- LoRA rank recommendation from @thelongestusernameofall:

  - A very small lora_rank can lead to slower convergence or worse training performance. It is recommended to set lora_rank to be>=32. Tests have shown that for a 0.5B model, with lora_rank=32,the training convergence speed and final performance are almost identical to non-LoRA training
  - For a 32B model,with lora_rank=128,the training convergence speed and final performance are also almost identical to non-LoRA training.
  - More comprehensive reference results are coming soon.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/f2b80b8b26829124dd393b7a795a0640eff11644/docs/lora.jpg?raw=true

3. Reference configuration for RL training with the Qwen2.5-72B model using 8 x 80GB GPUs (increase lora_rank if needed):

.. code-block::

    data.train_batch_size=64 \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \

Example Script
-------------------

For an end-to-end example, refer to the script below:

examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
