Train RL(HF) algorithms with LoRA Support
=================================

We have implemented LoRA support when training with RL(PPO/GRPO/DAPO,etc.).     

The benefits this brings include: 
    - the ability to train extremely large models(70B+), 
    - the capability to use larger batch sizes, 
    - easier transfer and deployment of the trained models, 
    - and the ability to run more different large model weights with limited resources using technologies like `SLoRA <https://arxiv.org/abs/2311.03285>`_ or `CCoE <https://arxiv.org/abs/2407.11686>`_

This document introduces how to enable LoRA support during RL training by configuring the parameters of the verl command.

Parameter Description
----------------
1. Ensure that `verl.trainer.main_ppo` is used as the training entry point.

2. Specify the reinforcement learning algorithm type(e.g., grpo/ppo/dapo/rloo,etc.) using `algorithm.adv_estimator`.

3. Note: Currently, LoRA training support is only implemented for the case where `strategy=fsdp` and `rollout.name=vllm`.

4. [Optional] Set `actor_rollout_ref.model.use_shm=True` to preload the model into `/dev/shm` to improve model loading speed.

5. Set `actor_rollout_ref.model.lora_rank` and `actor_rollout_ref.model.lora_alpha` to reasonable values greater than 0 (e.g., 8,16,32,64, etc.).

6. Set the value type of `actor_rollout_ref.rollout.load_format` to `safetensors` to specify that VLLM should load the base model rather than filling it with random numbers.

7. If the model is very large (70B+) or the GPU memory is limited (48GB-), it is recommended to set `actor_rollout_ref.rollout.layered_summon=True`. This will enable the actor-model to gather the FSDP shards in layers when synchronizing the LoRA Adapter to VLLM,thereby saving GPU memory.

Other command parameters are the same as those used during RL training.

Special Precautions
-------------------
1. When LoRA is enabled,it is recommended to increase the value of lr by an order of magnitude.

2. A very small lora_rank can lead to slower convergence or worse training performance.It is recommended to set lora_rank to be>=32.Tests have shown that for a 0.5B model,with lora_rank=32,the training convergence speed and final performance are almost identical to non-LoRA training;for a 32B model,with lora_rank=128,the training convergence speed and final performance are also almost identical to non-LoRA training.

3. When training the Qwen2.5-72B model using 8 x 80GB GPUs,the recommended parameter combination is as follows:
    • `data.train_batch_size=64`
    • `actor_rollout_ref.model.use_shm=True`
    • `actor_rollout_ref.model.lora_rank=32`
    • `actor_rollout_ref.model.lora_alpha=32`
    • `actor_rollout_ref.model.target_modules=all-linear`
    • `actor_rollout_ref.actor.optim.lr=3e-5`
    • `actor_rollout_ref.actor.fsdp_config.fsdp_size=8`
    • `actor_rollout_ref.actor.fsdp_config.param_offload=True`
    • `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`
    • `actor_rollout_ref.rollout.tensor_model_parallel_size=8`
    • `actor_rollout_ref.rollout.name=vllm`
    • `actor_rollout_ref.rollout.gpu_memory_utilization=0.4`
    • `actor_rollout_ref.rollout.n=5`
    • `actor_rollout_ref.rollout.max_num_seqs=64`
    • `actor_rollout_ref.rollout.max_model_len=1536`
    • `actor_rollout_ref.rollout.max_num_batched_tokens=1536`
    • `actor_rollout_ref.rollout.load_format=safetensors`
    • `actor_rollout_ref.rollout.layered_summon=True`
    • `actor_rollout_ref.ref.fsdp_config.param_offload=True`
    • `actor_rollout_ref.actor.ulysses_sequence_parallel_size=1`

Example Script
-------------------

For a practical training, refer to the example script:

examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
