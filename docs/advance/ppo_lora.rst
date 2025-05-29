Train RL(HF) algorithms with LoRA Support
=================================

We have implemented LoRA support when training with RL(PPO/GRPO/DAPO,etc.).     

The benefits this brings include: 
    - the ability to train extremely large models(70B+), 
    - the capability to use larger batch sizes, 
    - easier transfer and deployment of the trained models, 
    - and the ability to run more different large model weights with limited resources using technologies like [SLoRA](https://arxiv.org/abs/2311.03285) or [CCoE](https://arxiv.org/abs/2407.11686).

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

Example Script
-------------------

For a practical training, refer to the example script:

examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh