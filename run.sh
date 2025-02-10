git config --global credential.helper cache

# login to huggingface and wandb
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_TOKEN

ibstatus

nvidia-smi topo -m

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/data/data/aimo_math/train.parquet \
    data.val_files=/mnt/data/data/aimo_math/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/mnt/models/phi-4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_aimo_math' \
    trainer.experiment_name='phi-4-aimo-math-gropo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_local_dir='/mnt/output/checkpoints'\
    trainer.total_epochs=30 $@

    # actor_rollout_ref.rollout.kv_cache_dtype="fp8" \
    # +actor_rollout_ref.actor.optim.eight_bit=True \