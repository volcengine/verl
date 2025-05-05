# Discliamer: the model used in the script is only for academic purpose.
set -x

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
#   python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

gsm8k_train_path=$HOME/data/math/train.parquet
gsm8k_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

# prepare model ckpt
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct &
# huggingface-cli download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir $HOME/models/FsfairX-LLaMA3-RM-v0.1 &
wait

python3 -m recipe.sppo.main_sppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HOME/models/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang  \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='sppo-sglang' \
    trainer.val_before_train=True \
    trainer.experiment_name='Qwen2-7B-Instruct_hybrid_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1000 $@
    # Note that we set lr_warmup_steps = 15 in config/sppo_trainer.yaml
    # The experiment will converge to 0.656 on MATH dataset after 20 epochs