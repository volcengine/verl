# Discliamer: the model used in the script is only for academic purpose.
set -x
echo "=== GPU List ==="
nvidia-smi -L
echo "=== Ray Status ==="
ray status

export PATH="$PATH:/home/haoyuan/workspace/sop_yang_workspace/software/nsight/bin"
# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
#   python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
# ray start --head --num-gpus=8 --num-cpus=$(nproc) --disable-usage-stats
export DATA_DIR=/home/haoyuan/workspace/sop_yang_workspace/datasets
export MODEL_DIR=/home/haoyuan/workspace/sop_yang_workspace/hfmodel_ckpts
gsm8k_train_path=$DATA_DIR/gsm8k/train.parquet
gsm8k_test_path=$DATA_DIR/gsm8k/test.parquet
# math_train_path=$DATA_DIR/math/train.parquet
# math_test_path=$DATA_DIR/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
wandb login 73ab2750178b89b6c319f5dad835e03212572e94
# prepare model ckpt
# huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir $HOME/models/Qwen2-7B-Instruct &
# huggingface-cli download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir $HOME/models/FsfairX-LLaMA3-RM-v0.1 &
# wait

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/home/haoyuan/workspace/sop_yang_workspace/hfmodel_ckpts/Qwen2-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.ignore_eos=True \
    actor_rollout_ref.rollout.min_tokens=512 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="/home/haoyuan/workspace/sop_yang_workspace/hfmodel_ckpts/Qwen2-7B-Instruct" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_example' \
    trainer.val_before_train=False \
    trainer.experiment_name='Qwen2-7B-Instruct_hybrid' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.bench_steps=2 \
    trainer.total_epochs=1 $@

# cp -r /tmp/ray /home/haoyuan/workspace/sop_yang_workspace/rl_framework/verl/verl_log

    # reward_model.enable=True \
    # reward_model.model.path="/home/haoyuan/workspace/sop_yang_workspace/hfmodel_ckpts/FsfairX-LLaMA3-RM-v0.1" \
    # reward_model.model.use_remove_padding=True \
    # reward_model.model.fsdp_config.param_offload=True \
    # reward_model.micro_batch_size_per_gpu=32 \