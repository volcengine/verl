set -x
logs=/home/l00878165/sglang/repos/logs/jingdu0915/tmp10_2k32k_graph.log

export HCCL_CONNECT_TIMEOUT=1500

num_gpu=8
tp=4
sp_size=4
train_prompt_bsz=16
train_prompt_mini_bsz=16

use_dynamic_bsz=True
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

CKPTS_DIR=/home/l00878165/models/ckpt/qwen3_8b
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/l00878165/datasets/dapo/dapo-math-17k.parquet \
    data.val_files=/home/l00878165/datasets/dapo/aime-2024.parquet \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/l00878165/models/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=3600 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name='verl_grpo_example_dapo_2k_32k' \
    trainer.experiment_name='qwen3_8b_function_rm' \
    trainer.n_gpus_per_node=$num_gpu \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10000 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    trainer.device=npu  2>&1 | tee -i $logs