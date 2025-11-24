TASK=${1} # math, gsm8k, dapo-with-aime24
SPARSE_DIM=${2} # the original paper used 32 for math/gsm8k, 128 for dapo-with-aime24
BETA=${3} # 0.01
SEED=${4}

train_path=$HOME/data/${TASK}/train.parquet
dev_path=$HOME/data/${TASK}/dev.parquet

train_files="['$train_path']"
dev_files="['$dev_path']"

# Adjust things a bit for dapo-aime training since it has longer generations
# and hence is slower and consumes more memory
if [ ${TASK} == "dapo-with-aime24" ]; then
    TEST_FREQ=10
    SAVE_FREQ=10
    TRAIN_BATCH_SIZE=512
    PPO_MINI_BATCH_SIZE=128

    MAX_PROMPT_LENGTH=$((1024 * 2))
    MAX_RESPONSE_LENGTH=$((1024 * 8))
    MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    GPU_MEMORY_UTILIZATION=0.5
    PPO_MICRO_BATCH_SIZE_PER_GPU=8
    REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU=16
else
    TEST_FREQ=20
    SAVE_FREQ=20
    TRAIN_BATCH_SIZE=1024
    PPO_MINI_BATCH_SIZE=256

    MAX_PROMPT_LENGTH=1024
    MAX_RESPONSE_LENGTH=1024
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.6
    PPO_MICRO_BATCH_SIZE_PER_GPU=16
    REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU=32
fi

OFFLINE=True

PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=${OFFLINE} python3 -u -m recipe.rep_exp.main_rep_exp \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$dev_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.model.path=Qwen/Qwen2.5-7B-Instruct \
    reward_model.model.use_remove_padding=False \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=$REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.model.input_tokenizer=null \
    reward_model.elliptical.enable=True \
    reward_model.elliptical.sparse_dim=$SPARSE_DIM \
    reward_model.elliptical.reward_type=leverage \
    reward_model.elliptical.randomize_sparse_matrix=True \
    reward_model.elliptical.normalization=none \
    reward_model.elliptical.persist_covariance=False \
    reward_model.reward_manager=elliptical \
    reward_model.reward_kwargs.elliptical.beta=$BETA \
    reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_none_correct=True \
    reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_some_correct=False \
    reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_all_correct=False \
    reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_rollout_incorrect=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=True \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rep-exp' \
    trainer.experiment_name="${TASK}_elliptical_seed_${SEED}_beta_${BETA}_sparse_dim_${SPARSE_DIM}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1000 \
    trainer.resume_mode=disable \
    trainer.resume_from_path=''