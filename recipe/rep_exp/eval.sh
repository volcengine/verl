TASK=${1} # math, gsm8k, dapo-with-aime24

# Custom model path for evaluation after training
MODEL_PATH=${2} # /path/to/global_step_X/actor/hf, where X is the global step of the checkpoint with the best pass@1 on dev

# If you want to evaluate the base model before training
# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

train_path=$HOME/data/${TASK}/train.parquet
train_files="['$train_path']"
CHECKPOINT_SAVE_CONTENTS='["model"]'

if [ ${TASK} == "dapo-with-aime24" ]; then
    MAX_PROMPT_LENGTH=$((1024 * 2))
    MAX_RESPONSE_LENGTH=$((1024 * 8))
    MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    test_path=$HOME/data/${TASK}/dev.parquet
else
    MAX_PROMPT_LENGTH=1024
    MAX_RESPONSE_LENGTH=1024
    MAX_NUM_BATCHED_TOKENS=8192
    test_path=$HOME/data/${TASK}/test.parquet
fi

test_files="['$test_path']"

# If you're on a cluster with no internet access, set to OFFLINE=True
OFFLINE=False

PYTHONUNBUFFERED=1 WANDB_MODE=disabled TRANSFORMERS_OFFLINE=${OFFLINE} python3 -u -m recipe.rep_exp.main_rep_exp \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.val_batch_size=128 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_SAVE_CONTENTS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=256 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.model.path="$MODEL_PATH" \
    reward_model.model.use_remove_padding=False \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.input_tokenizer=null \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","json_eval"]' \
    trainer.project_name='rep-exp' \
    trainer.experiment_name="${TASK}_eval" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=100 \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.resume_from_path=''

exit 0
