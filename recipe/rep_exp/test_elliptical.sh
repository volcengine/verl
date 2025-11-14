set -x

# We'll use GSM8K as a good debugging task
train_path=$HOME/data/gsm8k/train.parquet
dev_path=$HOME/data/gsm8k/dev.parquet

train_files="['$train_path']"
dev_files="['$dev_path']"

# If you're on a cluster with no internet access, set to OFFLINE=True
OFFLINE=False

PYTHONUNBUFFERED=1 WANDB_MODE=disabled TRANSFORMERS_OFFLINE=${OFFLINE} python3 -m recipe.rep_exp.main_rep_exp \
 algorithm.adv_estimator=grpo \
 data.train_files="$train_files" \
 data.val_files="$dev_files" \
 data.train_batch_size=32 \
 data.max_prompt_length=1024 \
 data.max_response_length=1024 \
 data.filter_overlong_prompts=True \
 data.truncation='error' \
 actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]' \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.val_kwargs.n=2 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 reward_model.enable=True \
 reward_model.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 reward_model.model.use_remove_padding=False \
 reward_model.model.fsdp_config.param_offload=True \
 reward_model.micro_batch_size_per_gpu=32 \
 reward_model.model.input_tokenizer=null \
 reward_model.elliptical.sparse_dim=8 \
 reward_model.elliptical.enable=True \
 reward_model.elliptical.reward_type=leverage \
 reward_model.elliptical.normalization=none \
 reward_model.elliptical.persist_covariance=False \
 reward_model.reward_manager=elliptical \
 reward_model.reward_kwargs.elliptical.beta=1.0 \
 reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_none_correct=True \
 reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_some_correct=False \
 reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_all_correct=False \
 reward_model.reward_kwargs.elliptical.turn_off_elliptical_if_rollout_incorrect=False \
 algorithm.use_kl_in_reward=False \
 trainer.critic_warmup=0 \
 trainer.logger='["console"]' \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=1 \
 trainer.test_freq=1 \
 trainer.total_epochs=15 \
 trainer.resume_mode=disable \
 trainer.max_actor_ckpt_to_keep=null