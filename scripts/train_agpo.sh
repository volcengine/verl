# basics
project_name='AGPO'
exp_name='AGPO-R1-Distill-7B'

adv_estimator=agpo

max_prompt_length=2048
max_response_length=$((1024 * 32))
train_prompt_bsz=64
n_resp_per_prompt=16
train_prompt_mini_bsz=16

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Ray
NNODES=4

# Paths
RAY_DATA_HOME="/home/share/reasoning"
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/DeepSeek-R1-Distill-Qwen-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/rl_math_data.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/aime-2024-qwen3.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

val_temperature=0.6
val_top_p=0.95
val_top_k=20

# Performance Related Parameter
sp_size=1
gen_tp=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True


ray job submit --address="http://10.55.251.20:8265" \
    --runtime-env="./verl/trainer/runtime_env.yaml" \
    --no-wait \
    -- python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.train_batch_size=${train_prompt_bsz} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.max_num_batched_tokens=${actor_ppo_max_token_len} \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        reward_model.reward_manager=agpo \
        trainer.logger=['console','wandb'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.val_before_train=True \
        trainer.test_freq=5 \
        trainer.save_freq=20 \
        trainer.total_epochs=1 \
        trainer.resume_mode=auto