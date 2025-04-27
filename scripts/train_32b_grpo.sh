export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# basics
project_name='GRPO'
exp_name='GRPO-Qwen-32B'

adv_estimator=grpo

max_prompt_length=2192
max_response_length=12000
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# Ray
NNODES=4

# Paths
RAY_DATA_HOME="/home/share/reasoning"
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/Qwen2.5-32B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/sky_work_full_04_24.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/AIME_and_MATH_500.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

val_temperature=0.6
val_top_p=0.95
val_top_k=-1

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4


ray job submit --address="http://10.55.251.20:8265" \
    --runtime-env="./verl/trainer/runtime_env.yaml" \
    --no-wait \
    -- python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.001 \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.train_batch_size=${train_prompt_bsz} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        trainer.logger=['console'] \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=2 \
        trainer.resume_mode=auto