# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=de8d037779af70199106db0710f417f5157c1818
export TORCH_CPP_LOG_LEVEL="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export NCCL_DEBUG="WARN"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
# export VLLM_ATTENTION_BACKEND="XFORMERS"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0

export NCCL_P2P_DISABLE=1
# export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_USE_V1=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/vllm_multiturn/config"


TRAIN_DATA="/mnt/nas/bachvd/Code-Agent/verl/data/searchR1_processed_direct/train.parquet"
VAL_DATA="/mnt/nas/bachvd/Code-Agent/verl/data/searchR1_processed_direct/test.parquet"

TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

MODEL_PATH="/mnt/nas/bachvd/Code-Agent/model_zoo/Qwen3-VL-8B-Instruct"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    +actor_rollout_ref.rollout.presence_penalty=0.0 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name='jan-v2-instruct_function_rm-search-async-sgl-multi-w-searchtool-verify-n16' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1 $@

