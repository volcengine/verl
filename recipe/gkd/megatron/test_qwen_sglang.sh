set -x

# 0. download the config
# only need to download the `configuration_deepseek.py`, `config.json`, `tokenizer_config.json`, `tokenizer.json` and `generation_config.json`
# remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported

# huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the HF_MODEL_PATH to your own path
HF_MODEL_PATH=/path/to/Qwen3-0.6B
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=1 
export NVTE_DEBUG_LEVEL=2

# 2. run the script
gsm8k_train_path=/path/to/train.parquet
gsm8k_test_path=/path/to/test.parquet
train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

# 512 H20(96GB)
NODES=1
PP=1
TP=1
EP=1
ETP=1
INFER_TP=1
# consider TP/ETP, and enable recompute if short of memory

# full recompute
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}
# RAY_ADDRESS='auto' ray job submit --working-dir . --
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m main_gkd --config-name on_policy_distill_trainer \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    +teacher.server_ip=127.0.0.1 \
    +teacher.server_port=15555 \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    +actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.load_format='auto' \
    +algorithm.use_kl_in_reward=False \
    trainer.logger=['console'] \
    trainer.project_name='verl_examples' \
    trainer.experiment_name='qwen-distill' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$NODES \
    rollout.n_gpus_per_node=4 \
    rollout.nnodes=$NODES \
    trainer.save_freq=-1 \
    trainer.test_freq=25 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    trainer.val_before_train=False \
    trainer.total_training_steps=10 \
    trainer.total_epochs=1 $@
    # +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
