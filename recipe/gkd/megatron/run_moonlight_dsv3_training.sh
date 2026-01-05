#!/bin/bash
set -x

# 0. download the config
# only need to download the `configuration_deepseek.py`, `config.json`, `tokenizer_config.json`, `tokenizer.json` and `generation_config.json`
# remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported

# huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the HF_MODEL_PATH and DIST_CKPT_PATH to your own path

HF_MODEL_PATH=/path/to/Moonlight-16B-A3B-Instruct
DIST_CKPT_PATH=/path/to/Moonlight-16B-A3B-Instruct-MCORE

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=1 
export NVTE_DEBUG_LEVEL=2

# 2. run the script
gsm8k_train_path=/path/to/train.parquet
train_files=$gsm8k_train_path

# 512 H20(96GB)
NODES=1
PP=3
TP=1
EP=2
ETP=1
INFER_TP=1
SP=True
if [ $TP == 1 ]; then
    SP=False
fi
# consider TP/ETP, and enable recompute if short of memory

TEACHER_SERVER_HOST=127.0.0.1
TEACHER_SERVER_PORT=15555

function check_server_ready() {
    local server=$1
    local ip=$2
    local port=$3
    
    echo "check $server server ready at $ip:$port..."
    result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
    if [ $result -ne 1 ]; then
        echo "server $server is not ready at $ip:$port, exit..."
        exit 1
    fi
}

check_server_ready teacher $TEACHER_SERVER_HOST $TEACHER_SERVER_PORT

function now() {
    date '+%Y-%m-%d-%H-%M'
}

# full recompute
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m main_gkd --config-name on_policy_distill_trainer \
    data.train_files="$train_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=$SP \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_expert_capacity_factor=1.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.micro_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.max_token_len=6000 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
    actor_rollout_ref.actor.checkpoint.load_contents=[] \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    +actor_rollout_ref.actor.distill_loss.name=kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.load_format='dummy_megatron' \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.teacher.server_ip=$TEACHER_SERVER_HOST \
    actor_rollout_ref.teacher.server_port=$TEACHER_SERVER_PORT \
    trainer.logger=['console'] \
    trainer.project_name='on-policy-distill' \
    trainer.experiment_name="moonlight-dsv3-$(now)" \
    trainer.nnodes=$NODES \
    trainer.n_gpus_per_node=6 \
    rollout.nnodes=$NODES \
    rollout.n_gpus_per_node=2 \
    trainer.scheduler="two_step_off" \
    trainer.save_freq=100000 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=10 $@
