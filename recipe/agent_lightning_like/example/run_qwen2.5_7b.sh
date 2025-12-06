#!/usr/bin/env bash
# run on 8xA100
# make sure your current working directory is the root of the project
set -xeo pipefail
ulimit -n 65535

project_name='Lightning_like_demo'
exp_name='lightning_qwen25_7b_1'

rollout_engine=sglang
offload=False

# Algorithm
adv_estimator=grpo
rollout_n=16

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

lr=1e-6
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
entropy_coeff=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=5
max_prompt_length=2048
max_response_length=8192

train_bsz=128
train_mini_bsz=32
val_bsz=128
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

tp_size=2
sp_size=1

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
if [ -z "${N_GPUS_PER_NODE}" ]; then
    N_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi

# Paths
DFS_HOME=${DFS_HOME:-"${HOME}"}
DATA_HOME=${DATA_HOME:-"${DFS_HOME}/data"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME}/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_HOME}/gsm8k/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${DFS_HOME}/ckpts/${project_name}/${exp_name}"}
RUN_DIR=${RUN_DIR:-"${DFS_HOME}/run/${project_name}/${exp_name}"}
EXAMPLE_DIR=recipe/agent_lightning_like/example


export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
export VERL_LOGGING_LEVEL=WARN  # DO NOT enable DEBUG logging on large batch_size
export TENSORBOARD_DIR="${RUN_DIR}/tb"
mkdir -p ${TENSORBOARD_DIR}

# a file used to notify the llm server address for the agent server to read
# this file should be on a file system shared by the agent server and the ray cluster
export LLM_SERVER_NOTIFY_FILE="${RUN_DIR}/llm_server.notify"

pip3 install -r ${EXAMPLE_DIR}/requirements.txt

# Launch standalone Agent Server
_host=$(PYTHONPATH=recipe/agent_lightning_like python3 -c "from example.agent_server import _get_host_ip;print(_get_host_ip())")
_port=$(PYTHONPATH=recipe/agent_lightning_like python3 -c "from example.agent_server import _get_free_port;print(_get_free_port())")
agent_server_addr="${_host}:${_port}"
uvicorn recipe.agent_lightning_like.example.agent_server:app \
    --host 0.0.0.0 --port ${_port} --workers 16 --log-level warning 2>&1 &
server_pid=$!
trap "echo 'Killing agent server PID ${server_pid}' && kill ${server_pid}" EXIT


HYDRA_FULL_ERROR=1 python3 -m recipe.agent_lightning_like.main_lightning \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    data.custom_cls.path=${EXAMPLE_DIR}/dataset.py \
    data.custom_cls.name=CustomDataset \
    data.train_batch_size=${train_bsz} \
    data.val_batch_size=${val_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr="${lr}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${EXAMPLE_DIR}/agent_loop.yaml \
    lightning_trainer.agent_client_config_path=${EXAMPLE_DIR}/agent_client.yaml \
    lightning_trainer.agent_server_addr=${agent_server_addr} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=1 $@
