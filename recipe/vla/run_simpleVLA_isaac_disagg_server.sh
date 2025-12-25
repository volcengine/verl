#!/bin/bash
set -x

echo "remember to set ray param < --resources='{\"sim\"/\"actor_rollout\":1}' > if using disagg sim"

WORKSPACE=/workspace/verl_vla/

libero_train_path=$WORKSPACE/data/libero_rl/train.parquet
libero_test_path=$WORKSPACE/data/libero_rl/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$WORKSPACE/models/vla_libero_grpo_server"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$WORKSPACE"}/video/$(date +%Y%m%d_%H%M%S)
TIMELINE_FILE=${TIMELINE_FILE:-"$WORKSPACE/logs/ray_timeline_$(date +%Y%m%d_%H%M%S).json"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$WORKSPACE/data/Openvla-oft-SFT-libero10-trajall"}

# for rollout and train
NUM_NODES=1
# for simulator
SIM_NODES=1
NUM_ENV_GPUS=8
NUM_ROLLOUT_GPUS=8
STAGE_NUM=2
BATCH_SIZE=16
# rollout.n should equal to num_envs for isaac env
ROLLOUT_N=8
SERVER_GROUP_SIZE=16

# 512 is required for libero benchmark, but you can reduce it in debugging to run faster
MAX_EPISODE_STEPS=512

# Number of tasks in the benchmark
NUM_TASKS=10

# isaac or libero
# NOT SUPPORTED: libero means original libero benchmark with mujoco sim
# isaac means libero benchmark using isaac sim
SIM_TYPE=${SIM_TYPE:-"isaac"}
PROJECT_NAME="vla-disagg-isaac-server"
EXPERIMENT_NAME="${SIM_TYPE}_server_rl"

# ============================================
# Isaac Server Configuration
# ============================================
USE_SERVER_MODE=True

# Number of Isaac servers per group (one per GPU)
NUM_ISAAC_SERVERS=8

# Number of server groups - MUST match STAGE_NUM (pipeline_stage_num)
# Each pipeline stage uses its own server group for physical isolation
# This enables Gen-Sim parallel execution and prevents env state interference
NUM_SERVER_GROUPS=$STAGE_NUM

# Server address configuration
# Isaac Server runs on SIM_NODE, Client runs on TRAIN_NODE
SIM_NODE_IP="10.185.189.15"
ISAAC_SERVER_ADDRESS="tcp://${SIM_NODE_IP}"
ISAAC_SERVER_USE_TCP=True

# Base ports for each server group (auto-generated with 50-port spacing)
# Example with NUM_SERVER_GROUPS=2: [5556, 5606]
# Server group 0 (Stage 0): ports 5556-5563
# Server group 1 (Stage 1): ports 5606-5613
BASE_PORT=5556
PORT_SPACING=50
SERVER_BASE_PORTS="[$(seq -s, $BASE_PORT $PORT_SPACING $((BASE_PORT + (NUM_SERVER_GROUPS-1) * PORT_SPACING)))]"

# Calculate required envs on server:
# total_envs = NUM_ROLLOUT_GPUS * STAGE_NUM * ROLLOUT_N
REQUIRED_SERVER_ENVS=$((NUM_ROLLOUT_GPUS * STAGE_NUM * ROLLOUT_N))

echo "============================================"
echo "Isaac Server Configuration:"
echo "  Servers per group: ${NUM_ISAAC_SERVERS}"
echo "  Server groups: ${NUM_SERVER_GROUPS} (must match STAGE_NUM=${STAGE_NUM})"
echo "  Server base ports: ${SERVER_BASE_PORTS}"
echo "  Required server envs: $REQUIRED_SERVER_ENVS (${NUM_ROLLOUT_GPUS} rollout_gpus × ${STAGE_NUM} stages × ${ROLLOUT_N} envs)"
echo "  Envs per server: $((REQUIRED_SERVER_ENVS / NUM_ISAAC_SERVERS / NUM_SERVER_GROUPS))"
echo "============================================"
echo ""
echo "To start Isaac servers (in a separate terminal):"
echo "  NUM_SERVER_GROUPS=${NUM_SERVER_GROUPS} ./start_isaac_server.sh"
echo "============================================"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir -p /root/LIBERO/libero/libero/../datasets

SAVE_VIDEO=False

export PYTHONRECURSIONLIMIT=10000
# uncomment this to see full error messages
# export HYDRA_FULL_ERROR=1

$PYTHON -m recipe.vla.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    +data.num_tasks=${NUM_TASKS} \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$ROLLOUT_N \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$STAGE_NUM \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=8 \
    env.actor.model.action_dim=7 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=$SAVE_VIDEO \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    env.disagg_sim.enable=True \
    env.disagg_sim.nnodes=$SIM_NODES \
    env.train.use_server_mode=$USE_SERVER_MODE \
    env.train.isaac_server_address=$ISAAC_SERVER_ADDRESS \
    +env.train.num_isaac_servers=$NUM_ISAAC_SERVERS \
    +env.train.isaac_server_use_tcp=$ISAAC_SERVER_USE_TCP \
    +env.train.num_server_groups=$NUM_SERVER_GROUPS \
    +env.train.server_base_ports="$SERVER_BASE_PORTS" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_ROLLOUT_GPUS \
    env.train.server_group_size=$SERVER_GROUP_SIZE \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    +env.train.total_envs=$((NUM_ROLLOUT_GPUS * STAGE_NUM * ROLLOUT_N)) \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=20 \
    trainer.val_only=False \
    trainer.total_training_steps=10000 \
    algorithm.adv_estimator=reinforce_plus_plus \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    +ray_kwargs.timeline_json_file=$TIMELINE_FILE \
    $@ 2>&1 | tee $WORKSPACE/logs/vla_isaac_disagg_server_$(date +%Y%m%d_%H%M%S).log

