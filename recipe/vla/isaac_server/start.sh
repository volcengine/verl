#!/bin/bash
# Isaac Lab Multi-Server-Group Startup Script
# Starts multiple independent Isaac Server groups for pipeline parallel training
# Each server group corresponds to one pipeline stage for physical isolation
# All server groups share the same GPUs (time-interleaved)
#
# Note: NUM_SERVER_GROUPS must match env.rollout.pipeline_stage_num in training config
#
# Usage:
#   ./start_isaac_server.sh              # Normal start
#   CLEAN_CACHE=1 ./start_isaac_server.sh  # Clean caches before start

set -e

WORKSPACE=/workspace/verl_vla/

# Clean up any existing Isaac server processes
echo "Cleaning up existing Isaac server processes..."
pkill -f "isaac_server/server.py" 2>/dev/null || true
sleep 2

# ============================================
# Configuration
# ============================================

# Number of tasks: libero_10 benchmark has 10 different tasks
# check num_tasks in franka_libero_multitask_env_cfg.py
NUM_TASKS=10

# Envs per task: how many env instances to run per task in each server group
GROUP_SIZE=16

# Number of GPUs: all server groups share the same GPUs
NUM_GPUS=8

# Number of server groups - MUST match pipeline_stage_num in training config
# Each pipeline stage uses its own server group for physical isolation
NUM_SERVER_GROUPS=2

# Base port for server group 0 (subsequent groups use +50 spacing)
BASE_PORT=5556
PORT_SPACING=50

# Base master port for torch.distributed (subsequent groups use +1)
BASE_MASTER_PORT=29500

# Action mode
export LIBERO_OSC_TYPE=pose_rel

# Camera configuration
CAMERA_HEIGHT=256
CAMERA_WIDTH=256

# ============================================
# Calculate and Display Configuration
# ============================================

TOTAL_ENVS=$((NUM_TASKS * GROUP_SIZE))
TASKS_PER_GPU=$((NUM_TASKS / NUM_GPUS))
ENVS_PER_GPU=$((TASKS_PER_GPU * GROUP_SIZE))

echo "============================================"
echo "Isaac Lab Multi-Server-Group Mode"
echo "============================================"
echo "  GPU count:           ${NUM_GPUS} (shared by all groups)"
echo "  Total tasks:         ${NUM_TASKS}"
echo "  Tasks per GPU:       ${TASKS_PER_GPU}"
echo "  Envs per task:       ${GROUP_SIZE}"
echo "  Envs per GPU:        ${ENVS_PER_GPU}"
echo "  Total envs/group:    ${TOTAL_ENVS}"
echo "============================================"
echo "  Server groups:       ${NUM_SERVER_GROUPS}"
echo "  (Must match pipeline_stage_num in training config)"
echo "============================================"

for i in $(seq 0 $((NUM_SERVER_GROUPS - 1))); do
    SERVER_PORT=$((BASE_PORT + i * PORT_SPACING))
    MASTER_PORT=$((BASE_MASTER_PORT + i))
    echo "  Server Group $i (Stage $i):"
    echo "    Port range:        ${SERVER_PORT} - $((SERVER_PORT + NUM_GPUS - 1))"
    echo "    Master Port:       ${MASTER_PORT}"
done
echo "============================================"

# Python executable
PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"

# Create log directory
mkdir -p ${WORKSPACE}/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================
# Setup cache directories
# ============================================
# Each server group gets its own cache directories to avoid conflicts:
# 1. OptiX cache - prevents "DISKCACHE Failed to execute query: locking protocol"
# 2. NVIDIA Shader cache - prevents shader compilation conflicts
# 3. Omniverse cache - prevents general caching conflicts

OPTIX_CACHE_BASE="/tmp/optix_cache"
SHADER_CACHE_BASE="/tmp/nv_shader_cache"
OV_CACHE_BASE="/tmp/ov_cache"

mkdir -p ${OPTIX_CACHE_BASE}
mkdir -p ${SHADER_CACHE_BASE}
mkdir -p ${OV_CACHE_BASE}

echo "Cache directories:"
echo "  OptiX cache base:  ${OPTIX_CACHE_BASE}"
echo "  Shader cache base: ${SHADER_CACHE_BASE}"
echo "  OV cache base:     ${OV_CACHE_BASE}"

# Clean up caches if requested (set CLEAN_CACHE=1)
if [ "${CLEAN_CACHE}" = "1" ]; then
    echo ""
    echo "Cleaning cache directories..."
    rm -rf ${OPTIX_CACHE_BASE}/*
    rm -rf ${SHADER_CACHE_BASE}/*
    rm -rf ${OV_CACHE_BASE}/*
    # Also clean system-level caches
    rm -rf /root/.cache/ov/shaders/nv_shadercache 2>/dev/null || true
    rm -rf /root/.nv/ComputeCache 2>/dev/null || true
    echo "Cache directories cleaned."
fi

# ============================================
# Start all server groups
# ============================================

SERVER_PIDS=()

for i in $(seq 0 $((NUM_SERVER_GROUPS - 1))); do
    SERVER_PORT=$((BASE_PORT + i * PORT_SPACING))
    MASTER_PORT=$((BASE_MASTER_PORT + i))
    
    # Create per-group cache directories
    OPTIX_CACHE_GROUP="${OPTIX_CACHE_BASE}/group_${i}"
    SHADER_CACHE_GROUP="${SHADER_CACHE_BASE}/group_${i}"
    OV_CACHE_GROUP="${OV_CACHE_BASE}/group_${i}"
    mkdir -p ${OPTIX_CACHE_GROUP}
    mkdir -p ${SHADER_CACHE_GROUP}
    mkdir -p ${OV_CACHE_GROUP}
    
    echo "[$(date)] Starting Server Group $i (Stage $i)..."
    echo "  OptiX cache:  ${OPTIX_CACHE_GROUP}"
    echo "  Shader cache: ${SHADER_CACHE_GROUP}"
    
    # Set per-group environment variables for all caching systems:
    # - OPTIX_CACHE_PATH, OPTIX7_CACHE_PATH: OptiX denoiser cache
    # - __GL_SHADER_DISK_CACHE_PATH: NVIDIA driver shader cache
    # - OMNI_KIT_CACHE_DIR, OMNI_USER_CACHE_DIR: Omniverse Kit/User cache (including shader cache)
    # - CARB_DATA_PATH: Carbonite data path
    OPTIX_CACHE_PATH=${OPTIX_CACHE_GROUP} \
    OPTIX7_CACHE_PATH=${OPTIX_CACHE_GROUP} \
    __GL_SHADER_DISK_CACHE=1 \
    __GL_SHADER_DISK_CACHE_PATH=${SHADER_CACHE_GROUP} \
    __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1 \
    OMNI_KIT_CACHE_DIR=${OV_CACHE_GROUP} \
    OMNI_USER_CACHE_DIR=${OV_CACHE_GROUP} \
    CARB_DATA_PATH=${OV_CACHE_GROUP}/carb \
    ${PYTHON} -m torch.distributed.run \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${WORKSPACE}/verl/recipe/vla/isaac_server/server.py \
            --num_tasks ${NUM_TASKS} \
            --group_size ${GROUP_SIZE} \
            --port ${SERVER_PORT} \
            --use_tcp \
            --distributed \
            --camera_height ${CAMERA_HEIGHT} \
            --camera_width ${CAMERA_WIDTH} \
        2>&1 | tee ${WORKSPACE}/logs/isaac_server${i}_${TIMESTAMP}.log &
    
    SERVER_PIDS+=($!)
    echo "[$(date)] Server Group $i started with PID: ${SERVER_PIDS[$i]}"
    
    # Wait between server group starts to avoid initialization conflicts
    if [ $i -lt $((NUM_SERVER_GROUPS - 1)) ]; then
        sleep 10
    fi
done

echo "============================================"
echo "All ${NUM_SERVER_GROUPS} server groups started!"
for i in $(seq 0 $((NUM_SERVER_GROUPS - 1))); do
    echo "  Server Group $i PID: ${SERVER_PIDS[$i]}"
done
echo ""
echo "To stop all servers:"
echo "  kill ${SERVER_PIDS[*]}"
echo "============================================"

# Wait for all processes
wait ${SERVER_PIDS[*]}
