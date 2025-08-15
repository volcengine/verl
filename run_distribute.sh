#!/bin/bash
set -xeo pipefail


WORKING_DIR=${WORKING_DIR:-"/nfs/verl"}
cd ${WORKING_DIR}

MODEL_PATH=/nfs/volume-1615-2/models/lmsys/gpt-oss-20b-bf16

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_AGENT_CHECK_PARENT_INTERVAL_=S100000000i


export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
if [ -z "${DISTRIBUTED_NODE_RANK}" ]; then
    echo "DISTRIBUTED_NODE_RANK is not set, single node program"
    DISTRIBUTED_NODE_COUNT=1
    DISTRIBUTED_NODE_RANK=0
    DISTRIBUTED_MASTER_HOSTS=localhost
    dist_init_port=50010
else
    # dist_init_port=${LUBAN_AVAILABLE_PORT_1}
    dist_init_port=50010
    echo "dist_init_port: $dist_init_port"

    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1"
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_IB_TIMEOUT=22 # 10 -> 22
    export NCCL_PXN_DISABLE=0
    export NCCL_IB_SL=0
    # export NCCL_CUMEM_HOST_ENABLE=0
fi

if [ -z "${PET_MASTER_PORT}" ]; then
    echo "PET_MASTER_PORT is not set, not luban ddp environment, use default port 30000"
    PET_MASTER_PORT=21010
fi
## fix port 12001 conflict
PET_MASTER_PORT=21010

if [ -n "${RESOURCE_NUM_GPU}" ]; then
    NP=${RESOURCE_NUM_GPU}
    echo "NP is set to resource num gpu: $NP"
else
    echo "np set to gpu count"
    NP=$(nvidia-smi -L | wc -l)
fi
if [ -z "${NP}" ]; then
    echo "NP is not set, no gpu found"
    exit 1
fi

## echo variables
echo "DISTRIBUTED_NODE_COUNT: $DISTRIBUTED_NODE_COUNT"
echo "DISTRIBUTED_NODE_RANK: $DISTRIBUTED_NODE_RANK"
echo "DISTRIBUTED_MASTER_HOSTS: $DISTRIBUTED_MASTER_HOSTS"
echo "PET_MASTER_PORT: $PET_MASTER_PORT"
echo "NP: $NP"

echo -e "\n系统资源限制:"
ulimit -a

echo "123.207.209.134 llab-asst-pre-hna.xiaojukeji.com" | sudo tee -a /etc/hosts

echo "hosts 信息"
cat /etc/hosts


export TMPDIR=/tmp
pip list

export PYTHONPATH=$PYTHONPATH:/home/luban/Megatron-LM-v0.11.0
export VERL_PPO_LOGGING_LEVEL=INFO


if [ ${DISTRIBUTED_NODE_RANK} -eq 0 ]; then
    ray stop  # stop any existing ray instance
    ray start --head --port=${PET_MASTER_PORT}
    echo "ray start master in port: ${PET_MASTER_PORT}"

    TARGET_NODES=$DISTRIBUTED_NODE_COUNT
    while true; do
        CURRENT_NODES=$(ray list nodes | grep "ALIVE" | wc -l)
        echo "当前节点数: $CURRENT_NODES, 目标节点数: $TARGET_NODES"
        if [ "$CURRENT_NODES" -ge "$TARGET_NODES" ]; then
            echo "已达到目标节点数，开始提交任务..."
            break
        fi
        echo "等待更多节点加入..."
        sleep 10
    done
    echo "执行任务提交..."
    ray job submit --address="http://127.0.0.1:8265" \
        -- env PYTHONPATH=${PYTHONPATH} PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nfs/data/train.parquet \
    data.val_files=/nfs/data/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=["console"] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${DISTRIBUTED_NODE_COUNT} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@
else
    RAY_ADDRESS="${DISTRIBUTED_MASTER_HOSTS}:${PET_MASTER_PORT}"
    ray start --address="${RAY_ADDRESS}"
    echo "ray start worker join master ${RAY_ADDRESS}"
    sleep infinity
fi