#!/bin/bash
set -x

# set dist args
nproc_per_node=${ARNOLD_WORKER_GPU}
if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  MASTER_NODE_ID=${ARNOLD_ID}
  nnodes=1
  node_rank=0
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
fi
master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
master_addr=${!master_addr}
master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
master_port=${!master_port}
ports=(`echo $master_port | tr ',' ' '`)
master_port=${ports[0]}
echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

experiment_name=multiturn-sft-qwen-2.5-32b-instruct
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

TRAIN_DATA=$DATA_ROOT/dataset/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=$DATA_ROOT/dataset/wuxibin/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=$HDFS_ROOT/model/Qwen2.5-32B-Instruct
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

torchrun --nnodes=$ARNOLD_WORKER_NUM \
     --nproc_per_node=$ARNOLD_WORKER_GPU \
     --master-addr=$master_addr \
     --master-port=$master_port \
     --node-rank=$node_rank \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=wuxibin-multiturn-sft \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=6 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true