set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_vl_sft_3b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Changes to run orby.trainer.fsdp_sft_trainer
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
export PYTHONPATH="$PWD:$PYTHONPATH"


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m orby.trainer.fsdp_sft_trainer \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.train_files=$HOME/data/geo3k/train_clean.parquet \
    data.val_files=$HOME/data/geo3k/test_clean.parquet \
    data.prompt_key=prompt \
    data.response_key=extra_info \
    data.image_key=images \
    optim.lr=1e-4 \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=Qwen/Qwen2.5-VL-3B-Instruct \
    +name_or_path=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=geo3k-sft \
    trainer.experiment_name=geo3k-sft-qwen-2.5-0.5b-instruct-sp2 \
    trainer.logger=['console'] \
    trainer.total_training_steps=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false