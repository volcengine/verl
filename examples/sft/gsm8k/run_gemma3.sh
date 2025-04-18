set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=google/gemma-3-4b-it \
    model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft-gemma \
    trainer.experiment_name=google/gemma-3-4b-it \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@