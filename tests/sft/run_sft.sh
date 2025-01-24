# Tested with 2 & 4 GPUs

set -x

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=32 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$HOME/ckpts/ \
    trainer.project_name=qwen2.5-sft \
    trainer.experiment_name=gsm8k-sft-gemma-2b-it \
    trainer.total_training_steps=1 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@

rm -rf $HOME/ckpts/