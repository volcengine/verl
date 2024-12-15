set -x

hdfs_path=hdfs://user/verl/experiments/gsm8k/gemma-1.1-7b-it/ # replace to your own hdfs/local path

nproc_per_node=$1

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.micro_batch_size=8 \
    model.partial_pretrain=google/gemma-1.1-7b-it \
    trainer.default_hdfs_dir=$hdfs_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-1.1-7b-it \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']