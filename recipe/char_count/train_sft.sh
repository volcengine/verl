set -x

nproc_per_node=1
save_path=./models/sft

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/char_count/sft/train.parquet \
    data.val_files=$HOME/data/char_count/sft/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=256 \
    data.train_batch_size=256 \
    use_remove_padding=True \
    model.partial_pretrain=$HOME/models/SmolLM2-135M-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=char_count-sft \
    trainer.experiment_name=char_count-sft-SmolLM2-135M-Instruct \
    trainer.total_epochs=3 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null