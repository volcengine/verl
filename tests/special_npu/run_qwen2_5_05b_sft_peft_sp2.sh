set -x

mkdir -p ./save_ckpts

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=32 \
    model.partial_pretrain="${MODEL_PATH}" \
    trainer.default_local_dir=./save_ckpts \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=console \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.device=npu

rm -rf ./outputs ./save_ckpts
