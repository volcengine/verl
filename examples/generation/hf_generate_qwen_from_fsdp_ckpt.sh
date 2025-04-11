export HYDRA_FULL_ERROR=1
export NCCL_DEBUG='WARN'
export TOKENIZERS_PARALLELISM='true'

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=0

python3 -m verl.trainer.main_generation \
    trainer.n_gpus_per_node=1 \
    data.path='/data/countdown/test.parquet' \
    data.output_path='./out_test.parquet' \
    data.batch_size=8 \
    data.n_samples=1 \
    model.path='Qwen/Qwen2.5-3B' \
    model.actor_fsdp_model_path='/checkpoints/grpo-countdown-qwen2.5-3b/global_step_200/actor' \
    rollout.do_sample=False \
    rollout.response_length=1024
