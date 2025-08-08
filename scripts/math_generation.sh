set -x

python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$SAVE_PATH \
    model.path=$MODEL_DIR \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.95
