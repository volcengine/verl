MODEL_PATH=Qwen/DeepSeek-R1-Distill-Qwen-1.5B
DATA_PATH=/workspace/datasets/r1_bench

# Eval Data Process
python3 -m recipe.r1.data_process \
    --local_dir $DATA_PATH \
    --tasks all

# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=8 \
    data.output_path=$DATA_PATH/test-output-8.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=32768 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m recipe.r1.main_eval \
    data.path=$DATA_PATH/test-output-8.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func
