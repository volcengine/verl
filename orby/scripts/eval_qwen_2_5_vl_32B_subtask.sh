MODEL_PATH=Qwen/Qwen2.5-VL-32B-Instruct
DATA_PATH=~/data/subtask_direct_distill/mix/test/combined.parquet
REWARD_FILE=orby/reward/subtask.py
REWARD_FN=eval_reward_func
OUTPUT_FILE=test-output-subtask-1.parquet

# Generation
python3 -m orby.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.batch_size=256 \
    +data.max_prompt_length=7680 \
    data.n_samples=1 \
    data.output_path=$OUTPUT_FILE \
    model.path=$MODEL_PATH \
    rollout.temperature=0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=7680 \
    rollout.response_length=512 \
    rollout.tensor_model_parallel_size=8 \
    rollout.gpu_memory_utilization=0.6 \
    rollout.max_num_batched_tokens=65536 \
    +rollout.limit_images=3

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN
