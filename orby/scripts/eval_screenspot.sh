MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
DATA_PATH=~/data/screenspot
REWARD_FILE=orby/reward/screenspot.py
REWARD_FN=reward_func
OUTPUT_FILE=test-output-1.parquet

# Convert the dataset to parquet format
python3 -m orby.data.convert_screenspot

# Generation
python3 -m orby.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    +data.max_prompt_length=7936 \
    +data.image_key=images \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/$OUTPUT_FILE \
    model.path=$MODEL_PATH \
    rollout.temperature=0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=7936 \
    rollout.response_length=256 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$DATA_PATH/$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN
