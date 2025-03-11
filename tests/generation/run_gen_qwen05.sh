# Tested with 1 & 4 GPUs
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gen_qwen05.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$nproc_per_node \
    data.path=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
