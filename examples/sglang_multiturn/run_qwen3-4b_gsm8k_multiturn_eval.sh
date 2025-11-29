# Multi-turn tool evaluation script for GSM8K
# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/verl/trainer/config"

python3 -m verl.trainer.main_multiturn_eval \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_eval' \
    data.eval_files=$HOME/minhoucheng/data/gsm8k/test.parquet \
    data.eval_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.need_tools_kwargs=True \
    actor_rollout_ref.model.path=$HOME/minhoucheng/Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=null \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    reward_model.enable=False \
    reward_model.enable_resource_pool=False \
    checkpoint_dir=$HOME/minhoucheng/checkpoint/vllm_multiturn_async_gsm8k_code_interpreter_n8 \
    evaluation.max_batches=null \
    evaluation.max_samples=null \
    output.path=./eval_results \
    output.scores_path=evaluation_scores.parquet \
    output.trace_path=evaluation_trace.json \
    $@

