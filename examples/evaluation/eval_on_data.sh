export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export NCCL_DEBUG='WARN'
export TOKENIZERS_PARALLELISM='true'

export CUDA_VISIBLE_DEVICES=1

# Evaluate on Countdown dataset with actor_fsdp_model
python3 -m examples.evaluation.eval_on_data \
    data.val_files='/data/countdown/test.parquet' \
    actor_rollout_ref.rollout.micro_batch_size=4 \
    actor_rollout_ref.rollout.do_sample=False \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.model.hf_model_path='Qwen/Qwen2.5-3B' \
    actor_rollout_ref.model.actor_fsdp_model_path='/checkpoints/grpo-countdown-qwen2.5-3b/global_step_200/actor'

# Evaluate on Countdown dataset with Qwen
python3 -m examples.evaluation.eval_on_data \
    data.val_files='/data/countdown/test.parquet' \
    actor_rollout_ref.rollout.micro_batch_size=4 \
    actor_rollout_ref.rollout.do_sample=False \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.model.hf_model_path='Qwen/Qwen2.5-3B'
