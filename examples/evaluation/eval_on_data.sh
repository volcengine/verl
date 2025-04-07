export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

python3 -m verl.trainer.eval_on_data \
    data.val_files='/data/countdown/test.parquet' \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.do_sample=False \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.model.hf_model_path='Qwen/Qwen2.5-3B' \
    actor_rollout_ref.model.actor_fsdp_model_path='/checkpoints/grpo-countdown-qwen2.5-3b/global_step_200/actor'
