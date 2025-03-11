set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

math_train_path=/workspace/datasets/math/train.parquet
math_test_path=/workspace/datasets/math/test.parquet

python3 -m verl.trainer.main_ppo \
    data.train_files="$math_train_path" \
    data.val_files="$math_test_path" \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=/workspace/hf_models/Qwen2.5-Math-7B-Instruct \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0 \
    trainer.logger=['console'] \
    trainer.project_name='test-math-verify' \
    trainer.experiment_name='test-math-verify' \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=0 \
    data.train_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=grpo $@