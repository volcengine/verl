export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR=$PWD/data
export HF_HOME=$DATA_DIR
export PATH="$CONDA_PREFIX/bin:$PATH"
export VLLM_CACHE_ROOT=$DATA_DIR/vllm_cache
set -x


gsm8k_train_path=$DATA_DIR/gsm8k/train.parquet
gsm8k_test_path=$DATA_DIR/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

use_legacy_worker_impl=auto
use_legacy_worker_impl=disable

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.sft.enabled=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_sft_example_gsm8k' \
    trainer.experiment_name="qwen_05_use_legacy_worker_impl_$use_legacy_worker_impl" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=400 \
    trainer.test_freq=5 \
    trainer.resume_mode=disable \
    trainer.total_epochs=2 \
    trainer.use_legacy_worker_impl=$use_legacy_worker_impl $@
