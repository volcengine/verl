set -e
set -x
NUM_GPUS=${NUM_GPUS:-8}

exp_name="Qwen2.5-0.5B-Instruct-spin-minimal"

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.spin.main_spin \
  data.train_files="${HOME}/data/gsm8k/train.parquet" \
  data.val_files="${HOME}/data/gsm8k/test.parquet" \
  data.train_batch_size=1024 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  +trainer.log_freq=1 \
  trainer.ref_update_freq=1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1000 2>&1 | tee verl_demo.log