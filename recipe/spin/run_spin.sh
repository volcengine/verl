set -e
set -x
# export WANDB_API_KEY="324f1526c5559d81acdf0e40dddc9222f30965e1"
VISIBLE_DEVICES="4,5,6,7"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.spin.main_spin \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=1312 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=1 \
  +trainer.log_freq=1 \
  +trainer.ref_update_freq=1 \
  trainer.total_epochs=3 2>&1 | tee verl_demo.log