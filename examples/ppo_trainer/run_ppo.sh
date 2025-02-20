set -x

export NCCL_DEBUG=WARN
export WANDB_API_KEY='448ad9b79b563f75fbc01c9a69db00e98ffadae2'
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
 
PROJECT_NAME='Qwen-1.5B-Test'
EXPERIMENT_NAME='Qwen_1.5B'
DATA_PATH=/mnt/petrelfs/fanyuchen
SFT_MODEL_PATH=/mnt/petrelfs/fanyuchen/Scaling-PRIME/Qwen/Qwen2.5-1.5B
CKPT_PATH=/mnt/petrelfs/fanyuchen/verl/ckpt

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.val_batch_size=1024 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 +trainer.val_before_train=False \
 reward_model.reward_manager=generative \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$YOUR_PROJECT_NAME \
 trainer.experiment_name=$YOUR_RUN_NAME \