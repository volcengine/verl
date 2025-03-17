set -ex
source /map-vepfs/miniconda3/bin/activate verl
export PYTHONPATH=/map-vepfs/tianshun/verl
export WANDB_API_KEY=3f0344c93ff3094c21bab255ba9f0ce01b6427da
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=2

cd /map-vepfs/tianshun/verl
base_dir=$(pwd)
gsm8k_train_path="${base_dir}/data/gsm8k/train.parquet"
gsm8k_test_path="${base_dir}/data/gsm8k/test.parquet"

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

model_path=$1
exp_name=$2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_dag' \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15


# bash /map-vepfs/tianshun/verl/xrpo_script/qwen2.5-7b.sh /map-vepfs/huggingface/models/Qwen/Qwen2.5-Math-1.5B "test1"

# python3 -m debugpy --listen 58702 --wait-for-client verl/trainer/main_ppo.py \
#     algorithm.adv_estimator=grpo \
#     data.train_files="/map-vepfs/tianshun/verl/data/gsm8k/train.parquet" \
#     data.val_files="/map-vepfs/tianshun/verl/data/gsm8k/test.parquet" \
#     data.train_batch_size=128 \
#     data.val_batch_size=32 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=1024 \
#     actor_rollout_ref.model.path="/map-vepfs/tianyu/models/DAG/dag_qwen_sft_v1/checkpoint-1752" \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.grad_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='verl_grpo_dag' \
#     trainer.experiment_name="dag_debug" \
#     trainer.n_gpus_per_node=2 \
#     trainer.nnodes=1 \
#     trainer.save_freq=10 \
#     trainer.test_freq=-1 \
#     trainer.total_epochs=15


# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files="/map-vepfs/tianshun/verl/data/gsm8k/train.parquet" \
#     data.val_files="/map-vepfs/tianshun/verl/data/gsm8k/test.parquet" \
#     data.train_batch_size=128 \
#     data.val_batch_size=32 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=1024 \
#     actor_rollout_ref.model.path="/map-vepfs/huggingface/models/Qwen/Qwen2.5-Math-1.5B" \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.grad_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='verl_grpo_dag' \
#     trainer.experiment_name="dag_debug" \
#     trainer.n_gpus_per_node=4 \
#     trainer.nnodes=1 \
#     trainer.save_freq=10 \
#     trainer.test_freq=-1 \
#     trainer.total_epochs=15
