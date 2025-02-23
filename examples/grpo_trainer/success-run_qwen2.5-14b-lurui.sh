# cd /workspace/lurui/verl
# pip3 install -e .
# hostip=$(env | grep MLP_HOST=)
# hostip=${hostip#*=}
# echo $hostip
# pkill -9 ray
# pkill -9 python
# pkill -9 gcs_server
# if [ $hostip  = $MLP_WORKER_0_HOST ]; then
#     # 主节点
#     ray start --head --node-ip-address=$MLP_WORKER_0_HOST --port=6379 --block &
# else
#     # 其余节点
#     sleep 50
#     ray start --address $MLP_WORKER_0_HOST:6379 --block &
# fi
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/lurui-yun/deep_research/verl/datasets/test.parquet \
    data.val_files=/workspace/lurui-yun/deep_research/verl/datasets/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=/data/o1-cloud/checkpoints/rl/qwen-14b-general/epoch_1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    +trainer.val_before_train=False \
    trainer.total_epochs=15 $@