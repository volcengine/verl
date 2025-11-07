"""
Example configuration for PPO training on MATH dataset with vLLM backend.

This file shows a complete configuration for PPO with a critic model.
"""

# Hardware auto-detected in notebook
HARDWARE_INFO = {
    'gpu_count': 8,
    'gpu_model': 'NVIDIA A100-SXM4-80GB',
    'supports_bf16': True,
}

# Backend selection
BACKEND = 'vllm'

# Cluster configuration (single node, 8 GPUs)
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 8,
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
}

# Backend-specific settings (auto-configured)
BACKEND_CONFIG = {
    'actor_rollout_ref.rollout.name': 'vllm',
    'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
    'actor_rollout_ref.rollout.enable_chunked_prefill': True,
    'actor_rollout_ref.rollout.max_num_batched_tokens': 8192,
    'actor_rollout_ref.rollout.enable_prefix_caching': True,
}

# Data configuration (merged GSM8K + MATH)
DATA_CONFIG = {
    'train_files': '["/home/user/data/gsm8k/train.parquet", "/home/user/data/math/train.parquet"]',
    'val_files': '["/home/user/data/gsm8k/test.parquet", "/home/user/data/math/test.parquet"]',
    'max_prompt_length': 1024,  # Longer for MATH problems
    'max_response_length': 512,
}

# Model configuration
MODEL_CONFIG = {
    'model_path': 'Qwen/Qwen2-7B-Instruct',
    'output_dir': './checkpoints/ppo_math_vllm',
}

# Training hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 1e-6,
    'total_epochs': 15,
    'save_freq': 20,
    'test_freq': 5,
    'project_name': 'verl_notebook_math',
    'experiment_name': 'qwen2_7b_ppo_vllm',
}

# PPO-specific settings
PPO_SPECIFIC = {
    # Critic configuration
    'critic.optim.lr': 1e-5,
    'critic.model.path': 'Qwen/Qwen2-7B-Instruct',  # Same as actor
    'critic.model.use_remove_padding': True,
    'critic.model.enable_gradient_checkpointing': True,
    'critic.ppo_micro_batch_size_per_gpu': 16,
    'critic.model.fsdp_config.param_offload': False,
    'critic.model.fsdp_config.optimizer_offload': False,

    # Optional: Reward model (disabled by default)
    'reward_model.enable': False,

    # Training
    'actor_rollout_ref.actor.use_kl_loss': False,
    'algorithm.use_kl_in_reward': False,
    'trainer.critic_warmup': 0,
}

# Expected results:
# - GSM8K: ~78-80%
# - MATH: ~40-45%
# - Training time: ~4-5 hours on 8x A100-80GB
#
# Note: PPO requires more memory than GRPO due to the critic model
