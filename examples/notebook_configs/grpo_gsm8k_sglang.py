"""
Example configuration for GRPO training on GSM8K with SGLang backend.

This file shows a complete configuration example that you can copy into the notebook.
"""

# Hardware auto-detected in notebook
HARDWARE_INFO = {
    'gpu_count': 8,
    'gpu_model': 'NVIDIA A100-SXM4-80GB',
    'supports_bf16': True,
}

# Backend selection
BACKEND = 'sglang'

# Cluster configuration (single node, 8 GPUs)
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 8,
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
}

# Backend-specific settings (auto-configured)
BACKEND_CONFIG = {
    'actor_rollout_ref.rollout.name': 'sglang',
    'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
    'actor_rollout_ref.rollout.enable_flashinfer': True,
    'actor_rollout_ref.rollout.overlap_scheduler': True,
}

# Data configuration
DATA_CONFIG = {
    'train_files': '/home/user/data/gsm8k/train.parquet',
    'val_files': '/home/user/data/gsm8k/test.parquet',
    'max_prompt_length': 512,
    'max_response_length': 1024,
}

# Model configuration
MODEL_CONFIG = {
    'model_path': 'Qwen/Qwen3-8B',
    'output_dir': './checkpoints/grpo_gsm8k_sglang',
}

# Training hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 1e-6,
    'total_epochs': 15,
    'save_freq': 20,
    'test_freq': 5,
    'project_name': 'verl_notebook_gsm8k',
    'experiment_name': 'qwen3_8b_grpo_sglang',
}

# GRPO-specific settings
GRPO_SPECIFIC = {
    'actor_rollout_ref.rollout.n': 5,  # Sample 5 responses per prompt
    'actor_rollout_ref.actor.use_kl_loss': True,
    'actor_rollout_ref.actor.kl_loss_coef': 0.001,
    'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
    'actor_rollout_ref.actor.entropy_coeff': 0,
    'algorithm.use_kl_in_reward': False,
    'trainer.critic_warmup': 0,
}

# Expected results on GSM8K test set:
# - Baseline (Qwen3-8B): ~75%
# - After GRPO training: ~77-78%
# - Training time: ~3-4 hours on 8x A100-80GB
