"""
Example configuration for GRPO training on a single GPU (for testing).

This configuration is optimized for quick experimentation on limited hardware.
"""

# Hardware configuration (single RTX 4090 or similar)
HARDWARE_INFO = {
    'gpu_count': 1,
    'gpu_model': 'NVIDIA GeForce RTX 4090',
    'vram_per_gpu_gb': 24,
    'supports_bf16': True,
}

# Backend selection (vLLM recommended for stability)
BACKEND = 'vllm'

# Cluster configuration (single GPU)
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 1,
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
}

# Backend-specific settings
BACKEND_CONFIG = {
    'actor_rollout_ref.rollout.name': 'vllm',
    'actor_rollout_ref.rollout.gpu_memory_utilization': 0.5,  # Conservative for 24GB
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
    'actor_rollout_ref.rollout.enable_chunked_prefill': True,
    'actor_rollout_ref.rollout.max_num_batched_tokens': 4096,
}

# Data configuration (small subset for testing)
DATA_CONFIG = {
    'train_files': '/home/user/data/gsm8k/train.parquet',
    'val_files': '/home/user/data/gsm8k/test.parquet',
    'max_prompt_length': 512,
    'max_response_length': 512,  # Shorter to save memory
}

# Model configuration (smaller model)
MODEL_CONFIG = {
    'model_path': 'Qwen/Qwen2.5-3B',  # 3B model fits better on single GPU
    'output_dir': './checkpoints/grpo_single_gpu',
}

# Training hyperparameters (adjusted for single GPU)
TRAINING_CONFIG = {
    'learning_rate': 1e-6,
    'total_epochs': 10,  # Fewer epochs for testing
    'save_freq': 50,
    'test_freq': 10,
    'project_name': 'verl_notebook_single_gpu',
    'experiment_name': 'qwen2_5_3b_grpo',
}

# GRPO-specific settings (memory-optimized)
GRPO_SPECIFIC = {
    'actor_rollout_ref.rollout.n': 3,  # Fewer samples to save memory
    'actor_rollout_ref.actor.use_kl_loss': True,
    'actor_rollout_ref.actor.kl_loss_coef': 0.001,
    'actor_rollout_ref.actor.ppo_mini_batch_size': 64,  # Smaller batches
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 4,  # Small micro batch
    'actor_rollout_ref.actor.fsdp_config.param_offload': True,  # Enable offloading
    'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
    'data.train_batch_size': 256,  # Smaller overall batch
    'algorithm.use_kl_in_reward': False,
    'trainer.critic_warmup': 0,
}

# Expected results:
# - Training time: ~2-3 hours on single RTX 4090
# - Memory usage: ~20-22GB
# - GSM8K accuracy: ~72-74% (smaller model baseline)
#
# Tips for single GPU:
# 1. Use smaller models (3B-7B range)
# 2. Enable parameter offloading
# 3. Reduce batch sizes and sample count
# 4. Consider using LoRA for even lower memory
