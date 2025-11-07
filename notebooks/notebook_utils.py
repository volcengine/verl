"""
Utility functions for verl Jupyter notebooks.

This module provides helper functions for:
- Hardware detection and capability checking
- Backend (vLLM/SGLang) configuration
- Automatic configuration optimization
- Progress monitoring and visualization
"""

import importlib
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any

import torch


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware and compute capabilities.

    Returns:
        Dict containing:
        - gpu_count: Number of GPUs
        - gpu_model: GPU model name
        - total_vram_gb: Total VRAM across all GPUs
        - vram_per_gpu_gb: VRAM per GPU
        - supports_bf16: Whether bf16 is supported
        - supports_fp16: Whether fp16 is supported
        - cuda_version: CUDA version
        - compute_capability: GPU compute capability
        - cpu_count: Number of CPU cores
        - has_nvlink: Whether NVLink is available
        - has_infiniband: Whether InfiniBand is available
    """
    hardware_info = {}

    # GPU Detection
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        hardware_info['gpu_count'] = gpu_count

        # Get GPU model name
        if gpu_count > 0:
            hardware_info['gpu_model'] = torch.cuda.get_device_name(0)

            # VRAM
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            hardware_info['vram_per_gpu_gb'] = round(vram_gb, 2)
            hardware_info['total_vram_gb'] = round(vram_gb * gpu_count, 2)

            # Compute capability
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
            hardware_info['compute_capability'] = compute_capability

            # bf16 support (requires compute capability >= 8.0 for Ampere+)
            hardware_info['supports_bf16'] = major >= 8
            hardware_info['supports_fp16'] = True  # All modern GPUs support fp16

            # CUDA version
            hardware_info['cuda_version'] = torch.version.cuda
        else:
            hardware_info['gpu_model'] = 'No GPU detected'
            hardware_info['vram_per_gpu_gb'] = 0
            hardware_info['total_vram_gb'] = 0
            hardware_info['supports_bf16'] = False
            hardware_info['supports_fp16'] = False
            hardware_info['compute_capability'] = 'N/A'
    else:
        hardware_info['gpu_count'] = 0
        hardware_info['gpu_model'] = 'No CUDA available'
        hardware_info['vram_per_gpu_gb'] = 0
        hardware_info['total_vram_gb'] = 0
        hardware_info['supports_bf16'] = False
        hardware_info['supports_fp16'] = False
        hardware_info['cuda_version'] = 'N/A'
        hardware_info['compute_capability'] = 'N/A'

    # CPU info
    hardware_info['cpu_count'] = os.cpu_count() or 1

    # Check for NVLink (simplified check)
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '--status'],
                              capture_output=True, text=True, timeout=5)
        hardware_info['has_nvlink'] = result.returncode == 0 and 'Active' in result.stdout
    except:
        hardware_info['has_nvlink'] = False

    # Check for InfiniBand (simplified check)
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, timeout=5)
        hardware_info['has_infiniband'] = result.returncode == 0 and 'State: Active' in result.stdout
    except:
        hardware_info['has_infiniband'] = False

    return hardware_info


def detect_available_backends() -> Dict[str, bool]:
    """
    Detect which inference backends are installed.

    Returns:
        Dict with backend names as keys and availability as bool values
    """
    backends = {
        'vllm': importlib.util.find_spec('vllm') is not None,
        'sglang': importlib.util.find_spec('sglang') is not None,
    }
    return backends


def get_backend_config(backend: str, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get optimized configuration for specified backend.

    Args:
        backend: 'vllm' or 'sglang'
        hardware_info: Hardware information from detect_hardware()

    Returns:
        Configuration dict for the backend
    """
    if backend not in ['vllm', 'sglang']:
        raise ValueError(f"Unknown backend: {backend}. Must be 'vllm' or 'sglang'")

    # Calculate optimal tensor parallel size
    gpu_count = hardware_info.get('gpu_count', 1)
    if gpu_count >= 8:
        tp_size = 4
    elif gpu_count >= 4:
        tp_size = 2
    else:
        tp_size = 1

    # Base configuration
    base_config = {
        'actor_rollout_ref.rollout.name': backend,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': tp_size,
    }

    # Backend-specific settings
    if backend == 'sglang':
        sglang_config = {
            'actor_rollout_ref.rollout.enable_flashinfer': True,
            'actor_rollout_ref.rollout.overlap_scheduler': True,
            'actor_rollout_ref.rollout.enable_torch_compile': False,  # Can be enabled for speedup
        }
        base_config.update(sglang_config)

    elif backend == 'vllm':
        vllm_config = {
            'actor_rollout_ref.rollout.enable_chunked_prefill': True,
            'actor_rollout_ref.rollout.max_num_batched_tokens': 8192,
            'actor_rollout_ref.rollout.enable_prefix_caching': True,
        }
        base_config.update(vllm_config)

    return base_config


def get_recommended_config(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get recommended training configuration based on hardware.

    Args:
        hardware_info: Hardware information from detect_hardware()

    Returns:
        Recommended configuration dict
    """
    config = {}

    gpu_count = hardware_info.get('gpu_count', 1)
    vram_per_gpu = hardware_info.get('vram_per_gpu_gb', 0)
    supports_bf16 = hardware_info.get('supports_bf16', False)

    # Data type
    if supports_bf16:
        config['dtype'] = 'bf16'
    else:
        config['dtype'] = 'fp16'

    # Batch sizes based on VRAM
    if vram_per_gpu >= 80:  # A100 80GB, H100
        config['ppo_micro_batch_size_per_gpu'] = 32
        config['log_prob_micro_batch_size_per_gpu'] = 32
        config['recommended_train_batch_size'] = 1024
    elif vram_per_gpu >= 40:  # A100 40GB
        config['ppo_micro_batch_size_per_gpu'] = 16
        config['log_prob_micro_batch_size_per_gpu'] = 16
        config['recommended_train_batch_size'] = 512
    elif vram_per_gpu >= 24:  # RTX 4090, A5000
        config['ppo_micro_batch_size_per_gpu'] = 8
        config['log_prob_micro_batch_size_per_gpu'] = 8
        config['recommended_train_batch_size'] = 256
    else:  # Smaller GPUs
        config['ppo_micro_batch_size_per_gpu'] = 4
        config['log_prob_micro_batch_size_per_gpu'] = 4
        config['recommended_train_batch_size'] = 128

    # Gradient checkpointing (save memory)
    config['enable_gradient_checkpointing'] = vram_per_gpu < 80

    # Offloading recommendations
    if vram_per_gpu < 40:
        config['param_offload'] = True
        config['optimizer_offload'] = True
    else:
        config['param_offload'] = False
        config['optimizer_offload'] = False

    return config


def get_cluster_template(mode: str, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get cluster configuration template.

    Args:
        mode: 'single_gpu', 'single_node_multi_gpu', or 'multi_node_multi_gpu'
        hardware_info: Hardware information from detect_hardware()

    Returns:
        Cluster configuration dict
    """
    gpu_count = hardware_info.get('gpu_count', 1)

    if mode == 'single_gpu':
        return {
            'trainer.n_gpus_per_node': 1,
            'trainer.nnodes': 1,
            'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        }

    elif mode == 'single_node_multi_gpu':
        # Use detected GPU count
        tp_size = min(2, gpu_count)  # Tensor parallel size
        return {
            'trainer.n_gpus_per_node': gpu_count,
            'trainer.nnodes': 1,
            'actor_rollout_ref.rollout.tensor_model_parallel_size': tp_size,
        }

    elif mode == 'multi_node_multi_gpu':
        # Template for multi-node (user needs to specify node count)
        return {
            'trainer.n_gpus_per_node': gpu_count,
            'trainer.nnodes': 2,  # User should edit this
            'actor_rollout_ref.rollout.tensor_model_parallel_size': 4,
            'ray_kwargs.ray_init.address': 'auto',  # User should set head node address
            # Example: 'ray_kwargs.ray_init.address': '192.168.1.100:6379'
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")


def print_hardware_summary(hardware_info: Dict[str, Any]) -> None:
    """Print a formatted summary of detected hardware."""
    print("=" * 70)
    print("HARDWARE DETECTION SUMMARY")
    print("=" * 70)
    print(f"GPU Count:          {hardware_info['gpu_count']}")
    print(f"GPU Model:          {hardware_info['gpu_model']}")
    print(f"VRAM per GPU:       {hardware_info['vram_per_gpu_gb']:.2f} GB")
    print(f"Total VRAM:         {hardware_info['total_vram_gb']:.2f} GB")
    print(f"CUDA Version:       {hardware_info['cuda_version']}")
    print(f"Compute Capability: {hardware_info['compute_capability']}")
    print(f"BF16 Support:       {'✅ Yes' if hardware_info['supports_bf16'] else '❌ No'}")
    print(f"FP16 Support:       {'✅ Yes' if hardware_info['supports_fp16'] else '❌ No'}")
    print(f"CPU Cores:          {hardware_info['cpu_count']}")
    print(f"NVLink:             {'✅ Detected' if hardware_info['has_nvlink'] else '❌ Not detected'}")
    print(f"InfiniBand:         {'✅ Detected' if hardware_info['has_infiniband'] else '❌ Not detected'}")
    print("=" * 70)


def print_backend_summary(backends: Dict[str, bool]) -> None:
    """Print a formatted summary of available backends."""
    print("=" * 70)
    print("AVAILABLE INFERENCE BACKENDS")
    print("=" * 70)
    for backend, available in backends.items():
        status = "✅ Installed" if available else "❌ Not installed"
        print(f"{backend.upper():10s}: {status}")
    print("=" * 70)

    # Installation hints
    if not any(backends.values()):
        print("\n⚠️  No inference backends detected!")
        print("Install one with:")
        print("  pip install verl[vllm]     # For vLLM")
        print("  pip install verl[sglang]   # For SGLang")
        print("  pip install verl[vllm,sglang]  # For both")
        print()


def create_config_dict(
    algorithm: str,
    model_path: str,
    train_files: str,
    val_files: str,
    backend_config: Dict[str, Any],
    cluster_config: Dict[str, Any],
    recommended_config: Dict[str, Any],
    **overrides
) -> Dict[str, Any]:
    """
    Create a complete configuration dictionary for training.

    Args:
        algorithm: 'grpo', 'gae' (for PPO), 'reinforce_plus_plus', etc.
        model_path: Path to model (HuggingFace or local)
        train_files: Path to training data
        val_files: Path to validation data
        backend_config: Backend configuration from get_backend_config()
        cluster_config: Cluster configuration from get_cluster_template()
        recommended_config: Recommended config from get_recommended_config()
        **overrides: Additional config overrides

    Returns:
        Complete configuration dict
    """
    config = {
        # Algorithm
        'algorithm.adv_estimator': algorithm,

        # Data
        'data.train_files': train_files,
        'data.val_files': val_files,
        'data.train_batch_size': recommended_config['recommended_train_batch_size'],
        'data.max_prompt_length': 512,
        'data.max_response_length': 1024,
        'data.filter_overlong_prompts': True,
        'data.truncation': 'error',

        # Model
        'actor_rollout_ref.model.path': model_path,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': recommended_config['enable_gradient_checkpointing'],

        # Training
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 256,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': recommended_config['ppo_micro_batch_size_per_gpu'],
        'actor_rollout_ref.actor.fsdp_config.param_offload': recommended_config['param_offload'],
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': recommended_config['optimizer_offload'],

        # Rollout
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': recommended_config['log_prob_micro_batch_size_per_gpu'],

        # Reference
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': recommended_config['log_prob_micro_batch_size_per_gpu'],
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,

        # Trainer
        'trainer.logger': '["console","wandb"]',
        'trainer.project_name': 'verl_notebook_training',
        'trainer.save_freq': 20,
        'trainer.test_freq': 5,
        'trainer.total_epochs': 15,

        # Merge cluster, backend, and overrides
        **cluster_config,
        **backend_config,
        **overrides,
    }

    return config


def check_verl_installation() -> Tuple[bool, str]:
    """
    Check if verl is installed and return version info.

    Returns:
        (is_installed, version_or_message)
    """
    try:
        import verl
        version_file = os.path.join(os.path.dirname(verl.__file__), 'version', 'version')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = f.read().strip()
        else:
            version = 'unknown'
        return True, version
    except ImportError:
        return False, "verl is not installed. Run: pip install verl[vllm] or pip install verl[sglang]"


def check_dependencies() -> Dict[str, str]:
    """
    Check installation status of key dependencies.

    Returns:
        Dict of package: status
    """
    packages = ['torch', 'transformers', 'ray', 'hydra-core', 'vllm', 'sglang', 'wandb']
    status = {}

    for package in packages:
        spec = importlib.util.find_spec(package.replace('-', '_'))
        if spec is not None:
            try:
                mod = importlib.import_module(package.replace('-', '_'))
                version = getattr(mod, '__version__', 'unknown')
                status[package] = f"✅ {version}"
            except:
                status[package] = "✅ Installed (version unknown)"
        else:
            status[package] = "❌ Not installed"

    return status
