# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepSpeed utilities for VERL framework.
"""

import logging
import os
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logger.warning("DeepSpeed not available")


def get_deepspeed_config(
    optimizer_type: str = "AdamW",
    train_batch_size: int = 1,
    train_micro_batch_size_per_gpu: int = 1,
    gradient_accumulation_steps: int = 1,
    zero_stage: int = 2,
    lr: float = 1e-5,
    betas: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    warmup_min_lr: float = 0,
    warmup_max_lr: float = 1e-5,
    warmup_num_steps: int = 100,
    fp16_enabled: bool = True,
    bf16_enabled: bool = False,
    cpu_offload: bool = False,
    offload_optimizer: bool = False,
    disable_scheduler: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """
    Generate DeepSpeed configuration dictionary.

    Args:
        optimizer_type: Type of optimizer (e.g., "Adam", "AdamW")
        train_batch_size: Total batch size for training
        train_micro_batch_size_per_gpu: Micro batch size per GPU
        gradient_accumulation_steps: Number of gradient accumulation steps
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        lr: Learning rate
        betas: Adam optimizer betas
        eps: Adam optimizer epsilon
        weight_decay: Weight decay
        warmup_min_lr: Minimum learning rate for warmup
        warmup_max_lr: Maximum learning rate for warmup
        warmup_num_steps: Number of warmup steps
        fp16_enabled: Whether to enable FP16
        bf16_enabled: Whether to enable BF16
        cpu_offload: Whether to offload parameters to CPU
        offload_optimizer: Whether to offload optimizer to CPU
        disable_scheduler: If True, removes the scheduler from the config
        **kwargs: Additional configuration options

    Returns:
        DeepSpeed configuration dictionary
    """
    if betas is None:
        betas = (0.9, 0.999)
    config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": optimizer_type,
            "params": {
                "lr": lr,
                "betas": list(betas),
                "weight_decay": weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": warmup_min_lr,
                "warmup_max_lr": warmup_max_lr,
                "warmup_num_steps": warmup_num_steps,
            },
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e7,  # Match test script: 20M instead of 200M
            "overlap_comm": False,  # Match test script: False instead of True
            "reduce_scatter": True,
            "reduce_bucket_size": 2e7,  # Match test script: 20M instead of 200M
            "contiguous_gradients": True,
        },
        "steps_per_print": 1,  # Match test script configuration
    }

    if disable_scheduler:
        del config["scheduler"]

    # Configure precision
    if bf16_enabled:
        config["bf16"] = {"enabled": True}
    elif fp16_enabled:
        config["fp16"] = {"enabled": True}

    # Configure optimizer offloading for ZeRO-2/3
    if zero_stage >= 2 and offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    # Configure parameter offloading for ZeRO-3
    if zero_stage >= 3 and cpu_offload:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        # Add stage3_prefetch_bucket_size to match test script configuration
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 2e7

    # Apply additional configuration from kwargs, allowing overrides
    # This is useful for setting specific ZeRO params like overlap_comm
    if "zero_optimization" in kwargs:
        # Deep merge the zero_optimization dictionary
        for key, value in kwargs["zero_optimization"].items():
            config["zero_optimization"][key] = value
        del kwargs["zero_optimization"]

    config.update(kwargs)

    return config


def initialize_deepspeed_engine(
    model: torch.nn.Module,
    config: dict[str, Any],
    model_parameters: Optional[Any] = None,
    training_data: Optional[Any] = None,
    collate_fn: Optional[Any] = None,
    mpu: Optional[Any] = None,
    dist_init_required: Optional[bool] = None,
    config_params: Optional[str] = None,
) -> tuple:
    """
    Initialize DeepSpeed engine.

    Args:
        model: PyTorch model to wrap with DeepSpeed
        config: DeepSpeed configuration dictionary
        model_parameters: Model parameters (defaults to model.parameters())
        training_data: Training data loader
        collate_fn: Data collation function
        mpu: Model parallel unit
        dist_init_required: Whether distributed initialization is required
        config_params: Configuration parameters string

    Returns:
        Tuple of (engine, optimizer, training_dataloader, lr_scheduler)
    """
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not available. Please install DeepSpeed.")

    if model_parameters is None:
        model_parameters = model.parameters()

    # 构建参数字典，只包含有效参数
    init_kwargs = {
        "model": model,
        "config": config,
        "model_parameters": model_parameters,
    }

    # 只添加非None的可选参数
    if training_data is not None:
        init_kwargs["training_data"] = training_data
    if collate_fn is not None:
        init_kwargs["collate_fn"] = collate_fn
    if mpu is not None:
        init_kwargs["mpu"] = mpu
    if dist_init_required is not None:
        init_kwargs["dist_init_required"] = dist_init_required
    if config_params is not None:
        init_kwargs["config_params"] = config_params

    return deepspeed.initialize(**init_kwargs)


def load_deepspeed_model_to_gpu(engine: DeepSpeedEngine):
    """
    Load DeepSpeed model parameters to GPU.

    Args:
        engine: DeepSpeed engine
    """
    try:
        # For ZeRO-3, parameters are automatically managed by DeepSpeed
        # We should not manually move them as it can cause conflicts
        if hasattr(engine, "_config") and engine._config is not None:
            try:
                zero_config = getattr(engine._config, "zero_optimization", None)
                if zero_config is not None and getattr(zero_config, "stage", 0) == 3:
                    logger.debug("ZeRO-3 detected, skipping manual GPU loading")
                    return
            except AttributeError:
                # If we can't access the config, continue with safe loading
                pass

        # Use __dict__ to avoid triggering __getattr__ recursion
        if "module" in engine.__dict__ and engine.__dict__["module"] is not None:
            module = engine.__dict__["module"]
            # Check if any parameters exist and if they're on CPU
            try:
                param_list = list(module.parameters())
                if param_list and param_list[0].device.type == "cpu":
                    logger.debug("Moving DeepSpeed model from CPU to GPU")
                    engine.__dict__["module"] = module.cuda()
                else:
                    logger.debug("DeepSpeed model already on GPU or no parameters found")
            except (RuntimeError, StopIteration, AttributeError) as e:
                logger.debug(f"Could not check/move parameters: {e}")
        else:
            logger.debug("No module found in DeepSpeed engine")

    except Exception as e:
        logger.warning(f"Error in load_deepspeed_model_to_gpu: {e}")

    logger.debug("DeepSpeed model GPU loading completed")


def offload_deepspeed_model_to_cpu(engine: DeepSpeedEngine):
    """
    Offload DeepSpeed model parameters to CPU.

    Args:
        engine: DeepSpeed engine
    """
    try:
        # For ZeRO-3, parameters are automatically managed by DeepSpeed
        if hasattr(engine, "_config") and engine._config is not None:
            try:
                zero_config = getattr(engine._config, "zero_optimization", None)
                if zero_config is not None and getattr(zero_config, "stage", 0) == 3:
                    logger.debug("ZeRO-3 detected, skipping manual CPU offloading")
                    return
            except AttributeError:
                pass

        # Use __dict__ to avoid triggering __getattr__ recursion
        if "module" in engine.__dict__ and engine.__dict__["module"] is not None:
            module = engine.__dict__["module"]
            try:
                param_list = list(module.parameters())
                if param_list and param_list[0].device.type != "cpu":
                    logger.debug("Moving DeepSpeed model from GPU to CPU")
                    engine.__dict__["module"] = module.cpu()
                else:
                    logger.debug("DeepSpeed model already on CPU or no parameters found")
            except (RuntimeError, StopIteration, AttributeError) as e:
                logger.debug(f"Could not check/move parameters: {e}")
        else:
            logger.debug("No module found in DeepSpeed engine")

    except Exception as e:
        logger.warning(f"Error in offload_deepspeed_model_to_cpu: {e}")

    logger.debug("DeepSpeed model CPU offloading completed")


def get_deepspeed_memory_stats(engine: DeepSpeedEngine) -> dict[str, Any]:
    """
    Get memory statistics for DeepSpeed engine.

    Args:
        engine: DeepSpeed engine

    Returns:
        Dictionary with memory statistics
    """
    stats = {}

    if hasattr(engine, "zero_optimization_partition_weights"):
        stats["zero_stage"] = getattr(engine, "_config", {}).get("zero_optimization", {}).get("stage", 0)

    if torch.cuda.is_available():
        stats.update(
            {
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_reserved": torch.cuda.memory_reserved(),
                "cuda_max_allocated": torch.cuda.max_memory_allocated(),
            }
        )

    return stats


def save_deepspeed_checkpoint(
    engine: DeepSpeedEngine, save_dir: str, client_state: Optional[dict] = None, tag: Optional[str] = None
):
    """
    Save DeepSpeed checkpoint.

    Args:
        engine: DeepSpeed engine
        save_dir: Directory to save checkpoint
        client_state: Additional client state to save
        tag: Checkpoint tag
    """
    engine.save_checkpoint(save_dir, client_state=client_state, tag=tag)
    logger.info(f"DeepSpeed checkpoint saved to {save_dir}")


def load_deepspeed_checkpoint(
    engine: DeepSpeedEngine,
    load_dir: str,
    tag: Optional[str] = None,
    load_module_strict: bool = True,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
):
    """
    Load DeepSpeed checkpoint.

    Args:
        engine: DeepSpeed engine
        load_dir: Directory to load checkpoint from
        tag: Checkpoint tag
        load_module_strict: Whether to load module strictly
        load_optimizer_states: Whether to load optimizer states
        load_lr_scheduler_states: Whether to load learning rate scheduler states
    """
    _, client_state = engine.load_checkpoint(
        load_dir,
        tag=tag,
        load_module_strict=load_module_strict,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )
    logger.info(f"DeepSpeed checkpoint loaded from {load_dir}")
    return client_state


def zero_stage_3_gather_16bit_weights_on_model_save(engine: DeepSpeedEngine):
    """
    Context manager for gathering 16-bit weights in ZeRO-3.

    Args:
        engine: DeepSpeed engine
    """
    if hasattr(engine, "gather_16bit_weights_on_model_save"):
        return engine.gather_16bit_weights_on_model_save()
    else:
        # Fallback for older DeepSpeed versions
        from contextlib import nullcontext

        return nullcontext()


def get_deepspeed_config_from_args(args) -> dict[str, Any]:
    """
    Create DeepSpeed config from command line arguments or config object.

    Args:
        args: Arguments object with DeepSpeed configuration

    Returns:
        DeepSpeed configuration dictionary
    """
    # This can be extended to parse from argparse.Namespace or similar
    config = get_deepspeed_config()

    # Override with any provided arguments
    if hasattr(args, "zero_stage"):
        config["zero_optimization"]["stage"] = args.zero_stage
    if hasattr(args, "train_batch_size"):
        config["train_batch_size"] = args.train_batch_size
    if hasattr(args, "learning_rate"):
        config["optimizer"]["params"]["lr"] = args.learning_rate

    return config


def is_deepspeed_zero_3(engine: DeepSpeedEngine) -> bool:
    """
    Check if DeepSpeed engine is using ZeRO-3.

    Args:
        engine: DeepSpeed engine

    Returns:
        True if using ZeRO-3, False otherwise
    """
    if hasattr(engine, "_config"):
        zero_config = engine._config.get("zero_optimization", {})
        return zero_config.get("stage", 0) == 3
    return False


def deepspeed_gather_params(engine: DeepSpeedEngine, gather_16bit: bool = True):
    """
    Gather parameters for ZeRO-2/3 optimization.

    Args:
        engine: DeepSpeed engine
        gather_16bit: Whether to gather 16-bit weights
    """
    if is_deepspeed_zero_3(engine) and gather_16bit:
        return zero_stage_3_gather_16bit_weights_on_model_save(engine)
    else:
        from contextlib import nullcontext

        return nullcontext()
