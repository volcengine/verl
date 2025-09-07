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
The concrete Engine implementation using DeepSpeed ZeRO optimization
"""

import gc
import itertools
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.deepspeed_utils import (
    DEEPSPEED_AVAILABLE,
    get_deepspeed_config,
    initialize_deepspeed_engine,
    load_deepspeed_checkpoint,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
    save_deepspeed_checkpoint,
)
from verl.utils.device import (
    get_device_id,
    get_device_name,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.py_functional import append_to_dict, convert_to_regular_types
from verl.utils.seqlen_balancing import get_reverse_idx, prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    DeepSpeedEngineConfig,
    DeepSpeedOptimizerConfig,
    HFModelConfig,
)

from ..base import BaseEngine, EngineRegistry

logger = logging.getLogger("verl.engine.deepspeed")
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

if not DEEPSPEED_AVAILABLE:
    logger.error("DeepSpeed is not available. Please install DeepSpeed to use DeepSpeedEngine.")


from verl.utils.checkpoint.deepspeed_checkpoint_manager import DeepSpeedCheckpointManager  # noqa: E402


class DeepSpeedEngine(BaseEngine):
    """
    DeepSpeed-based training engine with ZeRO optimization for memory-efficient large model training.

    This engine provides a complete implementation of DeepSpeed ZeRO optimization stages
    for distributed training of large language models. It supports advanced features
    including parameter sharding, gradient accumulation, mixed precision training,
    and optional LoRA fine-tuning.

    Key Features:
        - ZeRO Stages 0-3: Progressive parameter, gradient, and optimizer state sharding
        - Mixed Precision: FP16/BF16 training with automatic loss scaling
        - Parameter Offloading: CPU/NVMe offloading for memory efficiency
        - LoRA Support: Low-rank adaptation for parameter-efficient fine-tuning
        - Ulysses Sequence Parallelism: Long sequence processing with attention parallelism
        - Remove Padding: Optimized attention computation for variable-length sequences
        - Gradient Checkpointing: Memory-time tradeoff for deeper models

    ZeRO Optimization Stages:
        - Stage 0: No sharding (baseline distributed training)
        - Stage 1: Optimizer state sharding across data parallel ranks
        - Stage 2: Optimizer + gradient sharding for reduced memory
        - Stage 3: Full parameter + optimizer + gradient sharding (maximum memory efficiency)

    Memory Management:
        - Automatic parameter gathering/scattering during forward/backward passes
        - CPU offloading for parameters and optimizer states
        - Dynamic memory allocation based on computation requirements
        - Gradient accumulation to simulate larger batch sizes

    Integration:
        - Compatible with HuggingFace transformers models
        - Supports custom model architectures with monkey patching
        - Integrates with VERL's training pipeline and data protocols
        - Provides checkpoint saving/loading with DeepSpeed native format

    Example:
        >>> model_config = HFModelConfig(path="gpt2", lora_rank=16)
        >>> engine_config = DeepSpeedEngineConfig(mixed_precision="bf16")
        >>> optimizer_config = DeepSpeedOptimizerConfig(lr=1e-5)
        >>>
        >>> engine = DeepSpeedEngine(
        ...     model_config=model_config,
        ...     engine_config=engine_config,
        ...     optimizer_config=optimizer_config,
        ...     checkpoint_config=CheckpointConfig(),
        ...     zero_stage=2,
        ...     gradient_accumulation_steps=4
        ... )
        >>> engine.initialize()
        >>> # Ready for training with ZeRO Stage 2 optimization

    Note:
        This engine requires DeepSpeed to be installed and properly configured.
        Distributed training requires proper process group initialization.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: DeepSpeedEngineConfig,
        optimizer_config: DeepSpeedOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        zero_stage: int = 2,
        gradient_accumulation_steps: int = 1,
        train_batch_size: Optional[int] = None,
        train_micro_batch_size_per_gpu: Optional[int] = None,
    ):
        """
        Initialize the DeepSpeed engine with comprehensive configuration.

        Sets up distributed training environment, configures DeepSpeed ZeRO optimization,
        and prepares the engine for memory-efficient large model training. The initialization
        handles device mesh setup, batch size calculations, and optimization strategies.

        Args:
            model_config (HFModelConfig): HuggingFace model configuration containing:
                - path: Model path or HuggingFace model identifier
                - lora_rank: LoRA rank (>0 enables LoRA fine-tuning)
                - use_remove_padding: Enable padding removal optimization
                - enable_gradient_checkpointing: Trade memory for computation
                - trust_remote_code: Allow remote code execution
                - External library configurations and model-specific settings

            engine_config (DeepSpeedEngineConfig): DeepSpeed engine configuration containing:
                - mixed_precision: Mixed precision settings ("fp16", "bf16", or dict)
                - param_offload: Enable parameter offloading to CPU/NVMe
                - optimizer_offload: Enable optimizer state offloading
                - ulysses_sequence_parallel_size: Sequence parallelism configuration
                - entropy_from_logits_with_chunking: Memory-efficient entropy computation
                - use_torch_compile: Enable PyTorch 2.0 compilation
                - forward_only: Inference-only mode configuration

            optimizer_config (DeepSpeedOptimizerConfig): Optimizer configuration containing:
                - lr: Learning rate for training
                - betas: Adam optimizer beta parameters
                - eps: Numerical stability epsilon
                - weight_decay: L2 regularization strength

            checkpoint_config (CheckpointConfig): Checkpoint management configuration
                for saving and loading model states during training.

            zero_stage (int): DeepSpeed ZeRO optimization stage. Options:
                - 0: No sharding (standard data parallelism)
                - 1: Optimizer state sharding only
                - 2: Optimizer + gradient sharding (recommended for most cases)
                - 3: Full parameter + optimizer + gradient sharding (maximum memory savings)

            gradient_accumulation_steps (int): Number of forward passes before optimizer step.
                Simulates larger batch sizes without increasing memory usage proportionally.
                Effective batch size = micro_batch_size * world_size * gradient_accumulation_steps.

            train_batch_size (Optional[int]): Global training batch size across all devices.
                If None, automatically calculated as gradient_accumulation_steps * world_size.

            train_micro_batch_size_per_gpu (Optional[int]): Micro batch size per GPU device.
                Controls memory usage per device. If None, defaults to 1.

        Raises:
            ImportError: If DeepSpeed is not installed or available
            ValueError: If configuration parameters are incompatible
            RuntimeError: If distributed environment setup fails

        Side Effects:
            - Initializes distributed training environment if not already done
            - Sets up device mesh for Ulysses sequence parallelism
            - Configures entropy computation optimization
            - Establishes batch size relationships and memory management

        Memory Considerations:
            - Higher ZeRO stages reduce memory usage but may increase communication overhead
            - Parameter/optimizer offloading trades GPU memory for CPU memory and I/O
            - Gradient accumulation allows larger effective batch sizes with fixed memory
            - Micro batch size directly controls peak GPU memory usage

        Example:
            >>> # Memory-efficient configuration for large models
            >>> engine = DeepSpeedEngine(
            ...     model_config=model_config,
            ...     engine_config=engine_config,
            ...     optimizer_config=optimizer_config,
            ...     checkpoint_config=checkpoint_config,
            ...     zero_stage=3,  # Maximum memory efficiency
            ...     gradient_accumulation_steps=8,  # Large effective batch
            ...     train_micro_batch_size_per_gpu=1  # Minimal memory per GPU
            ... )
        """
        super().__init__()

        # Validate DeepSpeed availability early to provide clear error messages
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available. Please install DeepSpeed.")

        # Store configuration objects for later use during model building
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.zero_stage = zero_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize training mode state (None = not initialized, "train"/"eval" = active mode)
        self.mode = None

        # Set up distributed training environment - handle both distributed and single-process scenarios
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            # Allow single-process usage for unit tests and debugging
            self.rank = 0
            self.world_size = 1

        # Extract DeepSpeed-specific optimization flags from engine configuration
        # These control memory management strategies and training optimizations
        self._is_offload_param = engine_config.param_offload  # CPU offloading for parameters
        self._is_offload_optimizer = engine_config.optimizer_offload  # CPU offloading for optimizer states
        self._is_lora = self.model_config.lora_rank > 0  # Low-rank adaptation fine-tuning

        # Extract sequence processing optimization settings
        self.use_remove_padding = self.model_config.use_remove_padding

        # Initialize device mesh for Ulysses Sequence Parallel
        # This enables processing of very long sequences by parallelizing across the sequence dimension
        if self.engine_config.ulysses_sequence_parallel_size > 1:
            assert self.world_size % self.engine_config.ulysses_sequence_parallel_size == 0, (
                f"world_size={self.world_size} must be divisible by ulysses_sequence_parallel_size="
                f"{self.engine_config.ulysses_sequence_parallel_size}"
            )
            assert self.engine_config.ulysses_sequence_parallel_size <= self.world_size, (
                "ulysses_sequence_parallel_size cannot exceed world_size. "
                f"Got sp={self.engine_config.ulysses_sequence_parallel_size}, world_size={self.world_size}"
            )
        self._init_device_mesh()

        # Calculate effective batch sizes for distributed training
        # Effective global batch size should always satisfy:
        #   train_batch_size == train_micro_batch_size_per_gpu * world_size * gradient_accumulation_steps
        if train_micro_batch_size_per_gpu is None:
            train_micro_batch_size_per_gpu = 1  # minimal default

        if train_batch_size is None:
            train_batch_size = train_micro_batch_size_per_gpu * self.world_size * gradient_accumulation_steps
        else:
            expected_global = train_micro_batch_size_per_gpu * self.world_size * gradient_accumulation_steps
            if train_batch_size != expected_global:
                # Prefer being strict to avoid silent mismatch causing wrong lr scaling.
                raise ValueError(
                    "Inconsistent batch size settings: expected train_batch_size == micro_batch_size_per_gpu * "
                    f"world_size * gradient_accumulation_steps = {train_micro_batch_size_per_gpu} * "
                    f"{self.world_size} * {gradient_accumulation_steps} = {expected_global}, but got "
                    f"train_batch_size={train_batch_size}. Either pass a consistent train_batch_size or set it to None."
                )

        self.train_batch_size = train_batch_size
        self.train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu

        # Configure entropy computation for policy training (PPO)
        # Entropy computation can be memory-intensive, so we provide chunked and compiled options
        if self.engine_config.entropy_from_logits_with_chunking:
            # Memory-efficient chunked computation for large vocabularies
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            # Standard entropy computation
            entropy_from_logits = verl_F.entropy_from_logits

        # Optionally compile entropy computation for performance (PyTorch 2.0+)
        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

        if self.rank == 0:
            try:
                mp_mode = engine_config.mixed_precision if engine_config.mixed_precision else "fp32"
                logger.info(
                    "[DeepSpeedEngine:init] zero=%s acc=%s mbsz=%s gbsz=%s world=%s mp=%s "
                    "offload(p=%s,o=%s) lora=%s sp=%s",
                    zero_stage,
                    gradient_accumulation_steps,
                    train_micro_batch_size_per_gpu if train_micro_batch_size_per_gpu is not None else 1,
                    train_batch_size if train_batch_size is not None else "auto",
                    self.world_size,
                    mp_mode,
                    self._is_offload_param,
                    self._is_offload_optimizer,
                    self._is_lora,
                    self.engine_config.ulysses_sequence_parallel_size,
                )
            except Exception:  # noqa: BLE001
                pass

    def is_collect(self):
        """
        Check if this rank should collect data for Ulysses Sequence Parallelism.

        In Ulysses SP, data collection is typically done by the first rank in each
        sequence parallel group to avoid redundant operations and communication overhead.

        Returns:
            bool: True if this rank should perform data collection operations,
                  False if data collection should be delegated to another rank.

        Note:
            When Ulysses SP is not enabled, all ranks return True (standard behavior).
        """
        if self.ulysses_device_mesh is not None:
            # In sequence parallel mode, only rank 0 in each SP group collects data
            return self.ulysses_device_mesh["sp"].get_local_rank() == 0
        # Standard mode: all ranks collect data
        return True

    def initialize(self):
        """
        Initialize the complete DeepSpeed training environment.

        This method orchestrates the full setup process including model loading,
        DeepSpeed engine initialization, memory management configuration, and
        checkpoint manager setup. The initialization follows a specific order
        to ensure proper distributed training setup.

        Initialization Steps:
            1. Import external libraries (custom kernels, optimizations)
            2. Build model architecture and apply optimizations
            3. Initialize DeepSpeed engine with ZeRO optimization
            4. Configure parameter/optimizer offloading if enabled
            5. Set up checkpoint management system

        Memory Management:
            - Parameters are moved to CPU if param_offload is enabled
            - Optimizer states are handled by DeepSpeed internal offloading
            - GPU memory usage is logged for monitoring

        Side Effects:
            - Creates self.engine (DeepSpeed engine instance)
            - Creates self.module (underlying PyTorch model)
            - Creates self.optimizer (DeepSpeed-managed optimizer)
            - Creates self.lr_scheduler (learning rate scheduler)
            - Creates self.checkpoint_manager (checkpoint management)
            - Allocates GPU/CPU memory for model and optimizer states
            - Initializes distributed process groups if needed

        Raises:
            ImportError: If required external libraries cannot be imported
            RuntimeError: If DeepSpeed initialization fails
            torch.cuda.OutOfMemoryError: If insufficient GPU memory

        Example:
            >>> engine = DeepSpeedEngine(model_config, engine_config, ...)
            >>> engine.initialize()  # Full setup
            >>> # Engine is now ready for training
            >>> with engine.train_mode():
            ...     loss = engine.forward_backward_batch(data, loss_fn)
        """
        # Step 1: Import external libraries for custom optimizations
        # This may include custom CUDA kernels, optimized attention implementations, etc.
        import_external_libs(self.model_config.external_lib)

        # Step 2: Build model architecture and initialize DeepSpeed engine
        # This is the core setup that creates the model, optimizer, and DeepSpeed wrapper
        self._build_model_optimizer()

        # Step 3: Handle parameter offloading for memory efficiency
        if self._is_offload_param and self.zero_stage < 3:
            # Only manually offload for ZeRO stages < 3
            # ZeRO-3 handles parameter management automatically
            offload_deepspeed_model_to_cpu(self.engine)
            log_gpu_memory_usage("After offload model during init", logger=logger)

        # Note: Optimizer state offloading is handled automatically by DeepSpeed
        # configuration and doesn't require explicit management here

        # Step 4: Initialize checkpoint management system
        # This sets up the infrastructure for saving and loading model states
        self.checkpoint_manager = self._create_checkpoint_manager()

    def _init_device_mesh(self):
        """
        Initialize device mesh for Ulysses Sequence Parallelism.

        Ulysses SP enables processing of very long sequences by parallelizing
        computation across the sequence dimension. This method sets up the
        device mesh that coordinates data parallel and sequence parallel ranks.

        Device Mesh Configuration:
            - Data Parallel (DP): Traditional data parallelism across samples
            - Sequence Parallel (SP): Parallelism across sequence tokens
            - Total devices = DP size × SP size = world_size

        Memory Benefits:
            - Reduces memory per GPU for long sequences
            - Enables training with sequences longer than single-GPU memory
            - Maintains computational efficiency through optimized communication

        Side Effects:
            - Sets self.ulysses_device_mesh: Device mesh for distributed operations
            - Sets self.ulysses_sequence_parallel_size: Number of sequence parallel ranks
            - Sets self.ulysses_sharding_manager: Manages parameter distribution
            - Sets self.use_ulysses_sp: Boolean flag for sequence parallelism

        Note:
            When sequence_parallel_size = 1, Ulysses SP is disabled and standard
            data parallelism is used across all devices.
        """
        from torch.distributed.device_mesh import init_device_mesh

        # Initialize device mesh configuration for Ulysses Sequence Parallelism
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size

        if self.ulysses_sequence_parallel_size > 1:
            # Calculate data parallel size: total devices divided by sequence parallel size
            # Example: 8 GPUs, SP=2 → DP=4, meaning 4 DP groups of 2 SP ranks each
            dp_size = self.world_size // self.ulysses_sequence_parallel_size

            # Create 2D device mesh: [data_parallel_size, sequence_parallel_size]
            # This mesh defines how devices are arranged for hybrid parallelism
            self.ulysses_device_mesh = init_device_mesh(
                get_device_name(),  # Device type ("cuda", "cpu", etc.)
                mesh_shape=(dp_size, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],  # Named dimensions for clarity
            )

        # Initialize sharding manager to handle parameter distribution across the device mesh
        # This manager coordinates how model parameters are distributed and synchronized
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # Set convenience flag for conditional sequence parallelism logic
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _create_deepspeed_config(self) -> dict[str, Any]:
        """
        Create comprehensive DeepSpeed configuration dictionary.

        Generates a complete DeepSpeed configuration by parsing mixed precision settings,
        batch size configurations, and optimization parameters. The configuration
        handles various input formats and provides robust error handling for
        mixed precision parsing.

        Mixed Precision Handling:
            - String format: "fp16", "bf16" → automatic conversion to boolean flags
            - Dict format: {"param_dtype": "fp16"} → extracts dtype from nested config
            - None: Disables mixed precision training

        Configuration Sections:
            - Training: Batch sizes, gradient accumulation, distributed settings
            - Optimization: Learning rate, optimizer parameters (Adam betas, weight decay)
            - Precision: FP16/BF16 settings with automatic loss scaling
            - Memory: ZeRO stage configuration, CPU/NVMe offloading settings

        Returns:
            Dict[str, Any]: Complete DeepSpeed configuration dictionary containing:
                - train_batch_size: Global batch size across all devices
                - train_micro_batch_size_per_gpu: Micro batch size per GPU
                - gradient_accumulation_steps: Steps before optimizer update
                - zero_optimization: ZeRO stage and offloading configuration
                - optimizer: Adam optimizer with specified parameters
                - scheduler: Learning rate scheduling configuration
                - fp16/bf16: Mixed precision training settings

        Example Output:
            >>> config = engine._create_deepspeed_config()
            >>> print(config.keys())
            dict_keys(['train_batch_size', 'train_micro_batch_size_per_gpu',
                      'gradient_accumulation_steps', 'zero_optimization',
                      'optimizer', 'fp16', 'bf16'])

        Note:
            The configuration is passed directly to DeepSpeed's initialize() function
            and must conform to DeepSpeed's configuration schema.
        """
        # Parse mixed precision configuration with robust error handling
        # Support multiple input formats: string, dict, or None
        mp = self.engine_config.mixed_precision
        fp16_enabled = False
        bf16_enabled = False

        if isinstance(mp, str):
            # Simple string format: "fp16" or "bf16"
            if mp.lower() == "fp16":
                fp16_enabled = True
            elif mp.lower() == "bf16":
                bf16_enabled = True
        elif isinstance(mp, dict):
            # Complex dict format: {"param_dtype": "fp16", ...}
            dtype = mp.get("param_dtype")
            if dtype == "fp16":
                fp16_enabled = True
            elif dtype == "bf16":
                bf16_enabled = True
        elif mp is None:
            # Explicit None: disable mixed precision
            fp16_enabled = False

        # Generate complete DeepSpeed configuration using utility function
        # Align with the native test by passing specific optimizer and ZeRO settings
        return get_deepspeed_config(
            # Training configuration
            train_batch_size=self.train_batch_size,
            train_micro_batch_size_per_gpu=self.train_micro_batch_size_per_gpu,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            # ZeRO optimization configuration
            zero_stage=self.zero_stage,
            # Optimizer configuration
            optimizer_type=self.optimizer_config.optimizer,
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
            # Mixed precision configuration
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            # Memory offloading configuration
            cpu_offload=self._is_offload_param,
            offload_optimizer=self._is_offload_optimizer,
            # Disable scheduler to match native test
            disable_scheduler=True,
            # Pass specific ZeRO params to match native test
            zero_optimization={
                "overlap_comm": False,
                "contiguous_gradients": True,
                "reduce_scatter": True,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e7,
                "reduce_bucket_size": 2e7,
            },
        )

    def _build_module(self):
        """
        Build and configure the base model module with optimizations.

        Creates the underlying PyTorch model from HuggingFace configurations,
        applies various optimizations including custom kernels, dtype conversions,
        and gradient checkpointing. The method handles both training and inference
        configurations with appropriate memory and performance optimizations.

        Model Loading Process:
            1. Determine appropriate data type based on training/inference mode
            2. Load model using HuggingFace AutoModel with specified configuration
            3. Apply Liger kernel optimizations if enabled
            4. Apply monkey patches for custom kernel integration
            5. Configure gradient checkpointing for memory efficiency

        Data Type Selection:
            - Training mode: FP32 for numerical stability
            - Inference mode: BF16 for memory efficiency
            - Override: Use explicitly specified model_dtype if provided

        Optimization Features:
            - Liger Kernel: High-performance kernel implementations
            - Remove Padding: Efficient attention computation for variable lengths
            - Fused Kernels: Combined operations for reduced memory bandwidth
            - Gradient Checkpointing: Trade computation for memory (configurable)

        Returns:
            torch.nn.Module: Configured model ready for DeepSpeed initialization

        Side Effects:
            - Loads model weights from disk/HuggingFace Hub
            - Applies in-place optimizations to model architecture
            - Configures model for specified precision and optimization settings
            - May download model files if not cached locally

        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If model loading fails
            ImportError: If required optimization libraries are missing

        Example:
            >>> module = engine._build_module()
            >>> print(f"Model dtype: {module.dtype}")
            >>> print(f"Model device: {next(module.parameters()).device}")
        """
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        # Step 1: Determine appropriate data type for model parameters
        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            # Automatic dtype selection based on mode:
            # Training: FP32 for numerical stability and gradient accuracy
            # Inference: BF16 for memory efficiency without significant accuracy loss
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Step 2: Get appropriate HuggingFace AutoModel class for the model architecture
        auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

        # Step 3: Load model with suppressed warnings (common for model loading)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load pretrained model with specified configuration
            module = auto_class.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            # Step 4: Apply Liger kernel optimizations if enabled
            # Liger provides high-performance kernel implementations for common operations
            if self.model_config.use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=module)

            # Step 5: Apply custom monkey patches for kernel optimizations
            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            # Apply comprehensive monkey patches for optimization
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,  # Efficient attention for variable lengths
                ulysses_sp_size=self.ulysses_sequence_parallel_size,  # Sequence parallelism configuration
                use_fused_kernels=self.model_config.use_fused_kernels,  # Enable fused operations
                fused_kernels_backend=fused_kernels_backend,  # Specific backend for fused kernels
            )

            # Step 6: Ensure model parameters are in the correct precision
            # This is important after applying patches that might change parameter types
            module.to(torch_dtype)

            # Step 7: Enable gradient checkpointing for memory efficiency if requested
            # Trade computation time for memory by recomputing activations during backward pass
            if self.model_config.enable_gradient_checkpointing:
                # Use non-reentrant checkpointing for better memory management
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        return module

    def _build_lora_module(self, module):
        """
        Apply Low-Rank Adaptation (LoRA) to the model for parameter-efficient fine-tuning.

        LoRA reduces the number of trainable parameters by factorizing weight updates
        into low-rank matrices. This enables fine-tuning large models with significantly
        reduced memory requirements and training time.

        Args:
            module (torch.nn.Module): Base model to apply LoRA to

        Returns:
            torch.nn.Module: Model wrapped with LoRA adapters

        LoRA Configuration:
            - rank (r): Bottleneck dimension for low-rank factorization
            - alpha: Scaling factor for LoRA updates
            - target_modules: Which model layers to apply LoRA to
            - exclude_modules: Layers to exclude from LoRA application
        """
        # Enable gradient computation for input embeddings (required for LoRA)
        module.enable_input_require_grads()

        # Configure LoRA parameters based on model configuration
        lora_config = {
            "task_type": TaskType.CAUSAL_LM,  # Causal language modeling task
            "r": self.model_config.lora_rank,  # Low-rank dimension
            "lora_alpha": self.model_config.lora_alpha,  # Scaling factor
            "target_modules": convert_to_regular_types(self.model_config.target_modules),  # Layers to adapt
            "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),  # Layers to skip
            "bias": "none",  # Don't adapt bias parameters
        }

        # Apply LoRA using PEFT library
        module = get_peft_model(module, LoraConfig(**lora_config))
        return module

    def _build_model_optimizer(self):
        """Build model and initialize DeepSpeed engine."""
        from verl.utils.model import print_model_size

        # Build base module
        module = self._build_module()

        # Apply LoRA if enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Ensure required DeepSpeed single-process environment vars if user didn't launch via deepspeed/torchrun
        if "LOCAL_RANK" not in os.environ:
            # Fallback for ad-hoc single-process usage (e.g. quick tests)
            os.environ.setdefault("LOCAL_RANK", str(self.rank))
            os.environ.setdefault("RANK", str(self.rank))
            os.environ.setdefault("WORLD_SIZE", str(self.world_size))
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            if not torch.distributed.is_initialized():
                try:
                    torch.distributed.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)
                except Exception as _e:  # noqa: F841
                    # If initialization fails we proceed; DeepSpeed may attempt its own init
                    pass  # Synchronize all processes if distributed is active
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Create DeepSpeed configuration
        ds_config = self._create_deepspeed_config()

        if self.rank == 0:
            logger.debug(
                "[DeepSpeedEngine:ds_config] %s", {k: v for k, v in ds_config.items() if k != "zero_optimization"}
            )
            if "zero_optimization" in ds_config:
                logger.debug(
                    "[DeepSpeedEngine:zero_optimization] %s",
                    ds_config["zero_optimization"],
                )

        # Initialize DeepSpeed engine
        log_gpu_memory_usage("Before DeepSpeed initialization", logger=logger)

        self.engine, self.optimizer, _, self.lr_scheduler = initialize_deepspeed_engine(
            model=module,
            config=ds_config,
            model_parameters=module.parameters(),
        )

        # References for clarity
        self.module = self.engine.module  # underlying nn.Module
        self.ds_engine = self.engine  # alias

        log_gpu_memory_usage("After DeepSpeed initialization", logger=logger)

    def _create_checkpoint_manager(self):
        """Instantiate a DeepSpeed checkpoint manager."""
        return DeepSpeedCheckpointManager(self)

    def train_mode(self):
        """Return a context manager that switches to training mode."""
        return EngineTrainModeCtx(self)

    def eval_mode(self):
        """Return a context manager that switches to evaluation mode."""
        return EngineEvalModeCtx(self)

    def get_data_parallel_rank(self):
        """Get data parallel rank."""
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        """Get data parallel size."""
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def prepare_micro_batches(self, data: DataProto):
        """Prepare micro batches from data (following FSDP pattern)."""
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", True)

        if use_dynamic_bsz:
            assert "max_token_len_per_gpu" in data.meta_info, (
                "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
            )
            max_token_len_per_gpu = data.meta_info.get("max_token_len_per_gpu")
            max_token_len = max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batch_size_per_gpu = data.meta_info.get("micro_batch_size_per_gpu")
            micro_batches = data.split(micro_batch_size_per_gpu)
            batch_idx_list = None

        return micro_batches, batch_idx_list

    def forward_backward_batch(self, data: DataProto, loss_function: Callable, forward_only: bool = False):
        """Forward (and optional backward) pass for a batch.

        Returns:
            dict: aggregated metrics when training (forward_only=False)
            dict[str, torch.Tensor]: concatenated model outputs (e.g. log_probs, entropy) when forward_only=True
        """
        micro_batches, indices = self.prepare_micro_batches(data=data)

        output = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, metrics = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    # For DeepSpeed, use engine.backward() instead of loss.backward()
                    self.engine.backward(loss)

            output.append(metrics)

        return self.postprocess_batch_func(output, indices, forward_only, data)

    def postprocess_batch_func(self, losses_reduced, indices, forward_only, data: DataProto):
        """Postprocess micro-batch outputs.

        Handles two modes:
          - forward_only=True: concatenate per-micro-batch tensors,
            optionally reorder back to original sample order if dynamic batch was used
          - training mode: merge dict metrics into a single aggregated dict (lists of values)
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", True)

        if forward_only:
            output = {}
            # Collect tensors
            for mb_out in losses_reduced:
                for key, val in mb_out.items():
                    output.setdefault(key, []).append(val)

            if use_dynamic_bsz:
                # indices is list of lists of original sample indices; flatten & build reverse mapping
                flat_indices = list(itertools.chain.from_iterable(indices))
                revert_indices = torch.tensor(
                    get_reverse_idx(flat_indices), dtype=torch.long, device=output[next(iter(output))][0].device
                )
            else:
                flat_indices = None
                revert_indices = None

            for key, pieces in output.items():
                cat_val = torch.cat(pieces, dim=0)
                if use_dynamic_bsz:
                    assert revert_indices is not None
                    assert len(flat_indices) == cat_val.size(0), f"{len(flat_indices)} vs. {cat_val.size()}"
                    cat_val = cat_val[revert_indices]
                output[key] = cat_val

            return output
        else:
            metrics = {}
            for metric in losses_reduced:
                append_to_dict(metrics, metric)
            return metrics

    def forward_step(self, micro_batch: DataProto, loss_function, forward_only):
        """Forward step - to be implemented in subclass."""
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """
        Zero out the gradients of the model parameters.
        This is handled by the DeepSpeed engine.
        """
        self.engine.zero_grad()

    def optimizer_step(self):
        """Optimizer step using DeepSpeed engine."""
        # DeepSpeed engine handles gradient clipping and optimizer step
        return self.engine.step()

    # Convenience wrappers for external code parity with FSDP engine usage
    def backward(self, loss: torch.Tensor):  # type: ignore[name-defined]
        """Backward pass through DeepSpeed engine (mirrors FSDP wrapper API)."""
        return self.engine.backward(loss)

    def set_train_mode(self):
        """Put model into train mode (non-context persistent)."""
        # For ZeRO-3 with parameter offload, let DeepSpeed handle device management
        if self.zero_stage >= 3:
            # DeepSpeed ZeRO-3 automatically manages parameter loading/offloading
            self.engine.module.train()
        else:
            # For other stages, we can handle loading manually if needed
            if self._is_offload_param:
                load_deepspeed_model_to_gpu(self.engine)
            self.engine.module.train()
        self.mode = "train"

    def set_eval_mode(self):
        """Put model into eval mode (non-context persistent)."""
        # For ZeRO-3 with parameter offload, let DeepSpeed handle device management
        if self.zero_stage >= 3:
            # DeepSpeed ZeRO-3 automatically manages parameter loading/offloading
            self.engine.module.eval()
        else:
            # For other stages, we can handle loading manually if needed
            if self._is_offload_param:
                load_deepspeed_model_to_gpu(self.engine)
            self.engine.module.eval()
        self.mode = "eval"

    def lr_scheduler_step(self):
        """Learning rate scheduler step."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_last_lr()[0]
        return self.optimizer_config.lr

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """Move DeepSpeed engine to CPU or GPU."""
        assert device in ("cuda", "cpu")

        if device == "cuda":
            if model and not self._is_offload_param:
                load_deepspeed_model_to_gpu(self.engine)
        elif device == "cpu":
            if model and not self._is_offload_param and self.zero_stage < 3:
                # Only manually offload for ZeRO stages < 3
                offload_deepspeed_model_to_cpu(self.engine)

        gc.collect()

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Delegate saving to checkpoint manager (with pruning)."""
        if self.checkpoint_manager is None:  # fallback safety
            if self._is_offload_param:
                load_deepspeed_model_to_gpu(self.engine)
            client_state = {"global_step": global_step, "hdfs_path": hdfs_path}
            save_deepspeed_checkpoint(
                engine=self.engine,
                save_dir=local_path,
                client_state=client_state,
                tag=f"step_{global_step}",
            )
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:  # noqa: BLE001
                    pass
            if self._is_offload_param and self.zero_stage < 3:
                # Only manually offload for ZeRO stages < 3
                offload_deepspeed_model_to_cpu(self.engine)
            return local_path
        return self.checkpoint_manager.save(
            root=local_path,
            global_step=global_step,
            hdfs_path=hdfs_path,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """Delegate loading to checkpoint manager; step auto-selected (latest)."""
        if self.checkpoint_manager is None:  # fallback
            if self._is_offload_param:
                load_deepspeed_model_to_gpu(self.engine)
            client_state = load_deepspeed_checkpoint(
                engine=self.engine,
                load_dir=local_path,
                tag=None,
                load_module_strict=True,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:  # noqa: BLE001
                    pass
            if self._is_offload_param and self.zero_stage < 3:
                # Only manually offload for ZeRO stages < 3
                offload_deepspeed_model_to_cpu(self.engine)
            return client_state
        return self.checkpoint_manager.load(
            root=local_path,
            step=None,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )


class EngineEvalModeCtx:
    """Context manager for evaluation mode."""

    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        self.engine.mode = "eval"
        if self.engine._is_offload_param:
            load_deepspeed_model_to_gpu(self.engine.engine)

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)

        if self.engine._is_offload_param and self.engine.zero_stage < 3:
            # Only manually offload for ZeRO stages < 3
            offload_deepspeed_model_to_cpu(self.engine.engine)
        self.engine.mode = None


class EngineTrainModeCtx:
    """Context manager for training mode."""

    def __init__(self, engine):  # Remove type hint to avoid confusion
        self.engine = engine

    def __enter__(self):
        self.engine.mode = "train"
        if self.engine._is_offload_param:
            load_deepspeed_model_to_gpu(self.engine.engine)

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        self.engine.optimizer_zero_grad()

        if self.engine._is_offload_param and self.engine.zero_stage < 3:
            # Only manually offload for ZeRO stages < 3
            offload_deepspeed_model_to_cpu(self.engine.engine)
        self.engine.mode = None


@EngineRegistry.register(model_type="language_model", backend=["deepspeed"])
class DeepSpeedEngineWithLMHead(DeepSpeedEngine):
    """DeepSpeed Engine with Language Model Head for text generation tasks."""

    def forward_step(self, micro_batch: DataProto, loss_function, forward_only):
        """Forward step implementation following FSDP pattern."""
        use_remove_padding = micro_batch.meta_info.get("use_remove_padding", True)
        use_fused_kernels = micro_batch.meta_info.get("use_fused_kernels", False)
        temperature = micro_batch.meta_info["temperature"]
        calculate_entropy = micro_batch.meta_info.get("calculate_entropy", False)

        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        micro_batch_tensor = micro_batch.batch.to(device_name)

        response_length = micro_batch_tensor["responses"].size(-1)
        # Defensive: ensure we have at least one context token before the response window
        # Window logic later slices logits[:, -response_length-1:-1] so require seqlen > response_length
        total_seq_len = micro_batch_tensor["input_ids"].size(-1)
        assert response_length < total_seq_len, (
            f"response_length ({response_length}) must be < total sequence length ({total_seq_len}). "
            "Check data construction (need at least one prompt token)."
        )
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch_tensor.keys():
            # Normalize: always concatenate along dim=0 for tensor inputs.
            # If items are non-tensor (e.g. list bounds), fall back to list form.
            first_item = micro_batch_tensor["multi_modal_inputs"][0]
            for key, val in first_item.items():
                collected = [sample[key] for sample in micro_batch_tensor["multi_modal_inputs"]]
                if torch.is_tensor(collected[0]):
                    multi_modal_inputs[key] = torch.cat(collected, dim=0)
                else:
                    multi_modal_inputs[key] = collected  # keep list (e.g. image_bound meta)

        # Determine autocast dtype based on engine configuration
        autocast_dtype = torch.bfloat16  # default
        mp = self.engine_config.mixed_precision
        if isinstance(mp, str):
            if mp.lower() == "fp16":
                autocast_dtype = torch.float16
            elif mp.lower() == "bf16":
                autocast_dtype = torch.bfloat16
        elif isinstance(mp, dict):
            dtype = mp.get("param_dtype", "bf16")
            if dtype == "fp16":
                autocast_dtype = torch.float16
            elif dtype == "bf16":
                autocast_dtype = torch.bfloat16

        with torch.autocast(device_type=device_name, dtype=autocast_dtype):
            input_ids = micro_batch_tensor["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch_tensor["attention_mask"]
            position_ids = micro_batch_tensor["position_ids"]
            entropy = None

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if use_remove_padding:
                # Use remove padding optimization
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # FSDP-style label shift: use torch.roll for next-token prediction.
                # Wrap-around last position (will later be sliced out by response window logic).
                # This avoids an explicit clone + manual -100 tail assignment.
                shifted_labels = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                assert shifted_labels.shape == input_ids_rmpad.shape

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch_tensor.keys()
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    shifted_labels, _, _ = ulysses_pad_and_slice_inputs(
                        shifted_labels,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                shifted_labels = shifted_labels.squeeze(0)

                extra_args = {}
                if use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # Use DeepSpeed engine's module for forward pass
                output = self.engine.module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0) / temperature

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False

                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=shifted_labels,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.engine_config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # Handle Ulysses sequence parallelism
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # Pad back to original shape
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                # Slice mapping: logits position (S-R-1 ... S-2) -> predicting tokens (S-R ... S-1)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                # Align with responses: verify shape
                assert log_probs.size(1) == response_length, "log_probs length mismatch after remove_padding path"

            else:
                # Standard forward pass without remove padding
                extra_args = {}
                if use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # Use DeepSpeed engine's module for forward pass
                output = self.engine.module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                # Non-remove_padding path - align with FSDP implementation
                if use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    if calculate_entropy:
                        entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits / temperature
                    logits = logits[:, -response_length - 1 : -1, :]  # Extract response portion
                    # Align with FSDP: Directly use "responses" from the batch as labels
                    log_probs = logprobs_from_logits(logits, micro_batch_tensor["responses"])

                    if calculate_entropy:
                        if not self.engine_config.entropy_checkpointing:
                            entropy = self.compute_entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits)

            # Build output dictionary
            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropy"] = entropy

            if forward_only:
                return None, outputs
            else:
                policy_loss, metrics = loss_function(model_output=outputs, data=micro_batch_tensor)
                return policy_loss, metrics
