#!/usr/bin/env python3
"""
Python launcher for VERL math RL training.

Usage:
    python launch_math_rl.py 2 16 --epochs=10
    python launch_math_rl.py 1 8 --epochs=5 --gsm8k=True  # Test on smaller data
    python launch_math_rl.py 2 16 --perf_preset=optimized --local=True
"""

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire


def calculate_batch_sizes(nodes: int, batch_size: int):
    """Calculate appropriate batch sizes based on number of nodes."""
    # Each node has 8 GPUs, so total GPUs = nodes * 8
    total_gpus = nodes * 8
    
    # PPO mini batch size should be a divisor of batch_size
    # and reasonable for the number of GPUs
    ppo_mini_batch_size = max(1, min(32, batch_size // 4))
    
    # PPO micro batch size per GPU should be small enough to fit in memory
    # but not too small to be inefficient
    ppo_micro_batch_size_per_gpu = max(1, min(4, batch_size // total_gpus))
    
    return ppo_mini_batch_size, ppo_micro_batch_size_per_gpu


def calculate_max_batched_tokens(max_prompt_length: int, max_response_length: int, rollout_max_bsz: int = 4) -> int:
    """Calculate max_num_batched_tokens based on prompt + response length and rollout batch size."""
    return (max_prompt_length + max_response_length) * rollout_max_bsz


def generate_experiment_name(
    nodes: int, 
    batch_size: int, 
    epochs: int,
    model_name: str,
    gsm8k: bool = False,
    custom_name: Optional[str] = None
) -> str:
    """Generate a descriptive experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.replace("Qwen/", "").replace("Qwen2.5-", "qwen2.5-").lower()
    
    dataset_suffix = ".gsm8k" if gsm8k else ""
    
    if custom_name:
        return f"{custom_name}.{model_short}.n{nodes}.bsz{batch_size}.ep{epochs}{dataset_suffix}.{timestamp}"
    else:
        return f"verl-math-rl.{model_short}.n{nodes}.bsz{batch_size}.ep{epochs}{dataset_suffix}.{timestamp}"


def create_log_directory(experiment_name: str) -> Path:
    """Create log directory structure."""
    log_base = Path.home() / "verl_log"
    log_dir = log_base / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_data_files(data_path: str, gsm8k: bool = False):
    """Get training and validation data file paths."""
    if gsm8k:
        # Use GSM8K data for testing (smaller dataset)
        train_files = f"{data_path}/gsm8k/train.parquet"
        val_files = f"{data_path}/gsm8k/test.parquet"
    else:
        # Use default polaris data
        train_files = f"{data_path}/polaris/train.parquet"
        val_files = f"{data_path}/polaris/test.parquet"
    
    return train_files, val_files


def format_verl_command(
    experiment_name: str,
    train_files: str,
    val_files: str,
    model_path: str,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size_per_gpu: int,
    total_epochs: int,
    nodes: int,
    max_prompt_length: int = 2048,
    max_response_length: int = 4096,
    project: str = "ssverl",
    # Performance parameters
    max_num_batched_tokens_override: Optional[int] = None,
    gpu_memory_utilization: float = 0.4,
    tensor_model_parallel_size: int = 2,
    use_remove_padding: bool = True,
    enable_gradient_checkpointing: bool = False,
    enable_activation_offload: bool = False,
    param_offload: bool = False,
    optimizer_offload: bool = False,
    forward_prefetch: bool = False,
    entropy_from_logits_with_chunking: bool = False,
    entropy_checkpointing: bool = False,
    use_dynamic_bsz: bool = False,
    ref_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    rollout_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    critic_forward_micro_batch_size_per_gpu: Optional[int] = None,
    critic_ppo_micro_batch_size_per_gpu: Optional[int] = None,
    ulysses_sequence_parallel_size: int = 1,
    **kwargs
) -> str:
    """Format the VERL training command with all parameters."""
    
    # Calculate max_num_batched_tokens automatically or use override
    if max_num_batched_tokens_override:
        max_num_batched_tokens = max_num_batched_tokens_override
    else:
        max_num_batched_tokens = calculate_max_batched_tokens(max_prompt_length, max_response_length)
    
    # Base command
    command = f"""python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files={train_files} \\
    data.val_files={val_files} \\
    data.train_batch_size={train_batch_size} \\
    data.max_prompt_length={max_prompt_length} \\
    data.max_response_length={max_response_length} \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    actor_rollout_ref.model.path={model_path} \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size} \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.000 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.actor.strategy=fsdp2 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.enforce_eager=False \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens} \\
    actor_rollout_ref.ref.strategy=fsdp2 \\
    algorithm.use_kl_in_reward=False \\
    trainer.critic_warmup=0 \\
    trainer.logger="['console','wandb']" \\
    trainer.project_name={project} \\
    trainer.experiment_name={experiment_name} \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes={nodes} \\
    trainer.save_freq=50 \\
    trainer.max_actor_ckpt_to_keep=3 \\
    trainer.max_critic_ckpt_to_keep=3 \\
    trainer.test_freq=3 \\
    trainer.total_epochs={total_epochs} \\
    trainer.val_before_train=True"""
    
    # Add performance optimization parameters
    # GPU memory utilization
    command += f" \\\n    actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}"
    
    # Tensor model parallel size
    command += f" \\\n    actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_model_parallel_size}"
    
    # Remove padding (sequence packing)
    if use_remove_padding:
        command += f" \\\n    actor_rollout_ref.model.use_remove_padding=True"
    
    # Gradient checkpointing
    if enable_gradient_checkpointing:
        command += f" \\\n    actor_rollout_ref.model.enable_gradient_checkpointing=True"
        command += f" \\\n    critic.model.enable_gradient_checkpointing=True"
    
    # Activation offloading
    if enable_activation_offload:
        command += f" \\\n    actor_rollout_ref.model.enable_activation_offload=True"
        command += f" \\\n    critic.model.enable_activation_offload=True"
    
    # Parameter and optimizer offloading
    if param_offload:
        command += f" \\\n    actor_rollout_ref.actor.fsdp_config.param_offload=True"
        command += f" \\\n    actor_rollout_ref.ref.fsdp_config.param_offload=True"
        command += f" \\\n    critic.model.fsdp_config.param_offload=True"
    
    if optimizer_offload:
        command += f" \\\n    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        command += f" \\\n    critic.model.fsdp_config.optimizer_offload=True"
    
    # Forward prefetch
    if forward_prefetch:
        command += f" \\\n    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
    
    # Entropy optimizations
    if entropy_from_logits_with_chunking:
        command += f" \\\n    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True"
    
    if entropy_checkpointing:
        command += f" \\\n    actor_rollout_ref.actor.entropy_checkpointing=True"
    
    # Dynamic batch size parameters
    if use_dynamic_bsz:
        command += f" \\\n    actor_rollout_ref.actor.use_dynamic_bsz=True"
        command += f" \\\n    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
        # Note: critic.use_dynamic_bsz might not be a valid parameter
        # command += f" \\\n    critic.use_dynamic_bsz=True"
        
        # Add dynamic batch size token limits
        for key, value in kwargs.items():
            if 'max_token_len_per_gpu' in key:
                if key == 'ppo_max_token_len_per_gpu':
                    command += f" \\\n    actor_rollout_ref.actor.ppo_max_token_len_per_gpu={value}"
                    # command += f" \\\n    critic.ppo_max_token_len_per_gpu={value}"
                elif key == 'ref_log_prob_max_token_len_per_gpu':
                    command += f" \\\n    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={value}"
                elif key == 'rollout_log_prob_max_token_len_per_gpu':
                    command += f" \\\n    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={value}"
                # elif key == 'critic_forward_max_token_len_per_gpu':
                #     command += f" \\\n    critic.forward_max_token_len_per_gpu={value}"
    else:
        # Standard micro batch size parameters
        command += f" \\\n    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu}"
        
        # Override micro batch sizes if provided
        if ref_log_prob_micro_batch_size_per_gpu:
            command += f" \\\n    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ref_log_prob_micro_batch_size_per_gpu}"
        else:
            command += f" \\\n    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu * 2}"
        
        if rollout_log_prob_micro_batch_size_per_gpu:
            command += f" \\\n    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout_log_prob_micro_batch_size_per_gpu}"
        else:
            command += f" \\\n    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu * 2}"
        
        if critic_forward_micro_batch_size_per_gpu:
            command += f" \\\n    critic.forward_micro_batch_size_per_gpu={critic_forward_micro_batch_size_per_gpu}"
        
        if critic_ppo_micro_batch_size_per_gpu:
            command += f" \\\n    critic.ppo_micro_batch_size_per_gpu={critic_ppo_micro_batch_size_per_gpu}"
    
    # Ulysses sequence parallel
    if ulysses_sequence_parallel_size > 1:
        command += f" \\\n    actor_rollout_ref.actor.ulysses_sequence_parallel_size={ulysses_sequence_parallel_size}"
        command += f" \\\n    actor_rollout_ref.ref.ulysses_sequence_parallel_size={ulysses_sequence_parallel_size}"
        command += f" \\\n    critic.ulysses_sequence_parallel_size={ulysses_sequence_parallel_size}"
    

    
    return command


def fill_template(
    template_path: Path,
    **kwargs
) -> str:
    """Fill the sbatch template with provided values."""
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    for key, value in kwargs.items():
        placeholder = f"${{{key}}}"
        template_content = template_content.replace(placeholder, str(value))
    
    return template_content


def submit_job(sbatch_content: str, experiment_name: str) -> str:
    """Submit the job using sbatch and return job ID."""
    # Create temporary file for sbatch script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(sbatch_content)
        temp_script_path = f.name
    
    try:
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        # Submit the job
        result = subprocess.run(
            ["sbatch", temp_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract job ID from output (typically "Submitted batch job 12345")
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id} for experiment: {experiment_name}")
        print(f"Log file: ~/verl_log/{experiment_name}/train.log")
        
        return job_id
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    finally:
        # Clean up temporary file
        os.unlink(temp_script_path)


def run_local_command(command: str, experiment_name: str, log_dir: Path, conda_env: str = "verl"):
    """Run the command locally instead of submitting to slurm."""
    print(f"Running locally: {experiment_name}")
    print(f"Log directory: {log_dir}")
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    
    # Prepare the command with conda activation
    full_command = f"""
    cd {Path.home() / "verl"}
    eval "$(/home/sam/miniconda3/bin/conda shell.bash hook)"
    conda activate {conda_env}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V1=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
    export HF_HOME=$TRANSFORMERS_CACHE
    
    # Run the command
    {command} 2>&1 | tee {log_file}
    """
    
    print(f"Executing command:")
    print(full_command)
    print(f"Logs will be written to: {log_file}")
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            check=False  # Don't raise exception on non-zero exit
        )
        print(f"Command finished with exit code: {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def main(
    # Required arguments
    nodes: int = 2,
    batch_size: int = 64,
    # Optional arguments
    epochs: int = 20,
    model_path: str = "Qwen/Qwen2.5-3B-Instruct",
    data_path: str = "$HOME/data",
    experiment_name: Optional[str] = None,
    project: str = "ssverl",
    max_prompt_length: int = 2048,
    max_response_length: int = 4096,
    gsm8k: bool = False,
    dry_run: bool = False,
    # Performance tuning arguments
    local: bool = False,
    use_remove_padding: bool = True,
    use_dynamic_bsz: bool = False,
    entropy_from_logits_with_chunking: bool = False,
    entropy_checkpointing: bool = False,
    enable_gradient_checkpointing: bool = False,
    enable_activation_offload: bool = False,
    forward_prefetch: bool = False,
    gpu_memory_utilization: float = 0.4,
    tensor_model_parallel_size: int = 2,
    # Batch size tuning arguments
    ppo_micro_batch_size_per_gpu: Optional[int] = None,
    ref_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    rollout_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    critic_forward_micro_batch_size_per_gpu: Optional[int] = None,
    critic_ppo_micro_batch_size_per_gpu: Optional[int] = None,
    # Dynamic batch size arguments
    ppo_max_token_len_per_gpu: Optional[int] = None,
    ref_log_prob_max_token_len_per_gpu: Optional[int] = None,
    rollout_log_prob_max_token_len_per_gpu: Optional[int] = None,
    critic_forward_max_token_len_per_gpu: Optional[int] = None,
    critic_ppo_max_token_len_per_gpu: Optional[int] = None,
    # Rollout tuning arguments
    max_num_batched_tokens_override: Optional[int] = None,
    disable_log_stats: bool = False,
    # Ulysses sequence parallel
    ulysses_sequence_parallel_size: int = 1,
    # Offloading arguments
    param_offload: bool = False,
    optimizer_offload: bool = False,
    # Performance preset arguments
    perf_preset: str = "baseline",
    # Conda environment
    conda_env: str = "verl",
):
    """Launch VERL math RL training.
    
    Args:
        nodes: Number of nodes
        batch_size: Training batch size
        epochs: Number of training epochs
        model_path: Path to the model
        data_path: Path to the data directory
        experiment_name: Custom experiment name (auto-generated if not provided)
        project: Project name for logging
        max_prompt_length: Maximum prompt length
        max_response_length: Maximum response length
        gsm8k: Use GSM8K dataset for testing (smaller dataset)
        dry_run: Generate script but don't submit job
        local: Run locally instead of submitting to slurm cluster
        max_steps: Maximum number of training steps (for performance testing)
        use_remove_padding: Enable sequence packing (remove padding)
        use_dynamic_bsz: Enable dynamic batch size for higher throughput
        use_liger_kernel: Enable LigerKernel for SFT performance optimization
        entropy_from_logits_with_chunking: Enable chunked entropy calculation to reduce memory
        entropy_checkpointing: Enable entropy checkpointing to reduce memory during training
        enable_gradient_checkpointing: Enable gradient checkpointing for larger batch sizes
        enable_activation_offload: Enable activation offloading to reduce memory
        forward_prefetch: Enable forward prefetch in FSDP for better efficiency
        gpu_memory_utilization: GPU memory utilization for vLLM rollout (default: 0.4)
        tensor_model_parallel_size: Tensor model parallel size for rollout (default: 2)
        ppo_micro_batch_size_per_gpu: PPO micro batch size per GPU (overrides auto-calculation)
        ref_log_prob_micro_batch_size_per_gpu: Reference log prob micro batch size per GPU
        rollout_log_prob_micro_batch_size_per_gpu: Rollout log prob micro batch size per GPU
        critic_forward_micro_batch_size_per_gpu: Critic forward micro batch size per GPU
        critic_ppo_micro_batch_size_per_gpu: Critic PPO micro batch size per GPU
        ppo_max_token_len_per_gpu: Max token length per GPU for PPO (used with dynamic batch size)
        ref_log_prob_max_token_len_per_gpu: Max token length per GPU for ref log prob (used with dynamic batch size)
        rollout_log_prob_max_token_len_per_gpu: Max token length per GPU for rollout log prob (used with dynamic batch size)
        critic_forward_max_token_len_per_gpu: Max token length per GPU for critic forward (used with dynamic batch size)
        critic_ppo_max_token_len_per_gpu: Max token length per GPU for critic PPO (used with dynamic batch size)
        max_num_batched_tokens_override: Override max_num_batched_tokens for rollout (default: auto-calculated)
        disable_log_stats: Disable rollout log stats
        ulysses_sequence_parallel_size: Ulysses sequence parallel size for long context training
        param_offload: Enable parameter offloading in FSDP
        optimizer_offload: Enable optimizer offloading in FSDP
        perf_preset: Performance preset configuration (baseline, optimized, memory_opt, speed_opt)
        conda_env: Conda environment to use (default: verl)
    """

    
    # Validate arguments
    if nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if max_prompt_length <= 0:
        raise ValueError("Max prompt length must be positive")
    if max_response_length <= 0:
        raise ValueError("Max response length must be positive")
    
    # Apply performance presets (directly modify variables)
    if perf_preset == "baseline":
        # Baseline configuration - no special optimizations
        pass
    elif perf_preset == "optimized":
        # Enable optimizations 1, 2, 4, 6, 8 from the performance guide
        use_remove_padding = True  # 2
        use_dynamic_bsz = True    # 4
        use_liger_kernel = True   # 6
        entropy_from_logits_with_chunking = True  # 8
        entropy_checkpointing = True  # 8
        gpu_memory_utilization = 0.6  # 1 - increase from default 0.4
        disable_log_stats = False  # 1 - enable log stats for tuning
        enable_gradient_checkpointing = True
        param_offload = True
        optimizer_offload = True
    elif perf_preset == "memory_opt":
        # Memory optimization focused
        use_remove_padding = True
        entropy_from_logits_with_chunking = True
        entropy_checkpointing = True
        enable_gradient_checkpointing = True
        enable_activation_offload = True
        param_offload = True
        optimizer_offload = True
        gpu_memory_utilization = 0.5
    elif perf_preset == "speed_opt":
        # Speed optimization focused
        use_remove_padding = True
        use_dynamic_bsz = True
        forward_prefetch = True
        gpu_memory_utilization = 0.7
        tensor_model_parallel_size = 1  # Prefer data parallelism
        disable_log_stats = False
    
    # Calculate dynamic batch size parameters
    dynamic_batch_params = {}
    if use_dynamic_bsz:
        # Calculate recommended token lengths based on sequence lengths
        base_token_len = max_prompt_length + max_response_length
        
        # Set default values if not provided
        if ppo_max_token_len_per_gpu is None:
            dynamic_batch_params['ppo_max_token_len_per_gpu'] = base_token_len * 2
        
        if ref_log_prob_max_token_len_per_gpu is None:
            dynamic_batch_params['ref_log_prob_max_token_len_per_gpu'] = base_token_len * 3
        
        if rollout_log_prob_max_token_len_per_gpu is None:
            dynamic_batch_params['rollout_log_prob_max_token_len_per_gpu'] = base_token_len * 3
        
        if critic_forward_max_token_len_per_gpu is None:
            dynamic_batch_params['critic_forward_max_token_len_per_gpu'] = base_token_len * 4
        
        if critic_ppo_max_token_len_per_gpu is None:
            dynamic_batch_params['critic_ppo_max_token_len_per_gpu'] = base_token_len * 4

    
    # Calculate batch sizes
    ppo_mini_batch_size, ppo_micro_batch_size_per_gpu = calculate_batch_sizes(
        nodes, batch_size
    )
    
    # Generate experiment name
    experiment_name = generate_experiment_name(
        nodes, batch_size, epochs, model_path, 
        gsm8k, experiment_name
    )
    
    # Create log directory
    log_dir = create_log_directory(experiment_name)
    
    # Get data files
    train_files, val_files = get_data_files(data_path, gsm8k)
    
    # Format the VERL command
    verl_command = format_verl_command(
        experiment_name=experiment_name,
        train_files=train_files,
        val_files=val_files,
        model_path=model_path,
        train_batch_size=batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        total_epochs=epochs,
        nodes=nodes,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        project=project,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_remove_padding=use_remove_padding,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        enable_activation_offload=enable_activation_offload,
        param_offload=param_offload,
        optimizer_offload=optimizer_offload,
        forward_prefetch=forward_prefetch,
        entropy_from_logits_with_chunking=entropy_from_logits_with_chunking,
        entropy_checkpointing=entropy_checkpointing,
        use_dynamic_bsz=use_dynamic_bsz,
        ref_log_prob_micro_batch_size_per_gpu=ref_log_prob_micro_batch_size_per_gpu,
        rollout_log_prob_micro_batch_size_per_gpu=rollout_log_prob_micro_batch_size_per_gpu,
        critic_forward_micro_batch_size_per_gpu=critic_forward_micro_batch_size_per_gpu,
        critic_ppo_micro_batch_size_per_gpu=critic_ppo_micro_batch_size_per_gpu,
        ulysses_sequence_parallel_size=ulysses_sequence_parallel_size,
        disable_log_stats=disable_log_stats,
        use_liger_kernel=use_liger_kernel,
        **dynamic_batch_params
    )
    
    # Calculate max_num_batched_tokens for display
    max_num_batched_tokens = calculate_max_batched_tokens(max_prompt_length, max_response_length)
    
    # Prepare template variables
    template_vars = {
        "JOB_NAME": experiment_name,
        "NUM_NODES": nodes,
        "LOGDIR": str(log_dir),
        "CODE_DIR": str(Path.home() / "verl"),
        "DATA_PATH": data_path,
        "MODEL_PATH": model_path,
        "EXPERIMENT_NAME": experiment_name,
        "VERL_COMMAND": verl_command,
        "CONDA_ENV": conda_env,
    }
    
    # Fill template
    template_path = Path(__file__).parent / "sbatch_template.sh"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    sbatch_content = fill_template(template_path, **template_vars)
    
    # Print configuration
    dataset_info = "GSM8K (test dataset)" if gsm8k else "Polaris (full dataset)"
    print(f"Configuration:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Dataset: {dataset_info}")
    print(f"  Nodes: {nodes}")
    print(f"  Total GPUs: {nodes * 8}")
    print(f"  Batch size: {batch_size}")
    print(f"  PPO mini batch size: {ppo_mini_batch_size}")
    print(f"  PPO micro batch size per GPU: {ppo_micro_batch_size_per_gpu}")
    print(f"  Epochs: {epochs}")
    print(f"  Model: {model_path}")
    print(f"  Data path: {data_path}")
    print(f"  Max prompt length: {max_prompt_length}")
    print(f"  Max response length: {max_response_length}")
    print(f"  Max batched tokens: {max_num_batched_tokens}")
    print(f"  Project: {project}")
    print(f"  Log directory: {log_dir}")
    print(f"  Performance preset: {perf_preset}")
    print(f"  Execution mode: {'Local' if local else 'Slurm'}")
    print(f"  Conda environment: {conda_env}")
    
    # Print enabled optimizations
    optimizations = []
    if use_remove_padding:
        optimizations.append("Remove padding (sequence packing)")
    if use_dynamic_bsz:
        optimizations.append("Dynamic batch size")
    if use_liger_kernel:
        optimizations.append("LigerKernel")
    if entropy_from_logits_with_chunking:
        optimizations.append("Chunked entropy calculation")
    if entropy_checkpointing:
        optimizations.append("Entropy checkpointing")
    if enable_gradient_checkpointing:
        optimizations.append("Gradient checkpointing")
    if enable_activation_offload:
        optimizations.append("Activation offloading")
    if param_offload:
        optimizations.append("Parameter offloading")
    if optimizer_offload:
        optimizations.append("Optimizer offloading")
    if forward_prefetch:
        optimizations.append("Forward prefetch")
    if ulysses_sequence_parallel_size > 1:
        optimizations.append(f"Ulysses sequence parallel (size={ulysses_sequence_parallel_size})")
    
    if optimizations:
        print(f"  Enabled optimizations: {', '.join(optimizations)}")
    else:
        print(f"  Enabled optimizations: None (baseline)")
    
    # Print performance-related parameters
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Tensor model parallel size: {tensor_model_parallel_size}")
    if max_steps:
        print(f"  Max steps (for testing): {max_steps}")
    
    # Print dynamic batch size parameters if enabled
    if use_dynamic_bsz:
        print(f"  Dynamic batch size parameters:")
        for key, value in dynamic_batch_params.items():
            print(f"    {key}: {value}")
    
    print()
    
    # Handle local execution
    if local:
        print("Running locally...")
        exit_code = run_local_command(verl_command, experiment_name, log_dir, conda_env)
        print(f"Local execution finished with exit code: {exit_code}")
        return
    
    if dry_run:
        print("Dry run mode - would submit the following sbatch script:")
        print("=" * 80)
        print(sbatch_content)
        print("=" * 80)
        print(f"VERL Command:")
        print(verl_command)
        print("=" * 80)
    else:
        # Submit the job
        job_id = submit_job(sbatch_content, experiment_name)
        print(f"Job submitted successfully with ID: {job_id}")
        print(f"Monitor with: squeue -j {job_id}")
        print(f"Check logs with: tail -f {log_dir}/train.log")


if __name__ == "__main__":
    fire.Fire(main)