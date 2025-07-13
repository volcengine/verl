#!/usr/bin/env python3
"""
Python launcher for VERL math RL training.

Usage:
    python launch_math_rl.py --nodes=2 --batch_size=16 --epochs=10
    python launch_math_rl.py --nodes=1 --batch_size=8 --epochs=5 --gsm8k  # Test on smaller data
"""

import argparse
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


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
    project: str = "ssverl"
) -> str:
    """Format the VERL training command with all parameters."""
    
    # Calculate max_num_batched_tokens automatically
    max_num_batched_tokens = calculate_max_batched_tokens(max_prompt_length, max_response_length)
    
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
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size} \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu} \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.actor.strategy=fsdp2 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=False \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.enforce_eager=False \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.ref.strategy=fsdp2 \\
    actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens} \\
    algorithm.use_kl_in_reward=False \\
    trainer.critic_warmup=0 \\
    trainer.logger="['console','wandb']" \\
    trainer.project_name={project} \\
    trainer.experiment_name={experiment_name} \\
    critic.model.fsdp_config.param_offload=True \\
    critic.model.fsdp_config.optimizer_offload=True \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes={nodes} \\
    trainer.save_freq=50 \\
    trainer.max_actor_ckpt_to_keep=3 \\
    trainer.max_critic_ckpt_to_keep=3 \\
    trainer.test_freq=3 \\
    trainer.total_epochs={total_epochs} \\
    trainer.val_before_train=True"""
    
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


def main():
    parser = argparse.ArgumentParser(description="Launch VERL math RL training")
    
    # Required arguments
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    
    # Optional arguments
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", 
                       help="Path to the model")
    parser.add_argument("--data_path", type=str, default="$HOME/data", 
                       help="Path to the data directory")
    parser.add_argument("--experiment_name", type=str, 
                       help="Custom experiment name (auto-generated if not provided)")
    parser.add_argument("--project", type=str, default="ssverl",
                       help="Project name for logging")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                       help="Maximum prompt length")
    parser.add_argument("--max_response_length", type=int, default=4096,
                       help="Maximum response length")
    parser.add_argument("--gsm8k", action="store_true", 
                       help="Use GSM8K dataset for testing (smaller dataset)")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Generate script but don't submit job")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.max_prompt_length <= 0:
        raise ValueError("Max prompt length must be positive")
    if args.max_response_length <= 0:
        raise ValueError("Max response length must be positive")
    
    # Calculate batch sizes
    ppo_mini_batch_size, ppo_micro_batch_size_per_gpu = calculate_batch_sizes(
        args.nodes, args.batch_size
    )
    
    # Generate experiment name
    experiment_name = generate_experiment_name(
        args.nodes, args.batch_size, args.epochs, args.model_path, 
        args.gsm8k, args.experiment_name
    )
    
    # Create log directory
    log_dir = create_log_directory(experiment_name)
    
    # Get data files
    train_files, val_files = get_data_files(args.data_path, args.gsm8k)
    
    # Format the VERL command
    verl_command = format_verl_command(
        experiment_name=experiment_name,
        train_files=train_files,
        val_files=val_files,
        model_path=args.model_path,
        train_batch_size=args.batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        total_epochs=args.epochs,
        nodes=args.nodes,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        project=args.project
    )
    
    # Calculate max_num_batched_tokens for display
    max_num_batched_tokens = calculate_max_batched_tokens(args.max_prompt_length, args.max_response_length)
    
    # Prepare template variables
    template_vars = {
        "JOB_NAME": experiment_name,
        "NUM_NODES": args.nodes,
        "LOGDIR": str(log_dir),
        "CODE_DIR": str(Path.home() / "verl"),
        "DATA_PATH": args.data_path,
        "MODEL_PATH": args.model_path,
        "EXPERIMENT_NAME": experiment_name,
        "VERL_COMMAND": verl_command,
    }
    
    # Fill template
    template_path = Path(__file__).parent / "sbatch_template.sh"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    sbatch_content = fill_template(template_path, **template_vars)
    
    # Print configuration
    dataset_info = "GSM8K (test dataset)" if args.gsm8k else "Polaris (full dataset)"
    print(f"Configuration:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Dataset: {dataset_info}")
    print(f"  Nodes: {args.nodes}")
    print(f"  Total GPUs: {args.nodes * 8}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PPO mini batch size: {ppo_mini_batch_size}")
    print(f"  PPO micro batch size per GPU: {ppo_micro_batch_size_per_gpu}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Max prompt length: {args.max_prompt_length}")
    print(f"  Max response length: {args.max_response_length}")
    print(f"  Max batched tokens: {max_num_batched_tokens}")
    print(f"  Project: {args.project}")
    print(f"  Log directory: {log_dir}")
    print()
    
    if args.dry_run:
        print("Dry run mode - would submit the following sbatch script:")
        print("=" * 80)
        print(sbatch_content)
        print("=" * 80)
    else:
        # Submit the job
        job_id = submit_job(sbatch_content, experiment_name)
        print(f"Job submitted successfully with ID: {job_id}")
        print(f"Monitor with: squeue -j {job_id}")
        print(f"Check logs with: tail -f {log_dir}/train.log")


if __name__ == "__main__":
    main() 