#!/usr/bin/env python3
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
Atropos-VeRL Launch Script

This script orchestrates the startup of both Atropos and VeRL together and auto-configures the environment

Usage:
    python launch_atropos_verl.py --config config/verl_grpo_atropos_config.yaml
    
Features:
- Automatic environment discovery and configuration
- Inference engine auto-pointing to VeRL
- Resource management and cleanup
- Monitoring and health checks
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass

import ray
import torch
from omegaconf import DictConfig, OmegaConf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("atropos_verl_launcher")


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    name: str
    process: subprocess.Popen
    port: Optional[int] = None
    pid: Optional[int] = None
    log_file: Optional[str] = None


class AtroposVeRLLauncher:
    """
    Launcher for coordinated Atropos-VeRL training.
    
    Manages:
    - Atropos environment processes
    - VeRL inference engines (vLLM/SGLang) 
    - Training process coordination
    - Resource cleanup and monitoring
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.processes: List[ProcessInfo] = []
        self.temp_dir = None
        self.ray_initialized = False
        
        # Extract configuration
        self.atropos_config = config.atropos
        self.trainer_config = config.trainer
        self.rollout_config = config.actor_rollout_ref.rollout
        
        # Process management
        self.cleanup_on_exit = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AtroposVeRL Launcher initialized")
        logger.info(f"  - Atropos groups: {self.atropos_config.num_groups}")
        logger.info(f"  - Inference engine: {self.rollout_config.name}")
        logger.info(f"  - Training steps: {self.trainer_config.total_training_steps}")
    
    def launch(self):
        """Launch the complete Atropos-VeRL training pipeline."""
        try:
            # Create temporary directory for coordination
            self.temp_dir = tempfile.mkdtemp(prefix="atropos_verl_")
            logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Step 1: Initialize Ray cluster
            self._initialize_ray()
            
            # Step 2: Start Atropos environment groups
            self._start_atropos_environments()
            
            # Step 3: Start VeRL inference engines
            self._start_inference_engines()
            
            # Step 4: Wait for all services to be ready
            self._wait_for_services()
            
            # Step 5: Update configuration with auto-discovered endpoints
            self._update_config_with_endpoints()
            
            # Step 6: Launch VeRL GRPO trainer with Atropos integration
            self._launch_trainer()
            
        except Exception as e:
            logger.error(f"Launch failed: {e}")
            self._cleanup()
            raise
        finally:
            if self.cleanup_on_exit:
                self._cleanup()
    
    def _initialize_ray(self):
        """Initialize Ray cluster for distributed training."""
        logger.info("Initializing Ray cluster...")
        
        # Ray configuration from trainer config
        ray_config = {
            "num_cpus": self.trainer_config.n_gpus_per_node * 2,  # 2 CPUs per GPU
            "num_gpus": self.trainer_config.n_gpus_per_node,
            "object_store_memory": 1000000000,  # 1GB object store
        }
        
        if not ray.is_initialized():
            ray.init(**ray_config)
            self.ray_initialized = True
            logger.info(f"Ray initialized with config: {ray_config}")
        else:
            logger.info("Ray already initialized")
    
    def _start_atropos_environments(self):
        """Start Atropos environment group processes."""
        logger.info("Starting Atropos environment groups...")
        
        # Use configurable host and port from config
        atropos_host = self.atropos_config.get("host", "localhost")
        base_port = self.atropos_config.get("port", 8000)
        
        for group_id in range(self.atropos_config.num_groups):
            port = base_port + group_id
            log_file = os.path.join(self.temp_dir, f"atropos_group_{group_id}.log")
            
            # Create Atropos environment configuration
            env_config = self._create_atropos_env_config(group_id, port, atropos_host)
            env_config_path = os.path.join(self.temp_dir, f"atropos_env_{group_id}.yaml")
            
            with open(env_config_path, 'w') as f:
                yaml.dump(env_config, f)
            
            # Start Atropos environment process
            cmd = [
                sys.executable, "-m", "atropos.server",
                "--config", env_config_path,
                "--host", atropos_host,
                "--port", str(port),
                "--group-id", str(group_id),
            ]
            
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=self.temp_dir
                )
            
            process_info = ProcessInfo(
                name=f"atropos_group_{group_id}",
                process=process,
                port=port,
                pid=process.pid,
                log_file=log_file
            )
            self.processes.append(process_info)
            
            logger.info(f"Started Atropos group {group_id} on {atropos_host}:{port} (PID: {process.pid})")
    
    def _start_inference_engines(self):
        """Start VeRL inference engines (vLLM/SGLang)."""
        logger.info(f"Starting {self.rollout_config.name} inference engines...")
        
        if self.rollout_config.name == "vllm":
            self._start_vllm_engines()
        elif self.rollout_config.name == "sglang":
            self._start_sglang_engines()
        else:
            logger.warning(f"Inference engine {self.rollout_config.name} not supported for auto-launch")
    
    def _start_vllm_engines(self):
        """Start vLLM inference engines."""
        base_port = 9000
        tensor_parallel_size = self.rollout_config.tensor_model_parallel_size
        
        for engine_id in range(self.trainer_config.nnodes):
            port = base_port + engine_id
            log_file = os.path.join(self.temp_dir, f"vllm_engine_{engine_id}.log")
            
            # vLLM server command
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.config.model.path,
                "--port", str(port),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(self.rollout_config.gpu_memory_utilization),
                "--max-model-len", str(self.config.data.max_prompt_length + self.config.data.max_response_length),
                "--disable-log-stats",
            ]
            
            # Add additional vLLM arguments
            if self.rollout_config.get("trust_remote_code", False):
                cmd.append("--trust-remote-code")
            
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=self._get_cuda_env(engine_id)
                )
            
            process_info = ProcessInfo(
                name=f"vllm_engine_{engine_id}",
                process=process,
                port=port,
                pid=process.pid,
                log_file=log_file
            )
            self.processes.append(process_info)
            
            logger.info(f"Started vLLM engine {engine_id} on port {port} (PID: {process.pid})")
    
    def _start_sglang_engines(self):
        """Start SGLang inference engines."""
        base_port = 9000
        tensor_parallel_size = self.rollout_config.tensor_model_parallel_size
        
        for engine_id in range(self.trainer_config.nnodes):
            port = base_port + engine_id
            log_file = os.path.join(self.temp_dir, f"sglang_engine_{engine_id}.log")
            
            # SGLang server command
            cmd = [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", self.config.model.path,
                "--port", str(port),
                "--tp-size", str(tensor_parallel_size),
                "--mem-fraction-static", str(self.rollout_config.gpu_memory_utilization),
            ]
            
            # Add additional SGLang arguments
            if self.rollout_config.get("trust_remote_code", False):
                cmd.append("--trust-remote-code")
            
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=self._get_cuda_env(engine_id)
                )
            
            process_info = ProcessInfo(
                name=f"sglang_engine_{engine_id}",
                process=process,
                port=port,
                pid=process.pid,
                log_file=log_file
            )
            self.processes.append(process_info)
            
            logger.info(f"Started SGLang engine {engine_id} on port {port} (PID: {process.pid})")
    
    def _wait_for_services(self):
        """Wait for all services to be ready."""
        logger.info("Waiting for services to be ready...")
        
        max_wait_time = 300  # 5 minutes
        check_interval = 5   # 5 seconds
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            all_ready = True
            
            # Check Atropos environments
            for process in self.processes:
                if "atropos" in process.name:
                    if not self._check_atropos_ready(process.port):
                        all_ready = False
                        break
            
            # Check inference engines
            for process in self.processes:
                if "vllm" in process.name or "sglang" in process.name:
                    if not self._check_inference_engine_ready(process.port):
                        all_ready = False
                        break
            
            if all_ready:
                logger.info("All services are ready!")
                return
            
            logger.info("Waiting for services...")
            time.sleep(check_interval)
        
        raise RuntimeError(f"Services not ready after {max_wait_time} seconds")
    
    def _check_atropos_ready(self, port: int) -> bool:
        """Check if Atropos environment is ready."""
        try:
            import requests
            atropos_host = self.atropos_config.get("host", "localhost")
            response = requests.get(f"http://{atropos_host}:{port}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_inference_engine_ready(self, port: int) -> bool:
        """Check if inference engine is ready."""
        try:
            import requests
            # Inference engines typically run on localhost
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _update_config_with_endpoints(self):
        """Update configuration with auto-discovered service endpoints."""
        logger.info("Updating configuration with service endpoints...")
        
        atropos_host = self.atropos_config.get("host", "localhost")
        
        # Update Atropos endpoints
        atropos_endpoints = []
        for process in self.processes:
            if "atropos" in process.name:
                atropos_endpoints.append(f"http://{atropos_host}:{process.port}")
        
        # Update inference engine endpoints (usually localhost)
        inference_endpoints = []
        for process in self.processes:
            if "vllm" in process.name or "sglang" in process.name:
                inference_endpoints.append(f"http://localhost:{process.port}")
        
        # Create configuration with service endpoints
        self.config.atropos.endpoints = atropos_endpoints
        self.config.actor_rollout_ref.rollout.inference_endpoints = inference_endpoints
        
        # Save configuration for training
        config_path = os.path.join(self.temp_dir, "training_config.yaml")
        with open(config_path, 'w') as f:
            OmegaConf.save(self.config, f)
        
        logger.info(f"Training configuration saved to: {config_path}")
        logger.info(f"  - Atropos endpoints: {len(atropos_endpoints)}")
        logger.info(f"  - Inference endpoints: {len(inference_endpoints)}")
    
    def _launch_trainer(self):
        """Launch the VeRL GRPO trainer with Atropos integration."""
        logger.info("Launching VeRL GRPO trainer with Atropos integration...")
        
        # Import and initialize trainer
        from recipe.atropos.verl_grpo_atropos_trainer import VeRLGRPOAtroposTrainer
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.path, 
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Create datasets
        train_dataset = create_rl_dataset(
            self.config.data.train_files, 
            self.config.data, 
            tokenizer, 
            processor=None
        )
        val_dataset = create_rl_dataset(
            self.config.data.val_files, 
            self.config.data, 
            tokenizer, 
            processor=None
        )
        
        # Create resource pool manager
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [self.trainer_config.n_gpus_per_node] * self.trainer_config.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
        )
        resource_pool_manager.create_resource_pool()
        
        # Define worker classes
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        # Initialize trainer
        trainer = VeRLGRPOAtroposTrainer(
            config=self.config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        
        # Initialize workers and start training
        trainer.init_workers()
        logger.info("Starting Atropos-VeRL integrated training...")
        trainer.fit()
        
        logger.info("Training completed successfully!")
    
    def _create_atropos_env_config(self, group_id: int, port: int, atropos_host: str) -> Dict[str, Any]:
        """Create configuration for an Atropos environment group."""
        env_config = {
            "group_id": group_id,
            "host": atropos_host,
            "port": port,
            "environment": {
                "type": self.atropos_config.environment_type,
                "max_steps": self.atropos_config.max_env_steps,
                "timeout": 30,
            },
            "evaluation": {
                "mode": self.atropos_config.advantage_mode,
                "normalize_advantages": self.atropos_config.normalize_advantages,
                "clip_range": self.atropos_config.advantage_clip_range,
            },
            "rewards": {
                "shaping": self.atropos_config.reward_shaping,
                "success_reward": self.atropos_config.success_reward,
                "failure_penalty": self.atropos_config.failure_penalty,
            },
            "api": {
                "base_url": f"http://{atropos_host}:{port}",
                "timeout": self.atropos_config.get("timeout", 30),
                "retry_attempts": self.atropos_config.get("retry_attempts", 3),
                "api_key": self.atropos_config.get("api_key", None),
            }
        }
        
        # Add environment-specific configuration
        if self.atropos_config.environment_type == "code_execution":
            env_config["environment"].update({
                "allowed_modules": ["math", "random", "itertools"],
                "memory_limit": "1GB",
                "cpu_limit": 1,
            })
        
        return env_config
    
    def _get_cuda_env(self, engine_id: int) -> Dict[str, str]:
        """Get CUDA environment variables for an engine."""
        env = os.environ.copy()
        
        # Set CUDA device visibility
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpu_start = (engine_id * self.rollout_config.tensor_model_parallel_size) % n_gpus
            gpu_end = gpu_start + self.rollout_config.tensor_model_parallel_size
            gpu_ids = list(range(gpu_start, min(gpu_end, n_gpus)))
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        return env
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup_on_exit = True
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Clean up all processes and resources."""
        logger.info("Cleaning up processes and resources...")
        
        # Terminate all processes
        for process_info in self.processes:
            try:
                if process_info.process.poll() is None:  # Process is still running
                    logger.info(f"Terminating {process_info.name} (PID: {process_info.pid})")
                    process_info.process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process_info.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {process_info.name}")
                        process_info.process.kill()
                        
            except Exception as e:
                logger.error(f"Error terminating {process_info.name}: {e}")
        
        # Shutdown Ray
        if self.ray_initialized:
            try:
                ray.shutdown()
                logger.info("Ray cluster shutdown")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}")
        
        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {e}")
        
        logger.info("Cleanup completed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed processes."""
        status = {
            "processes": [],
            "ray_initialized": self.ray_initialized,
            "temp_dir": self.temp_dir,
        }
        
        for process_info in self.processes:
            process_status = {
                "name": process_info.name,
                "pid": process_info.pid,
                "port": process_info.port,
                "running": process_info.process.poll() is None,
                "log_file": process_info.log_file,
            }
            status["processes"].append(process_status)
        
        return status


def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="Launch Atropos-VeRL integrated training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-cleanup", 
        action="store_true",
        help="Don't cleanup processes on exit (for debugging)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create and run launcher
    launcher = AtroposVeRLLauncher(config)
    launcher.cleanup_on_exit = not args.no_cleanup
    
    try:
        launcher.launch()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 