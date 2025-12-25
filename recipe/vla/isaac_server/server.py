# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Isaac Lab Multi-Task Server

A standalone server that runs a multi-task Isaac Lab environment and exposes
a ZMQ-based API for remote clients (EnvWorkers) to interact with.

Usage:
    # Single GPU mode:
    python isaac_server.py --num_tasks 40 --group_size 2 --port 5555

    # Multi-GPU distributed mode (recommended):
    python -m torch.distributed.run --nproc_per_node=8 isaac_server.py \\
        --num_tasks 10 --group_size 4 --base_port 5555 --distributed

Features:
    - Single Isaac instance handles all tasks simultaneously
    - No task-switching overhead (no reset needed)
    - ZMQ REP socket for request-response communication
    - Support for step, reset, and close commands
    - Multi-GPU support via torchrun (each GPU runs independent server)
"""

import argparse
import logging
import os
import pickle
import signal
from typing import Any

import numpy as np
import torch
import zmq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IsaacServer")


def setup_per_process_caches():
    """Setup per-process cache directories to avoid locking conflicts.

    When multiple Isaac processes run on the same node, they can conflict
    when accessing shared cache databases. This function creates unique
    cache directories for each process based on LOCAL_RANK.

    Caches isolated:
    1. OptiX cache - prevents denoiser database locking
    2. NVIDIA Shader cache - prevents shader compilation conflicts
    3. Omniverse Kit cache - prevents general caching conflicts

    Must be called BEFORE importing any Isaac/Omniverse modules.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # === OptiX Cache ===
    # Get base cache path from environment (set by start_isaac_server.sh)
    optix_base = os.environ.get("OPTIX_CACHE_PATH", "/tmp/optix_cache")
    optix_rank_cache = os.path.join(optix_base, f"rank_{local_rank}")
    os.makedirs(optix_rank_cache, exist_ok=True)
    os.environ["OPTIX_CACHE_PATH"] = optix_rank_cache
    os.environ["OPTIX7_CACHE_PATH"] = optix_rank_cache

    # === NVIDIA Driver Shader Cache ===
    shader_base = os.environ.get("__GL_SHADER_DISK_CACHE_PATH", "/tmp/nv_shader_cache")
    shader_rank_cache = os.path.join(shader_base, f"rank_{local_rank}")
    os.makedirs(shader_rank_cache, exist_ok=True)
    os.environ["__GL_SHADER_DISK_CACHE"] = "1"
    os.environ["__GL_SHADER_DISK_CACHE_PATH"] = shader_rank_cache
    os.environ["__GL_SHADER_DISK_CACHE_SKIP_CLEANUP"] = "1"

    # === Omniverse Kit Cache ===
    ov_base = os.environ.get("OMNI_KIT_CACHE_DIR", "/tmp/ov_cache")
    ov_rank_cache = os.path.join(ov_base, f"rank_{local_rank}")
    os.makedirs(ov_rank_cache, exist_ok=True)
    os.environ["OMNI_KIT_CACHE_DIR"] = ov_rank_cache

    logger.info(f"[Rank {local_rank}] Cache directories:")
    logger.info(f"[Rank {local_rank}]   OptiX:  {optix_rank_cache}")
    logger.info(f"[Rank {local_rank}]   Shader: {shader_rank_cache}")
    logger.info(f"[Rank {local_rank}]   OV Kit: {ov_rank_cache}")


# Setup per-process caches before any Isaac imports
setup_per_process_caches()


class IsaacMultiTaskServer:
    """Multi-task Isaac Lab server with ZMQ interface.

    Supports two modes:
    1. Single-GPU mode: One server handles all tasks
    2. Distributed mode: Multiple servers (one per GPU), each handles a subset of tasks

    In distributed mode (torchrun), each process:
    - Gets assigned a unique GPU via LOCAL_RANK
    - Listens on a unique port (base_port + LOCAL_RANK)
    - Handles a subset of tasks (tasks are split evenly across GPUs)
    """

    def __init__(
        self,
        env_id: str = "Isaac-Libero-Franka-OscPose-All-Tasks-v0",
        num_tasks: int = 40,
        group_size: int = 8,
        port: int = 5555,
        use_ipc: bool = True,
        distributed: bool = False,
        render_last_only: bool = True,
        camera_height: int = 256,
        camera_width: int = 256,
    ):
        """
        Initialize the Isaac server.

        Args:
            env_id: Gymnasium environment ID for multi-task Isaac env
            num_tasks: Number of different tasks (total across all GPUs in distributed mode)
            group_size: Number of parallel envs per task
            port: ZMQ port number (base port in distributed mode)
            use_ipc: If True, use Unix IPC socket; otherwise use TCP
            distributed: If True, enable multi-GPU distributed mode via torchrun
            render_last_only: If True, only render on the last step of action chunks (saves GPU)
            camera_height: Camera image height (default: 256)
            camera_width: Camera image width (default: 256)
        """
        self.env_id = env_id
        self.group_size = group_size
        self.use_ipc = use_ipc
        self.distributed = distributed
        self.render_last_only = render_last_only
        self.camera_height = camera_height
        self.camera_width = camera_width

        # Distributed mode settings
        if distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.global_rank = int(os.environ.get("RANK", 0))

            # Each GPU handles a subset of tasks
            # Split tasks evenly across GPUs
            tasks_per_gpu = num_tasks // self.world_size
            remainder = num_tasks % self.world_size

            # Distribute remainder tasks to first few GPUs
            if self.local_rank < remainder:
                self.num_tasks = tasks_per_gpu + 1
                self.task_offset = self.local_rank * (tasks_per_gpu + 1)
            else:
                self.num_tasks = tasks_per_gpu
                self.task_offset = remainder * (tasks_per_gpu + 1) + (self.local_rank - remainder) * tasks_per_gpu

            # Each GPU uses its own port
            self.port = port + self.local_rank
            self.device_id = self.local_rank

            logger.info(f"[Rank {self.local_rank}/{self.world_size}] Distributed mode enabled")
            logger.info(
                f"[Rank {self.local_rank}] Handling tasks {self.task_offset} to {self.task_offset + self.num_tasks - 1}"
            )
            logger.info(f"[Rank {self.local_rank}] Using GPU {self.device_id}, port {self.port}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            self.num_tasks = num_tasks
            self.task_offset = 0
            self.port = port
            self.device_id = 0

        self.total_envs = self.num_tasks * group_size
        self.total_num_tasks_global = num_tasks  # Total tasks across all GPUs

        # Task ID to env indices mapping (local to this server)
        # In distributed mode, task_id is local (0 to num_tasks-1)
        # Client needs to add task_offset to get global task_id
        self.task_to_env_indices = {
            task_id: list(range(task_id * group_size, (task_id + 1) * group_size)) for task_id in range(self.num_tasks)
        }

        self.env = None
        self.socket = None
        self.ctx = None
        self.running = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _init_env(self):
        """Initialize the Isaac Lab multi-task environment."""
        import torch

        num_gpus = torch.cuda.device_count()
        logger.info(f"[Rank {self.local_rank}] Initializing Isaac environment: {self.env_id}")
        logger.info(f"[Rank {self.local_rank}] Available GPUs: {num_gpus}, using GPU {self.device_id}")
        logger.info(
            f"[Rank {self.local_rank}] Requested config: "
            f"{self.num_tasks} tasks x {self.group_size} group_size = {self.total_envs} envs"
        )

        # Import Isaac Lab components (requires Isaac Sim to be running)
        import gymnasium as gym
        from isaaclab.app import AppLauncher

        # Create argument namespace for AppLauncher
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.headless = True
        args.enable_cameras = True

        # In distributed mode, each process uses a specific GPU
        if self.distributed:
            args.device = f"cuda:{self.device_id}"
            # Set distributed flag for AppLauncher to properly configure GPU
            args.distributed = True
            logger.info(f"[Rank {self.local_rank}] Distributed mode: using device {args.device}")
        else:
            args.device = "cuda:0"
            args.distributed = False

        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app

        # Store device for later use
        self.device = f"cuda:{self.device_id}" if self.distributed else "cuda:0"

        # Now import Isaac Lab task utilities
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # Parse environment config - this will use our GROUP_SIZE env var
        env_cfg = parse_env_cfg(self.env_id, device=self.device, num_envs=self.total_envs)

        # Override camera dimensions if the config supports it
        if hasattr(env_cfg, "camera_height"):
            env_cfg.camera_height = self.camera_height
            env_cfg.camera_width = self.camera_width
            env_cfg.recreate_cameras()
            logger.info(f"[Rank {self.local_rank}] Set camera dimensions: {self.camera_width}x{self.camera_height}")

        # In distributed mode, we need to adjust the task configuration
        # Each server only handles a subset of tasks
        if self.distributed and hasattr(env_cfg, "task_offset"):
            env_cfg.task_offset = self.task_offset
            logger.info(f"[Rank {self.local_rank}] Set task_offset={self.task_offset} in env config")

        # Verify the config picked up our group_size
        actual_group_size = getattr(env_cfg, "group_size", None)
        actual_num_envs = getattr(env_cfg, "num_envs", None)
        logger.info(
            f"[Rank {self.local_rank}] Config created: group_size={actual_group_size}, num_envs={actual_num_envs}"
        )

        if actual_num_envs != self.total_envs:
            logger.warning(
                f"[Rank {self.local_rank}] Config num_envs mismatch: "
                f"expected {self.total_envs}, got {actual_num_envs}. "
                f"Forcing scene.num_envs={self.total_envs}"
            )
            env_cfg.scene.num_envs = self.total_envs

        # Create environment with config
        self.env = gym.make(self.env_id, cfg=env_cfg, render_mode=None).unwrapped
        self.action_dim = self.env.action_space.shape[-1]

        # Debug: print action space info
        logger.info(f"[Rank {self.local_rank}] Action space: {self.env.action_space}")
        logger.info(f"[Rank {self.local_rank}] Action dim: {self.action_dim}")
        logger.info(f"[Rank {self.local_rank}] LIBERO_OSC_TYPE env var: {os.getenv('LIBERO_OSC_TYPE', 'not set')}")

        # Verify actual number of envs created
        actual_num_envs = self.env.num_envs
        if actual_num_envs != self.total_envs:
            logger.error(
                f"[Rank {self.local_rank}] MISMATCH: Requested {self.total_envs} envs "
                f"but Isaac Lab created {actual_num_envs}! "
                f"Check GROUP_SIZE environment variable and Isaac Lab config."
            )
            # Update total_envs to reflect reality
            self.total_envs = actual_num_envs
            self.group_size = actual_num_envs // self.num_tasks if self.num_tasks > 0 else actual_num_envs
            # Rebuild task mapping
            self.task_to_env_indices = {
                task_id: list(range(task_id * self.group_size, (task_id + 1) * self.group_size))
                for task_id in range(self.num_tasks)
            }
            logger.warning(
                f"[Rank {self.local_rank}] Adjusted: group_size={self.group_size}, total_envs={self.total_envs}"
            )

        # Initial reset
        self.env.reset()
        logger.info(f"[Rank {self.local_rank}] Isaac environment initialized successfully on GPU {self.device_id}")
        logger.info(
            f"[Rank {self.local_rank}] Actual envs created: {actual_num_envs} "
            f"(requested: {self.num_tasks} tasks × {self.group_size} group_size)"
        )

    def _init_zmq(self):
        """Initialize ZMQ socket."""
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)

        if self.use_ipc:
            # In distributed mode, each process uses a unique socket file
            if self.distributed:
                address = f"ipc:///tmp/isaac_server_{self.local_rank}.sock"
            else:
                address = "ipc:///tmp/isaac_server.sock"
        else:
            # In distributed mode, port is already adjusted (base_port + local_rank)
            address = f"tcp://*:{self.port}"

        self.socket.bind(address)
        logger.info(f"[Rank {self.local_rank}] ZMQ server bound to {address}")

    def start(self):
        """Start the server."""
        logger.info("Starting Isaac Multi-Task Server...")

        # Initialize environment first
        self._init_env()

        # Initialize ZMQ
        self._init_zmq()

        self.running = True
        logger.info("Server is ready to accept requests")

        # Main server loop
        while self.running:
            try:
                # Wait for request with timeout to allow checking self.running
                if self.socket.poll(timeout=1000):  # 1 second timeout
                    message = self.socket.recv()
                    request = pickle.loads(message)
                    response = self._handle_request(request)
                    self.socket.send(pickle.dumps(response))
            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ error: {e}")
            except Exception as e:
                logger.exception(f"Error handling request: {e}")
                if self.running:
                    # Send error response
                    error_response = {"status": "error", "message": str(e)}
                    self.socket.send(pickle.dumps(error_response))

        self._cleanup()

    def _handle_request(self, request: dict) -> dict:
        """Handle incoming request."""
        cmd = request.get("cmd")

        if cmd == "step":
            return self._handle_step(request)
        elif cmd == "reset":
            return self._handle_reset(request)
        elif cmd == "get_task_mapping":
            return self._handle_get_task_mapping(request)
        elif cmd == "ping":
            return {"status": "ok", "message": "pong"}
        elif cmd == "close":
            self.running = False
            return {"status": "ok", "message": "server shutting down"}
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    def _handle_step(self, request: dict) -> dict:
        """
        Handle step request with action chunk support.

        Request format:
            {
                "cmd": "step",
                "actions": np.ndarray,  # shape: (batch_size, action_dim) or (batch_size, num_chunks, action_dim)
                "env_indices": list[int],  # which envs to step
            }

        Response format:
            {
                "status": "ok",
                "obs": dict,  # observation dict
                "rewards": np.ndarray,  # shape: (batch_size,) or (batch_size, num_chunks)
                "terminations": np.ndarray,
                "truncations": np.ndarray,
                "infos": dict,
            }
        """
        actions = np.array(request["actions"])
        env_indices = request["env_indices"]

        # Log step info
        print(f"[Isaac Rank {self.local_rank}] Step: {len(env_indices)} envs, indices={env_indices}", flush=True)

        # Check if actions have chunk dimension: [num_envs, num_chunks, action_dim]
        if len(actions.shape) == 3:
            return self._handle_chunk_step(actions, env_indices)

        # Single step: [num_envs, action_dim]
        # Build full action tensor (use zeros for non-selected envs)
        full_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
        full_actions[env_indices] = torch.tensor(actions, device=self.device, dtype=torch.float32)

        # Step all envs
        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

        # Extract results only for requested env indices
        response = {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
            "rewards": rewards[env_indices].cpu().numpy(),
            "terminations": terminations[env_indices].cpu().numpy(),
            "truncations": truncations[env_indices].cpu().numpy(),
            "infos": self._extract_infos(infos, env_indices),
        }

        return response

    def _handle_chunk_step(self, chunk_actions: np.ndarray, env_indices: list) -> dict:
        """
        Handle action chunks: execute each chunk sequentially.

        Args:
            chunk_actions: [num_envs, num_chunks, action_dim]
            env_indices: list of env indices to step

        Returns:
            dict with accumulated rewards and final obs
        """
        num_chunks = chunk_actions.shape[1]

        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []

        # Save original render_interval to restore later
        original_render_interval = None
        if self.render_last_only and hasattr(self.env.unwrapped, "cfg"):
            original_render_interval = self.env.unwrapped.cfg.sim.render_interval

        for chunk_idx in range(num_chunks):
            is_last_chunk = chunk_idx == num_chunks - 1

            # Control rendering: disable for intermediate steps, enable for last step
            if self.render_last_only and original_render_interval is not None:
                if is_last_chunk:
                    # Restore original render_interval for the last step
                    self.env.unwrapped.cfg.sim.render_interval = original_render_interval
                else:
                    # Disable rendering for intermediate steps (set to a large value)
                    self.env.unwrapped.cfg.sim.render_interval = 999999

            # Get actions for this chunk
            actions = chunk_actions[:, chunk_idx, :]  # [num_envs, action_dim]

            # Build full action tensor
            full_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
            full_actions[env_indices] = torch.tensor(actions, device=self.device, dtype=torch.float32)

            # Step all envs
            obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

            # Collect results for this chunk
            chunk_rewards.append(rewards[env_indices].cpu().numpy())
            chunk_terminations.append(terminations[env_indices].cpu().numpy())
            chunk_truncations.append(truncations[env_indices].cpu().numpy())

        # Restore original render_interval after all chunks
        if self.render_last_only and original_render_interval is not None:
            self.env.unwrapped.cfg.sim.render_interval = original_render_interval

        # Stack chunk results: [num_envs, num_chunks]
        stacked_rewards = np.stack(chunk_rewards, axis=1)
        stacked_terminations = np.stack(chunk_terminations, axis=1)
        stacked_truncations = np.stack(chunk_truncations, axis=1)

        response = {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),  # Final obs after all chunks
            "rewards": stacked_rewards,
            "terminations": stacked_terminations,
            "truncations": stacked_truncations,
            "infos": self._extract_infos(infos, env_indices),
        }

        return response

    def _handle_reset(self, request: dict) -> dict:
        """
        Handle reset request.

        Request format:
            {
                "cmd": "reset",
                "env_indices": list[int],  # which envs to reset (optional, all if not specified)
                "stabilize": bool,  # whether to run stabilization steps (default: True)
            }

        Response format:
            {
                "status": "ok",
                "obs": dict,  # observation dict
            }

        Note:
            In multi-task Isaac, env_indices are pre-allocated to specific tasks.
            Reset will initialize the robot to initial state for those envs.
            Stabilization runs 10 zero-action steps to let physics settle.
        """
        env_indices = request.get("env_indices", list(range(self.total_envs)))
        stabilize = request.get("stabilize", True)

        # Log reset info
        suffix = "..." if len(env_indices) > 3 else ""
        print(
            f"[Isaac Rank {self.local_rank}] Reset: {len(env_indices)} envs, indices={env_indices}{suffix}",
            flush=True,
        )

        # Get actual number of envs from the Isaac Lab environment
        actual_num_envs = self.env.unwrapped.num_envs

        # Validate env_indices are within bounds
        max_idx = max(env_indices) if env_indices else 0
        min_idx = min(env_indices) if env_indices else 0
        if max_idx >= actual_num_envs:
            raise RuntimeError(
                f"env_indices out of bounds: requested indices [{min_idx}-{max_idx}] "
                f"but environment only has {actual_num_envs} envs. "
                f"Server config: num_tasks={self.num_tasks}, group_size={self.group_size}, "
                f"total_envs={self.total_envs}. "
                f"Check that Isaac Lab actually created {self.total_envs} envs, not {actual_num_envs}."
            )

        # Use Isaac Lab's internal _reset_idx for partial reset
        # This properly resets robot state, scene objects, and managers
        reset_env_ids = torch.tensor(env_indices, device=self.device, dtype=torch.long)

        logger.debug(f"Resetting envs: {env_indices} (total envs: {actual_num_envs})")

        # Call _reset_idx to reset specific environments
        # This resets: scene, event_manager, observation_manager, action_manager, etc.
        self.env.unwrapped._reset_idx(reset_env_ids)

        # Stabilize: run zero-action steps to let physics settle
        # This is important for robot initialization (same as standard IsaacEnv)
        if stabilize:
            zero_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
            for _ in range(10):
                obs, _, _, _, infos = self.env.step(zero_actions)
        else:
            # Get current observations without stepping
            obs = self.env.unwrapped.observation_manager.compute()

        response = {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
        }

        return response

    def _handle_get_task_mapping(self, request: dict) -> dict:
        """Return the task ID to env indices mapping."""
        return {
            "status": "ok",
            "task_to_env_indices": self.task_to_env_indices,
            "num_tasks": self.num_tasks,
            "group_size": self.group_size,
            "total_envs": self.total_envs,
            # Distributed mode info
            "distributed": self.distributed,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "task_offset": self.task_offset if self.distributed else 0,
            "total_num_tasks_global": self.total_num_tasks_global,
        }

    def _to_cpu_numpy(self, value: Any) -> Any:
        """Recursively convert any CUDA tensors to CPU numpy arrays."""
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        elif isinstance(value, dict):
            return {k: self._to_cpu_numpy(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return type(value)(self._to_cpu_numpy(v) for v in value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            # For unknown types, try to convert if it has a tensor-like interface
            if hasattr(value, "cpu") and callable(value.cpu):
                try:
                    return value.cpu().numpy()
                except Exception:
                    pass
            return value

    def _extract_obs(self, obs: Any, env_indices: list) -> dict:
        """Extract observations for specified env indices and ensure all data is CPU numpy."""
        if isinstance(obs, dict):
            result = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value[env_indices].cpu().numpy()
                elif isinstance(value, dict):
                    result[key] = self._extract_obs(value, env_indices)
                elif isinstance(value, np.ndarray):
                    result[key] = value[env_indices]
                else:
                    # Ensure any other types are also converted to CPU
                    result[key] = self._to_cpu_numpy(value)
            return result
        elif isinstance(obs, torch.Tensor):
            return obs[env_indices].cpu().numpy()
        elif isinstance(obs, np.ndarray):
            return obs[env_indices]
        else:
            return self._to_cpu_numpy(obs)

    def _extract_infos(self, infos: dict, env_indices: list) -> dict:
        """Extract infos for specified env indices and ensure all data is CPU numpy."""
        result = {}
        if infos:
            for key, value in infos.items():
                if isinstance(value, torch.Tensor):
                    # Check if tensor is indexable (has at least 1 dimension)
                    if value.dim() > 0 and value.shape[0] >= max(env_indices) + 1:
                        result[key] = value[env_indices].cpu().numpy()
                    else:
                        # Scalar tensor or tensor with fewer elements than needed
                        result[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    # Check if array is indexable
                    if value.ndim > 0 and value.shape[0] >= max(env_indices) + 1:
                        result[key] = value[env_indices]
                    else:
                        result[key] = value
                elif isinstance(value, dict):
                    result[key] = self._extract_infos(value, env_indices)
                else:
                    # Ensure any other types are also converted to CPU
                    result[key] = self._to_cpu_numpy(value)
        return result

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")

        if self.socket:
            self.socket.close()
        if self.ctx:
            self.ctx.term()
        if self.env:
            self.env.close()
        if hasattr(self, "simulation_app") and self.simulation_app:
            self.simulation_app.close()

        logger.info("Server shutdown complete")


def main():
    import os

    parser = argparse.ArgumentParser(description="Isaac Lab Multi-Task Server")
    parser.add_argument(
        "--env_id", type=str, default="Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0", help="Environment ID"
    )
    parser.add_argument("--num_tasks", type=int, default=40, help="Number of tasks (total across all GPUs)")
    parser.add_argument("--group_size", type=int, default=8, help="Number of envs per task")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (base port in distributed mode)")
    parser.add_argument("--use_tcp", action="store_true", help="Use TCP instead of IPC")
    parser.add_argument(
        "--distributed", action="store_true", help="Enable multi-GPU distributed mode (use with torchrun)"
    )
    parser.add_argument(
        "--no_render_last_only",
        action="store_true",
        help="Disable render_last_only optimization (render every step of action chunks)",
    )
    parser.add_argument("--camera_height", type=int, default=256, help="Camera image height")
    parser.add_argument("--camera_width", type=int, default=256, help="Camera image width")

    args = parser.parse_args()

    # Get distributed info from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Auto-detect distributed mode if launched via torchrun
    if world_size > 1 and not args.distributed:
        logger.info(f"Auto-detected distributed mode: WORLD_SIZE={world_size}")
        args.distributed = True

    # Set GROUP_SIZE env var BEFORE any Isaac Lab imports
    # Isaac Lab's MultiTaskLiberoTaskConfig reads this in __post_init__
    os.environ["GROUP_SIZE"] = str(args.group_size)

    # In distributed mode, also set the task offset and num_tasks for this rank
    if args.distributed:
        tasks_per_gpu = args.num_tasks // world_size
        remainder = args.num_tasks % world_size
        if local_rank < remainder:
            local_num_tasks = tasks_per_gpu + 1
            task_offset = local_rank * (tasks_per_gpu + 1)
        else:
            local_num_tasks = tasks_per_gpu
            task_offset = remainder * (tasks_per_gpu + 1) + (local_rank - remainder) * tasks_per_gpu

        # Set environment variables for Isaac Lab's MultiTaskLiberoTaskConfig
        os.environ["TASK_OFFSET"] = str(task_offset)
        os.environ["NUM_TASKS"] = str(local_num_tasks)
        logger.info(
            f"[Rank {local_rank}] Set GROUP_SIZE={args.group_size}, "
            f"TASK_OFFSET={task_offset}, NUM_TASKS={local_num_tasks}"
        )
    else:
        logger.info(f"Set GROUP_SIZE={args.group_size} for Isaac Lab config")

    server = IsaacMultiTaskServer(
        env_id=args.env_id,
        num_tasks=args.num_tasks,
        group_size=args.group_size,
        port=args.port,
        use_ipc=not args.use_tcp,
        distributed=args.distributed,
        render_last_only=not args.no_render_last_only,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )

    server.start()


if __name__ == "__main__":
    main()
