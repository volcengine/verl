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
EnvWorker Server Mode

A lightweight EnvWorker that connects to the Isaac Multi-Task Server
instead of managing its own Isaac instance.

Architecture:
    - Single EnvWorkerServer instance (Coordinator Mode)
    - Handles ALL envs from ALL generation workers
    - Each env produces one trajectory (rollout)
    - Uses hash_key (trajectory_key) to uniquely identify each env

Key Concept:
    - One worker manages multiple envs
    - Each env corresponds to one trajectory (one rollout sequence)
    - Each env has a unique trajectory_key (hash-based UUID)
    - trajectory_key maps to: {env_index, task_id, server_rank}

Env Mapping with Hash Keys:
    Instead of relying on position indices, each env gets a unique hash_key:

    Reset:
        1. Generate unique hash_key for each env (UUID)
        2. Allocate env_index on server based on task_id
        3. Store mapping: hash_key → (env_index, task_id, server_rank)
        4. Return hash_keys to caller (one per env)

    Step:
        1. Receive hash_keys with actions (one action per env)
        2. Validate hash_keys exist in mapping
        3. Dispatch actions to correct env_indices on server
        4. Return observations ordered by hash_keys

    This ensures:
        - Explicit env identity (not position-dependent)
        - Validation at every step (detect mismatches)
        - task_id ↔ env_index binding is maintained throughout trajectory

Usage:
    Replace EnvWorker import with EnvWorkerServer in main_ppo.py:

    # from recipe.vla.workers.env.env_worker import EnvWorker
    from recipe.vla.workers.env import EnvWorkerServer as EnvWorker
"""

import logging
import os
import uuid

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

from recipe.vla.envs.action_utils import prepare_actions, put_info_on_image, save_rollout_video, tile_images
from recipe.vla.isaac_server import IsaacMultiServerClient
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group_ray

logger = logging.getLogger(__name__)


def extract_images_and_states(obs):
    """Extract images and states from Isaac Lab MultiTaskObservationsCfg format.

    Supports two rgb_camera formats:
    1. concatenate_terms=False (dict): {"agentview_cam": [N, H, W, 3], "eye_in_hand_cam": [N, H, W, 3]}
    2. concatenate_terms=True (array): [N, H, W, 6] (agentview + eye_in_hand concatenated on channel dim)

    Expected input format:
    {
        "policy": {"eef_pose": [N, 7], "gripper_pos": [N, 1], ...} or [N, state_dim] (concatenated)
        "rgb_camera": dict or numpy array
    }

    Returns:
    {
        "full_image": [N, H, W, 3] (agentview RGB image for VLA)
        "state": [N, state_dim] (concatenated eef_pose + gripper_pos)
    }
    """
    result = {}

    # Extract image from rgb_camera
    if "rgb_camera" in obs:
        rgb_camera = obs["rgb_camera"]
        if isinstance(rgb_camera, dict):
            # Format 1: concatenate_terms=False, dict with camera names
            if "agentview_cam" in rgb_camera:
                result["full_image"] = rgb_camera["agentview_cam"]  # [N, H, W, 3]
                result["camera_name"] = "agentview"
        elif isinstance(rgb_camera, np.ndarray):
            # Format 2: concatenate_terms=True, concatenated array [N, H, W, C]
            # agentview_cam is the first camera, take first 3 channels
            if rgb_camera.ndim == 4 and rgb_camera.shape[-1] >= 3:
                result["full_image"] = rgb_camera[:, :, :, :3]  # [N, H, W, 3]
                result["camera_name"] = "agentview"

    # Extract state from policy
    if "policy" in obs:
        policy = obs["policy"]
        if isinstance(policy, dict):
            # Format 1: concatenate_terms=False, dict with state terms
            state_list = []
            for key in ["eef_pose", "gripper_pos"]:
                if key in policy:
                    val = policy[key]
                    if hasattr(val, "shape"):
                        if len(val.shape) > 2:
                            val = val.reshape(val.shape[0], -1)
                        state_list.append(val)
            if state_list:
                result["state"] = np.concatenate(state_list, axis=-1)
        elif isinstance(policy, np.ndarray):
            # Format 2: concatenate_terms=True, already concatenated
            result["state"] = policy

    return result


def create_env_batch_dataproto(obs, rews, terminations, truncations, infos, meta=None, task_descriptions=None):
    """Create DataProto from environment outputs."""
    ret_dict = {"obs": obs, "rews": rews, "terminations": terminations, "truncations": truncations, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    # Extract images_and_states from various possible formats
    images_and_states = extract_images_and_states(ret_dict["obs"])

    # Handle numpy arrays (from server response)
    tensor_batch = {
        "full_image": torch.from_numpy(images_and_states["full_image"]).cpu().contiguous(),
        "state": torch.from_numpy(images_and_states["state"]).cpu().contiguous(),
        "rews": torch.from_numpy(ret_dict["rews"]).cpu().contiguous(),
        "terminations": torch.from_numpy(ret_dict["terminations"]).cpu().contiguous(),
        "truncations": torch.from_numpy(ret_dict["truncations"]).cpu().contiguous(),
    }
    # Use provided task_descriptions, fallback to obs, then empty
    if task_descriptions is None:
        task_descriptions = obs.get("task_descriptions", [])
    non_tensor_batch = {"task_descriptions": task_descriptions}
    output = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    return output


class EnvWorkerServer(Worker):
    """
    EnvWorker that connects to Isaac Multi-Task Server.

    This is a drop-in replacement for EnvWorker when using server mode.
    The main differences:
    1. No local Isaac instance - connects to remote server
    2. Uses task_id to select correct env_indices on server
    3. Compatible with the same EnvLoop pipeline as standard EnvWorker

    Supports two server configurations:
    1. Single server mode: One Isaac server handles all tasks
    2. Distributed mode: Multiple Isaac servers (one per GPU), each handles subset of tasks

    Task-Env Binding:
        In multi-task Isaac, each task has a fixed set of env_indices.
        During reset, we allocate env_indices based on task_id.
        This binding is maintained throughout all subsequent step calls.

        Example (10 tasks, group_size=8):
            task_id=0 → env_indices [0-7]
            task_id=1 → env_indices [8-15]
            ...

        When a stage is assigned task_id=3, it will use env_indices from [24-31].
    """

    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.cfg = config

        # Server configuration
        self.server_address = config.train.get("isaac_server_address", "ipc:///tmp/isaac_server")
        self.num_servers_per_group = config.train.get(
            "num_isaac_servers", 8
        )  # Servers per group (typically = num GPUs)
        self.use_tcp = config.train.get("isaac_server_use_tcp", True)

        # Multi-server-group configuration
        # num_server_groups MUST match env.rollout.pipeline_stage_num
        # Each pipeline stage uses its own server group for physical isolation
        self.num_server_groups = config.train.get("num_server_groups", 2)
        # Base ports for each server group, auto-generated if not provided
        # Default: [5556, 5606, 5656, ...] with 50-port spacing
        default_base_ports = [5556 + i * 50 for i in range(self.num_server_groups)]
        self.server_base_ports = config.train.get("server_base_ports", default_base_ports)

        # Client (will be initialized in init_worker)
        self.client: IsaacMultiServerClient = None

        # Stage configuration
        self.stage_num = self.cfg.rollout.pipeline_stage_num  # stages per original worker (e.g., 2)
        self.num_envs = self.cfg.train.num_envs  # envs per stage (e.g., 8)
        self.total_envs = self.cfg.train.get("total_envs", 128)

        # Initialize distributed
        initialize_global_process_group_ray(timeout_second=None)
        device_name = get_device_name()
        env_device_mesh = init_device_mesh(device_name, mesh_shape=(self.world_size, 1), mesh_dim_names=["dp", "tp"])
        self._register_dispatch_collect_info("env", dp_rank=env_device_mesh["dp"].get_local_rank(), is_collect=True)

        # Hash-key based env mapping (set during reset, used during step)
        # Key: trajectory_key (str) - unique ID for each env
        # Value: dict with env_index, task_id, server_rank
        # Note: Each env produces one trajectory, so trajectory_key == env_key
        self._trajectory_registry: dict[str, dict] = {}

        # Track allocation offset within each task's env pool
        # (to handle multiple envs sharing the same task_id)
        self._task_allocation_offset: dict[int, int] = {}

        # Current active trajectory_keys (ordered list, set during reset)
        # One key per env, len == total_envs
        self._active_trajectory_keys: list[str] = []

        # Video saving - organized by stage and task_id
        # Directory structure: video_base_dir/train/stage_{id}/task_{id}/
        self.video_cfg = self.cfg.train.get("video_cfg", {})
        self.save_video = self.video_cfg.get("save_video", False)
        self.video_base_dir = self.video_cfg.get("video_base_dir", "/tmp/videos")
        # render_images[stage_id][task_id] = list of frames
        self.render_images: dict[int, dict[int, list[np.ndarray]]] = {}
        self.video_cnt: dict[int, dict[int, int]] = {}  # video_cnt[stage_id][task_id]
        self.camera_view = "agentview"  # Primary camera used for video

        logger.info(f"[rank={self.rank}] EnvWorkerServer initialized (Coordinator Mode)")
        logger.info(f"[rank={self.rank}] Config: server={self.server_address}, tcp={self.use_tcp}")
        logger.info(
            f"[rank={self.rank}] Server groups: {self.num_server_groups} "
            f"(must match pipeline_stage_num={self.stage_num})"
        )
        logger.info(f"[rank={self.rank}] Servers per group: {self.num_servers_per_group}")
        logger.info(f"[rank={self.rank}] Server base ports: {self.server_base_ports}")
        logger.info(f"[rank={self.rank}] Total envs: {self.total_envs}, video_saving={self.save_video}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_worker(self):
        """Initialize the client connection to Isaac server groups.

        Uses IsaacMultiServerClient which supports configurable number of server groups.
        Each server group corresponds to one pipeline stage for physical isolation.

        Note:
            num_server_groups MUST match env.rollout.pipeline_stage_num.
            If they don't match, training will fail with incorrect stage routing.
        """
        # Validate configuration
        if self.num_server_groups != self.stage_num:
            raise ValueError(
                f"num_server_groups ({self.num_server_groups}) must match "
                f"pipeline_stage_num ({self.stage_num}). "
                f"Each pipeline stage requires its own server group for physical isolation."
            )

        logger.info(f"[rank={self.rank}] Connecting to Isaac server groups")
        for i, port in enumerate(self.server_base_ports):
            logger.info(f"[rank={self.rank}] Server group {i} base port: {port}")

        self.client = IsaacMultiServerClient(
            num_server_groups=self.num_server_groups,
            num_servers_per_group=self.num_servers_per_group,
            base_ports=self.server_base_ports,
            base_address=self.server_address,
            use_tcp=self.use_tcp,
        )

        if not self.client.connect():
            raise RuntimeError("Failed to connect to Isaac server groups")

        logger.info(
            f"[rank={self.rank}] Connected to Isaac servers: "
            f"{self.num_server_groups} groups × {self.num_servers_per_group} servers/group, "
            f"{self.client.num_tasks} tasks × {self.client.group_size} group_size"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_simulator(self):
        """No-op for server mode - server manages the simulator."""
        logger.info(f"[rank={self.rank}] init_simulator called (no-op in server mode)")
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def env_interact_step(self, data: DataProto) -> DataProto:
        """
        Interact with the environment through the server.

        Coordinator Mode: Receives actions with trajectory_keys for validation.
        Each trajectory_key identifies one env (one env = one trajectory).
        Uses the env_indices that were assigned during reset_envs_to_state_ids.
        This ensures task_id-env_index binding is maintained throughout the trajectory.

        For distributed mode:
        - Groups actions by server_rank
        - Sends batched requests to each server
        - Aggregates responses in the original order

        Args:
            data: DataProto containing:
                - non_tensor_batch["actions"]: actions array (N, action_dim)
                - non_tensor_batch["trajectory_keys"]: list of hash_keys for validation
                - meta_info["stage_id"]: stage identifier (for compatibility)

        Returns:
            DataProto with observations, rewards, dones (ordered by trajectory_keys)
        """
        chunk_actions = data.non_tensor_batch["actions"]
        stage_id: int = data.meta_info["stage_id"]

        # Get trajectory_keys if provided (new method), otherwise fallback to stage-based
        trajectory_keys = data.non_tensor_batch.get("trajectory_keys", None)

        if trajectory_keys is not None:
            # Validate trajectory_keys exist in registry
            self._validate_trajectory_keys(trajectory_keys)
        else:
            # Fallback: use active keys in order (for backward compatibility)
            # Actions should be in the same order as reset
            logger.warning(
                f"[rank={self.rank}] trajectory_keys not provided in step, "
                f"using order from reset. This may cause misalignment!"
            )
            # Calculate which portion of active keys this stage_id corresponds to
            # stage_id -> which slice of _active_trajectory_keys
            stage_start = stage_id * self.num_envs
            stage_end = (stage_id + 1) * self.num_envs
            trajectory_keys = self._active_trajectory_keys[stage_start:stage_end]
            self._validate_trajectory_keys(trajectory_keys)

        # Prepare actions
        chunk_actions = prepare_actions(
            simulator_type=self.cfg.train.simulator_type,
            raw_chunk_actions=chunk_actions,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
        )

        # Convert to numpy for transmission
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # Get step progress info for logging
        step_idx = data.meta_info.get("step_idx", 0)
        max_steps = data.meta_info.get("max_steps", 1)

        # Route to stage-specific server group
        response = self._server_group_step(chunk_actions, trajectory_keys, stage_id, step_idx, max_steps)

        if response is None or response.get("status") != "ok":
            raise RuntimeError(f"[rank={self.rank}] Step request failed: {response}")

        # Build response DataProto
        env_info_list = {}

        terminations = response["terminations"]
        truncations = response["truncations"]
        chunk_dones = np.logical_or(terminations, truncations)

        if chunk_dones.any():
            infos = response.get("infos", {})
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info.get("episode", {}):
                    env_info_list[key] = final_info["episode"][key][chunk_dones[:, -1]]

        # Generate task_descriptions from trajectory registry
        task_descriptions = []
        for key in trajectory_keys:
            task_id = self._trajectory_registry[key]["task_id"]
            task_descriptions.append(f"Task {task_id}")

        env_batch = create_env_batch_dataproto(
            obs=response["obs"],
            rews=response["rewards"],
            terminations=terminations,
            truncations=truncations,
            infos=response.get("infos", {}),
            meta=env_info_list,
            task_descriptions=task_descriptions,
        )

        # Collect images for video saving - organized by stage and task_id
        if self.save_video:
            images_and_states = extract_images_and_states(response["obs"])
            if "full_image" in images_and_states:
                # Track camera name for video naming
                if "camera_name" in images_and_states:
                    self.camera_view = images_and_states["camera_name"]

                full_images = images_and_states["full_image"]  # [N, H, W, C]
                rewards = response.get("rewards", np.zeros(len(trajectory_keys)))
                terminations = response.get("terminations", np.zeros(len(trajectory_keys)))

                # Group images by stage_id and task_id
                if stage_id not in self.render_images:
                    self.render_images[stage_id] = {}
                    self.video_cnt[stage_id] = {}

                # Group images by task_id within this stage, with info overlay
                task_id_to_images = {}
                for i, key in enumerate(trajectory_keys):
                    task_id = self._trajectory_registry[key]["task_id"]
                    img = full_images[i]

                    # Add info overlay (like IsaacEnv.add_new_frames)
                    # Handle chunk rewards: take last step or mean
                    if len(rewards.shape) > 1:
                        reward_val = float(rewards[i, -1])  # Last chunk step
                    else:
                        reward_val = float(rewards[i])

                    if len(terminations.shape) > 1:
                        term_val = bool(terminations[i, -1])
                    else:
                        term_val = bool(terminations[i])

                    plot_info = {
                        "reward": reward_val,
                        "done": term_val,
                        "task": task_id,
                    }
                    img = put_info_on_image(img, plot_info)

                    if task_id not in task_id_to_images:
                        task_id_to_images[task_id] = []
                    task_id_to_images[task_id].append(img)

                # Tile and store images for each task_id
                for task_id, task_images in task_id_to_images.items():
                    if task_id not in self.render_images[stage_id]:
                        self.render_images[stage_id][task_id] = []
                        self.video_cnt[stage_id][task_id] = 0

                    nrows = int(np.ceil(np.sqrt(len(task_images))))
                    tiled = tile_images(task_images, nrows=nrows)
                    self.render_images[stage_id][task_id].append(tiled)

        return env_batch

    def _server_group_step(
        self,
        chunk_actions: np.ndarray,
        trajectory_keys: list,
        stage_id: int,
        step_idx: int = 0,
        max_steps: int = 1,
        render_last_only: bool = True,
    ) -> dict:
        """
        Execute step on the stage-specific server group.

        Each pipeline stage has its own server group for physical isolation:
        - Stage 0 → Server Group 0
        - Stage 1 → Server Group 1
        - ...

        This provides physical isolation - one stage's sim won't affect other stages' env states.

        Args:
            chunk_actions: Actions array [N, ...]
            trajectory_keys: List of hash_keys (used to look up env_indices from registry)
            stage_id: Pipeline stage ID (determines which server group to use)
            step_idx: Current step index (for logging)
            max_steps: Total number of steps (for logging)
            render_last_only: If True, only render the last step of the action chunk

        Returns:
            Aggregated response dict with obs, rewards, terminations, truncations, infos
        """
        from collections import defaultdict

        # === Phase 1: Group by server_rank within this stage's server group ===
        server_groups = defaultdict(lambda: {"indices": [], "actions": [], "original_positions": []})

        for i, key in enumerate(trajectory_keys):
            info = self._trajectory_registry[key]
            server_rank = info["server_rank"]
            server_groups[server_rank]["indices"].append(info["env_index"])
            server_groups[server_rank]["actions"].append(chunk_actions[i])
            server_groups[server_rank]["original_positions"].append(i)

        # Log step info
        server_summary = {rank: len(group["indices"]) for rank, group in server_groups.items()}
        print(
            f"[EnvWorker] Step {step_idx + 1}/{max_steps} Stage {stage_id}: "
            f"{len(trajectory_keys)} envs -> {len(server_groups)} servers {server_summary}",
            flush=True,
        )

        # === Phase 2: Build batched requests ===
        server_requests = {}
        position_map = {}
        for rank, group in server_groups.items():
            actions = np.array(group["actions"])
            indices = group["indices"]
            server_requests[rank] = (actions, indices)
            position_map[rank] = group["original_positions"]

        # === Phase 3: Send requests to stage-specific server group ===
        # Key difference: pass stage_id to route to correct server group
        # render_last_only: only render the last step of the action chunk for efficiency
        responses = self.client.step_batched(server_requests, stage_id=stage_id, render_last_only=render_last_only)

        # Validate responses
        all_responses = {}
        for rank, response in responses.items():
            if response is None or response.get("status") != "ok":
                raise RuntimeError(
                    f"[rank={self.rank}] Step request to stage {stage_id} server {rank} failed: {response}"
                )
            all_responses[rank] = {
                "response": response,
                "original_positions": position_map[rank],
            }

        # === Phase 4: Aggregate responses ===
        num_envs = len(trajectory_keys)
        first_response = next(iter(all_responses.values()))["response"]
        rewards_shape = first_response["rewards"].shape[1:] if len(first_response["rewards"].shape) > 1 else ()
        term_shape = first_response["terminations"].shape[1:] if len(first_response["terminations"].shape) > 1 else ()

        aggregated_rewards = np.zeros((num_envs,) + rewards_shape, dtype=first_response["rewards"].dtype)
        aggregated_terminations = np.zeros((num_envs,) + term_shape, dtype=first_response["terminations"].dtype)
        aggregated_truncations = np.zeros((num_envs,) + term_shape, dtype=first_response["truncations"].dtype)

        def init_aggregated_obs(obs_dict, num_envs):
            result = {}
            for key, value in obs_dict.items():
                if isinstance(value, dict):
                    result[key] = init_aggregated_obs(value, num_envs)
                elif isinstance(value, np.ndarray):
                    result[key] = np.zeros((num_envs,) + value.shape[1:], dtype=value.dtype)
                else:
                    result[key] = [None] * num_envs
            return result

        aggregated_obs = init_aggregated_obs(first_response["obs"], num_envs)

        def fill_aggregated_obs(agg_obs, response_obs, positions):
            for key, value in response_obs.items():
                if isinstance(value, dict):
                    fill_aggregated_obs(agg_obs[key], value, positions)
                elif isinstance(value, np.ndarray):
                    for i, pos in enumerate(positions):
                        agg_obs[key][pos] = value[i]
                else:
                    for i, pos in enumerate(positions):
                        agg_obs[key][pos] = value[i] if hasattr(value, "__getitem__") else value

        for server_rank, data in all_responses.items():
            response = data["response"]
            positions = data["original_positions"]

            for i, pos in enumerate(positions):
                aggregated_rewards[pos] = response["rewards"][i]
                aggregated_terminations[pos] = response["terminations"][i]
                aggregated_truncations[pos] = response["truncations"][i]

            fill_aggregated_obs(aggregated_obs, response["obs"], positions)

        return {
            "status": "ok",
            "obs": aggregated_obs,
            "rewards": aggregated_rewards,
            "terminations": aggregated_terminations,
            "truncations": aggregated_truncations,
            "infos": {},
        }

    def _validate_trajectory_keys(self, trajectory_keys: list[str]) -> None:
        """
        Validate that all trajectory_keys exist in the registry.

        Each trajectory_key uniquely identifies one env. One env produces one trajectory.

        Args:
            trajectory_keys: List of hash_keys to validate (one key per env/trajectory)

        Raises:
            RuntimeError if any hash_key is invalid
        """
        invalid_keys = [key for key in trajectory_keys if key not in self._trajectory_registry]

        if invalid_keys:
            suffix = "..." if len(invalid_keys) > 5 else ""
            raise RuntimeError(
                f"[rank={self.rank}] Invalid trajectory_keys: {invalid_keys[:5]}{suffix}. "
                f"Valid keys: {list(self._trajectory_registry.keys())[:5]}..."
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_all_state_ids(self):
        """Get all available state IDs."""
        # In multi-task mode, state_id maps to task_id
        return list(range(self.client.num_tasks))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def reset_envs_to_state_ids(self, data: DataProto):
        """
        Reset environments to specified state IDs (task IDs).

        Coordinator Mode: This single worker receives ALL state_ids from ALL generation gpus.

        Key Concept:
            - Each env produces one trajectory
            - trajectory_key is a unique ID for each env
            - total_envs = num_workers × envs_per_worker

        For each env in the batch:
            1. Generate a unique trajectory_key (UUID) per env
            2. Allocate env_index on server based on task_id
            3. Store mapping: trajectory_key → {env_index, task_id, server_rank}
            4. Return trajectory_keys (one per env) for use in subsequent steps

        The per-env trajectory_key design:
            - Each env has its own UUID
            - len(trajectory_keys) == total_envs == batch_size of observations
            - One trajectory_key identifies one env, which produces one trajectory

        Example (128 envs total, each env produces 1 trajectory):
            trajectory_key "abc12345": task_id=3 → server_rank=0, env_index=39
            trajectory_key "def67890": task_id=3 → server_rank=0, env_index=40
            ...
        """
        state_ids_list = list(data.non_tensor_batch["state_ids"])
        task_ids_list = list(data.non_tensor_batch["task_ids"])

        logger.debug(
            f"[rank={self.rank}] reset_envs_to_state_ids: {len(state_ids_list)} envs, "
            f"unique_tasks={len(set(task_ids_list))}, server_groups={self.num_server_groups}"
        )

        # Coordinator mode: we receive ALL state_ids (total_envs)
        expected_count = self.total_envs  # e.g., 128
        assert len(state_ids_list) == expected_count, (
            f"[rank={self.rank}] Expected {expected_count} state_ids, got {len(state_ids_list)}. "
            f"(total_envs={self.total_envs})"
        )

        # Clear previous mappings
        self._trajectory_registry.clear()
        self._task_allocation_offset.clear()
        self._active_trajectory_keys.clear()

        trajectory_keys = []  # One key per env (not per trajectory)

        # Number of trajectories = total_envs / num_envs (for reset grouping)
        num_trajectories = self.total_envs // self.num_envs

        # === Phase 1: Collect all env_indices and build trajectory registry ===
        # Group by server_rank for batched reset
        from collections import defaultdict

        server_env_groups = defaultdict(lambda: {"env_indices": [], "traj_indices": []})
        traj_to_server = {}  # traj_idx -> server_rank

        for traj_idx in range(num_trajectories):
            # Get task ids for this trajectory group
            traj_start = traj_idx * self.num_envs
            traj_end = (traj_idx + 1) * self.num_envs
            traj_task_ids = task_ids_list[traj_start:traj_end]

            # For Isaac multi-task, all envs in a reset group must have the same task_id
            unique_task_ids = set(traj_task_ids)
            if len(unique_task_ids) != 1:
                raise RuntimeError(
                    f"[rank={self.rank}] Trajectory {traj_idx} has mixed task_ids: {unique_task_ids}. "
                    f"Isaac requires all envs in a trajectory to run the same task."
                )

            task_id = traj_task_ids[0]

            # Get server's env_indices for this task
            task_env_indices = self.client.get_env_indices_for_task(task_id)
            group_size = len(task_env_indices)

            # Get which server handles this task (same across all server groups)
            server_rank = self.client.get_server_for_task(task_id)

            # Use sequential allocation within each task's env pool
            offset = self._task_allocation_offset.get(task_id, 0)

            if offset + self.num_envs > group_size:
                raise RuntimeError(
                    f"[rank={self.rank}] Task {task_id} has only {group_size} envs on server, "
                    f"but need {offset + self.num_envs} (offset={offset}, num_envs={self.num_envs}). "
                    f"Increase server group_size or reduce trajectories sharing this task."
                )

            env_indices = task_env_indices[offset : offset + self.num_envs]
            self._task_allocation_offset[task_id] = offset + self.num_envs

            # Generate unique hash_key for EACH env (not per trajectory)
            traj_env_keys = []
            for i, env_idx in enumerate(env_indices):
                env_key = str(uuid.uuid4())[:8]  # Short UUID for readability
                self._trajectory_registry[env_key] = {
                    "env_index": env_idx,  # Single env index (local to server)
                    "task_id": task_id,
                    "traj_idx": traj_idx,
                    "server_rank": server_rank,  # Which server handles this env
                }
                traj_env_keys.append(env_key)

            # Add all env keys from this trajectory to the full list
            trajectory_keys.extend(traj_env_keys)

            # Group env_indices by server for batched reset
            server_env_groups[server_rank]["env_indices"].extend(env_indices)
            server_env_groups[server_rank]["traj_indices"].append(traj_idx)
            traj_to_server[traj_idx] = server_rank

            logger.info(
                f"[rank={self.rank}] Traj {traj_idx}: task_id={task_id}, server_rank={server_rank}, "
                f"env_indices [{env_indices[0]}-{env_indices[-1]}], keys: {traj_env_keys[:2]}..."
            )

        # === Phase 2: Batched reset - Reset ALL server groups ===
        # Each stage has its own server group, we need to reset all of them
        # because each stage will need the same initial states
        print(
            f"[EnvWorker] Batched Reset: {num_trajectories} trajectories -> "
            f"{len(server_env_groups)} server(s) x {self.num_server_groups} groups",
            flush=True,
        )

        server_requests = {rank: group["env_indices"] for rank, group in server_env_groups.items()}

        # Reset all server groups (one per stage)
        # Store responses per stage: server_responses[stage_id][rank] = obs
        server_responses_per_stage: dict[int, dict[int, dict]] = {}
        for stage_id in range(self.num_server_groups):
            print(f"[EnvWorker] Reset: Server group {stage_id}", flush=True)
            stage_responses = self.client.reset_batched(server_requests, stage_id=stage_id)

            # Validate and store responses from each server group
            server_responses_per_stage[stage_id] = {}
            for rank, response in stage_responses.items():
                if response is None or response.get("status") != "ok":
                    raise RuntimeError(
                        f"[rank={self.rank}] Reset to server group {stage_id} server {rank} failed: {response}"
                    )
                server_responses_per_stage[stage_id][rank] = response["obs"]

        # === Phase 3: Reconstruct result_list in trajectory order ===
        # Each trajectory uses observations from its corresponding stage's server group
        # Trajectories are interleaved across stages: traj 0 → stage 0, traj 1 → stage 1, ...
        # Track position within each (stage, server_rank) response
        server_positions_per_stage = {
            stage_id: {rank: 0 for rank in server_env_groups.keys()} for stage_id in range(self.num_server_groups)
        }
        result_list = []

        for traj_idx in range(num_trajectories):
            # Determine which stage will handle this trajectory (interleaved assignment)
            stage_id = traj_idx % self.num_server_groups
            server_rank = traj_to_server[traj_idx]
            obs = server_responses_per_stage[stage_id][server_rank]
            pos = server_positions_per_stage[stage_id][server_rank]

            # Extract this trajectory's observations from server response
            traj_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    traj_obs[key] = value[pos : pos + self.num_envs]
                elif isinstance(value, dict):
                    traj_obs[key] = {}
                    for k2, v2 in value.items():
                        if isinstance(v2, np.ndarray):
                            traj_obs[key][k2] = v2[pos : pos + self.num_envs]
                        else:
                            traj_obs[key][k2] = v2
                else:
                    traj_obs[key] = value

            result_list.append(traj_obs)
            server_positions_per_stage[stage_id][server_rank] += self.num_envs

        # Store active keys for fallback (backward compatibility)
        self._active_trajectory_keys = trajectory_keys

        logger.info(
            f"[rank={self.rank}] Reset complete: {len(trajectory_keys)} env keys registered "
            f"({num_trajectories} trajectories × {self.num_envs} envs)"
        )

        # Concatenate results from all trajectories
        output_tensor_dict = {}
        output_non_tensor_dict = {}

        # Use global extract_images_and_states to handle various obs formats
        images_and_states_list = [extract_images_and_states(obs) for obs in result_list]
        if images_and_states_list and images_and_states_list[0]:
            for k in images_and_states_list[0].keys():
                if isinstance(images_and_states_list[0][k], np.ndarray):
                    output_tensor_dict[k] = torch.from_numpy(
                        np.concatenate([obs[k] for obs in images_and_states_list], axis=0)
                    )

        # Handle 'task_descriptions'
        task_descriptions = []
        for traj_idx, obs in enumerate(result_list):
            if "task_descriptions" in obs:
                task_descriptions.extend(obs["task_descriptions"])
            else:
                # Fallback: generate from task_ids
                # trajectory_keys is per-env, so get the first env key of this trajectory
                env_key = trajectory_keys[traj_idx * self.num_envs]
                task_id = self._trajectory_registry[env_key]["task_id"]
                task_descriptions.extend([f"Task {task_id}"] * self.num_envs)
        output_non_tensor_dict["task_descriptions"] = task_descriptions

        # trajectory_keys: one key per env, same length as batch size
        output_non_tensor_dict["trajectory_keys"] = trajectory_keys

        output = DataProto.from_dict(tensors=output_tensor_dict, non_tensors=output_non_tensor_dict)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_rollout(self, mode="train"):
        """Finish rollout - save videos organized by stage and task_id.

        Directory structure: video_base_dir/mode/stage_{id}/task_{id}/
        """
        if self.save_video and self.render_images:
            total_videos = 0
            for stage_id, stage_data in self.render_images.items():
                for task_id, frames in stage_data.items():
                    if not frames:
                        continue

                    # Directory: video_base_dir/train/stage_0/task_3/
                    output_dir = os.path.join(self.video_base_dir, mode, f"stage_{stage_id}", f"task_{task_id}")
                    os.makedirs(output_dir, exist_ok=True)

                    # Get video count for this stage/task
                    cnt = self.video_cnt.get(stage_id, {}).get(task_id, 0)
                    video_name = f"rollout_{cnt:04d}_{self.camera_view}"

                    save_rollout_video(frames, output_dir, video_name)

                    # Increment counter
                    self.video_cnt[stage_id][task_id] = cnt + 1
                    total_videos += 1

            # Clear for next rollout
            self.render_images = {}
            logger.info(f"[rank={self.rank}] Saved {total_videos} videos to {self.video_base_dir}/{mode}/")
        return

    def __del__(self):
        """Clean up client connection."""
        if self.client:
            self.client.close()
