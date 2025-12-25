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
Isaac Lab Server Client

A lightweight client for communicating with the Isaac Lab Multi-Task Server.
Used by EnvWorker to send step/reset commands to the server.

Architecture:
    - Multiple server groups (one per pipeline stage)
    - Each group has N servers (one per GPU)
    - All groups share the same GPUs (time-multiplexed)

Note:
    num_server_groups must match env.rollout.pipeline_stage_num in training config.
"""

import logging
import pickle
import time
from typing import Optional

import numpy as np
import zmq

logger = logging.getLogger(__name__)


class IsaacClient:
    """Client for communicating with Isaac Multi-Task Server."""

    def __init__(
        self,
        server_address: str = "ipc:///tmp/isaac_server.sock",
        timeout_ms: int = 30000,
        max_retries: int = 3,
    ):
        """
        Initialize the client.

        Args:
            server_address: ZMQ address of the server (ipc:// or tcp://)
            timeout_ms: Timeout for requests in milliseconds
            max_retries: Maximum number of retries for failed requests
        """
        self.server_address = server_address
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

        self.ctx = zmq.Context()
        self.socket = None
        self._connected = False

        # Cached task mapping
        self._task_to_env_indices = None
        self._num_tasks = None
        self._group_size = None
        self._total_envs = None

    def connect(self) -> bool:
        """Connect to the server."""
        if self._connected:
            return True

        try:
            self.socket = self.ctx.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(self.server_address)

            # Test connection with ping
            response = self._send_request({"cmd": "ping"})
            if response and response.get("status") == "ok":
                self._connected = True
                logger.info(f"Connected to Isaac server at {self.server_address}")

                # Get task mapping
                self._fetch_task_mapping()
                return True
            else:
                logger.error("Failed to ping server")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    def _fetch_task_mapping(self):
        """Fetch task to env indices mapping from server."""
        response = self._send_request({"cmd": "get_task_mapping"})
        if response and response.get("status") == "ok":
            self._task_to_env_indices = response["task_to_env_indices"]
            self._num_tasks = response["num_tasks"]
            self._group_size = response["group_size"]
            self._total_envs = response["total_envs"]
            logger.info(
                f"Task mapping: {self._num_tasks} tasks, {self._group_size} group_size, {self._total_envs} total envs"
            )

    def disconnect(self):
        """Disconnect from the server."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self._connected = False

    def _send_request(self, request: dict, retries: int = None) -> Optional[dict]:
        """Send a request to the server and return the response."""
        if retries is None:
            retries = self.max_retries

        for attempt in range(retries):
            try:
                self.socket.send(pickle.dumps(request))
                response = pickle.loads(self.socket.recv())
                return response

            except zmq.Again:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    # Recreate socket for retry
                    self._reconnect()
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff

            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < retries - 1:
                    self._reconnect()
                    time.sleep(0.1 * (attempt + 1))

        return None

    def _reconnect(self):
        """Reconnect to the server."""
        if self.socket:
            self.socket.close()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.server_address)

    def step(self, actions: np.ndarray, env_indices: list, render_last_only: bool = True) -> Optional[dict]:
        """
        Send step command to server.

        Args:
            actions: Actions array, shape (len(env_indices), action_dim)
            env_indices: List of env indices to step
            render_last_only: If True, only render the last step of action chunk

        Returns:
            dict with obs, rewards, terminations, truncations, infos
            or None if request failed
        """
        request = {
            "cmd": "step",
            "actions": actions,
            "env_indices": env_indices,
            "render_last_only": render_last_only,
        }
        return self._send_request(request)

    def reset(self, env_indices: Optional[list] = None) -> Optional[dict]:
        """
        Send reset command to server.

        Args:
            env_indices: List of env indices to reset (None for all)

        Returns:
            dict with obs, or None if request failed
        """
        request = {"cmd": "reset"}
        if env_indices is not None:
            request["env_indices"] = env_indices
        return self._send_request(request)

    def get_env_indices_for_task(self, task_id: int) -> list:
        """Get env indices for a given task ID."""
        if self._task_to_env_indices is None:
            self._fetch_task_mapping()
        return self._task_to_env_indices.get(task_id, [])

    @property
    def task_to_env_indices(self) -> dict:
        """Get task to env indices mapping."""
        if self._task_to_env_indices is None:
            self._fetch_task_mapping()
        return self._task_to_env_indices

    @property
    def num_tasks(self) -> int:
        """Get number of tasks."""
        if self._num_tasks is None:
            self._fetch_task_mapping()
        return self._num_tasks

    @property
    def group_size(self) -> int:
        """Get group size (envs per task)."""
        if self._group_size is None:
            self._fetch_task_mapping()
        return self._group_size

    @property
    def total_envs(self) -> int:
        """Get total number of envs."""
        if self._total_envs is None:
            self._fetch_task_mapping()
        return self._total_envs

    def close(self):
        """Close the client and optionally the server."""
        self.disconnect()
        if self.ctx:
            self.ctx.term()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class IsaacDistributedClient:
    """
    Client for connecting to multiple distributed Isaac servers.

    In distributed mode, there are N Isaac servers (one per GPU), each handling
    a subset of tasks. This client:
    1. Connects to all N servers
    2. Routes requests to the correct server based on task_id
    3. Provides a unified interface that looks like a single server

    Architecture:
        Client ─┬─> Server 0 (GPU 0): tasks 0-4,  ports 5555, socket isaac_server_0.sock
                ├─> Server 1 (GPU 1): tasks 5-9,  ports 5556, socket isaac_server_1.sock
                ├─> Server 2 (GPU 2): tasks 10-14, ports 5557, socket isaac_server_2.sock
                └─> ...
    """

    def __init__(
        self,
        base_address: str = "ipc:///tmp/isaac_server",
        num_servers: int = 8,
        timeout_ms: int = 30000,
        max_retries: int = 3,
        use_tcp: bool = False,
        base_port: int = 5555,
    ):
        """
        Initialize the distributed client.

        Args:
            base_address: Base address for IPC sockets (will append _{rank}.sock)
            num_servers: Number of servers to connect to
            timeout_ms: Timeout for requests in milliseconds
            max_retries: Maximum number of retries for failed requests
            use_tcp: If True, use TCP instead of IPC
            base_port: Base port for TCP mode (each server uses base_port + rank)
        """
        from concurrent.futures import ThreadPoolExecutor

        self.num_servers = num_servers
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.use_tcp = use_tcp
        self.base_port = base_port
        self.base_address = base_address

        # Persistent thread pool for concurrent requests (one thread per server)
        self._executor = ThreadPoolExecutor(max_workers=num_servers)

        # Create individual clients for each server
        self.clients: list[IsaacClient] = []
        for rank in range(num_servers):
            if use_tcp:
                # Extract host from base_address if it contains one
                # e.g., "tcp://10.0.0.1" -> "tcp://10.0.0.1:5555"
                if base_address.startswith("tcp://"):
                    host = base_address.replace("tcp://", "")
                    address = f"tcp://{host}:{base_port + rank}"
                else:
                    address = f"tcp://localhost:{base_port + rank}"
            else:
                address = f"{base_address}_{rank}.sock"

            print(f"[DEBUG isaac_client] Creating client for server {rank}: {address}", flush=True)
            client = IsaacClient(
                server_address=address,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
            )
            self.clients.append(client)

        # Global task mapping (task_id -> server_rank)
        self._global_task_to_server: dict[int, int] = {}
        # Global task to env indices (global task_id -> local env indices on that server)
        self._global_task_to_env_indices: dict[int, list[int]] = {}
        # Server info
        self._server_task_offsets: list[int] = []
        self._total_tasks = 0
        self._group_size = 0
        self._total_envs = 0

    def connect(self) -> bool:
        """Connect to all servers."""
        all_connected = True
        for rank, client in enumerate(self.clients):
            if not client.connect():
                logger.error(f"Failed to connect to server {rank}")
                all_connected = False
            else:
                logger.info(f"Connected to server {rank}")

        if all_connected:
            self._build_global_task_mapping()

        return all_connected

    def _build_global_task_mapping(self):
        """Build global task mapping from all servers."""
        self._global_task_to_server.clear()
        self._global_task_to_env_indices.clear()
        self._server_task_offsets.clear()

        for rank, client in enumerate(self.clients):
            # Fetch task mapping from this server
            response = client._send_request({"cmd": "get_task_mapping"})
            if response and response.get("status") == "ok":
                task_offset = response.get("task_offset", 0)
                num_tasks = response["num_tasks"]
                group_size = response["group_size"]
                task_to_env_indices = response["task_to_env_indices"]

                self._server_task_offsets.append(task_offset)
                self._group_size = group_size

                # Map global task_ids to this server
                for local_task_id in range(num_tasks):
                    global_task_id = task_offset + local_task_id
                    self._global_task_to_server[global_task_id] = rank
                    # Store local env indices for this global task
                    self._global_task_to_env_indices[global_task_id] = task_to_env_indices[local_task_id]

                logger.info(
                    f"Server {rank}: tasks {task_offset} to {task_offset + num_tasks - 1}, "
                    f"{num_tasks * group_size} envs"
                )

        self._total_tasks = len(self._global_task_to_server)
        self._total_envs = self._total_tasks * self._group_size

        logger.info(
            f"Global mapping built: {self._total_tasks} tasks, "
            f"{self._total_envs} envs across {self.num_servers} servers"
        )

    def get_env_indices_for_task(self, global_task_id: int) -> list:
        """
        Get env indices for a given global task ID.

        Note: Returns LOCAL env indices on the appropriate server.
        Use get_server_for_task() to know which server to send requests to.
        """
        return self._global_task_to_env_indices.get(global_task_id, [])

    def get_server_for_task(self, global_task_id: int) -> int:
        """Get the server rank that handles a given global task ID."""
        return self._global_task_to_server.get(global_task_id, 0)

    def step(
        self, actions: np.ndarray, env_indices: list, server_rank: int, render_last_only: bool = True
    ) -> Optional[dict]:
        """
        Send step command to a specific server.

        Args:
            actions: Actions array
            env_indices: LOCAL env indices on that server
            server_rank: Which server to send to
            render_last_only: If True, only render the last step of action chunk

        Returns:
            Response dict or None
        """
        if server_rank >= len(self.clients):
            logger.error(f"Invalid server rank: {server_rank}")
            return None
        return self.clients[server_rank].step(actions, env_indices, render_last_only)

    def reset(self, env_indices: list, server_rank: int) -> Optional[dict]:
        """
        Send reset command to a specific server.

        Args:
            env_indices: LOCAL env indices on that server
            server_rank: Which server to send to

        Returns:
            Response dict or None
        """
        if server_rank >= len(self.clients):
            logger.error(f"Invalid server rank: {server_rank}")
            return None
        return self.clients[server_rank].reset(env_indices)

    def step_batched(
        self, server_requests: dict[int, tuple[np.ndarray, list]], render_last_only: bool = True
    ) -> dict[int, dict]:
        """
        Send step commands to multiple servers CONCURRENTLY.

        Uses persistent thread pool (created at init) for efficiency.

        Args:
            server_requests: Dict mapping server_rank -> (actions, env_indices)
                e.g., {0: (actions_0, [0,1,2]), 2: (actions_2, [8,9,10])}
            render_last_only: If True, only render the last step of action chunk

        Returns:
            Dict mapping server_rank -> response dict
        """
        from concurrent.futures import as_completed

        def step_on_server(rank: int, actions: np.ndarray, indices: list):
            response = self.clients[rank].step(actions, indices, render_last_only)
            return rank, response

        # Submit all tasks to persistent executor
        futures = {
            self._executor.submit(step_on_server, rank, actions, indices): rank
            for rank, (actions, indices) in server_requests.items()
        }

        # Collect results
        results = {}
        for future in as_completed(futures):
            rank, response = future.result()
            results[rank] = response

        return results

    def reset_batched(self, server_requests: dict[int, list]) -> dict[int, dict]:
        """
        Send reset commands to multiple servers CONCURRENTLY.

        Uses persistent thread pool (created at init) for efficiency.

        Args:
            server_requests: Dict mapping server_rank -> env_indices
                e.g., {0: [0,1,2,3], 2: [8,9,10,11]}

        Returns:
            Dict mapping server_rank -> response dict
        """
        from concurrent.futures import as_completed

        def reset_on_server(rank: int, indices: list):
            response = self.clients[rank].reset(indices)
            return rank, response

        # Submit all tasks to persistent executor
        futures = {
            self._executor.submit(reset_on_server, rank, indices): rank for rank, indices in server_requests.items()
        }

        # Collect results
        results = {}
        for future in as_completed(futures):
            rank, response = future.result()
            results[rank] = response

        return results

    @property
    def num_tasks(self) -> int:
        """Get total number of tasks across all servers."""
        return self._total_tasks

    @property
    def group_size(self) -> int:
        """Get group size (envs per task)."""
        return self._group_size

    @property
    def total_envs(self) -> int:
        """Get total number of envs across all servers."""
        return self._total_envs

    def close(self):
        """Close all client connections and shutdown thread pool."""
        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        # Close all client connections
        for client in self.clients:
            client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class IsaacMultiServerClient:
    """
    Client for connecting to multiple independent Isaac server groups.

    Architecture:
        - Multiple server groups (one per pipeline stage)
        - Each group has N servers (one per GPU)
        - All groups share the same GPUs (time-multiplexed)
        - Example with num_server_groups=2:
            - Stage 0 → Server Group 0 (ports 5556-5563)
            - Stage 1 → Server Group 1 (ports 5600-5607)

    This enables:
        - Physical isolation between stages (no env interference)
        - Full GPU utilization through time-interleaving
        - Stage i sim uses GPU while Stage j generates, and vice versa

    Note:
        num_server_groups MUST match env.rollout.pipeline_stage_num in training config.
        This ensures each pipeline stage has its own dedicated server group.
    """

    def __init__(
        self,
        num_server_groups: int = 2,
        num_servers_per_group: int = 8,
        base_ports: list[int] = None,  # e.g., [5556, 5600] for 2 groups
        base_address: str = "ipc:///tmp/isaac_server",
        timeout_ms: int = 30000,
        max_retries: int = 3,
        use_tcp: bool = True,
    ):
        """
        Initialize the multi-server client.

        Args:
            num_server_groups: Number of server groups. MUST match env.rollout.pipeline_stage_num.
                               Each pipeline stage uses its own server group for physical isolation.
                               Default: 2 (for 2-stage pipeline)
            num_servers_per_group: Number of servers per group (typically = num GPUs)
            base_ports: List of base ports, one per group (e.g., [5556, 5600])
                        If None, auto-generated starting from 5556 with 50-port spacing.
            base_address: Base address for IPC sockets
            timeout_ms: Timeout for requests in milliseconds
            max_retries: Maximum number of retries
            use_tcp: If True, use TCP instead of IPC
        """
        self.num_server_groups = num_server_groups
        self.num_servers_per_group = num_servers_per_group
        self.use_tcp = use_tcp

        # Auto-generate base ports if not provided (50-port spacing to avoid conflicts)
        if base_ports is None:
            base_ports = [5556 + i * 50 for i in range(num_server_groups)]

        if len(base_ports) != num_server_groups:
            raise ValueError(f"Expected {num_server_groups} base_ports, got {len(base_ports)}")

        self.base_ports = base_ports

        # Create one IsaacDistributedClient per server group (one per pipeline stage)
        self.stage_clients: list[IsaacDistributedClient] = []
        for group_id in range(num_server_groups):
            print(
                f"[IsaacMultiServerClient] Creating client for server group {group_id}, "
                f"base_port={base_ports[group_id]}",
                flush=True,
            )
            client = IsaacDistributedClient(
                base_address=base_address,
                num_servers=num_servers_per_group,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                use_tcp=use_tcp,
                base_port=base_ports[group_id],
            )
            self.stage_clients.append(client)

        # Cached info (assumed same across all stages)
        self._num_tasks = 0
        self._group_size = 0
        self._total_envs = 0

    def connect(self) -> bool:
        """Connect to all server groups."""
        all_connected = True
        for stage_id, client in enumerate(self.stage_clients):
            logger.info(f"Connecting to server group for stage {stage_id}...")
            if not client.connect():
                logger.error(f"Failed to connect to server group for stage {stage_id}")
                all_connected = False
            else:
                logger.info(f"Connected to server group for stage {stage_id}")

        if all_connected and self.stage_clients:
            # Cache info from first stage (assumed same for all)
            first_client = self.stage_clients[0]
            self._num_tasks = first_client.num_tasks
            self._group_size = first_client.group_size
            self._total_envs = first_client.total_envs

        return all_connected

    def get_env_indices_for_task(self, global_task_id: int) -> list:
        """Get env indices for a task (same across all stages)."""
        return self.stage_clients[0].get_env_indices_for_task(global_task_id)

    def get_server_for_task(self, global_task_id: int) -> int:
        """Get server rank for a task (same across all stages)."""
        return self.stage_clients[0].get_server_for_task(global_task_id)

    def step(
        self,
        actions: np.ndarray,
        env_indices: list,
        server_rank: int,
        stage_id: int,
        render_last_only: bool = True,
    ) -> Optional[dict]:
        """
        Send step command to a specific server in a specific stage's server group.

        Args:
            actions: Actions array
            env_indices: LOCAL env indices on that server
            server_rank: Which server in the group
            stage_id: Which pipeline stage (determines which server group)
            render_last_only: If True, only render the last step of action chunk

        Returns:
            Response dict or None
        """
        return self.stage_clients[stage_id].step(actions, env_indices, server_rank, render_last_only)

    def reset(self, env_indices: list, server_rank: int, stage_id: int) -> Optional[dict]:
        """
        Send reset command to a specific server in a specific stage's server group.

        Args:
            env_indices: LOCAL env indices on that server
            server_rank: Which server in the group
            stage_id: Which pipeline stage

        Returns:
            Response dict or None
        """
        return self.stage_clients[stage_id].reset(env_indices, server_rank)

    def step_batched(
        self,
        server_requests: dict[int, tuple[np.ndarray, list]],
        stage_id: int,
        render_last_only: bool = True,
    ) -> dict[int, dict]:
        """
        Send batched step commands to a specific stage's server group.

        Args:
            server_requests: Dict mapping server_rank -> (actions, env_indices)
            stage_id: Which pipeline stage
            render_last_only: If True, only render the last step of action chunk

        Returns:
            Dict mapping server_rank -> response dict
        """
        return self.stage_clients[stage_id].step_batched(server_requests, render_last_only)

    def reset_batched(self, server_requests: dict[int, list], stage_id: int) -> dict[int, dict]:
        """
        Send batched reset commands to a specific stage's server group.

        Args:
            server_requests: Dict mapping server_rank -> env_indices
            stage_id: Which pipeline stage

        Returns:
            Dict mapping server_rank -> response dict
        """
        return self.stage_clients[stage_id].reset_batched(server_requests)

    @property
    def num_tasks(self) -> int:
        """Get total number of tasks."""
        return self._num_tasks

    @property
    def group_size(self) -> int:
        """Get group size (envs per task)."""
        return self._group_size

    @property
    def total_envs(self) -> int:
        """Get total number of envs per stage."""
        return self._total_envs

    def close(self):
        """Close all client connections."""
        for client in self.stage_clients:
            client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
