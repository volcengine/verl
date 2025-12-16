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
Cache Manager for distributed suffix cache in PPO training.
Encapsulates cache servers, storage, and updater logic.
"""

import socket
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import psutil
import ray

from verl.trainer.ppo.utils import Role


@ray.remote(num_cpus=1)
class CacheWorker:
    """Ray remote worker for running a gRPC-based rollout cache server on each GPU node.

    This worker deploys a SuffixCache and RolloutCacheServer on each compute node
    (excluding the master node). The cache server provides suffix caching capabilities
    via gRPC to accelerate rollout generation during PPO training.
    """

    def __init__(self, port: int = 6378):
        """Initialize and start the cache server.

        Args:
            port: Port number for the gRPC server (default: 6378)
        """

        self.port = port

        from specrl.suffix_cache import RolloutCacheServer

        # Initialize the rollout cache server with IPv6 support ([::])
        self.server = RolloutCacheServer(f"[::]:{port}")
        self.server.initialize()

        # Start server in a separate process with CPU affinity to avoid interference with GPU workers
        self.cache_server_process = Process(target=self._run_cache_server)
        self.cache_server_process.daemon = True
        self.cache_server_process.start()

        # Set CPU affinity to cores 0-20 to keep cache server on separate CPU cores
        process = psutil.Process(self.cache_server_process.pid)
        affinity_cores = min(psutil.cpu_count() // 2, 21)
        process.cpu_affinity(list(range(affinity_cores)))
        print(f"Rollout cache server started on port {port} (PID: {self.cache_server_process.pid})")
        print(f"CPU affinity set up to core {affinity_cores - 1}")

    def _run_cache_server(self):
        """Run the cache server in a separate process"""
        try:
            # Set CPU affinity for this process (additional safety measure)
            current_process = psutil.Process()
            affinity_cores = min(psutil.cpu_count() // 2, 21)
            current_process.cpu_affinity(list(range(affinity_cores)))
            print(f"Cache server process CPU affinity set up to core {affinity_cores - 1}")

            self.server.start()
            self.server.wait()
        except Exception as e:
            print(f"Cache server error: {e}")

    def get_node_ip(self) -> str:
        """Get the IPv6 address of the node this worker is running on.

        Returns:
            IPv6 address of the current node
        """
        # Get all address info for the hostname, filtering for IPv6
        hostname = socket.gethostname()
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET6)
        # Return the first IPv6 address found
        if addr_info:
            return addr_info[0][4][0]
        # Fallback to localhost IPv6 if no address found
        return "::1"

    def shutdown(self):
        """Shutdown the cache server and cleanup resources."""
        if hasattr(self, "cache_server_process") and self.cache_server_process.is_alive():
            try:
                # Terminate the server process
                self.cache_server_process.terminate()
                self.cache_server_process.join(timeout=5)
                if self.cache_server_process.is_alive():
                    self.cache_server_process.kill()
                print(f"Cache server process terminated (PID: {self.cache_server_process.pid})")
            except Exception as e:
                print(f"Error terminating cache server process: {e}")

        if hasattr(self, "server"):
            try:
                self.server.shutdown()
            except Exception as e:
                print(f"Error shutting down cache server: {e}")

    def __del__(self):
        """Clean up when the worker is destroyed."""
        self.shutdown()


class CacheManager:
    """Manager for distributed suffix cache infrastructure.

    This class encapsulates all cache-related components:
    - Cache servers: One gRPC server per GPU node
    - Cache storage: SuffixCache for storing prompt/response pairs
    - Cache updater: Client for distributed async cache updates

    Provides simple interface for initialization, updates, and cleanup.
    """

    def __init__(
        self,
        config,
        role_worker_mapping: dict,
        resource_pool_manager,
        port: int = 6378,
    ):
        """Initialize cache manager if speculative decoding is enabled.

        Args:
            config: Training configuration
            role_worker_mapping: Mapping from roles to worker types
            resource_pool_manager: Ray resource pool manager
        """
        self.config = config
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        # Internal state
        self._cache_servers = None
        self._cache_updater = None
        self._cache_update_futures = []
        self._max_futures = 5
        self._executor = None
        self.port = port

        # Check if cache is enabled
        self._enabled = self._should_enable_cache()

        if self._enabled:
            self._initialize()

    def _should_enable_cache(self) -> bool:
        """Check if cache should be enabled based on configuration.

        Returns:
            True if speculative decoding with suffix cache is enabled
        """
        # Check if ActorRolloutRef role exists and has spec decoding enabled
        from verl.trainer.ppo.utils import Role

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if actor_role not in self.role_worker_mapping:
            return False

        rollout_config = self.config.actor_rollout_ref.rollout
        enable_spec = rollout_config.get("enable_spec_decoding", True)

        return enable_spec

    def _initialize(self):
        """Initialize cache servers, storage, and updater."""
        # Get resource pool for actor/rollout workers
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)

        # Create cache servers (one per GPU node)
        self._cache_servers = self._create_cache_servers(resource_pool, self.port)

        # Collect server addresses for distributed updates
        server_addresses = self._get_server_addresses()

        from specrl.cache_updater import SuffixCacheUpdater

        # Initialize cache updater (it manages its own thread pool internally)
        self._cache_updater = SuffixCacheUpdater(server_addresses=server_addresses)

        # Thread pool executor for async cache updates from trainer
        self._executor = ThreadPoolExecutor(max_workers=self._max_futures)

        print(f"Cache manager initialized with {len(self._cache_servers)} servers on ports {self.port}")
        print(f"Server addresses: {server_addresses}")

    def _create_cache_servers(self, resource_pool, port: int) -> list[dict]:
        """Create cache server workers on each GPU node.

        Args:
            resource_pool: Ray resource pool for placement
            port: gRPC server port

        Returns:
            List of dicts with {server, ip, port} for each node
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        # Get placement groups and extract unique node IDs
        pgs = resource_pool.get_placement_groups()

        # Get node IDs from placement groups
        node_ids = set()
        for pg in pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            # All bundles in a placement group should be on the same node
            node_id = specs["bundles_to_node_id"][0]
            node_ids.add(node_id)

        servers = []
        for node_id in node_ids:
            # Create cache server worker on specific node
            # Server starts automatically in __init__
            strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            server = CacheWorker.options(
                scheduling_strategy=strategy,
                name=f"cache_server_{node_id}",
            ).remote(port=port)

            # Get node's IPv6 address
            ip = ray.get(server.get_node_ip.remote())

            servers.append(
                {
                    "server": server,
                    "ip": ip,
                    "port": port,
                }
            )

        return servers

    def _get_server_addresses(self) -> list[str]:
        """Get formatted gRPC addresses for all cache servers.

        Returns:
            List of addresses in format '[<ipv6>]:<port>'
        """
        if not self._cache_servers:
            return []

        addresses = []
        for server_info in self._cache_servers:
            ip = server_info["ip"]
            port = server_info["port"]
            # Format IPv6 address with brackets for gRPC
            address = f"[{ip}]:{port}"
            addresses.append(address)

        return addresses

    def update_cache(
        self,
        batch,
        responses_per_prompt: int,
    ):
        """Update the suffix cache with new generation results asynchronously.

        This method extracts prompts and responses from the batch and submits them
        to the cache updater for async processing. The cache is updated across all
        cache servers in a distributed manner.

        Args:
            batch: DataProto containing prompts, responses, and attention masks
            responses_per_prompt: Number of responses generated per prompt
        """
        if not self._enabled:
            return

        # Extract response length from the batch
        response_length = batch.batch["responses"].shape[-1]

        # Split attention mask into prompt and response parts
        prompt_mask = batch.batch["attention_mask"][:, :-response_length]
        response_mask = batch.batch["attention_mask"][:, -response_length:]

        # Calculate actual lengths (excluding padding)
        prompt_length = prompt_mask.sum(-1).float()
        response_length_tensor = response_mask.sum(-1).float()  # (batch_size,)

        # Convert tensors to Python lists for gRPC transmission
        prompts_ = batch.batch["prompts"].tolist()
        responses_ = batch.batch["responses"].tolist()
        prompt_lengths_ = prompt_length.tolist()
        response_lengths_ = response_length_tensor.tolist()

        # Limit concurrent futures to prevent memory overflow
        # Wait for oldest future if we've reached the limit
        if len(self._cache_update_futures) >= self._max_futures:
            oldest_future = self._cache_update_futures.pop(0)
            oldest_future.result()  # Block until oldest update completes

        # Submit cache update task to thread pool for async execution
        # This allows training to continue while cache is being updated
        future = self._executor.submit(
            self._cache_updater.update_response_cache,
            prompts=prompts_,
            responses=responses_,
            prompt_lengths=prompt_lengths_,
            response_lengths=response_lengths_,
            responses_per_prompt=responses_per_prompt,
        )
        self._cache_update_futures.append(future)

    def get_server_addresses(self) -> list[str] | None:
        """Get cache server addresses for rollout workers to connect.

        Returns:
            List of gRPC addresses in format '[<ipv6>]:<port>' or None if disabled
        """
        if not self._enabled:
            return None
        return self._get_server_addresses()

    @property
    def enabled(self) -> bool:
        """Check if cache manager is enabled.

        Returns:
            True if cache is initialized and active
        """
        return self._enabled

    def shutdown(self):
        """Clean up cache updater and server resources."""
        if not self._enabled:
            return

        # Wait for all pending futures
        for future in self._cache_update_futures:
            if not future.done():
                try:
                    future.result(timeout=5)
                except Exception as e:
                    print(f"Cache update future failed: {e}")

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=True)

        # Shutdown cache servers
        if self._cache_servers:
            shutdown_futures = []
            for server_info in self._cache_servers:
                try:
                    # Call shutdown method asynchronously
                    future = server_info["server"].shutdown.remote()
                    shutdown_futures.append(future)
                except Exception as e:
                    print(f"Failed to initiate cache server shutdown: {e}")

            # Wait for all shutdowns to complete
            if shutdown_futures:
                try:
                    ray.get(shutdown_futures, timeout=10)
                except Exception as e:
                    print(f"Error waiting for cache server shutdowns: {e}")

        print("Cache manager shutdown complete")

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.shutdown()
