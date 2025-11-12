# Copyright 2025 Meituan Ltd. and/or its affiliates
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


import logging
import os

import yaml
import ray

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def update_prometheus_config(server_addresses: list[str]):
    """
    Update Prometheus configuration file with server addresses and reload on first node.

    server_addresses: vllm or sglang server addresses
    """

    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    try:
        # Read existing Prometheus config or create default one
        prometheus_config_path = str(
            os.getenv("PROMETHEUS_FILE", "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml")
        )
        prometheus_config = {
            "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
            "scrape_configs": [
                {
                    "job_name": "ray",
                    "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],
                },
                {"job_name": "rollout", "static_configs": [{"targets": server_addresses}]},
            ],
        }

        # Write the configuration to file on all nodes
        @ray.remote(num_cpus=0)
        def write_config_file(config_data, config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            return True

        # Call prometheus on all nodes, just master node should success, we skip other node error.
        @ray.remote(num_cpus=0)
        def reload_prometheus():
            import os
            import socket
            import subprocess

            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            port = int(os.getenv("PROMETHEUS_PORT", "9090"))

            reload_url = f"http://{ip_address}:{port}/-/reload"

            try:
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)
                print(f"Reloading Prometheus on node: {reload_url}")
            except Exception:
                pass

        # Schedule task on each specific node

        # Get all available nodes and schedule task on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        write_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = write_config_file.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(prometheus_config, prometheus_config_path)
            write_tasks.append(task)

        ray.get(write_tasks)

        print(f"Updated Prometheus configuration at {prometheus_config_path} with {len(server_addresses)} VLLM servers")

        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote()
            reload_tasks.append(task)

        ray.get(reload_tasks)

    except Exception as e:
        logger.error(f"Failed to update Prometheus configuration: {e}")
