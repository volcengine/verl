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
Run a command on all nodes of a Ray cluster.

Usage:
    python run_on_all_nodes.py "echo hello"
    python run_on_all_nodes.py "pip install numpy" --timeout 300
    python run_on_all_nodes.py "ls -la /data" --ray-address auto
"""

import argparse
import subprocess
import socket

import ray


@ray.remote(num_cpus=0)
def run_command_on_node(command: str, timeout: int = 60) -> dict:
    """Run a shell command on a specific node and return the result."""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "hostname": hostname,
            "ip": ip_address,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "hostname": hostname,
            "ip": ip_address,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
        }
    except Exception as e:
        return {
            "hostname": hostname,
            "ip": ip_address,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


def run_on_all_nodes(command: str, timeout: int = 60, verbose: bool = True) -> list[dict]:
    """
    Run a command on all nodes of a Ray cluster.

    Args:
        command: The shell command to run on each node.
        timeout: Timeout in seconds for each command execution.
        verbose: Whether to print progress and results.

    Returns:
        List of result dictionaries from each node.
    """
    # Get all available nodes and schedule tasks on each node
    nodes = ray.nodes()
    alive_nodes = [node for node in nodes if node["Alive"]]

    if verbose:
        print(f"Found {len(alive_nodes)} alive nodes in the Ray cluster:")
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            print(f"  - {node_ip}")
        print(f"\nRunning command: {command}\n")

    # Run command on all nodes
    tasks = []
    for node in alive_nodes:
        node_ip = node["NodeManagerAddress"]
        task = run_command_on_node.options(
            resources={"node:" + node_ip: 0.001}  # Schedule to specific node
        ).remote(command, timeout)
        tasks.append(task)

    # Wait for all tasks to complete
    results = ray.get(tasks)

    if verbose:
        print("=" * 60)
        print("Results:")
        print("=" * 60)

        success_count = 0
        for result in results:
            status = "✓" if result["success"] else "✗"
            print(f"\n[{status}] {result['hostname']} ({result['ip']}) - exit code: {result['returncode']}")

            if result["stdout"].strip():
                print("  stdout:")
                for line in result["stdout"].strip().split("\n"):
                    print(f"    {line}")

            if result["stderr"].strip():
                print("  stderr:")
                for line in result["stderr"].strip().split("\n"):
                    print(f"    {line}")

            if result["success"]:
                success_count += 1

        print("\n" + "=" * 60)
        print(f"Summary: {success_count}/{len(results)} nodes succeeded")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run a command on all nodes of a Ray cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_on_all_nodes.py "echo hello"
    python run_on_all_nodes.py "pip install numpy" --timeout 300
    python run_on_all_nodes.py "ls -la /data" --ray-address auto
    python run_on_all_nodes.py "nvidia-smi" --quiet
        """,
    )
    parser.add_argument("command", type=str, help="The shell command to run on all nodes")
    parser.add_argument(
        "--timeout", type=int, default=60, help="Timeout in seconds for each command execution (default: 60)"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address to connect to (default: auto-detect or start local)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print summary, not detailed output")

    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        if args.ray_address:
            ray.init(address=args.ray_address)
        else:
            ray.init()

    results = run_on_all_nodes(args.command, timeout=args.timeout, verbose=not args.quiet)

    # Exit with non-zero if any node failed
    if not all(r["success"] for r in results):
        exit(1)


if __name__ == "__main__":
    main()
