# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import os
import subprocess
import sys
from typing import Optional

import torch
import torch.multiprocessing as mp


def force_gc_tensor(tensor):
    if not torch.is_tensor(tensor):
        return

    try:
        ref_count = sys.getrefcount(tensor)
        for _ in range(ref_count + 10):
            ctypes.pythonapi.Py_DecRef(ctypes.py_object(tensor))

    except Exception as e:
        print(f"Error during force delete: {e}")


def cleanup_cuda_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            force_gc_tensor(obj)
    gc.collect()
    torch.cuda.empty_cache()


def get_gpu_numa_node(gpu_id: int) -> int:
    try:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            # Get PCI bus info
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId.decode("utf-8")
        except ImportError:
            # Fallback to nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            pci_bus_id = result.stdout.strip()

        # Extract bus number from PCI bus ID (format: 0000:XX:YY.Z)
        bus_number = pci_bus_id.split(":")[1]

        # Get NUMA node from sysfs
        numa_node_path = f"/sys/bus/pci/devices/0000:{bus_number}:00.0/numa_node"
        if os.path.exists(numa_node_path):
            with open(numa_node_path) as f:
                numa_node = int(f.read().strip())
                if numa_node >= 0:
                    return numa_node

        # Fallback: try to get from lscpu
        result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
        numa_nodes = 0
        for line in result.stdout.split("\n"):
            if "NUMA node(s):" in line:
                numa_nodes = int(line.split(":")[1].strip())
                break

        # If we can't determine the exact NUMA node, distribute evenly
        return gpu_id % numa_nodes if numa_nodes > 0 else 0

    except Exception as e:
        print(f"Warning: Could not determine NUMA node for GPU {gpu_id}: {e}")
        return 0


def get_numa_cpus(numa_node: int) -> list:
    try:
        # Read from sysfs
        cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        if os.path.exists(cpulist_path):
            with open(cpulist_path) as f:
                cpulist = f.read().strip()

            # Parse CPU list (e.g., "0-7,16-23" or "0,1,2,3")
            cpus = []
            for part in cpulist.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    cpus.extend(range(start, end + 1))
                else:
                    cpus.append(int(part))
            return cpus
    except Exception as e:
        print(f"Warning: Could not get CPU list for NUMA node {numa_node}: {e}")

    # Fallback: return all available CPUs
    return list(range(os.cpu_count() or 1))


def set_process_numa_affinity(gpu_id: int) -> None:
    try:
        numa_node = get_gpu_numa_node(gpu_id)
        cpus = get_numa_cpus(numa_node)

        if not cpus:
            print(f"Warning: No CPUs found for NUMA node {numa_node}")
            return

        os.sched_setaffinity(0, cpus)
        try:
            subprocess.run(
                ["numactl", "--membind", str(numa_node), "--"],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass  # numactl not available, that's ok

    except Exception as e:
        print(f"Warning: Could not set NUMA affinity for GPU {gpu_id}: {e}")


def recursive_to_own(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_shared() else obj
    elif isinstance(obj, list):
        return [recursive_to_own(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_own(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_own(v) for k, v in obj.items()}
    else:
        return obj


class EnvManager:
    def __init__(self, cfg, rank, world_size, env_cls):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.process: Optional[mp.Process] = None
        self.command_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.state_buffer: Optional[bytes] = None

        self.env_cls = env_cls
        self.env = self.env_cls(cfg, rank, world_size)

    def start_simulator(self):
        """Start simulator process with shared memory queues"""
        if self.env is not None:
            return

        if self.process is not None and self.process.is_alive():
            raise RuntimeError("Simulator already running")

        self.context = mp.get_context("spawn")
        # Create shared memory queues
        self.command_queue = self.context.Queue()
        self.result_queue = self.context.Queue()

        # Start simulator process
        self.process = self.context.Process(
            target=_simulator_worker,
            args=(
                self.cfg,
                self.rank,
                self.world_size,
                self.env_cls,
                self.command_queue,
                self.result_queue,
                self.state_buffer,
                True,
            ),
        )
        self.process.start()

        # Wait for initialization
        result = self.result_queue.get(timeout=60)
        if result["status"] != "ready":
            raise RuntimeError(f"Simulator initialization failed: {result}")

    def stop_simulator(self):
        if self.env is not None:
            return

        if self.process is None or not self.process.is_alive():
            raise RuntimeError("No simulator running")

        # Request state save
        self.command_queue.put({"method": "get_state", "args": [], "kwargs": {}})

        # Get saved state
        result = self.result_queue.get(timeout=60)
        if result["status"] == "success":
            self.state_buffer = result["data"]

        self.command_queue.put({"method": "shutdown"})
        self.command_queue.close()
        self.result_queue.close()
        self.command_queue = None
        self.result_queue = None
        self.process.join(timeout=5)

        self.command_queue = None
        self.result_queue = None
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.process = None

    def __getattr__(self, name):
        if self.env is not None:
            return getattr(self.env, name)

        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def method_proxy(*args, **kwargs):
            if self.process is None or not self.process.is_alive():
                raise RuntimeError("Simulator not running")

            args = recursive_to_own(args)
            kwargs = recursive_to_own(kwargs)
            self.command_queue.put({"method": name, "args": args, "kwargs": kwargs})

            result = self.result_queue.get()
            result = recursive_to_own(result)
            if result["status"] == "error":
                raise Exception(result["error"])
            return result["data"]

        return method_proxy

    def __setattr__(self, name, value):
        # Handle special attributes that should be set on self
        if name in [
            "cfg",
            "rank",
            "world_size",
            "process",
            "command_queue",
            "result_queue",
            "state_buffer",
            "env_cls",
            "env",
            "context",
        ]:
            super().__setattr__(name, value)
            return

        # If env is directly available, set attribute on it
        if self.env is not None:
            setattr(self.env, name, value)
            return

        # For offloaded environments, send attribute set command
        if name.startswith("_"):
            raise AttributeError(f"Cannot set private attribute '{name}' on offloaded environment")

        if self.process is None or not self.process.is_alive():
            raise RuntimeError("Simulator not running")

        value = recursive_to_own(value)
        self.command_queue.put(
            {
                "method": "__setattr__",
                "args": [name, value],
                "kwargs": {},
            }
        )

        result = self.result_queue.get()
        result = recursive_to_own(result)
        if result["status"] == "error":
            raise Exception(result["error"])


def _simulator_worker(
    cfg,
    rank,
    world_size,
    env_cls,
    command_queue,
    result_queue,
    state_buffer,
    bind_numa=True,
):
    """Worker process for simulator"""
    # Set NUMA affinity for the process to match the GPU rank
    if bind_numa:
        set_process_numa_affinity(rank)
    try:
        simulator = env_cls(cfg, rank, world_size)

        if state_buffer:
            simulator.load_state(state_buffer)

        # Signal ready
        result_queue.put({"status": "ready"})

        # Main command processing loop
        while True:
            try:
                command = command_queue.get()

                if command["method"] == "shutdown":
                    break

                method_name = command["method"]
                args = command.get("args", [])
                kwargs = command.get("kwargs", {})

                if method_name == "__setattr__":
                    # Handle attribute setting
                    attr_name, attr_value = args
                    setattr(simulator, attr_name, attr_value)
                    result_queue.put({"status": "success", "data": None})
                elif hasattr(simulator, method_name):
                    method = getattr(simulator, method_name)
                    assert callable(method), f"Method {method_name} is not callable"
                    result = method(*args, **kwargs)
                    result_queue.put({"status": "success", "data": result})
                else:
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"Method '{method_name}' not found",
                        }
                    )

            except Exception as e:
                result_queue.put({"status": "error", "error": str(e)})

    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})

    finally:
        command_queue.close()
        result_queue.close()
        cleanup_cuda_tensors()
