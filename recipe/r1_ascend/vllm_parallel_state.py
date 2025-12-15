# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from https://gitcode.com/Ascend/MindSpeed-RL/blob/2.1.0/mindspeed_rl/utils/utils.py

import logging
import os
import re
import socket
import subprocess

import torch
import vllm.envs as envs
from vllm.distributed import parallel_state as vllm_ps
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_ip_by_ifname():
    """
    Get IPv4 address by interface name (e.g. eth0, en0)
    returns IP string on success, and None on failure
    """
    try:
        # Execute `ifconfig` and capture its output
        ifname = os.environ.get("HCCL_SOCKET_IFNAME", 0)
        if ifname:
            output = subprocess.check_output(["ifconfig", ifname], stderr=subprocess.STDOUT).decode()
            # Match IPv4 addresses using regex, and exclude 127.0.0.1
            matches = re.findall(r"inet (?:addr:)?((?:\d{1,3}\.){3}\d{1,3})", output)
            for ip in matches:
                if ip != "127.0.0.1":
                    return ip
        return None
    except subprocess.CalledProcessError:
        return None


def _get_current_node_ip() -> str:
    try:
        # Create UDP socket (Only used to get info of interface).
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to external address, without actual communication.
            s.connect(("8.8.8.8", 80))  # Google DNS Server.
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = _get_ip_by_ifname()
        if not local_ip:
            # Fallback to iterative search on failure.
            local_ip = "127.0.0.1"
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                ip = addr[4][0]
                if not ip.startswith("::"):
                    local_ip = ip
                    break
    return local_ip


def get_cluster_info():
    # Ensure initialization of distributed env.
    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed environment not initialized")

    world_size = torch.distributed.get_world_size()

    # Get IP address of current node.
    ip_address = _get_current_node_ip()

    # Collect IP addresses of all ranks.
    ip_list = [None] * world_size
    torch.distributed.all_gather_object(ip_list, ip_address)

    return ip_list


### init DP group ranks for vLLM ascend
def init_parallel_state(tensor_parallel_size):
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size: int = torch.distributed.get_world_size()
    distributed_init_method = "env://"
    backend = "hccl"
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)

    initialize_model_parallel(tensor_parallel_size)
    logger.info(
        f"[DEBUG]: RANK[{rank}]: TP group: {vllm_ps._TP.ranks}\n"
        f"[DEBUG]: RANK[{rank}]: PP group: {vllm_ps._PP.ranks}\n"
        f"[DEBUG]: RANK[{rank}]: DP group: {vllm_ps._DP.ranks}\n"
        f"[DEBUG]: RANK[{rank}]: EP group: {vllm_ps._EP.ranks}\n"
    )

    os.environ["VLLM_DP_RANK"] = str(vllm_ps._DP.rank_in_group)
    envs.VLLM_DP_RANK = int(os.environ["VLLM_DP_RANK"])

    ip_list = get_cluster_info()

    rank_0 = vllm_ps._DP.ranks[0]
    index = rank_0
    os.environ["VLLM_DP_MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1 + index)
    os.environ["VLLM_DP_MASTER_IP"] = ip_list[rank_0]
    envs.VLLM_DP_MASTER_PORT = int(os.environ["VLLM_DP_MASTER_PORT"])
    envs.VLLM_DP_MASTER_IP = os.environ["VLLM_DP_MASTER_IP"]
