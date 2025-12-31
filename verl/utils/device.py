# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging
import types
from typing import Any

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of Ascend NPU.

    Returns:
        bool: True if torch.npu is available and functional, False otherwise.
    """
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_visible_devices_keyword() -> str:
    """Get the environment variable name for visible devices.

    Returns:
        str: 'CUDA_VISIBLE_DEVICES' for CUDA devices, 'ASCEND_RT_VISIBLE_DEVICES' for NPU.
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Get the device type string based on available hardware.

    This function detects the available accelerator and returns the appropriate
    device name. Currently supports CUDA GPUs, Ascend NPUs, and CPU fallback.

    Returns:
        str: The device type name ('cuda', 'npu', or 'cpu').
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> types.ModuleType:
    """Get the torch device module corresponding to the detected hardware.

    Returns the appropriate torch device namespace (e.g., torch.cuda, torch.npu)
    based on the detected hardware. Falls back to torch.cuda if the device
    namespace is not found.

    Returns:
        types.ModuleType: The torch device module (e.g., torch.cuda or torch.npu).
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Get the current device index.

    Returns:
        int: The index of the current device (e.g., GPU index).
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Get the collective communication backend name for the current device.

    Returns:
        str: 'hccl' for Ascend NPU, 'nccl' for CUDA devices.
    """
    if is_npu_available:
        return "hccl"
    else:
        # default to nccl
        return "nccl"


def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


def auto_set_ascend_device_name(config: Any) -> None:
    """Automatically set the device name to 'npu' when running on Ascend hardware.

    If an Ascend NPU is detected and the config has a different device setting,
    this function updates the config to use 'npu' and logs a warning.

    Args:
        config (Any): Configuration object with trainer.device attribute.
    """
    if config and config.trainer and config.trainer.device:
        if is_torch_npu_available():
            if config.trainer.device != "npu":
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for Ascend NPU, maybe"
                    f"from default value in config file, automatically set to `npu` instead."
                )

            config.trainer.device = "npu"


def get_device_capability(device_id: int = 0) -> tuple[int | None, int | None]:
    """Get the compute capability of a CUDA device.

    Args:
        device_id (int): The device index to query. Defaults to 0.

    Returns:
        tuple[int | None, int | None]: A tuple of (major, minor) version numbers for CUDA devices,
            or (None, None) if CUDA is not available.
    """
    major, minor = None, None
    if is_cuda_available:
        major, minor = torch.cuda.get_device_capability(device_id)

    return major, minor
