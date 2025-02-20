# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py

import os
import logging
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Function that takes an optional device string, verifies it's correct and available given the machine and
    distributed settings, and returns a :func:`~torch.device`. If device string is not provided, this function will
    infer the device based on the environment.
    If CUDA-like is available and being used, this function also sets the CUDA-like device.
    Args:
        device (Optional[str]): The name of the device to use, e.g. "cuda" or "cpu" or "npu".
    Example:
        >>> device = get_device("cuda")
        >>> device
        device(type='cuda', index=0)
    Returns:
        torch.device: Device
    """
    if device_name is None:
        device_name = get_device_name()
    device = torch.device(device_name)
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(
            f"Device namespace '{device_name}' not found in torch, try to load torch.cuda."
        )
        return torch.cuda