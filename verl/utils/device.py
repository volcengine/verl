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

import torch
from .tpu_utils import Tpu

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False
    

def is_torch_tpu_available() -> bool:
    """Check the availability of TPUs"""
    global _is_tpu_available_cached
    if _is_tpu_available_cached is not None:
        return _is_tpu_available_cached
    try:
        import torch_xla.core.xla_model as xm
        xm.xla_device()
        _is_tpu_available_cached = True
    except:
        _is_tpu_available_cached = False
    return _is_tpu_available_cached

is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()
_is_tpu_available_cached = None


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU, TPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    elif is_torch_tpu_available():
        device = "xla"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    if device_name == "xla":
        return Tpu(device_name)
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda
