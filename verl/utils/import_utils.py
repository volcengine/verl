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
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

import importlib
import importlib.util
import os
import warnings
from functools import cache, wraps
from typing import Optional


@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None


@cache
def is_vllm_available():
    try:
        vllm_spec = importlib.util.find_spec("vllm")
    except ModuleNotFoundError:
        vllm_spec = None
    return vllm_spec is not None


@cache
def is_sglang_available():
    try:
        sglang_spec = importlib.util.find_spec("sglang")
    except ModuleNotFoundError:
        sglang_spec = None
    return sglang_spec is not None


@cache
def is_nvtx_available():
    try:
        nvtx_spec = importlib.util.find_spec("nvtx")
    except ModuleNotFoundError:
        nvtx_spec = None
    return nvtx_spec is not None


@cache
def is_trl_available():
    try:
        trl_spec = importlib.util.find_spec("trl")
    except ModuleNotFoundError:
        trl_spec = None
    return trl_spec is not None


def import_external_libs(external_libs=None):
    if external_libs is None:
        return
    if not isinstance(external_libs, list):
        external_libs = [external_libs]
    import importlib

    for external_lib in external_libs:
        importlib.import_module(external_lib)


def load_extern_type(file_path: Optional[str], type_name: Optional[str]) -> type:
    """Load a external data type based on the file path and type name"""
    if not file_path:
        return None

    if file_path.startswith("pkg://"):
        # pkg://verl.utils.dataset.rl_dataset
        # pkg://verl/utils/dataset/rl_dataset
        module_name = file_path[6:].replace("/", ".")
        module = importlib.import_module(module_name)

    else:
        # file://verl/utils/dataset/rl_dataset
        # file:///path/to/verl/utils/dataset/rl_dataset.py
        # or without file:// prefix
        if file_path.startswith("file://"):
            file_path = file_path[7:]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Custom type file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}'") from e

    if not hasattr(module, type_name):
        raise AttributeError(f"Custom type '{type_name}' not found in '{file_path}'.")

    return getattr(module, type_name)


def _get_qualified_name(func):
    """Get full qualified name including module and class (if any)."""
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}.{qualname}"


def deprecated(replacement: str = ""):
    """Decorator to mark functions or classes as deprecated."""

    def decorator(obj):
        qualified_name = _get_qualified_name(obj)

        if isinstance(obj, type):
            original_init = obj.__init__

            @wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                msg = f"Warning: Class '{qualified_name}' is deprecated."
                if replacement:
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            return obj

        else:

            @wraps(obj)
            def wrapped(*args, **kwargs):
                msg = f"Warning: Function '{qualified_name}' is deprecated."
                if replacement:
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapped

    return decorator
