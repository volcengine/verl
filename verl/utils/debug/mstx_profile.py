# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import os
import functools
from contextlib import contextmanager
from typing import Callable, Dict, Optional

import torch_npu
from torch_npu.npu import mstx

from .profile import DistProfiler, ProfilerConfig


def mark_start_range(message: Optional[str] = None) -> None:
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """
    return mstx.range_start(message=message)


def mark_end_range(range_id: str) -> None:
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return mstx.range_end(range_id)


def mark_annotate(message: Optional[str] = None) -> Callable:
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """

    def decorator(func):
        profile_message = message or func.__name__
        return mstx.mstx_range(profile_message)(func)

    return decorator


@contextmanager
def marked_timer(name: str, timing_raw: Dict[str, float], **kwargs):
    """Context manager for timing with MSTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds MSTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    mark_range = mark_start_range(message=name)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


def get_npu_profiler(role: Optional[str] = None, profile_step: Optional[str] = None):
    """Generate and return an NPU profiler object.

    Args:
        role (str, optional):
            The role of the current data collection. Defaults to None.
        profile_step(str, optional):
            The current training step. Defaults to None.
    """
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        export_type=torch_npu.profiler.ExportType.Text,
        data_simplification=True,
        msprof_tx=True
    )

    profile_save_path = "./profiling_result"
    if profile_step:
        profile_save_path = os.path.join(profile_save_path, profile_step)
    if role:
        profile_save_path = os.path.join(profile_save_path, role)

    return torch_npu.profiler.profile(
        with_modules=False,
        record_shapes=False,
        profile_memory=False,
        activities=[torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path,
                                                                    analyse_flag=True),
        experimental_config=experimental_config
    )


class NPUProfiler(DistProfiler):
    """
    NPU profiler. Installed in a worker to control the NPU profiler.
    """

    def __init__(self, rank: int, config: ProfilerConfig):
        config = config
        self.this_step: bool = False
        self.discrete: bool = config.discrete
        self.this_rank: bool = False
        self.profile_npu = None
        if config.all_ranks:
            self.this_rank = True
        if config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        profile_step = kwargs.get('profile_step', None)
        role = kwargs.get('role', None)
        if self.this_rank:
            self.this_step = True
            if not self.discrete:
                self.profile_npu = get_npu_profiler(role, profile_step)
                self.profile_npu.start()

    def stop(self):
        if self.this_rank:
            self.this_step = False
            if not self.discrete:
                self.profile_npu.step()
                self.profile_npu.stop()

    @staticmethod
    def annotate(message: Optional[str] = None, role: Optional[str] = None, **kwargs) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker, which has a member field `profiler` with NPUProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                profile_name = message or func.__name__

                if self.profiler.this_step:
                    if self.profiler.discrete:
                        profile_npu = get_npu_profiler(role)
                        profile_npu.start()
                    mark_range = mark_start_range(message=profile_name)

                result = func(self, *args, **kwargs)

                if self.profiler.this_step:
                    mark_end_range(mark_range)
                    if self.profiler.discrete:
                        profile_npu.step()
                        profile_npu.stop()

                return result

            return wrapper

        return decorator

