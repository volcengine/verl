# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import functools
from contextlib import contextmanager
from typing import Callable, Optional

import nvtx
import torch

from .profile import DistProfiler, ProfilerConfig


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """
    return nvtx.start_range(message=message, color=color, domain=domain, category=category)


def mark_end_range(range_id: str) -> None:
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return nvtx.end_range(range_id)


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """

    def decorator(func):
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=domain, category=category)(func)

    return decorator


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with NVTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds NVTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the NVTX marker. Defaults to None.
        domain (Optional[str]): Domain for the NVTX marker. Defaults to None.
        category (Optional[str]): Category for the NVTX marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    mark_range = mark_start_range(message=name, color=color, domain=domain, category=category)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


class NsightSystemsProfiler(DistProfiler):
    """Nsight system profiler. Installed in a worker to control the Nsight system profiler."""

    def __init__(self, rank: int, config: Optional[ProfilerConfig], **kwargs):
        """Initialize the NsightSystemsProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
        """
        # If no configuration is provided, create a default ProfilerConfig with an empty list of ranks
        if not config:
            config = ProfilerConfig(ranks=[])
        self.this_step: bool = False
        self.discrete: bool = config.discrete
        self.this_rank: bool = False
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        if self.this_rank:
            self.this_step = True
            if not self.discrete:
                torch.cuda.profiler.start()

    def stop(self):
        if self.this_rank:
            self.this_step = False
            if not self.discrete:
                torch.cuda.profiler.stop()

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker, which has a member field `profiler` with
        NightSystemsProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            color (str, optional):
                The color of the range. Defaults to None.
            domain (str, optional):
                The domain of the range. Defaults to None.
            category (str, optional):
                The category of the range. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                profile_name = message or func.__name__

                if self.profiler.this_step:
                    if self.profiler.discrete:
                        torch.cuda.profiler.start()
                    mark_range = mark_start_range(message=profile_name, color=color, domain=domain, category=category)

                result = func(self, *args, **kwargs)

                if self.profiler.this_step:
                    mark_end_range(mark_range)
                    if self.profiler.discrete:
                        torch.cuda.profiler.stop()

                return result

            return wrapper

        return decorator
