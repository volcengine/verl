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

import os
from typing import Callable, Optional

import torch
import torch.distributed

from .config import ProfilerConfig


class Profiler:
    """A PyTorch profiler wrapper class for collecting performance metrics.

    TODO(haibin.lin): this should implement the DistProfiler interface, and the config should be unified.

    This profiler provides a convenient interface for profiling PyTorch operations,
    with support for:

    - CPU and CUDA activity profiling
    - Configurable profiling schedule (wait/warmup/active steps)
    - Multi-rank profiling support
    - Chrome trace export

    Args:
        config: Configuration object containing profiling parameters
    """

    def __init__(self, config):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        self.config = config
        self.skip_prof = False
        self.saved = False
        self.prof = None
        self.rank = torch.distributed.get_rank()
        # we need to validate the config before using the profiler
        self._validate()
        if config.use_profile and self.rank in self.config.profile_ranks:
            print(f"[Profiler] Profiler init for rank {self.rank}")

            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=max(self.config.step_start - 1, 0),
                    warmup=1 if self.config.step_start > 0 else 0,
                    active=self.config.step_end - self.config.step_start,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    def _validate(self):
        if self.config.use_profile:
            if self.config.profile_ranks is None:
                print("[WARNING] Profile ranks is not set, default to rank 0")
                self.config.profile_ranks = [0]
            assert self.config.step_start >= 0, "[ERROR] Profile step start must be greater than 0"
            assert self.config.step_end >= 0, "[ERROR] Profile step end must be greater than 0"
            assert self.config.step_start < self.config.step_end, (
                "[ERROR] Profile step start must be less than step end"
            )

    def check(self):
        return self.prof is not None and not self.skip_prof

    def start(self):
        if self.check():
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None and not self.saved:
            if not os.path.exists(self.config.save_path):
                os.makedirs(self.config.save_path)
            save_file_name = f"/prof_start_{self.config.step_start}_end_{self.config.step_end}_rank_{self.rank}.json"
            print(f"[Profiler] Saving trace to {self.config.save_path + save_file_name}")
            self.prof.export_chrome_trace(self.config.save_path + save_file_name)
            self.skip_prof = True
            self.saved = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        if self.check():
            print(f"[Profiler] Trace stopped for rank {self.rank}")
            self.skip_prof = True


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Start a profiling range marker (no-op implementation).

    Args:
        message (Optional[str]): Message to associate with the range marker.
        color (Optional[str]): Color for the marker visualization.
        domain (Optional[str]): Domain for the marker.
        category (Optional[str]): Category for the marker.
    """
    pass


def mark_end_range(range_id: str) -> None:
    """End a profiling range marker (no-op implementation).

    Args:
        range_id (str): Identifier of the range to end.
    """
    pass


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorator to annotate a function with profiling markers (no-op implementation).

    Args:
        message (Optional[str]): Message to associate with the annotation.
        color (Optional[str]): Color for the marker visualization.
        domain (Optional[str]): Domain for the marker.
        category (Optional[str]): Category for the marker.

    Returns:
        Callable: Decorator function that returns the original function unchanged.
    """

    def decorator(func):
        return func

    return decorator


class DistProfiler:
    """A distributed profiler class for collecting performance metrics across multiple ranks.

    This profiler is designed to work in distributed training environments, allowing selective
    profiling of specific ranks or all ranks. It provides basic start/stop functionality and
    supports annotation of code sections for detailed profiling.

    Args:
        rank (int): The rank of the current process
        config (ProfilerConfig, optional): Configuration for the profiler.
    """

    def __init__(self, rank: int, config: Optional[ProfilerConfig] = None, **kwargs):
        pass

    def start(self, **kwargs):
        pass

    def stop(self):
        pass

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> Callable:
        def decorator(func):
            return func

        return decorator


class DistProfilerExtension:
    """An extension class for DistProfiler that provides distributed profiling capabilities.

    It is intended for workers in verl that single controller invokes.

    This class wraps a DistProfiler instance and provides methods to start/stop profiling
    that can be dispatched across multiple ranks in a distributed training environment.

    Args:
        profiler (DistProfiler): The base distributed profiler instance to extend
    """

    def __init__(self, profiler: DistProfiler):
        self.profiler = profiler

    from verl.single_controller.base.decorator import Dispatch, register

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()
