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

import functools
import os
from typing import Callable, Optional

import torch
import torch.distributed

from ..memory_utils import MemorySnapshotSampler, enable_memory_visualize
from .config import ProfilerConfig, TorchMemoryToolConfig, TorchProfilerToolConfig


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

    def __init__(self, config: ProfilerConfig, tool_config: Optional[TorchProfilerToolConfig] = None):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        if not config:
            config = ProfilerConfig(ranks=[], enable=False)
        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"
        self.prof = None
        self.saved = False
        self.enable = config.enable
        if not config.enable:
            return
        self.config = config
        self.tool_config = tool_config
        self.rank = torch.distributed.get_rank()
        # we need to validate the config before using the profiler
        self._validate()
        if self.rank in self.config.profile_ranks:
            print(f"[Profiler] Profiler init for rank {self.rank}")

            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=max(self.tool_config.step_start - 1, 0),
                    warmup=1 if self.tool_config.step_start > 0 else 0,
                    active=self.tool_config.step_end - self.tool_config.step_start,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    def _validate(self):
        if self.enable:
            if self.config.profile_ranks is None:
                print("[WARNING] Profile ranks is not set, default to rank 0")
                self.config.profile_ranks = [0]
            assert self.tool_config.step_start >= 0, "[ERROR] Profile step start must be greater than 0"
            assert self.tool_config.step_end >= 0, "[ERROR] Profile step end must be greater than 0"
            assert self.tool_config.step_start < self.tool_config.step_end, (
                "[ERROR] Profile step start must be less than step end"
            )

    def check(self):
        return self.prof is not None and self.enable

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
            self.enable = False
            self.saved = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        if self.check():
            print(f"[Profiler] Trace stopped for rank {self.rank}")
            self.enable = False


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
    """A dispatcher that delegates to specific profilers based on config.tool.

    Supported tools:
    - nsys: NsightSystemsProfiler
    - npu: NPUProfiler (Ascend)
    - torch: PyTorch torch.profiler wrapper
    - torch_memory: Torch CUDA memory snapshot dump
    """

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig] = None, tool_config: Optional[object] = None, **kwargs
    ):
        # Default config
        if not config:
            config = ProfilerConfig(ranks=[], enable=False)

        self._impl = None
        self._tool = getattr(config, "tool", None)

        # Normalize rank selection
        self._this_rank = False
        if config.all_ranks:
            self._this_rank = True
        elif config.ranks:
            self._this_rank = rank in config.ranks
        else:
            # default rank 0 if enabled but ranks unspecified
            self._this_rank = (rank == 0) if config.enable else False

        # Lazy import to avoid circular deps
        if self._tool == "nsys":
            from .nvtx_profile import NsightSystemsProfiler as _Nsight

            self._impl = _Nsight(rank=rank, config=config, tool_config=tool_config, **kwargs)
        elif self._tool == "npu":
            from .mstx_profile import NPUProfiler as _Npu

            self._impl = _Npu(rank=rank, config=config, tool_config=tool_config, **kwargs)
        elif self._tool == "torch":
            # Use the torch profiler wrapper defined above
            self._impl = Profiler(config=config, tool_config=tool_config)
        elif self._tool == "torch_memory":
            self._impl = TorchMemoryProfiler(rank=rank, config=config, tool_config=tool_config)
        else:
            # Fallback to a no-op impl
            self._impl = _NoOpProfiler()

    def start(self, **kwargs):
        return getattr(self._impl, "start", lambda **_: None)(**kwargs)

    def stop(self):
        return getattr(self._impl, "stop", lambda: None)()

    @classmethod
    def annotate(
        cls,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                profiler = getattr(self_instance, "profiler", None)
                if not profiler:
                    return func(self_instance, *args, **kwargs_inner)

                impl = profiler._impl
                if hasattr(impl, "annotate"):
                    try:
                        actual_decorator = impl.annotate(
                            message=message, color=color, domain=domain, category=category, **kwargs_outer
                        )

                        return actual_decorator(func)(self_instance, *args, **kwargs_inner)
                    except Exception:
                        return func(self_instance, *args, **kwargs_inner)
                return func(self_instance, *args, **kwargs_inner)

            return wrapper

        return decorator


class _NoOpProfiler:
    def start(self, **kwargs):
        return

    def stop(self):
        return


class TorchMemoryProfiler:
    """Profiler that dumps CUDA memory snapshots at step boundaries.

    Behavior:
    - On first construction (per process), enable memory history recording if CUDA is available
    - On start(step=X), remember sub_dir for this step
    - On stop(), dump a memory snapshot into config.save_path under the remembered sub_dir
    """

    _memory_history_enabled: bool = False

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[TorchMemoryToolConfig] = None
    ):
        # Always respond to explicit start/stop calls for torch_memory tool,
        # regardless of per-role enable flag, to align with global step control.
        self.enable = True
        if not config:
            config = ProfilerConfig(ranks=[])
        self.config = config
        self.rank = rank
        self.this_step = False
        self.sub_dir = None
        self.sampler = MemorySnapshotSampler()

        # Get parameters from tool_config, with fallback to defaults
        if tool_config:
            trace_alloc_max_entries = tool_config.trace_alloc_max_entries
            stack_depth = tool_config.stack_depth
        else:
            trace_alloc_max_entries = 100_000
            stack_depth = 32

        # Best-effort enable memory history once
        if not TorchMemoryProfiler._memory_history_enabled:
            try:
                enable_memory_visualize(trace_alloc_max_entries=trace_alloc_max_entries, stack_depth=stack_depth)
            except Exception:
                # silently ignore if not supported
                pass
            TorchMemoryProfiler._memory_history_enabled = True

    def start(self, **kwargs):
        if not self.enable:
            return
        if not self._should_profile_this_rank():
            return
        profile_step = kwargs.get("profile_step", None)
        # Keep ranks aligned under same folder name
        self.sub_dir = f"step{profile_step}" if profile_step is not None else None
        self.this_step = True

    def stop(self):
        if not self.enable or not self.this_step:
            return
        self.this_step = False
        if not self._should_profile_this_rank():
            return
        out_dir = self.config.save_path or "outputs/profile"
        tag = "torch_memory"
        # Dump snapshot; all ranks write into same sub_dir
        try:
            self.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=self.sub_dir)
        except Exception:
            pass

    def _should_profile_this_rank(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        # default rank 0
        return self.rank == 0


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
