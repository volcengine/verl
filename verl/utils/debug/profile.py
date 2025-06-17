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
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed


class Profiler:
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
            assert self.config.step_start < self.config.step_end, "[ERROR] Profile step start must be less than step end"

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


def mark_start_range(message: Optional[str] = None, color: Optional[str] = None, domain: Optional[str] = None, category: Optional[str] = None) -> None:
    pass


def mark_end_range(range_id: str) -> None:
    pass


def mark_annotate(message: Optional[str] = None, color: Optional[str] = None, domain: Optional[str] = None, category: Optional[str] = None) -> Callable:
    def decorator(func):
        return func

    return decorator


@dataclass
class ProfilerConfig:
    """
    Worker profiler config. Currently only support Nsight system profiler.
    """

    all_ranks: bool = False
    ranks: Optional[list[int]] = None
    discrete: bool = False

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            discrete=self.discrete or other.discrete,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            discrete=self.discrete and other.discrete,
        )


class WorkerProfiler:
    def __init__(self, rank: int, config: Optional[ProfilerConfig] = None):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    @staticmethod
    def annotate(message: Optional[str] = None, color: Optional[str] = None, domain: Optional[str] = None, category: Optional[str] = None) -> Callable:
        def decorator(func):
            return func

        return decorator


class WorkerProfilerExtension:
    def __init__(self, profiler: WorkerProfiler):
        self.profiler = profiler

    from verl.single_controller.base.decorator import Dispatch, register

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()
