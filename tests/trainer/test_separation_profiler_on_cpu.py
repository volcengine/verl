# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl.experimental.separation import ray_trainer
from verl.single_controller.ray.base import RayWorkerGroup


@pytest.mark.parametrize(
    "continuous, expected_starts, expected_stops",
    [
        (False, [1, 2, 5], [1, 2, 5]),
        (True, [1, 5], [2, 5]),
    ],
    ids=["per-step", "continuous-window"],
)
def test_worker_capture_count_matches_separation_windows(monkeypatch, continuous, expected_starts, expected_stops):
    trainer = ray_trainer.SeparateRayPPOTrainer.__new__(ray_trainer.SeparateRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {},
            "global_profiler": {
                "tool": "nsys",
                "steps": [1, 2, 5],
                "profile_continuous_steps": continuous,
                "global_tool_config": {
                    "nsys": {"discrete": False, "worker_nsight_options": {"capture-range-end": None}}
                },
            },
        }
    )
    trainer.device_name = "cpu"
    trainer.resource_pool_to_cls = {"pool": {"actor": object()}}
    groups = []

    def create_worker_group(resource_pool, ray_cls_with_init, **kwargs):
        # Attach to placeholder handles to exercise real capture configuration without starting Ray actors.
        group = RayWorkerGroup(worker_names=["worker"], worker_handles=[object()], **kwargs)
        group.spawn = Mock(return_value={})
        groups.append(group)
        return group

    trainer.ray_worker_group_cls = create_worker_group
    monkeypatch.setattr(ray_trainer, "create_colocated_worker_cls", Mock())
    trainer._init_worker_groups()

    starts, stops = [], []

    def start_profiling(enabled):
        if enabled:
            starts.append(trainer.global_steps)

    def stop_profiling(enabled):
        if enabled:
            stops.append(trainer.global_steps)

    trainer._start_profiling = start_profiling
    trainer._stop_profiling = stop_profiling
    trainer.timing_raw = {}
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    for step in range(1, 6):
        trainer.global_steps = step
        trainer._fit_start_profile()
        trainer._fit_stop_profile()

    assert starts == expected_starts
    assert stops == expected_stops
    assert len(starts) == len(stops)
    assert groups[0].worker_nsight_options["capture-range-end"] == f"repeat-shutdown:{len(starts)}"
