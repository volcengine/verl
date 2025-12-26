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

import pytest
import ray

from tests.checkpoint_engine.test_utils import Worker


@pytest.mark.parametrize("rebuild_group, num_rollout", [(False, 6), (True, 6)])
def test_nccl_checkpoint_engine(rebuild_group, num_rollout, num_nodes=1, num_gpus_per_node=8):
    ray.init(
        runtime_env={
            "env_vars": {
                "UCX_TLS": "rc,tcp,cuda",
                "UCX_MAX_RNDV_RAILS": "4",
                "UCX_LOG_LEVEL": "INFO",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    checkpoint_kwargs = {
        "bucket_size": 3 * 1024 * 1024 * 1024,  # 3GB
        "rebuild_group": rebuild_group,
    }
    trainer = [
        Worker.options(
            runtime_env={
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        ).remote("trainer", "nccl", {**checkpoint_kwargs, "is_master": rank == 0}, rank, 2)
        for rank in range(2)
    ]

    rollout = [Worker.remote("rollout", "nccl", checkpoint_kwargs, rank, num_rollout) for rank in range(num_rollout)]
    workers = trainer + rollout

    for _ in range(3):
        # 1. prepare all workers
        metadata = ray.get([worker.execute_checkpoint_engine.remote("prepare") for worker in workers])
        metadata = [metadata[0]] + metadata[-len(rollout) :]
        kwargs = []
        for rank in range(len(metadata)):
            kwargs.append(
                {
                    "rank": rank,
                    "world_size": len(metadata),
                    "master_metadata": metadata[0],
                }
            )

        dummy = {"rank": -1, "world_size": len(metadata), "master_metadata": metadata[0]}
        kwargs = [kwargs[0]] + [dummy] * (len(trainer) - 1) + kwargs[1:]
        assert len(kwargs) == len(workers), f"kwargs length must be {len(workers)}, but got {len(kwargs)}"

        # 2. init process group between all workers
        ray.get(
            [
                worker.execute_checkpoint_engine.remote("init_process_group", **kwargs[rank])
                for rank, worker in enumerate(workers)
            ]
        )

        # 3. update weights of all workers
        ray.get([worker.update_weights.remote() for worker in workers])

        # 4. finish all workers
        ray.get([worker.execute_checkpoint_engine.remote("finish") for worker in workers])

        # 5. check weights of all workers
        ray.get([worker.check_weights.remote() for worker in workers])

    ray.shutdown()


if __name__ == "__main__":
    test_nccl_checkpoint_engine(14)
