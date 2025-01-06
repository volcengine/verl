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
import json
import os

import ray

# compatiblity cern
from verl.single_controller.base.decorator import *


def maybe_remote(main):
    """Schedule main function as ray remote task if VERL_DRIVER_NUM_GPUS or VERL_DRIVER_RESOURCES specified in config.
       - VERL_DRIVER_NUM_GPUS: number of GPUs for driver task.
       - VERL_DRIVER_RESOURCES: custom resources for driver task, e.g {"verl_driver": 1.0}.

    For job submission to ray cluster, you can specify these two envs in runtime.yaml.
    ```yaml
    working_dir: "."
    env_vars:
      VERL_DRIVER_NUM_GPUS: "1"
      VERL_DRIVER_RESOURCES: '{"verl_driver": 1.0}'
    ```

    ray job submit --runtime-env=runtime.yaml -- python3 test.py

    Args:
        main (Callable): main function to be schedule.
    """

    num_gpus = 0
    resources = {}
    env_num_gpus = os.getenv("VERL_DRIVER_NUM_GPUS")
    if env_num_gpus:
        num_gpus = int(env_num_gpus)
    env_resources = os.getenv("VERL_DRIVER_RESOURCES")
    if env_resources:
        resources = json.loads(env_resources)
    print(f"verl driver num_gpus: {num_gpus}, resources={resources}")
    assert isinstance(resources, dict), f"resources must be dict, got {type(resources)}"

    @functools.wraps(main)
    def _main(*args, **kwargs):
        # Run main function locally.
        if num_gpus == 0 and len(resources) == 0:
            return main(*args, **kwargs)

        # Run main function remotely as ray task.
        f = ray.remote(num_gpus=num_gpus, resources=resources)(main)
        return ray.get(f.remote(*args, **kwargs))

    return _main
