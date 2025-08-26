# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

os.environ['NCCL_DEBUG'] = 'WARN'


from verl.workers.roles import ActorWorker
from verl.workers.config import ActorConfig, HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


import ray

if __name__ == "__main__":
    path = '/mnt/hdfs/zhangchi.usc1992_lf_lq/models/Qwen2.5-0.5B-Instruct'
    model_config = HFModelConfig(path=path)
    engine_config = McoreEngineConfig(forward_only=False, use_mbridge=True)
    optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    config = ActorConfig(model_config=model_config, 
                         engine=engine_config, 
                         strategy="megatron", 
                         ppo_micro_batch_size_per_gpu=256,
                         optim=optimizer_config,
                         n=1)
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[1])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)

    wg.init_model()