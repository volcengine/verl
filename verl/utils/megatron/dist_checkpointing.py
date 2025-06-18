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

from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper


def async_save_dist_checkpointing(async_save_requests, sharded_state_dict, ckpt_name):
    async_sharded_save = True
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(with_context_parallel=True))

    # Save model sharded state dicts
    async_save_request = dist_checkpointing.save(sharded_state_dict, ckpt_name, save_strategy=save_strategy, async_sharded_save=async_sharded_save, validate_sharding_integrity=validate_sharding_integrity)
    async_save_requests.append(async_save_request)

    return async_save_requests
