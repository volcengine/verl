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
"""
The main entry point to run the PPO algorithm
"""

import os
import logging
import ray
import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig

from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.sharding_manager import AllGatherPPModel
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel

from verl.single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.model import load_megatron_model_weights
from verl.utils.flops_counter import FlopsCounter
from verl.utils.megatron_utils import init_model_parallel_config
from verl.utils.megatron_utils import offload_megatron_param_and_grad, load_megatron_param_and_grad
from verl.utils import hf_tokenizer

from codetiming import Timer

from megatron.core import parallel_state as mpu
from megatron.core import ModelParallelConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def set_random_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class ActorRolloutRefWorker(MegatronWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ['LOCAL_RANK'])
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

            if self.config.actor.megatron.sequence_parallel:
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            
            from megatron.training.arguments import parse_args
            from megatron.training.global_vars import set_args
            args = parse_args(ignore_unknown_args=True)
            set_args(args)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.actor.megatron.seed)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor and self._is_rollout:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get('ppo_micro_batch_size', None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.get('param_offload', False)
            self._is_offload_grad = self.config.actor.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.get('optimizer_offload', False)
        elif self._is_ref:
            if self.config.ref.get('ppo_micro_batch_size', None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.ppo_micro_batch_size
            self._is_offload_param = self.config.ref.get('param_offload', False)