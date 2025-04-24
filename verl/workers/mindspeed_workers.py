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

    def _build_model_optimizer(self,
                               model_path,
                               megatron_config: ModelParallelConfig,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False):
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from transformers import AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path)
        self.tokenizer.max_token_id = max(self.tokenizer.get_vocab().values())

        # Step 2: get the actor_model_config
        actor_model_config = AutoConfig.from_pretrained(local_path)

        self.generation_config = get_generation_config(local_path)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        self.share_embeddings_and_output_weights = getattr(actor_model_config, "tie_word_embeddings", False)
        self.architecture = getattr(actor_model_config, "architecture", None)

        def megatron_actor_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_model_from_config
            # vpp is not supported yet because it will hang for some reason. Need debugging
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            parallel_model = get_parallel_model_from_config(
                config=actor_model_config,
                megatron_config=megatron_config,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                value=False)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        if self._is_actor and self._is_rollout:
            # Initialize the 3D HybridEngine
            hybrid_engine = AllGatherPPModel(model_provider=megatron_actor_model_provider)
            # Fetch the model at current rank
            actor_module = hybrid_engine.this_rank_models
            actor_modules_list = []
            if isinstance(actor_module, nn.ModuleList):
                for module in actor_module:
                    actor_modules_list.append(module)
            actor_module = actor_modules_list
            print(f'actor_module: {len(actor_module)}')
            if self.config.actor.load_weight:
                self.hf_config = load_megatron_model_weights(self.config,
                                                             actor_model_config,
                                                             actor_module,
                                                             params_dtype=megatron_config.params_dtype,
                                                             is_value_model=False)

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage('After AllGatherPPModel init', logger=logger)
        elif self._is_ref:
            print(f'self.config.ref.load_weight: {self.config.ref.load_weight}')
            ref_module = get_model(model_provider_func=megatron_actor_model_provider,
                                   model_type=ModelType.encoder_or_decoder,
                                   wrap_with_ddp=False)
            # ref_module = nn.ModuleList(ref_module)

            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print(f'load ref weight start')
                self.hf_config = load_megatron_model_weights(self.config,
                                                             actor_model_config,
                                                             ref_module,
                                                             params_dtype=megatron_config.params_dtype,
                                                             is_value_model=False)
            log_gpu_memory_usage('After ref module init', logger=logger)
            return ref_module, actor_model_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config = init_megatron_optim_config(optim_config)
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        else:
            optim_config = None
            actor_optimizer = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module, hybrid_engine, actor_optimizer, actor_model_config, optim_config
    
    def _build_rollout(self):
        if self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import MegatronVLLMShardingManager
            from verl.utils.model import normalize_pp_vpp_params

            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.
            layer_name_mapping = {
                "qkv_layer_name":
                    self.config.rollout.layer_name_map.get("qkv_layer_name", "qkv"),
                "gate_proj_layer_name":
                    self.config.rollout.layer_name_map.get("gate_proj_layer_name", "linear_fc1.weight"),
            }

            # reshard the weight partition from actor to rollout to initialize the rollout class
            # create a new cuda space for parameters not in this pp rank
            self.hybrid_engine.load_params_to_cuda()
            # broadcast the parameters from pp rank to other ranks
            self.hybrid_engine.allgather_params()
            # obtain name to parameters in pp/vpp
            params = self.hybrid_engine.get_all_params()
            # update the param name for the
            params = normalize_pp_vpp_params(params=params,
                                             num_hidden_layers=self.actor_model_config.num_hidden_layers,
                                             layer_name='layers')
            assert vllm_mode == 'customized', "Support for vllm>=0.7 for Megatron-LM backend has not been implemented yet."
            rollout = vLLMRollout(actor_module=params,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config,
                                  train_tp=mpu.get_tensor_model_parallel_world_size())
            log_gpu_memory_usage('After building vllm rollout', logger=logger)

            # perform weight resharding between actor and rollout
            sharding_manager = MegatronVLLMShardingManager(module=self.hybrid_engine,
                                                           inference_engine=rollout.inference_engine,
                                                           model_config=self.actor_model_config,
                                                           layer_name_mapping=layer_name_mapping)
            log_gpu_memory_usage('After building sharding manager', logger=logger)
        else:
            NotImplementedError('Only vllmRollout is supported with Megatron now')

        return rollout, sharding_manager
