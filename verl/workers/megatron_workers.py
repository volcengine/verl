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
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.model import load_megatron_model_weights
from verl.utils.megatron_utils import init_model_parallel_config
from verl.utils.megatron_utils import offload_megatron_param_and_grad, load_megatron_param_and_grad
from verl.utils import hf_tokenizer

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
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=None,
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
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self._is_offload_param = self.config.actor.get('param_offload', False)
            self._is_offload_grad = self.config.actor.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.get('optimizer_offload', False)
        elif self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self._is_offload_param = self.config.ref.get('param_offload', False)

    def _build_model_optimizer(self,
                               model_path,
                               megatron_config: ModelParallelConfig,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False):
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_local_path_from_hdfs(model_path)
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the actor_model_config
        actor_model_config = AutoConfig.from_pretrained(local_path)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        def megatron_actor_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_model_from_config
            # vpp is not supported yet because it will hang for some reason. Need debugging
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            parallel_model = get_parallel_model_from_config(config=actor_model_config,
                                                            megatron_config=megatron_config,
                                                            pre_process=pre_process,
                                                            post_process=post_process,
                                                            value=False)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        if self._is_actor and self._is_rollout:
            # Initialize the 3D HybridEngine
            hybrid_engine = AllGatherPPModel(model_provider=megatron_actor_model_provider)
            # Fetch the model at current rank
            actor_module = hybrid_engine.this_rank_models
            if isinstance(actor_module, nn.ModuleList):
                actor_module = [actor_module[0]]
            if self.config.actor.load_weight:
                load_megatron_model_weights(self.config,
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
                load_megatron_model_weights(self.config,
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
            from verl.workers.rollout.vllm_rollout import vLLMRollout
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        torch_dtype = torch.bfloat16

        megatron_config = OmegaConf.create({
            'sequence_parallel': self.config.actor.megatron.get('sequence_parallel', True),
            'param_dtype': PrecisionType.to_str(torch_dtype),
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        })

        megatron_config = init_model_parallel_config(megatron_config)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
            else:
                optim_config = None
            self.actor_module, self.hybrid_engine, self.actor_optimizer, \
            self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                megatron_config=megatron_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
            )

        if self._is_actor:
            self.actor = MegatronPPOActor(config=self.config.actor,
                                          model_config=self.actor_model_config,
                                          megatron_config=megatron_config,
                                          actor_module=self.actor_module,
                                          actor_optimizer=self.actor_optimizer,
                                          actor_optimizer_config=self.actor_optim_config)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                megatron_config=megatron_config,
                optim_config=None,
                override_model_config=override_model_config,
            )
            self.ref_policy = MegatronPPOActor(config=self.config.ref,
                                               model_config=self.ref_model_config,
                                               megatron_config=megatron_config,
                                               actor_module=self.ref_module,
                                               actor_optimizer=None,
                                               actor_optimizer_config=None)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        dataloader = self.actor.make_minibatch_iterator(data=data)
        metrics = self.actor.update_policy(dataloader=dataloader)

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)

            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)

        validate = prompts.meta_info.get('validate', False)
        if self._is_actor and not validate:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch['old_log_probs'] = old_log_probs

        output = output.to('cpu')
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_ref
        if self._is_offload_param:
            load_megatron_param_and_grad(self.ref_module, torch.cuda.current_device(), self._is_offload_grad)

        micro_batch_size = self.config.rollout.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})
        output = output.to('cpu')
        if self._is_offload_param:
            offload_megatron_param_and_grad(self.ref_module, self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path):
        assert self._is_actor
        pass


class CriticWorker(MegatronWorker):

    def __init__(self, config):
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

            if self.config.megatron.sequence_parallel:
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(self,
                                      model_path,
                                      megatron_config: ModelParallelConfig,
                                      optim_config,
                                      override_model_config,
                                      enable_gradient_checkpointing=False):
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config, init_model_parallel_config
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_local_path_from_hdfs(model_path)
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the actor_model_config
        critic_model_config = AutoConfig.from_pretrained(local_path)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f'Model config after override: {critic_model_config}')

        def megatron_critic_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_model_from_config
            # TODO: support vpp here
            # vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            parallel_model = get_parallel_model_from_config(config=critic_model_config,
                                                            megatron_config=megatron_config,
                                                            pre_process=pre_process,
                                                            post_process=post_process,
                                                            value=True)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        critic_module = get_model(model_provider_func=megatron_critic_model_provider,
                                  model_type=ModelType.encoder_or_decoder,
                                  wrap_with_ddp=True)
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            load_megatron_model_weights(self.config,
                                        critic_model_config,
                                        critic_module,
                                        params_dtype=megatron_config.params_dtype,
                                        is_value_model=True)
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config)
        torch.cuda.empty_cache()
        return critic_module, critic_optimizer, critic_model_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic
        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        torch_dtype = torch.bfloat16

        megatron_config = OmegaConf.create({
            'sequence_parallel': self.config.megatron.get('sequence_parallel', True),
            'param_dtype': PrecisionType.to_str(torch_dtype),
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        })

        megatron_config = init_model_parallel_config(megatron_config)

        critic_module, critic_optimizer, critic_model_config, critic_optimizer_config = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            megatron_config=megatron_config,
            optim_config=self.config.optim,
            override_model_config=override_model_config)
        self.critic = MegatronPPOCritic(config=self.config,
                                        model_config=critic_model_config,
                                        megatron_config=megatron_config,
                                        critic_module=critic_module,
                                        critic_optimizer=critic_optimizer,
                                        critic_optimizer_config=critic_optimizer_config)

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={'values': values})
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        dataloader = self.critic.make_minibatch_iterator(data)
        metrics = self.critic.update_critic(dataloader=dataloader)
        output = DataProto(batch=None, meta_info={'metrics': metrics})
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path):
        pass


class RewardModelWorker(MegatronWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
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

            if self.config.megatron.sequence_parallel:
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        self.config.micro_batch_size //= mpu.get_data_parallel_world_size()

    def _build_rm_model(self, model_path, megatron_config: ModelParallelConfig, override_model_config):
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.megatron_utils import get_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_local_path_from_hdfs(model_path)
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the actor_model_config
        rm_model_config = AutoConfig.from_pretrained(local_path)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(rm_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f'Model config after override: {rm_model_config}')

        def megatron_rm_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_model_from_config
            # vpp is not supported yet because it will hang for some reason. Need debugging
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            parallel_model = get_parallel_model_from_config(config=rm_model_config,
                                                            megatron_config=megatron_config,
                                                            pre_process=pre_process,
                                                            post_process=post_process,
                                                            value=True)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        reward_model = get_model(model_provider_func=megatron_rm_model_provider,
                                 model_type=ModelType.encoder_or_decoder,
                                 wrap_with_ddp=False)
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # reward_model = nn.ModuleList(reward_model)

        if self.config.load_weight:
            load_megatron_model_weights(self.config,
                                        rm_model_config,
                                        reward_model,
                                        params_dtype=megatron_config.params_dtype,
                                        is_value_model=True)

        # TODO: add more optimizer args into config
        torch.cuda.empty_cache()
        return reward_model, rm_model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic
        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoTokenizer

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        sft_tokenizer_local_path = copy_local_path_from_hdfs(self.config.model.input_tokenizer)
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get('rm_tokenizer', None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_local_path_from_hdfs(rm_tokenizer_path)
            rm_tokenizer = hf_tokenizer(rm_tokenizer_local_path)

        torch_dtype = torch.bfloat16

        megatron_config = OmegaConf.create({
            'sequence_parallel': self.config.megatron.get('sequence_parallel', True),
            'param_dtype': PrecisionType.to_str(torch_dtype),
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        })

        megatron_config = init_model_parallel_config(megatron_config)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            megatron_config=megatron_config,
            override_model_config=override_model_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        self.rm = MegatronRewardModel(config=self.config,
                                      reward_model_module=reward_model_module,
                                      model_config=reward_model_config,
                                      megatron_config=megatron_config,
                                      sft_tokenizer=sft_tokenizer,
                                      rm_tokenizer=rm_tokenizer)

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data.batch = data.batch.cuda()
        output = self.rm.compute_reward(data)
        output = output.to('cpu')
        return output
