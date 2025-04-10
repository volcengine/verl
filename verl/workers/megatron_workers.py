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
from verl.utils.model import load_megatron_model_weights, load_megatron_gptmodel_weights
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.megatron_utils import mcore_model_parallel_config
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
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
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
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config, convert_config
        from transformers import AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path)

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
        self.architectures = getattr(actor_model_config, "architectures", None)

        tfconfig = convert_config(actor_model_config, megatron_config)
        if enable_gradient_checkpointing:
            gradient_checkpointing_cfg = dict(self.config.model.get('gradient_checkpointing_kwargs', dict()))
            tfconfig.recompute_method = gradient_checkpointing_cfg['activations_checkpoint_method']
            tfconfig.recompute_granularity = gradient_checkpointing_cfg['activations_checkpoint_granularity']
            tfconfig.recompute_num_layers = gradient_checkpointing_cfg['activations_checkpoint_num_layers']
        print(f'TF config: {tfconfig}')
        self.hf_config = actor_model_config

        def megatron_actor_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_gptmodel_from_config
            parallel_model = get_parallel_gptmodel_from_config(
                tfconfig,
                actor_model_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                value=False)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        if self._is_actor and self._is_rollout:
            # Initialize the 3D HybridEngine
            hybrid_engine = AllGatherPPModel(
                model_provider=megatron_actor_model_provider,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer)
            # Fetch the model at current rank
            actor_module = hybrid_engine.this_rank_models
            actor_modules_list = []
            if isinstance(actor_module, nn.ModuleList):
                for module in actor_module:
                    actor_modules_list.append(module)
            actor_module = actor_modules_list
            print(f'actor_module: {len(actor_module)}')
            if self.config.actor.load_weight:
                load_megatron_gptmodel_weights(self.config,
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
                                   wrap_with_ddp=False,
                                   use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer)
            # ref_module = nn.ModuleList(ref_module)

            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print(f'load ref weight start')
                load_megatron_gptmodel_weights(self.config,
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
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.weight",
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
            raise NotImplementedError('Only vllmRollout is supported with Megatron now')

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
        self.param_dtype = torch.bfloat16

        megatron_config = mcore_model_parallel_config(sequence_parallel=self.config.actor.megatron.get(
            'sequence_parallel', True),
                                                      params_dtype=PrecisionType.to_dtype(self.param_dtype))

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
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False)
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
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False))
            self.ref_policy = MegatronPPOActor(config=self.config.ref,
                                               model_config=self.ref_model_config,
                                               megatron_config=megatron_config,
                                               actor_module=self.ref_module,
                                               actor_optimizer=None,
                                               actor_optimizer_config=None)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                model_config=self.actor_model_config,
                role='actor',
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                tokenizer=self.tokenizer,
                optimizer=self.actor_optimizer,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                checkpoint_contents=self.config.actor.checkpoint.contents)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name='update_policy', logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info['global_token_num']
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics['perf/mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

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
        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)

            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After generate_sequences', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_ref
        if self._is_offload_param:
            load_megatron_param_and_grad(self.ref_module, torch.cuda.current_device(), self._is_offload_grad)

        micro_batch_size = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})
        output = output.to('cpu')
        if self._is_offload_param:
            offload_megatron_param_and_grad(self.ref_module, self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        data = data.to('cuda')
        output = data
        # we should always recompute old_log_probs when it is HybridEngine
        output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        output.meta_info['temperature'] = self.config.rollout.temperature
        old_log_probs = self.actor.compute_log_prob(data=output)
        output.batch['old_log_probs'] = old_log_probs
        output = output.to('cpu')
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After generate_sequences', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path,
                                                  hdfs_path=hdfs_path,
                                                  del_local_after_load=del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path,
                                                  hdfs_path=hdfs_path,
                                                  global_step=global_step,
                                                  max_ckpt_to_keep=max_ckpt_to_keep)


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
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get('ppo_micro_batch_size', None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

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
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config, convert_config
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the critic_model_config
        critic_model_config = AutoConfig.from_pretrained(local_path)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        self.share_embeddings_and_output_weights = getattr(critic_model_config, "tie_word_embeddings", False)
        update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(critic_model_config, "architectures", None)
        if self.rank == 0:
            print(f'Model config after override: {critic_model_config}')
        tfconfig = convert_config(critic_model_config, megatron_config)
        if enable_gradient_checkpointing:
            gradient_checkpointing_cfg = dict(self.config.model.get('gradient_checkpointing_kwargs', dict()))
            tfconfig.recompute_method = gradient_checkpointing_cfg['activations_checkpoint_method']
            tfconfig.recompute_granularity = gradient_checkpointing_cfg['activations_checkpoint_granularity']
            tfconfig.recompute_num_layers = gradient_checkpointing_cfg['activations_checkpoint_num_layers']
        print(f'Critic TF config: {tfconfig}')
        self.hf_config = critic_model_config

        def megatron_critic_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_gptmodel_from_config
            parallel_model = get_parallel_gptmodel_from_config(tfconfig,
                                                               critic_model_config,
                                                               pre_process,
                                                               post_process,
                                                               share_embeddings_and_output_weights=False,
                                                               value=True)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        critic_module = get_model(model_provider_func=megatron_critic_model_provider,
                                  model_type=ModelType.encoder_or_decoder,
                                  wrap_with_ddp=True,
                                  use_distributed_optimizer=self.config.megatron.use_distributed_optimizer)
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            load_megatron_gptmodel_weights(self.config,
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
        self.param_dtype = torch.bfloat16

        megatron_config = mcore_model_parallel_config(sequence_parallel=self.config.megatron.get(
            'sequence_parallel', True),
                                                      params_dtype=PrecisionType.to_dtype(self.param_dtype))
        self.critic_module, self.critic_optimizer, self.critic_model_config, critic_optimizer_config = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            megatron_config=megatron_config,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False))
        self.critic = MegatronPPOCritic(config=self.config,
                                        model_config=self.critic_model_config,
                                        megatron_config=megatron_config,
                                        critic_module=self.critic_module,
                                        critic_optimizer=self.critic_optimizer,
                                        critic_optimizer_config=critic_optimizer_config)
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            model_config=self.critic_model_config,
            role='critic',
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            tokenizer=self.tokenizer,
            optimizer=self.critic_optimizer,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            checkpoint_contents=self.config.checkpoint.contents)

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
        with Timer(name='update_critic', logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info['global_token_num']
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics['perf/mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        output = DataProto(batch=None, meta_info={'metrics': metrics})
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path,
                                                  hdfs_path=hdfs_path,
                                                  del_local_after_load=del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path,
                                                  hdfs_path=hdfs_path,
                                                  global_step=global_steps,
                                                  max_ckpt_to_keep=max_ckpt_to_keep)


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
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, megatron_config: ModelParallelConfig, override_model_config):
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.model import update_model_config
        from verl.utils.megatron_utils import get_model
        from transformers import AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_to_local(model_path)
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
            print(f'Model config after override: rm_model_config {rm_model_config}')

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
                                                            share_embeddings_and_output_weights=False,
                                                            value=True)
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        reward_model = get_model(model_provider_func=megatron_rm_model_provider,
                                 model_type=ModelType.encoder_or_decoder,
                                 wrap_with_ddp=False,
                                 use_distributed_optimizer=self.config.reward_model.use_distributed_optimizer)
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

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer)
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get('rm_tokenizer', None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path)
            rm_tokenizer = hf_tokenizer(rm_tokenizer_local_path)

        self.param_dtype = torch.bfloat16

        megatron_config = mcore_model_parallel_config(sequence_parallel=self.config.megatron.get(
            'sequence_parallel', True),
                                                      params_dtype=PrecisionType.to_dtype(self.param_dtype))

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
