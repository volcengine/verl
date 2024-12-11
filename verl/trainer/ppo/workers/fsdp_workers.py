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
import warnings
import ray
import torch
import torch.distributed
from omegaconf import DictConfig, open_dict

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.actor import DataParallelPPOActor
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, load_fsdp_grad, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
import verl.utils.hdfs_io as hdfs_io
from verl.utils import set_pad_token_id

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


@ray.remote
class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        set_pad_token_id(self.tokenizer)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl.trainer.ppo.rollout import HFRollout
            from verl.trainer.ppo.hybrid_engine import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.trainer.ppo.rollout.vllm_rollout import vLLMRollout
            from verl.trainer.ppo.hybrid_engine import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                       inference_engine=rollout.inference_engine,
                                                       model_config=self.actor_model_config,
                                                       full_params='hf' in self.config.rollout.load_format)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        metrics = self.actor.update_policy(data=data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)

            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch['old_log_probs'] = old_log_probs

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


@ray.remote
class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size()
        self.config.ppo_micro_batch_size //= torch.distributed.get_world_size()

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: support loading critic weights from RM. Support using AutoModelForTokenClassification
        from transformers import AutoTokenizer

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=config.model.get('trust_remote_code', False))
        set_pad_token_id(self.tokenizer)

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                 torch_dtype=torch_dtype,
                                                                 config=critic_model_config,
                                                                 attn_implementation='flash_attention_2',
                                                                 trust_remote_code=trust_remote_code)
            critic_module.lm_head = nn.Sequential(nn.Linear(critic_model_config.hidden_size, 1, dtype=torch_dtype),
                                                  LambdaLayer(fn=squeeze))

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable()
        if self.rank == 0:
            print_model_size(critic_module)

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.trainer.ppo.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.ppo_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={'values': values})
        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())
        metrics = self.critic.update_critic(data=data)

        self.critic_lr_scheduler.step()
        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics['critic/lr(1e-4)'] = lr * 1e4

        output = DataProto(batch=None, meta_info={'metrics': metrics})
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


@ray.remote
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_local_path,
                                                                 trust_remote_code=config.model.get(
                                                                     'trust_remote_code', False))
            self.tokenizer = AutoTokenizer.from_pretrained(local_path,
                                                           trust_remote_code=config.model.get(
                                                               'trust_remote_code', False))
            set_pad_token_id(self.tokenizer)
            set_pad_token_id(self.input_tokenizer)

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                               torch_dtype=torch.bfloat16,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload))

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.reward_module(input_ids=micro_batch['input_ids'],
                                        attention_mask=micro_batch['attention_mask'],
                                        position_ids=micro_batch['position_ids'])
            rm_score = output.logits  # (batch_size,)
            rm_score = rm_score.squeeze(-1)
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()
        micro_batches = rm_data.batch.split(self.config.micro_batch_size)
        output = []
        for micro_batch in micro_batches:
            rm_score = self._forward_micro_batch(micro_batch)
            output.append(rm_score)
        scores = torch.cat(output, dim=0)  # (batch_size)
        token_level_scores = self._expand_to_token_level(data, scores)
        # Note that this is only the scores, may not be the final rewards used to train RL
        output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
