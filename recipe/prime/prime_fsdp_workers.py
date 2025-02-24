# Copyright 2024 PRIME team and/or its affiliates
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

import logging
import os
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer
from verl.workers.fsdp_workers import create_device_mesh

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class PRIMERewardModelWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)
            self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                      self.ulysses_sequence_parallel_size)
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

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

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

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

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)

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

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module,
                                                        optimizer=self.critic_optimizer,
                                                        lr_scheduler=self.critic_lr_scheduler,
                                                        tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
