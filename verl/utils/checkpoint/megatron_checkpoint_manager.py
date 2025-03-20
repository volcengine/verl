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

import ray
import os
import random
import numpy as np

import warnings
from typing import Union
import torch
import torch.distributed

from verl.utils.fs import copy_to_local, is_non_local
from verl.models.weight_loader_registry import get_weight_saver
from verl.models.weight_loader_registry import get_weight_loader
from verl.utils.model import load_megatron_model_weights
from verl.utils.megatron_utils import TransformerConfig, get_model_checkpoint_path, get_optimizer_checkpoint_path, get_rng_states_checkpoint_path

from .checkpoint_manager import BaseCheckpointManager
from transformers import AutoModelForCausalLM

from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import \
    get_default_save_sharded_strategy, get_default_load_sharded_strategy

class MegatronCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save 
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(self,
                 config,
                 model_config,
                 role,
                 model: torch.nn.ModuleList,
                 arch: str,
                 hf_config,
                 param_dtype: torch.dtype,
                 share_embeddings_and_output_weights: bool,
                 tokenizer,
                 optimizer,
                 use_distributed_optimizer: bool,
                 async_save: bool=True,
                 checkpoint_contents: list=['model', 'optimizer', 'extra'],
                 **kwargs):

        super().__init__(model, checkpoint_contents=checkpoint_contents)
        self.arch = arch
        self.config = config
        self.role = role
        self.is_value_model = False
        if self.role in ["reward", "critic"]:
            self.is_value_model = True
        self.model_config = model_config
        self.hf_config = hf_config
        self.param_dtype = param_dtype
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.model_path = self.config.model.path
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.use_distributed_optimizer = use_distributed_optimizer
        self.async_save = async_save
        
        self.rank = torch.distributed.get_rank()
        
        self.weight_saver = get_weight_saver(self.arch)
        self.weight_loader = get_weight_loader(self.arch)


    def get_rng_state(self, use_dist_ckpt: bool = False, data_parallel_random_init: bool = False):
        """ collect rng state across data parallel ranks """
        rng_state = {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

        rng_state_list = None
        if torch.distributed.is_initialized() and \
                mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
            rng_state_list = \
                [None for i in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                rng_state_list,
                rng_state,
                group=mpu.get_data_parallel_group())
        else:
            rng_state_list = [rng_state]

        if use_dist_ckpt:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            rng_state_list = ShardedObject('rng_state', rng_state_list, (pp_size, tp_size), (pp_rank, tp_rank),
                                        replica_id=mpu.get_data_parallel_rank(with_context_parallel=True))

        return rng_state_list

    def get_checkpoint_name(self, checkpoints_path, 
                            pipeline_parallel=None,
                            tensor_rank=None, pipeline_rank=None,
                            expert_parallel=None, expert_rank=None, basename="model.pt"):
        """Determine the directory name for this rank's checkpoint."""
        # Use both the tensor and pipeline MP rank.
        if pipeline_parallel is None:
            pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
        if tensor_rank is None:
            tensor_rank = mpu.get_tensor_model_parallel_rank()
        if pipeline_rank is None:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        if expert_parallel is None:
            expert_parallel = (mpu.get_expert_model_parallel_world_size() > 1)
        if expert_rank is None:
            expert_rank = mpu.get_expert_model_parallel_rank()

        # Use both the tensor and pipeline MP rank. If using the distributed
        # optimizer, then the optimizer's path must additionally include the
        # data parallel rank.
        if not pipeline_parallel:
            common_path = os.path.join(checkpoints_path, f'mp_rank_{tensor_rank:02d}')
        else:
            common_path = os.path.join(checkpoints_path, f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

        if expert_parallel:
            common_path = common_path + f'_{expert_rank:03d}'
        
        os.makedirs(common_path, exist_ok=True)

        return os.path.join(common_path, basename)

    def load_optimizer(self, ckpt_path):
        # TODO: Check Optimizer format and distributed optimizer
        optimizer_path = get_optimizer_checkpoint_path(ckpt_path)
        print(f"Loading actor optimizer from {optimizer_path}")
        self.optimizer.load_parameter_state(optimizer_path)


    def load_rng_states(self, ckpt_path, data_parallel_random_init=False, use_dist_ckpt=False):
        rng_state_path = get_rng_states_checkpoint_path(ckpt_path)
        print(f"Loading actor rng states from {rng_state_path}")
        rng_state = torch.load(rng_state_path)
        # access rng_state for data parallel rank
        if not use_dist_ckpt:
            if data_parallel_random_init:
                rng_state = rng_state[mpu.get_data_parallel_rank()]
            else:
                rng_state = rng_state[0]
        random.setstate(rng_state['random_rng_state'])
        np.random.set_state(rng_state['np_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])
        torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
        # Check for empty states array
        if not rng_state['rng_tracker_states']:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_state['rng_tracker_states'])


    def load_checkpoint(self, local_path: str, hdfs_path: str=None, del_local_after_load=False):
        local, ckpt_path = self.checkpath(local_path, hdfs_path)
        
        if ckpt_path is None:
            return

        if 'model' in self.checkpoint_contents:
            model_path = get_model_checkpoint_path(ckpt_path)
            ckpt_name = get_checkpoint_name(model_path)
            # self.model = torch.load(os.path.join(model_path, 'model_state_dict.pt'))
            load_strategy = get_default_load_sharded_strategy(ckpt_name)
            state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_name, load_strategy, strict='assume_ok_unexpected')
            self.weight_loader(state_dict,
                                self.model,
                                self.config,
                                self.param_dtype,
                                is_value_model=self.is_value_model,
                                tie_word_embeddings=self.tie_word_embeddings)
        
        if 'optimizer' in self.checkpoint_contents:
            self.load_optimizer(ckpt_path)
        
        if 'extra' in self.checkpoint_contents:
            self.load_rng_states(ckpt_path)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )

    def save_checkpoint(self, local_path: str, hdfs_path: str=None, global_step: int=0, remove_previous_ckpt=False):
        local, ckpt_path = self.checkpath(local_path, hdfs_path)
        
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        if local:
            local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()


        # Save Model
        if 'model' in self.checkpoint_contents:
            state_dict = self.weight_saver(self.model,
                                        self.hf_config,
                                        dtype=self.param_dtype,
                                        is_value_model=self.is_value_model,
                                        tie_word_embeddings=self.share_embeddings_and_output_weights)

            # wait for everyone to dump to local
            torch.distributed.barrier()

            if self.rank == 0:
                print(f'Saving actor checkpoint to {ckpt_path}')
                model_ckpt_path = get_model_checkpoint_path(ckpt_path)
                # torch.save(state_dict, os.path.join(model_path, 'model_state_dict.pt'))
                save_strategy = get_default_save_sharded_strategy()
                ckpt_name = self.get_checkpoint_name(model_ckpt_path)
                async_save_request = dist_checkpointing.save(state_dict, ckpt_name, save_strategy,
                                                         async_sharded_save=self.async_save)
                def after_saving():
                    print(f'Finish model state_dict')
                async_save_request.add_finalize_fn(after_saving)
                self.tokenizer.save_pretrained(model_ckpt_path)
                print(f'Saved actor checkpoint to {model_ckpt_path}')
                if hdfs_path is not None:
                    print(f'Uploading actor checkpoint to {hdfs_path}')
                    from verl.utils import hdfs_io
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)

        # Save Optimizer
        if 'optimizer' in self.checkpoint_contents:
            torch.distributed.barrier()
            
            optimizer_path = get_optimizer_checkpoint_path(ckpt_path)
            self.optimizer.save_parameter_state(optimizer_path)
            if self.rank == 0:
                print(f"saving critic optimizer state to {optimizer_path}")
        
        # Save RNG States
        if 'extra' in self.checkpoint_contents:
            torch.distributed.barrier()
            
            rng_state_path = get_rng_states_checkpoint_path(ckpt_path)
            rng_state = self.get_rng_state()
            torch.save(rng_state, rng_state_path)
            if self.rank == 0:
                print(f"saving critic rng states to {rng_state_path}")
    
        self.previous_saved_path = ckpt_path
