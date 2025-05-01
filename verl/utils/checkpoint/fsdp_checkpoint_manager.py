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

import os
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.fs import copy_to_local, is_non_local

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
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

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

        assert self.save_model, f"FSDPCheckpointManager must include ['model'], got {self.checkpoint_contents}"
        if self.save_optimizer:
            assert optimizer is not None, "optimizer must be provided when checkpoint_contents includes ['optimizer']"

        if self.optimizer is not None and not self.save_optimizer:
            print("Warning: optimizer is managed by FSDPCheckpointManager, but 'optimizer' not in checkpoint_contents. optimizer state will not be saved or loaded.")

        if self.lr_scheduler is not None and not self.save_extra:
            print("Warning: lr_scheduler is managed by FSDPCheckpointManager, but 'extra' not in checkpoint_contents. lr_scheduler state will not be saved or loaded.")

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is None:
            return

        # every rank download its own checkpoint
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True) if self.save_model else None
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True) if self.save_optimizer else None
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_config=state_dict_cfg, optim_state_dict_config=optim_cfg):
            if self.save_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                print(f"[rank-{self.rank}]: Loading model from {remote_model_path}")

            if self.save_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                print(f"[rank-{self.rank}]: Loading optimizer from {remote_optim_path}")

        if self.save_extra:
            remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                print(f"[rank-{self.rank}]: Loading rng from {remote_extra_state_path}")

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                print(f"[rank-{self.rank}]: Loading lr_scheduler from {remote_extra_state_path}")

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path, only rank 0 should do this
        if self.rank == 0 and max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True) if self.save_model else None
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True) if self.save_optimizer else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_config=state_dict_cfg, optim_state_dict_config=optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.save_model:
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")

                if self.save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")

                if self.save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")

        if "hf_model" in self.checkpoint_contents:
            # wait for everyone to dump to local
            torch.distributed.barrier()

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)
                self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
                self.processing_class.save_pretrained(hf_local_path)

        torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)
