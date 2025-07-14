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

import logging
import os

import torch
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo import core_algos
from verl.utils.device import (
    get_device_id,
    is_cuda_available,
)
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.engine import get_training_engine


from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.workers.roles.processor import get_processor_cls

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config):
        Worker.__init__(self)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        self.config = config
        engine_cls = get_training_engine(self.config.strategy)
        self.engine = engine_cls(self.config)
        self.engine.set_loss_fn(self.loss_fn)


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.init_model()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        response_mask = data.batch["response_mask"]
        processor = get_processor_cls("critic", self.config)(self.config)

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output = self.engine.infer_batch(
                data, processor=processor
            )     
        
        values = output * response_mask  # Only action tokens have values
        output = DataProto.from_dict(tensors={"values": values})

        output = self.engine.unshard_data(data=output)
        output = output.to("cpu")
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            data = self.engine.shard_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                select_keys = [
                    "input_ids",
                    "responses",
                    "response_mask",
                    "attention_mask",
                    "position_ids",
                    "values",
                    "returns",
                ]
                batch = data.select(batch_keys=select_keys).batch
                has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

                # Split to make minibatch iterator for updating the actor
                # See PPO paper for details. https://arxiv.org/abs/1707.06347
                if has_multi_modal_inputs:
                    num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
                    non_tensor_select_keys = ["multi_modal_inputs"]
                    dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
                else:
                    dataloader = batch.split(self.config.ppo_mini_batch_size)

                metrics = {}
                processor = get_processor_cls("critic", self.config)(self.config)
                for epoch in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(dataloader):
                        self.engine.optimizer_zero_grad()
                        vpreds_list, losses, metrics = self.engine.train_batch(
                            mini_batch, metrics=metrics, processor=processor
                        )
                        grad_norm = self.engine.optimizer_step()
                        mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                        append_to_dict(metrics, mini_batch_metrics)
                self.engine.optimizer_zero_grad()
            delta_time = timer.last

            # TODO: should not access engine's flops_counter
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.engine.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            metrics["critic/lr"] = self.engine.lr_scheduler_step()[0]
            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.engine.unshard_data(data=output)

        output = output.to("cpu")
        return output


    def loss_fn(self, batch, vpreds, metrics):
        values = batch["values"]
        returns = batch["returns"]
        response_mask = batch["response_mask"]
        micro_batch_metrics = {}
        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=self.config.cliprange_value,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        if self.config.use_dynamic_bsz:
            # relative to the dynamic bsz
            loss = vf_loss * (len(batch) / self.config.ppo_mini_batch_size)
        else:
            gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            loss = vf_loss / gradient_accumulation

        micro_batch_metrics = {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }

        append_to_dict(metrics, micro_batch_metrics)
        return loss, metrics


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)
