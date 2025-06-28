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

from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device, is_cuda_available, is_npu_available
from verl.trainer.ppo import core_algos
from verl.workers.engine.fsdp import FSDPEngine

from verl.utils.torch_functional import masked_mean
from verl.utils.py_functional import append_to_dict


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        self.config = config
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.forward_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"

        self.engine = FSDPEngine(self.config)

        def loss_fn(batch, vpreds, ctx):
            responses = batch["responses"]
            attention_mask = batch["attention_mask"]
            values = batch["values"]
            returns = batch["returns"]
            response_length = responses.size(1)
            response_mask = attention_mask[:, -response_length:]
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
                loss = vf_loss / ctx["gradient_accumulation"]

            info = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
            }
            
            append_to_dict(ctx["metrics"], info)

            return loss, ctx
        self.engine.set_loss_fn(loss_fn)


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.init_model()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output, _ = self.engine.forward_backward_step(data,
                                                            forward_only=True)
            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")
        return output



    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        # perform forward computation
        with self.engine.train_mode():
            data = self.engine.shard_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
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
                for epoch in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(dataloader):   
                        ctx = {}
                        ctx["metrics"] = metrics                     
                        self.engine.optimizer_zero_grad()
                        vpreds_list, losses, ctx = self.engine.forward_backward_step(mini_batch,
                                                                                         ctx=ctx,
                                                                                         forward_only=False)
                        metrics = ctx["metrics"]
                        grad_norm = self.engine.optimizer_step() 
                        append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})
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


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)

