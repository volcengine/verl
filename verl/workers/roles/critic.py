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
from verl.trainer.ppo import core_algos
from verl.workers.engine.fsdp import FSDPEngine

from verl.utils.torch_functional import masked_mean
from verl.utils.py_functional import append_to_dict
import torch
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.utils.device import (
    get_device_id,
    is_cuda_available,
    is_npu_available,
)

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.utils.config import omega_conf_to_dataclass

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
        self.engine = FSDPEngine(self.config)

        def loss_fn(batch, vpreds, ctx):
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
                loss = vf_loss / ctx["gradient_accumulation"]

            micro_batch_metrics = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
            }

            append_to_dict(ctx["metrics"], micro_batch_metrics)

            return loss, ctx

        self.engine.set_loss_fn(loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.init_model()

    def get_microbatch_process_fn(self):
        def preprocess_fn_without_rmpad(batch, ctx):
            ctx["response_length"] = batch["responses"].size(-1)

            inputs = {}
            if "multi_modal_inputs" in batch.keys():
                for key in batch["multi_modal_inputs"][0].keys():
                    inputs[key] = torch.cat([inputs[key] for inputs in batch["multi_modal_inputs"]], dim=0)

            inputs["input_ids"] = batch["input_ids"]
            inputs["attention_mask"] = batch["attention_mask"]
            position_ids = batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            inputs["position_ids"] = position_ids
            return inputs, ctx

        def postprocess_fn_without_rmpad(outputs, ctx):
            response_length = ctx["response_length"]
            use_value_head_model = ctx["use_value_head_model"]
            if use_value_head_model:
                # For trl.AutoModelForCausalLMWithValueHead
                values = outputs[2]
            else:
                values = outputs.logits
            values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values, ctx

        def preprocess_fn_with_rmpad(batch, ctx):
            ctx["response_length"] = batch["responses"].size(-1)

            inputs = {}
            if "multi_modal_inputs" in batch.keys():
                for key in batch["multi_modal_inputs"][0].keys():
                    inputs[key] = torch.cat([inputs[key] for inputs in batch["multi_modal_inputs"]], dim=0)

            input_ids = batch["input_ids"]
            bs, seqlen = input_ids.shape
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # pad and slice the inputs if sp > 1
            if ctx["ulysses_sequence_parallel_size"] > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                )
                ctx["pad_size"] = pad_size

            inputs["input_ids"] = input_ids_rmpad
            inputs["attention_mask"] = None
            inputs["position_ids"] = position_ids_rmpad

            ctx["indices"] = indices
            ctx["seqlen"] = seqlen
            ctx["bs"] = bs
            return inputs, ctx

        def postprocess_fn_with_rmpad(outputs, ctx):
            response_length = ctx["response_length"]
            use_value_head_model = ctx["use_value_head_model"]
            if use_value_head_model:
                # For trl.AutoModelForCausalLMWithValueHead
                values_rmpad = outputs[2].squeeze(0).unsqueeze(-1)
            else:
                values_rmpad = outputs.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

            # gather output if sp > 1
            if ctx["ulysses_sequence_parallel_size"] > 1:
                values_rmpad = gather_outpus_and_unpad(
                    values_rmpad, gather_dim=0, unpad_dim=0, padding_size=ctx["pad_size"]
                )

            # pad it back
            values = pad_input(values_rmpad, indices=ctx["indices"], batch=ctx["bs"], seqlen=ctx["seqlen"]).squeeze(-1)
            values = values[:, -response_length - 1 : -1]
            return values, ctx

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        if self.use_remove_padding:
            return preprocess_fn_with_rmpad, postprocess_fn_with_rmpad
        else:
            return preprocess_fn_without_rmpad, postprocess_fn_without_rmpad

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        preprocess_fn, postprocess_fn = self.get_microbatch_process_fn()
        ctx = self.engine.get_default_ctx()

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output, _ = self.engine.infer_batch(
                data, ctx=ctx, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn
            )
            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        preprocess_fn, postprocess_fn = self.get_microbatch_process_fn()
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
                for epoch in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(dataloader):
                        ctx = self.engine.get_default_ctx()
                        ctx["metrics"] = metrics
                        self.engine.optimizer_zero_grad()
                        vpreds_list, losses, ctx = self.engine.train_batch(
                            mini_batch, ctx=ctx, preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn
                        )
                        metrics = ctx["metrics"]
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)
