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
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_nccl_backend,
)
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.engine import EngineRegistry
import verl.workers.engine.config as engine_cfg

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
            torch.distributed.init_process_group(backend=get_nccl_backend())

        self.config = omega_conf_to_dataclass(config)
        assert self.config.ppo_micro_batch_size is None and \
            self.config.forward_micro_batch_size is None, \
            "new engine implementation does not support ppo_micro_batch_size and forward_micro_batch_size"
        # if strategy is fsdp, offload_config should be None
        
        engine_config = self.create_engine_config(self.config)
        self.engine = EngineRegistry.new(self.config.strategy, engine_config)


    def create_engine_config(self, config):
        # ModelConfig Setup
        override_config = {
            "num_labels": 1,
            "classifier_dropout": 0.0,
            "hidden_dropout": "0",
            "summary_dropout_prob": 0.0,
        }

        model_config = engine_cfg.ModelConfig(
            path=config.model.path,
            module_type="token_classification",
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            target_modules=config.model.target_modules,
            trust_remote_code=config.model.trust_remote_code,
            use_shm=config.model.use_shm,
            external_lib=config.model.external_lib,
            override_config=override_config,
            custom_chat_template=None,
            tokenizer_path=None,
            use_liger=False,
            use_fused_kernels=False,
            fused_kernel_options=None,
        )

        # OptimConfig Setup
        lr_scheduler_style = config.optim.get("warmup_style", "constant")
        lr_scheduler_args = None
        if lr_scheduler_style == "consine":
            lr_scheduler_args = {
                "min_lr_ratio": config.optim.get("min_lr_ratio", 0.0),
                "num_cycles": config.optim.get("num_cycles", 0.5),
            }

        optim_config = engine_cfg.OptimConfig(
            grad_clip=config.grad_clip,  
            betas=config.optim.get("betas", (0.9, 0.999)),    
            weight_decay=config.optim.get("weight_decay", 1e-2),  
            lr=config.optim.lr,  
            lr_warmup_steps=config.optim.get("lr_warmup_steps", -1),    
            lr_warmup_steps_ratio=config.optim.lr_warmup_steps_ratio,
            lr_scheduler_style=lr_scheduler_style,
            lr_scheduler_args=lr_scheduler_args,
        )

        # SystemConfig Setup
        fsdp_config = config.model.fsdp_config

        system_config = engine_cfg.SystemConfig(
            fsdp_size=fsdp_config.fsdp_size,
            model_dtype=fsdp_config.get("model_dtype", "fp32"),
            param_offload=fsdp_config.param_offload,
            optimizer_offload=fsdp_config.optimizer_offload,
            wrap_policy=fsdp_config.wrap_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=fsdp_config.offload_policy,
            forward_prefetch=fsdp_config.forward_prefetch,
            ulysses_sequence_parallel_size=config.ulysses_sequence_parallel_size,
            enable_gradient_checkpointing=config.model.enable_gradient_checkpointing,
            enable_activation_offload=config.model.enable_activation_offload,
            use_remove_padding=config.model.use_remove_padding,
            mixed_precision=fsdp_config.get("mixed_precision", None),
            use_orig_params=fsdp_config.get("use_orig_params", False),
        )

        # CheckpointConfig Setup
        ckpt_config = engine_cfg.CheckpointConfig(
            save_contents=config.checkpoint.save_contents,
            load_contents=config.checkpoint.load_contents,
            async_save=config.checkpoint.async_save,
        )

        train_mini_batch_size = config.ppo_mini_batch_size * config.rollout_n

        engine_config = engine_cfg.EngineConfig(
            strategy=config.strategy,
            model=model_config,
            optim=optim_config,
            system=system_config,
            checkpoint=ckpt_config,
            total_training_steps=config.optim.get("total_training_steps", -1),
            use_dynamic_bsz=config.use_dynamic_bsz,
            train_mini_batch_size=train_mini_batch_size,
            train_micro_batch_size_per_gpu=config.ppo_micro_batch_size_per_gpu,
            train_max_token_len_per_gpu=config.ppo_max_token_len_per_gpu,
            infer_micro_batch_size_per_gpu=config.forward_micro_batch_size_per_gpu,
            infer_max_token_len_per_gpu=config.forward_max_token_len_per_gpu,
        )

        return engine_config


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.initialize()


    def _post_fn_values(self, micro_batch, preds):
        response_length = micro_batch["responses"].size(-1)
        values = preds[:, -response_length - 1 : -1]
        values = values.squeeze(-1)
        return values, {"values": values.clone().detach()}


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output = self.engine.forward_step(data, post_fn=self._post_fn_values)
            response_mask = data.batch["response_mask"]
            values = output["values"] * response_mask  # Only action tokens have values
            output = DataProto.from_dict(tensors={"values": values})

            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")
        return output

    def loss_fn(
        self, batch: DataProto, vpreds: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        old_values = batch["values"]
        returns = batch["returns"]
        response_mask = batch["response_mask"]
        micro_batch_metrics = {}

        values, _ = self._post_fn_values(batch, vpreds)

        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=values,
            values=old_values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=self.config.cliprange_value,
            loss_agg_mode=self.config.loss_agg_mode,
        )

        micro_batch_metrics = {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(values, response_mask).detach().item(),
        }

        return vf_loss, micro_batch_metrics

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        metrics = {}
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
                has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
                non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
                data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

                # calculate mini batch size
                mini_bsz = self.config.ppo_mini_batch_size * self.config.rollout_n
                mini_bsz = mini_bsz // self.engine.get_data_parallel_size()
                # Split to make minibatch iterator for updating the actor
                # See PPO paper for details. https://arxiv.org/abs/1707.06347
                mini_batches = data.split(self.config.ppo_mini_batch_size)

                for epoch in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(mini_batches):
                        mini_batch_metrics = self.engine.train_step(mini_batch, self.loss_fn)
                        # renaming metrics for critic specific
                        mini_batch_metrics["critic/grad_norm"] = mini_batch_metrics.pop("grad_norm")
                        append_to_dict(metrics, mini_batch_metrics)
            delta_time = timer.last

            # TODO: should not access engine's flops_counter
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.engine.estimate_flops(global_num_tokens, delta_time)
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
