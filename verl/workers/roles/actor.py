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

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Any

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.engine import EngineRegistry
import verl.workers.engine.config as engine_cfg
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def debug_print(msg):
    print(f"[ActorWorker] {msg}")


class ActorWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config, role, **kwargs):
        debug_print(f"__init__ role: {role}")
        Worker.__init__(self)
        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        self.profile_option = kwargs.get("profile_option", None)
        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, option=self.profile_option)
        )

        import torch.distributed
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # if strategy is fsdp, offload_config should be None

        self.role = role
        assert self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]

        self.config = config
        assert not self.config.model.use_fused_kernels and not self.config.actor.entropy_checkpointing, \
            "fused kernels and entropy checkpointing are not supported in the new worker implementation yet"

        engine_config = self.create_engine_config(self.config)
        self.engine = EngineRegistry.new(self.config.actor.strategy, engine_config)
    

    def create_engine_config(self, actor_config):
        print(actor_config)
        model_config = engine_cfg.get_model_config(actor_config.model)
        print(f"override_config")
        print(model_config.override_config)
        optim_config = engine_cfg.get_optim_config(actor_config.actor.optim)
        system_config = engine_cfg.get_system_config(actor_config.actor.fsdp_config)
        ckpt_config = engine_cfg.get_checkpoint_config(actor_config.actor.checkpoint)

        ret = engine_cfg.get_engine_config(actor_config.actor,
                                            model_config,
                                            optim_config,
                                            system_config,
                                            ckpt_config,
                                            module_type="causal_lm",
                                            rollout_n=actor_config.rollout.n,
                                            infer_micro_batch_size_per_gpu=actor_config.rollout.log_prob_micro_batch_size_per_gpu,
                                            infer_max_token_len_per_gpu=actor_config.rollout.log_prob_max_token_len_per_gpu)
        return ret


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        debug_print(f"init_model")
        self.engine.init_model()

    # TODO: temporary
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        self.engine.send_params()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        return self.engine.get_params_meta_info()

    def _post_fn_log_prob(self, micro_batch, preds):
        # assert not use_fused_kernels and not entropy_checkpointing
        response_length = micro_batch["responses"].size(-1)
        temperature = self.config.rollout.temperature

        logits = preds                                          # (bsz, seqlen, vocab_size)
        debug_print(f"logits: {logits.shape}")
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]                     # (bsz, response_length, vocab_size)
        debug_print(f"logits after transform: {logits.shape}")
        debug_print(f"responses: {micro_batch['responses'].shape}")
        log_probs = logprobs_from_logits(logits, micro_batch["responses"])   # (bsz, response_length)
        debug_print(f"log_probs: {log_probs.shape}")
        entropy = verl_F.entropy_from_logits(logits)                         # (bsz, response_length)

        outputs = {
            "log_probs": log_probs.clone().detach(),
            "entropy": entropy.clone().detach(),
        }

        return (log_probs, entropy), outputs

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # TODO: adapter_ctx support for lora model

        # Support all hardwares
        data = data.to(get_device_id())

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output = self.engine.infer_batch(data, post_fn=self._post_fn_log_prob)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output["log_probs"], "entropys": output["entropy"]},
                meta_info={"temperature": self.config.rollout.temperature},
            )

            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")

        # TODO: check this
        # # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # # unshard the root FSDP module
        # if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
        #     self.actor.actor_module._handle.reshard(True)
        return output

    
    def loss_fn(
        self, batch: DataProto, vpreds: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        response_mask = batch["response_mask"]
        old_log_prob = batch["old_log_probs"]
        advantages = batch["advantages"]
        micro_batch_metrics = {}


        entropy_coeff = self.config.actor.entropy_coeff
        loss_agg_mode = self.config.actor.loss_agg_mode

        # all return: (bsz, response_length)
        calculate_entropy = False
        if entropy_coeff != 0:
            calculate_entropy = True

        (log_prob, entropy), _ = self._post_fn_log_prob(batch, vpreds)

        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.config.actor,
        )

        if entropy_coeff != 0:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            # compute policy loss
            policy_loss = pg_loss - entropy_loss * entropy_coeff
        else:
            policy_loss = pg_loss

        if self.config.actor.use_kl_loss:
            ref_log_prob = batch["ref_log_prob"]
            # compute kl loss
            kld = kl_penalty(
                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.actor.kl_loss_type
            )
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

            policy_loss = policy_loss + kl_loss * self.config.actor.kl_loss_coef
            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
            micro_batch_metrics["actor/kl_coef"] = self.config.actor.kl_loss_coef

        micro_batch_metrics.update(
            {
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                "actor/ppo_kl": ppo_kl.detach().item(),
                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            }
        )
        return policy_loss, micro_batch_metrics


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            data = self.engine.shard_data(data=data)

            with Timer(name="update_policy", logger=None) as timer:
                select_keys = [
                    "responses",
                    "response_mask",
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "old_log_probs",
                    "advantages",
                ]

                if self.config.actor.use_kl_loss:
                    select_keys.append("ref_log_prob")

                has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
                non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
                data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

                # calculate mini batch size
                mini_bsz = self.config.actor.ppo_mini_batch_size * self.config.rollout.n
                mini_bsz = mini_bsz // self.engine.get_data_parallel_size()
                # Split to make minibatch iterator for updating the actor
                # See PPO paper for details. https://arxiv.org/abs/1707.06347
                mini_batches = data.split(mini_bsz)

                metrics = {}
                for epoch in range(self.config.actor.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(mini_batches):
                        self.engine.optimizer_zero_grad()
                        mini_batch_metrics = self.engine.train_batch(mini_batch, self.loss_fn)
                        grad_norm = self.engine.optimizer_step()
                        mini_batch_metrics["critic/grad_norm"] = grad_norm.detach().item()
                        append_to_dict(metrics, mini_batch_metrics)
                self.engine.optimizer_zero_grad()
            delta_time = timer.last

            # TODO: should not access engine's flops_counter
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.engine.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.engine.lr_scheduler_step()[0]
            metrics["actor/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.engine.unshard_data(data=output)

        output = output.to("cpu")
        raise ValueError
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        raise NotImplementedError

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        raise NotImplementedError
