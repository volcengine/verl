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
import logging
import os
import warnings
from functools import partial
from typing import Any, Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.flops_counter import FlopsCounter
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import ActorConfig, CriticConfig, HFModelConfig, RolloutConfig, TrainingWorkerConfig
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss, value_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrainingWorker(Worker):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        # we use the one defined in model
        self.engine_config.use_remove_padding = self.model_config.use_remove_padding

        # TODO: add DistProfilerExtension
        # self.profiler_config = self.config.profiler_config
        # tool_config = self.profiler_config.tool_config
        # DistProfilerExtension.__init__(
        #     self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        # )

        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        self.flops_counter = FlopsCounter(self.model_config.hf_config)

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """
        # TODO: whether to log memory
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        torch.distributed.all_reduce(
            loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
        )
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group
        final_metrics = allgather_dict_into_dict(data=metrics, group=self.engine.get_data_parallel_group())
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr
        # compute mfu
        if global_token_num is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.use_remove_padding,
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output, global_token_num=global_token_num, delta_time=delta_time, forward_only=False
            )
        else:
            final_output = None
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)

        default_keys = dict(
            use_remove_padding=self.model_config.use_remove_padding,
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output, global_token_num=global_token_num, delta_time=delta_time, forward_only=True
            )
        else:
            final_output = None
        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: ActorConfig):
        self.config = config
        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

        self.loss_fn = partial(ppo_loss, config=self.config)

    def _build_engine(self):
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine
        self.optimizer_config = self.config.optim
        self.checkpoint_config = self.config.checkpoint

        from verl.workers.engine import BaseEngine, EngineRegistry

        self.engine: BaseEngine = EngineRegistry.new(
            model_type="language_model",
            backend=self.config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="actor",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        # aggregate with bon sampling
        self.ppo_mini_batch_size = self.config.ppo_mini_batch_size * self.config.rollout_n
        assert self.ppo_mini_batch_size % self.engine.get_data_parallel_size() == 0, (
            f"{self.ppo_mini_batch_size=} is not divisible by {self.engine.get_data_parallel_size()=}"
        )
        self.ppo_mini_batch_size_per_dp = self.ppo_mini_batch_size // self.engine.get_data_parallel_size()

        # setup flops counter
        self.flops_counter = FlopsCounter(self.model_config.hf_config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_engine()
        self.engine.initialize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data.meta_info["use_fused_kernels"] = self.config.use_fused_kernels
        if "calculate_entropy" not in data.meta_info:
            data.meta_info["calculate_entropy"] = True
        calculate_entropy = data.meta_info["calculate_entropy"]

        if self.config.use_dynamic_bsz:
            data.meta_info["max_token_len_per_gpu"] = self.config.ppo_infer_max_token_len_per_gpu
        else:
            data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_infer_micro_batch_size_per_gpu

        with self.engine.eval_mode():
            # TODO: make worker API to accept TensorDict as well
            data = data.to_tensordict()
            data = left_right_2_no_padding(data)
            output = self.engine.infer_batch(data)

        if self.engine.is_mp_src_rank_with_outputs():
            output = output["model_output"]
            log_probs = output["log_probs"]
            log_probs = no_padding_2_padding(log_probs, data)  # (bsz, response_length)

            tensors = {"old_log_probs": log_probs.float()}
            if calculate_entropy:
                entropy = no_padding_2_padding(output["entropy"], data)  # (bsz, response_length)
                tensors["entropys"] = entropy.float()

            # in megatron, only last pp contains valid data and returned to the single controller
            output = DataProto.from_dict(tensors=tensors)
            output = output.to("cpu")

        return output if self.engine.is_mp_src_rank_with_outputs() else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data.meta_info["use_fused_kernels"] = self.config.use_fused_kernels
        data.meta_info["calculate_entropy"] = self.config.entropy_coeff != 0.0
        if self.config.use_dynamic_bsz:
            data.meta_info["max_token_len_per_gpu"] = self.config.ppo_max_token_len_per_gpu
        else:
            data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_micro_batch_size_per_gpu

        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            dataloader = data.make_iterator(
                mini_batch_size=self.ppo_mini_batch_size_per_dp,
                epochs=self.config.ppo_epochs,
                seed=self.config.data_loader_seed + self.engine.get_data_parallel_rank(),
                dataloader_kwargs={"shuffle": self.config.shuffle},
            )
            with Timer(name="update_policy", logger=None) as timer:
                for batch_idx, mini_batch in enumerate(dataloader):
                    mini_batch.meta_info["global_batch_size"] = self.ppo_mini_batch_size
                    # TODO: make worker API to accept TensorDict as well
                    mini_batch = mini_batch.to_tensordict()
                    mini_batch = left_right_2_no_padding(mini_batch)
                    output = self.engine.train_batch(mini_batch, self.loss_fn)
                    mini_batch_metrics = output.get("metrics", {})
                    append_to_dict(metrics, mini_batch_metrics, prefix="actor/")

            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.engine.lr_scheduler_step()
            metrics["actor/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class CriticWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: CriticConfig):
        self.config = config
        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

        self.loss_fn = partial(value_loss, config=self.config)

    def _build_engine(self):
        from copy import copy, deepcopy

        self.model_config = copy(self.config.model_config)
        self.model_config.hf_config = deepcopy(self.config.model_config.hf_config)
        self.engine_config = self.config.engine
        self.optimizer_config = self.config.optim
        self.checkpoint_config = self.config.checkpoint

        from verl.workers.engine import BaseEngine, EngineRegistry

        # replace AutoModelForSequenceClassification to AutoModelForTokenClassification
        hf_config = self.model_config.hf_config

        arch = hf_config.architectures[0]
        # This logic assumes the critic is a token classification model.
        # If the provided model is a CausalLM, we adapt it.
        if "ForCausalLM" in arch:
            model_name = arch.split("ForCausalLM")[0]
            new_arch = f"{model_name}ForTokenClassification"
            warnings.warn(f"Implicitly changing critic architecture from '{arch}' to '{new_arch}'", stacklevel=2)
            hf_config.architectures[0] = new_arch
        elif "ForTokenClassification" not in arch and "ForSequenceClassification" not in arch:
            raise ValueError(
                f"Unsupported critic architecture: {arch}. "
                f"Critic worker expects an architecture suitable for value function estimation, "
                f"such as '...ForTokenClassification' or '...ForSequenceClassification'."
            )

        # make sure output dropout is 0
        hf_config.classifier_dropout = 0

        self.engine: BaseEngine = EngineRegistry.new(
            model_type="value_model",
            backend=self.config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="critic",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        # aggregate with bon sampling
        self.ppo_mini_batch_size = self.config.ppo_mini_batch_size * self.config.rollout_n
        assert self.ppo_mini_batch_size % self.engine.get_data_parallel_size() == 0, (
            f"{self.ppo_mini_batch_size=} is not divisible by {self.engine.get_data_parallel_size()=}"
        )
        self.ppo_mini_batch_size_per_dp = self.ppo_mini_batch_size // self.engine.get_data_parallel_size()

        # setup flops counter
        self.flops_counter = FlopsCounter(self.model_config.hf_config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_engine()
        self.engine.initialize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="blue", role="critic_compute_values")
    def compute_values(self, data: DataProto):
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        if self.config.use_dynamic_bsz:
            data.meta_info["max_token_len_per_gpu"] = self.config.ppo_infer_max_token_len_per_gpu
        else:
            data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_infer_micro_batch_size_per_gpu

        with self.engine.eval_mode():
            # TODO: make worker API to accept TensorDict as well
            data = data.to_tensordict()
            data = left_right_2_no_padding(data)
            output = self.engine.infer_batch(data)

        if self.engine.is_mp_src_rank_with_outputs():
            # in megatron, only last pp contains valid data and returned to the single controller
            output = output["model_output"]
            values = output["values"]
            values = no_padding_2_padding(values, data)  # (bsz, response_length)

            output = DataProto.from_dict(
                tensors={"values": values.float()},
            )
            output = output.to("cpu")

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="red", role="critic_update")
    def update_critic(self, data: DataProto):
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        if self.config.use_dynamic_bsz:
            data.meta_info["max_token_len_per_gpu"] = self.config.ppo_max_token_len_per_gpu
        else:
            data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_micro_batch_size_per_gpu

        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            dataloader = data.make_iterator(
                mini_batch_size=self.ppo_mini_batch_size_per_dp,
                epochs=self.config.ppo_epochs,
                seed=self.config.data_loader_seed + self.engine.get_data_parallel_rank(),
                dataloader_kwargs={"shuffle": self.config.shuffle},
            )
            with Timer(name="update_policy", logger=None) as timer:
                for batch_idx, mini_batch in enumerate(dataloader):
                    mini_batch.meta_info["global_batch_size"] = self.ppo_mini_batch_size
                    # TODO: make worker API to accept TensorDict as well
                    mini_batch = mini_batch.to_tensordict()
                    mini_batch = left_right_2_no_padding(mini_batch)
                    output = self.engine.train_batch(mini_batch, self.loss_fn)
                    mini_batch_metrics = output.get("metrics", {})
                    append_to_dict(metrics, mini_batch_metrics, prefix="critic/")

            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.engine.lr_scheduler_step()
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.role = role
        self.actor: ActorWorker = None
        self.ref: ActorWorker = None
        self.rollout: BaseRollout = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)
            ref_config.model_config = model_config

            self.ref = ActorWorker(ref_config)
            self.ref.init_model()
            self.ref.engine.to("cpu")
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config

            self.actor = ActorWorker(actor_config)
            self.actor.init_model()
            self.actor.engine.to("cpu")
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        # - vllm: vLLMAsyncRollout
        # - sglang: ServerAdapter
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 init trainer and rollout random states
            self.torch_random_states = get_torch_device().get_rng_state()
            gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

            # 3.3 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        data.meta_info["calculate_entropy"] = False
        output = self.ref.compute_log_prob(data)
        if output is not None:
            output.batch["ref_log_prob"] = output.batch.pop("old_log_probs")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        return self.actor.compute_log_prob(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        return self.actor.update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        """Context switch from rollout mode to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        """Context switch trainer mode to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        # 1. get per tensor generator from engine, this will load model to gpu
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param()

        # 2. resume weights and update weights
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)
        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
