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
from functools import partial

import psutil
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.flops_counter import FlopsCounter
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig
from verl.workers.roles.utils.losses import ppo_loss
from verl.workers.roles.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class ActorWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
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
        data.meta_info["calculate_entropy"] = True
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

            entropy = output["entropy"]
            if entropy is not None:
                entropy = no_padding_2_padding(entropy, data)  # (bsz, response_length)

            # in megatron, only last pp contains valid data and returned to the single controller
            output = DataProto.from_dict(
                tensors={"old_log_probs": log_probs.float(), "entropy": entropy.float()},
            )
            output = output.to("cpu")
        return output

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
                    mini_batch.meta_info["global_batch_size"] = self.config.ppo_mini_batch_size
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
