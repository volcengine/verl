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

import contextlib
import inspect
import logging
import os

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from torch.distributed.fsdp._unshard_param_utils import _get_module_fsdp_state, _unshard_params_for_summon
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device, set_expandable_segments
from verl.utils.fsdp_utils import (
    fsdp_version,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler
from verl.workers.config import HFModelConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class RobActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    original_generate_sequences = inspect.unwrap(ActorRolloutRefWorker.generate_sequences)
    fsdp_unshard_exit_stack = contextlib.ExitStack()

    def _build_rollout(self, trust_remote_code=False):
        from recipe.vla.naive_rollout_rob import NaiveRolloutRob

        self.base_sync_done = False
        world_size = torch.distributed.get_world_size()
        dp = world_size
        infer_tp = self.config.rollout.tensor_model_parallel_size
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        self.rollout = NaiveRolloutRob(module=self.actor_module_fsdp, model_config=self.config.model)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        fsdp_unshard_exit_stack = contextlib.ExitStack()
        optional_state = _get_module_fsdp_state(self.actor_module_fsdp)
        if optional_state is None:
            self.fsdp_unshard_exit_stack = fsdp_unshard_exit_stack
        states_and_modules = ([optional_state], [self.actor_module_fsdp])

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)
        for state, fsdp_module in zip(*states_and_modules, strict=False):
            fsdp_unshard_exit_stack.enter_context(
                _unshard_params_for_summon(
                    module=fsdp_module,
                    state=state,
                    writeback=False,
                    rank0_only=False,
                    offload_to_cpu=False,
                    with_grads=False,
                )
            )

        self.fsdp_unshard_exit_stack = fsdp_unshard_exit_stack

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
        if self.fsdp_unshard_exit_stack is not None:
            self.fsdp_unshard_exit_stack.close()
            self.fsdp_unshard_exit_stack = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from recipe.vla.dp_rob import RobDataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        from verl.models.openvla_oft.configuration_prismatic import OpenVLAConfig
        from verl.models.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from verl.models.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                )
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                # get the original unwrapped module
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        return self.original_generate_sequences(prompts)

    # @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # def compute_ref_log_prob(self, data: DataProto):
    #     return data
    #     # pass

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # def save_checkpoint(self, local_path, hdfs_path=None):
    #     pass
