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

import datetime
import logging
import os
import time
from typing import Any

import psutil
import torch
import torch.distributed
from codetiming import Timer
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig, OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import reduce_timing
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        MegatronWorker.__init__(self)
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self.config.actor.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.actor.megatron.seed)

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor and self._is_rollout:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            else:
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, (
                    "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and "
                    "`log_prob_micro_batch_size` should not be None at the same time."
                )
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    def _build_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron.use_mbridge,
        )
        self.generation_config = get_generation_config(self.local_path)

        def make_model(wrap_with_ddp=False):
            if self.bridge is not None:
                from verl.models.mcore.mbridge import freeze_moe_router

                post_model_creation_callbacks = []
                if override_model_config.get("moe_config", {}).get("freeze_moe_router", False):
                    post_model_creation_callbacks.append(freeze_moe_router)
                return self.bridge.get_model(
                    post_model_creation_callbacks=post_model_creation_callbacks, wrap_with_ddp=wrap_with_ddp
                )
            else:

                def megatron_actor_model_provider(pre_process, post_process):
                    from verl.models.mcore import init_mcore_model

                    parallel_model = init_mcore_model(
                        self.tf_config,
                        self.hf_config,
                        pre_process,
                        post_process,
                        share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                        value=False,
                        freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False),
                    )
                    parallel_model.to(get_device_name())
                    return parallel_model

                override_ddp_config = OmegaConf.to_container(
                    self.config.actor.megatron.get("override_ddp_config", OmegaConf.create()), resolve=True
                )
                return get_model(
                    megatron_actor_model_provider,
                    wrap_with_ddp=wrap_with_ddp,
                    use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                    override_ddp_config=override_ddp_config,
                )

        if self._is_actor or self._is_rollout:
            actor_module = make_model(wrap_with_ddp=True)
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        self.bridge.load_weights(actor_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                        )

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self._is_ref:
            print(f"self.config.ref.load_weight: {self.config.ref.load_weight}")
            ref_module = make_model(wrap_with_ddp=False)
            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        self.bridge.load_weights(ref_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                        )
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(optim_config)
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        if self.config.rollout.name == "vllm":
            from torch.distributed.device_mesh import init_device_mesh

            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
            )
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

            # perform weight resharding between actor and rollout
            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            sharding_manager = MegatronVLLMShardingManager(
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                rollout_config=self.config.rollout,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                bridge=self.bridge,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif self.config.rollout.name == "sglang":
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's
            # model_runner would check CUDA device capability.
            # However, due to verl's setting, the main process of ray can not find any CUDA device, which would
            # potentially lead to: "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it
            # here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.megatron_sglang import MegatronSGLangShardingManager

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            )
            rollout_device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp")
            )

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)

            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            sharding_manager = MegatronSGLangShardingManager(
                actor_module=self.actor.actor_module,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                weight_converter=weight_converter,
                bridge=self.bridge,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        else:
            raise NotImplementedError("Only vllmRollout is supported with Megatron now")
        print(f"rollout and sharding manager init done sharding_manager: {sharding_manager}")
        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                self.config.ref.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
            )
        else:
            override_transformer_config = {}
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                use_fused_kernels = self.config.model.get("use_fused_kernels", False)
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = MegatronPPOActor(
                config=self.config.actor,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )
            # used for sleep/wake_up
            self.rollout.sharding_manager = self.sharding_manager
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
            )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        data.batch = data.batch.to(get_device_name())

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts.batch = prompts.batch.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        timing_generate = {}
        with self.sharding_manager:
            log_gpu_memory_usage("After entering sharding manager", logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)
            output = self.sharding_manager.postprocess_data(output)
            log_gpu_memory_usage("After rollout generation", logger=logger)

        timing_generate.update(self.sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    @DistProfiler.annotate(color="olive")
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data = data.to(get_device_id())
        output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    @DistProfiler.annotate(color="blue")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=logger)
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data = data.to(get_device_id())
        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        output = DataProto.from_dict(
            tensors={"old_log_probs": output, "entropys": entropys},
            meta_info={"temperature": self.config.rollout.temperature},
        )
        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: str | bytes, *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(
                f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: "
                f"{method if isinstance(method, str) else 'Callable'}"
            )
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.sleep()
        # return something to block the caller
        return True


class CriticWorker(MegatronWorker, DistProfilerExtension):
    def __init__(self, config):
        MegatronWorker.__init__(self)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config
    ):
        from megatron.core.models.gpt.gpt_model import ModelType

        from verl.utils.megatron.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.tokenizer_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron.use_mbridge,
        )

        if self.bridge is not None:
            from verl.models.mcore.mbridge import freeze_moe_router, make_value_model

            post_model_creation_callbacks = [make_value_model]
            if override_model_config.get("moe_config", {}).get("freeze_moe_router", False):
                post_model_creation_callbacks.append(freeze_moe_router)
            critic_module = self.bridge.get_model(
                post_model_creation_callbacks=post_model_creation_callbacks, wrap_with_ddp=True
            )
        else:

            def megatron_critic_model_provider(pre_process, post_process):
                from verl.models.mcore import init_mcore_model

                parallel_model = init_mcore_model(
                    self.tf_config,
                    self.hf_config,
                    pre_process,
                    post_process,
                    share_embeddings_and_output_weights=False,
                    value=True,
                    freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False),
                )
                parallel_model.to(get_device_name())
                return parallel_model

            override_ddp_config = OmegaConf.to_container(
                self.config.megatron.get("override_ddp_config", OmegaConf.create()), resolve=True
            )
            # Step 3: initialize the megatron model
            critic_module = get_model(
                model_provider_func=megatron_critic_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
                override_ddp_config=override_ddp_config,
            )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module, self.config.megatron.dist_checkpointing_path, is_value_model=True
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(critic_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()
        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(
            self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
        )
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
        )

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
        self.critic_optimizer_scheduler.step(1)

        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)


class RewardModelWorker(MegatronWorker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        MegatronWorker.__init__(self)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, tokenizer, override_model_config, override_transformer_config):
        from megatron.core.models.gpt.gpt_model import ModelType

        from verl.utils.megatron_utils import get_model

        self._init_hf_config_and_tf_config(
            model_path,
            tokenizer,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron.use_mbridge,
        )
        if self.bridge is not None:
            from verl.models.mcore.mbridge import freeze_moe_router, make_value_model

            post_model_creation_callbacks = [make_value_model]
            if override_model_config.get("moe_config", {}).get("freeze_moe_router", False):
                post_model_creation_callbacks.append(freeze_moe_router)
            reward_model = self.bridge.get_model(
                post_model_creation_callbacks=post_model_creation_callbacks, wrap_with_ddp=False
            )
        else:

            def megatron_rm_model_provider(pre_process, post_process):
                from verl.models.mcore import init_mcore_model

                parallel_model = init_mcore_model(
                    self.tf_config,
                    self.hf_config,
                    pre_process,
                    post_process,
                    share_embeddings_and_output_weights=False,
                    value=True,
                )
                parallel_model.to(get_device_name())
                return parallel_model

            # Step 3: initialize the megatron model
            reward_model = get_model(
                model_provider_func=megatron_rm_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            )
            # note that here reward_model will be a list to be compatible with the construction of interleaved pp (vpp)
            # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
            # reward_model = nn.ModuleList(reward_model)

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(reward_model, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(reward_model, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, reward_model, params_dtype=self.dtype, is_value_model=True
                    )

        # TODO: add more optimizer args into config
        get_torch_device().empty_cache()
        return reward_model, self.hf_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(
            self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
        )

        use_shm = self.config.model.get("use_shm", False)
        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer, use_shm=use_shm)
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get("rm_tokenizer", None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path, use_shm=use_shm)
            rm_tokenizer = hf_tokenizer(
                rm_tokenizer_local_path, trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            tokenizer=rm_tokenizer,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        output = self.rm.compute_reward(data)
        output = output.to("cpu")
        return output
