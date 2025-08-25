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
from typing import Callable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager

from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device
from verl.utils.model import load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)

from ..base import BaseEngine, EngineRegistry

from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register("megatron")
class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

    def _init_device_mesh(self):
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )

    def _build_tf_config(self):
        from verl.models.mcore import hf_to_mcore_config
        from verl.utils.torch_dtypes import PrecisionType

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        tf_config = hf_to_mcore_config(
            self.model_config.hf_config, self.dtype, **self.engine_config.override_transformer_config
        )

        use_mbridge = self.engine_config.use_mbridge
        if use_mbridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config)
            bridge.set_extra_args(**self.engine_config.override_transformer_config)
            tf_config = bridge.config
            self.bridge = bridge
        else:
            self.bridge = None

        print(f"TF config: {tf_config}")
        self.tf_config = tf_config

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        is_value_model = False

        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=is_value_model,  # actor is not value model
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
        )
        module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=self.engine_config.override_ddp_config,
        )
        print(f"actor_module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(module, self.engine_config.dist_checkpointing_path, is_value_model=is_value_model)
        else:
            if self.bridge is not None:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                # (vermouth1992) this is a workaround to be compatible with the old API
                tmp_config = OmegaConf.create(
                    {"model": {"path": self.model_config.local_path, "use_shm": self.model_config.use_shm}}
                )

                load_megatron_gptmodel_weights(
                    tmp_config,
                    self.model_config.hf_config,
                    module,
                    params_dtype=self.dtype,
                    is_value_model=is_value_model,
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        return module

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            init_megatron_optim_config,
        )

        optim_config_megatron = init_megatron_optim_config(self.optimizer_config)
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer_param_scheduler,
        )

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_lr_scheduler()
        else:
            self.optimizer = None
            self.lr_scheduler = None

        tmp_config = OmegaConf.create({"model": {"path": self.model_config.local_path}})

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role="actor",
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
        )

    def train_mode(self):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self)

    def eval_mode(self):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self)

    def forward_step(
        self,
        data: DataProto,
        post_fn: Callable[[DataProto, torch.Tensor], tuple[list[torch.Tensor], dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        Perform inference on a mini batch of data.

        Args:
            data: The input data for inference, typically containing tensors and metadata.
            post_fn: A post-processing function that takes a micro-batch and predictions as input,
                     and returns a tuple containing processed predictions and a dictionary of outputs.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the predictions for the entire batch.
        """
        raise NotImplementedError

    def train_step(
        self,
        data: DataProto,
        loss_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """
        Perform a training step on a mini-batch of data.

        Args:
            data (DataProto): The input data for training, typically containing tensors and metadata.
            loss_fn (Callable): A function that computes the loss and metrics given a micro-batch and predictions.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the mini-batch.
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        assert device in ("cuda", "cpu")
        if device == "cuda":
            if not self.engine_config.param_offload:
                if model:
                    load_megatron_model_to_gpu(self.module, load_grad=True)
                if optimizer and self.optimizer is not None:
                    load_megatron_optimizer(self.optimizer, device)
        elif device == "cpu":
            if not self.engine_config.param_offload:
                if model:
                    offload_megatron_model_to_cpu(self.module)
                if optimizer and self.optimizer is not None:
                    offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        raise NotImplementedError

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)




class EngineEvalModeCtx:
    def __init__(self, engine: MegatronEngine):
        self.engine = engine

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)

        self.engine.mode = "eval"
        if self.engine._is_offload_param:
            load_megatron_model_to_gpu(self.engine.module, load_grad=True)

        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine._is_offload_param:
            offload_megatron_model_to_cpu(self.engine.fsdp_module)
        self.engine.mode = None


class EngineTrainModeCtx:
    def __init__(self, engine: MegatronEngine):
        self.engine = engine

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)

        self.engine.mode = "train"
        if self.engine._is_offload_param:
            load_megatron_model_to_gpu(self.engine.module, load_grad=True)
        if self.engine._is_offload_optimizer:
            load_megatron_optimizer(optimizer=self.engine.optimizer)

        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine._is_offload_param:
            offload_megatron_model_to_cpu(self.engine.module)
        if self.engine._is_offload_optimizer:
            offload_megatron_optimizer(optimizer=self.engine.optimizer)
        self.engine.mode = None