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
from typing import Any, Callable, Iterator

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.model import load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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

        # TODO: add more cases
        is_value_model = (
            "ForTokenClassification" in self.model_config.architectures[0]
            or "ForSequenceClassification" in self.model_config.architectures[0]
        )

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
        print(f"module: {len(module)}")

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

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

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

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
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

    def get_data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

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

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=False,
            min_num_micro_batch=None,
        )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(self.forward_step, postprocess_micro_batch_func=postprocess_micro_batch_func)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )
        # loss_reduces contains the stats returned from loss_func
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            return postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
        else:
            return {}

    def forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx:
    def __init__(self, engine: MegatronEngine):
        self.engine = engine

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)

        self.engine.mode = "eval"
        if self.engine._is_offload_param:
            load_megatron_model_to_gpu(self.engine.module, load_grad=True)

        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine._is_offload_param:
            offload_megatron_model_to_cpu(self.engine.module)
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

        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.engine._is_offload_param:
            offload_megatron_model_to_cpu(self.engine.module)
        if self.engine._is_offload_optimizer:
            offload_megatron_optimizer(optimizer=self.engine.optimizer)
        self.engine.mode = None


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        batch = batch.to(get_device_id())
        batch = batch.contiguous()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(bool)
        position_ids = batch["position_ids"]

        # process vlm inputs
        batch["attention_mask"] = batch["attention_mask"].to(bool)
        has_multi_modal_inputs = "multi_modal_inputs" in batch.keys()
        if has_multi_modal_inputs:
            batch["multi_modal_inputs"] = batch["multi_modal_inputs"]
            batch["multi_modal_inputs_idx"] = torch.Tensor(list(range(len(batch["multi_modal_inputs"])))).to(
                torch.int64
            )

        if batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            batch["position_ids"] = batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                idxs = batch["multi_modal_inputs_idx"]
                mmi = batch["multi_modal_inputs"]
                multi_modal_inputs[key] = torch.cat(
                    [mmi[idx].get(key).to(input_ids.device) for idx in idxs if mmi[idx].get(key) is not None], dim=0
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        calculate_entropy = tu.get_non_tensor_data(data, key="calculate_entropy", default=False)
        responses = data["responses"]
        response_length = responses.size(1)

        log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()
        model_output = {"log_probs": log_prob}
        if calculate_entropy:
            entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, batch_iter: Iterator[TensorDict], model, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        temperature = batch["temperature"]

        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        position_ids = model_inputs["position_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        responses = batch["responses"]
        response_length = responses.size(1)
        label = position_ids.clone()
        label[:, -response_length - 1 : -1] = responses
        label_mask = attention_mask.clone()
        label_mask[:, : -response_length - 1] = False
        label_mask[:, -1] = False

        from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

        if use_fused_kernels:
            forward_fn = get_mcore_forward_fused_fn(self.model_config.hf_config)
            # return dict of [logits, entropy]
            output = forward_fn(
                model,
                input_ids,
                position_ids,
                attention_mask,
                sequence_parallel=self.tf_config.sequence_parallel,
                multi_modal_inputs=multi_modal_inputs,
                labels=label,
                labels_mask=label_mask,
                temperature=temperature,
            )
        else:
            forward_fn = get_mcore_forward_fn(self.model_config.hf_config)

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape
                logits.div_(temperature)
                ret = {}
                if calculate_entropy:
                    logits_bak = logits.clone()
                    if torch.distributed.get_rank() == 0:
                        logger.warning_once(
                            "For memory-efficient computation, enable fused kernels via "
                            "`actor_rollout_ref.model.use_fused_kernels=True`. "
                            "The current `clone()` operation ensures correctness but increases memory usage."
                        )
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits
                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                log_probs = log_probs.masked_fill(~label_mask, 0.0)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}
            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                multi_modal_inputs=multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        local_micro_bsz = data["input_ids"].shape[0]
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch and cp size inside pp schedule
            n_micro_batch = data["num_micro_batch"]
            loss = loss * n_micro_batch / mpu.get_context_parallel_world_size()
            global_bsz = data["global_batch_size"]
            loss_scale_factor = local_micro_bsz / (global_bsz / self.get_data_parallel_size())
            loss = loss * loss_scale_factor
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            metrics = {}

        output = {
            "model_output": model_output,
            "loss": loss,
            "metrics": metrics,
        }

        # return loss and stats
        return loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        position_ids = model_inputs["position_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        from verl.models.mcore import get_mcore_forward_fn

        forward_fn = get_mcore_forward_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            attention_mask,
            position_ids,
            sequence_parallel=self.tf_config.sequence_parallel,
            multi_modal_inputs=multi_modal_inputs,
            value_model=True,
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        responses = data["responses"]
        response_length = responses.size(1)
        output = output[:, -response_length - 1 : -1].contiguous()
        return {"values": output}
