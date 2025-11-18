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


import os
from functools import partial

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler, default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group
from verl.utils.flops_counter import FlopsCounter
from verl.utils.logger import log_with_rank
from verl.utils.tracking import Tracking

if is_cuda_available:
    pass
elif is_npu_available:
    pass

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def patch_llama_attention_for_non_causal():
    """
    Monkey patch LlamaAttention to use non-causal attention.
    This modifies the forward method to change causal attention to non-causal.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError:
        print("Warning: Could not import LlamaAttention. Skipping monkey patch.")
        return

    # Store the original forward method
    original_forward = LlamaAttention.forward

    def non_causal_forward(self, *args, **kwargs):
        # Get the original implementation result but modify the attention mechanism
        # We need to intercept before the scaled_dot_product_attention call

        # Check if this is the SDPA version (transformers >= 4.36)
        if hasattr(self, "_update_causal_mask"):
            # For newer transformers versions, we need to modify the causal mask
            def patched_update_causal_mask(
                self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions
            ):
                # Get the original causal mask
                causal_mask = self._update_causal_mask.__wrapped__(
                    self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions
                )

                # Modify causal mask to be non-causal as per your reference code
                if causal_mask is not None:
                    D = causal_mask.shape[-1]
                    last_row = causal_mask[:, :, -1, :].clone()
                    new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
                    causal_mask = new_mask

                return causal_mask

            # Wrap the original _update_causal_mask method
            if not hasattr(self._update_causal_mask, "__wrapped__"):
                self._update_causal_mask.__wrapped__ = self._update_causal_mask
                self._update_causal_mask = patched_update_causal_mask.__get__(self, type(self))

        # For direct SDPA calls, we need to patch the actual attention computation
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def non_causal_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            # Force is_causal to False and modify the mask if needed
            if attn_mask is not None:
                D = attn_mask.shape[-1] if attn_mask.ndim >= 2 else None
                if D is not None and attn_mask.ndim >= 4:
                    last_row = attn_mask[:, :, -1, :].clone()
                    new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
                    attn_mask = new_mask

            return original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,  # Force non-causal
                **kwargs,
            )

        # Temporarily replace SDPA
        torch.nn.functional.scaled_dot_product_attention = non_causal_sdpa

        try:
            # Call the original forward method
            result = original_forward(self, *args, **kwargs)
        finally:
            # Restore the original SDPA
            torch.nn.functional.scaled_dot_product_attention = original_sdpa

        return result

    # Apply the monkey patch
    LlamaAttention.forward = non_causal_forward
    print("Successfully applied monkey patch to LlamaAttention for non-causal attention.")


patch_llama_attention_for_non_causal()


def multi_modal_collect(data):
    multi_modal_data = [i.pop("multi_modal_inputs", None) for i in data]
    others = default_collate(data)
    if multi_modal_data[0] is not None:
        others["multi_modal_inputs"] = multi_modal_data
    return others


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        from verl.workers.roles.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        self.flops_counter = FlopsCounter(self.model_config.hf_config)

        if self.rank == 0:
            print(self.config)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
        )

    def _build_config(self):
        from verl.trainer.config import CheckpointConfig
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import (
            FSDPEngineConfig,
            FSDPOptimizerConfig,
            HFModelConfig,
            McoreEngineConfig,
            McoreOptimizerConfig,
        )

        self.model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
        self.engine_config: FSDPEngineConfig | McoreEngineConfig = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config: FSDPOptimizerConfig | McoreOptimizerConfig = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config: CheckpointConfig = omega_conf_to_dataclass(self.config.checkpoint)

    def _build_engine(self):
        from verl.workers.engine import BaseEngine, EngineRegistry

        self.engine: BaseEngine = EngineRegistry.new(
            model_type="language_model",
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = min(self.total_training_steps, len(self.train_dataloader))

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.engine.initialize()

    def _build_dataset(self):
        config = self.config

        tokenizer = self.model_config.tokenizer
        train_dataset = create_sft_dataset(
            config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)

                with self.engine.train_mode():
                    with Timer(name="update_policy", logger=None) as timer:
                        output = self.engine.train_batch(data=data, loss_function=self.loss_fn)
                lr = self.engine.lr_scheduler_step()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = output["metrics"]

                    loss = torch.sum(torch.tensor(metrics["loss"], device=self.device_name))

                    # mean over dp group
                    is_nested = data["input_ids"].is_nested
                    if is_nested:
                        batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
                    else:
                        batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
                    batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)

                    output_tensor = torch.randint(
                        0,
                        100,
                        (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
                        device=self.device_name,
                    )  # (global_bsz,)

                    torch.distributed.all_gather_into_tensor(
                        output_tensor=output_tensor,
                        input_tensor=batch_seqlens,
                        group=self.engine.get_data_parallel_group(),
                    )
                    torch.distributed.all_reduce(
                        loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
                    )

                    batch_seqlens = output_tensor.tolist()
                    loss = loss.item()

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                    metrics["loss"] = loss
                    metrics["train/loss"] = metrics.pop("loss")
                    metrics["train/grad_norm"] = metrics.pop("grad_norm")
                    metrics["train/lr"] = lr
                    metrics["train/global_tokens"] = output_tensor.sum().item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9
                    # mfu
                    delta_time = timer.last
                    estimated_flops, promised_flops = self.flops_counter.estimate_flops(batch_seqlens, delta_time)
                    metrics["train/mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        with self.engine.eval_mode():
                            # construct tensordict
                            val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                            output = self.engine.infer_batch(data=val_data, loss_function=self.loss_fn)
                            if self.engine.is_mp_src_rank_with_outputs():
                                val_losses.extend(output["metrics"]["loss"])

                #     if self.engine.is_mp_src_rank_with_outputs():
                #         val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                #         # average over data parallel group
                #         torch.distributed.all_reduce(
                #             val_loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
                #         )

                #     if is_logging:
                #         metric = {"val/loss": val_loss.detach().item()}
                #         tracking.log(data=metric, step=global_step)
                #         last_valid_metric = metric
                #     torch.distributed.barrier()

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return dataset


if __name__ == "__main__":
    main()
