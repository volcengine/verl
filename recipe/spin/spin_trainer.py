# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
import traceback
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from recipe.spin import core_algos
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different
            # WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this "
                    f"ray cluster"
                )


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """Placeholder: Computes prompt and response lengths."""
    try:
        # Assuming 'prompts' and 'responses' keys exist after generation/union
        prompt_len = batch.batch["prompts"].shape[1]
        resp_len = batch.batch["responses"].shape[1]
        # This is simplified - real implementation might use attention masks
        # to get actual lengths per sample.
        batch_size = batch.batch.batch_size[0]
        prompt_lengths_tensor = torch.full((batch_size,), prompt_len, dtype=torch.float32, device=batch.batch.device)
        response_lengths_tensor = torch.full((batch_size,), resp_len, dtype=torch.float32, device=batch.batch.device)

        # Try getting actual lengths from attention mask if possible (more accurate)
        if "response_mask" in batch.batch:
            response_lengths_tensor = batch.batch["response_mask"].sum(dim=1).float()
            # if "attention_mask" in batch.batch and "response_mask" in batch.batch:
            # full_mask = batch.batch["attention_mask"]
            # resp_mask = batch.batch["response_mask"]
            # Infer prompt mask length based on where response mask starts or total length
            # This logic depends heavily on how your masks are constructed.
            # Example: prompt_lengths_tensor = full_mask.sum(dim=1).float() - response_lengths_tensor
            # Fallback to using prompt shape if mask logic is complex:
            prompt_lengths_tensor = torch.tensor(
                [batch.batch["prompts"].shape[1]] * batch_size, dtype=torch.float32, device=batch.batch.device
            )

        return {
            "prompt_length": prompt_lengths_tensor,
            "response_length": response_lengths_tensor,
            "max_response_length": resp_len,
            "max_prompt_length": prompt_len,  # Or from config if fixed padding
        }
    except KeyError as e:
        print(f"Warning: Missing key in _compute_response_info: {e}. Returning defaults.")
        # Return default/dummy values if keys are missing
        b_size = batch.batch.batch_size[0] if batch.batch.batch_size else 1
        max_resp = batch.batch.get("responses").shape[1] if batch.batch.get("responses") is not None else 0
        max_prompt = batch.batch.get("prompts").shape[1] if batch.batch.get("prompts") is not None else 0
        return {
            "prompt_length": torch.zeros(b_size),
            "response_length": torch.zeros(b_size),
            "max_response_length": max_resp,
            "max_prompt_length": max_prompt,
        }


# --- Modified Metric Function ---
def compute_dpo_data_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Computes and returns metrics relevant for the DPO-like process.
    Assumes 'batch' contains results after generation and preference marking,
    potentially including 'dpo_logits', 'preferences', 'chosen_logps', etc.
    Removes PPO-specific advantage/return/critic metrics.
    """
    print("---- [DEBUG] Computing DPO Data Metrics ----")
    metrics = {}
    try:
        # --- Scores and Rewards (from reward_fn) ---
        if "token_level_scores" in batch.batch and batch.batch["token_level_scores"] is not None:
            sequence_score = batch.batch["token_level_scores"].sum(-1)
            metrics.update(
                {
                    "reward/score/mean": torch.mean(sequence_score).item(),
                    "reward/score/max": torch.max(sequence_score).item(),
                    "reward/score/min": torch.min(sequence_score).item(),
                }
            )
        else:
            print("DEBUG compute_dpo_data_metrics: 'token_level_scores' not found.")

        if "token_level_rewards" in batch.batch and batch.batch["token_level_rewards"] is not None:
            sequence_reward = batch.batch["token_level_rewards"].sum(-1)
            metrics.update(
                {
                    "reward/rewards/mean": torch.mean(sequence_reward).item(),
                    "reward/rewards/max": torch.max(sequence_reward).item(),
                    "reward/rewards/min": torch.min(sequence_reward).item(),
                }
            )
        else:
            print("DEBUG compute_dpo_data_metrics: 'token_level_rewards' not found.")

        # --- DPO Specific Metrics (if stored previously) ---
        if "dpo_logits" in batch.batch and batch.batch["dpo_logits"] is not None:
            metrics["actor/dpo_logits"] = batch.batch["dpo_logits"].mean().item()
        else:
            print("DEBUG compute_dpo_data_metrics: 'dpo_logits' not found.")

        if "chosen_logps" in batch.batch and batch.batch["chosen_logps"] is not None:
            metrics["actor/chosen_logps"] = batch.batch["chosen_logps"].mean().item()
        else:
            print("DEBUG compute_dpo_data_metrics: 'chosen_logps' not found.")

        if "rejected_logps" in batch.batch and batch.batch["rejected_logps"] is not None:
            metrics["actor/rejected_logps"] = batch.batch["rejected_logps"].mean().item()
        else:
            print("DEBUG compute_dpo_data_metrics: 'rejected_logps' not found.")

        # Add metrics based on the 'preferences' mask if available
        # if "preferences" in batch.batch and batch.batch["preferences"] is not None:
        # prefs_mask = batch.batch["preferences"]  # Shape [batch_size * n]
        # Calculate accuracy based on RM scores (assuming higher score -> True in mask)
        # Requires chosen/rejected scores to be available or recalculated
        # This is complex here, better calculated in the main loop or update function

        # --- Length Metrics ---
        response_info = _compute_response_info(batch)
        prompt_length = response_info["prompt_length"]
        response_length = response_info["response_length"]
        max_response_length = response_info["max_response_length"]
        max_prompt_length = response_info["max_prompt_length"]  # Use calculated or from config

        metrics.update(
            {
                "response_length/mean": torch.mean(response_length).item(),
                "response_length/max": torch.max(response_length).item(),
                "response_length/min": torch.min(response_length).item(),
                "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).item(),
                "prompt_length/mean": torch.mean(prompt_length).item(),
                "prompt_length/max": torch.max(prompt_length).item(),
                "prompt_length/min": torch.min(prompt_length).item(),
                # Prompt clip ratio might need adjustment based on how max_prompt_length is defined
                "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).item(),
            }
        )

    except KeyError as e:
        print(f"ERROR in compute_dpo_data_metrics: Missing key {e}")
    except Exception as e:
        print(f"ERROR in compute_dpo_data_metrics: {e}")
        traceback.print_exc()

    print(f"---- [DEBUG] Calculated DPO Data Metrics: {list(metrics.keys())} ----")
    return metrics


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_onlineDPO_pref(data: DataProto):
    """
    Wrapper to compute DPO preference and add it to the DataProto batch.
    Includes debugging prints.
    """
    # print(f"\n---- [DEBUG] Entering compute_onlineDPO_pref ----")
    # print(f"  Input batch keys: {list(data.batch.keys())}")

    # Check inputs
    rewards_tensor = data.batch.get("token_level_rewards")
    mask_tensor = data.batch.get("response_mask")

    if rewards_tensor is None or mask_tensor is None:
        print("  ERROR: Missing 'token_level_rewards' or 'response_mask' in input data!")
        # Handle error case - maybe return original data or raise?
        # Returning original data for now to potentially allow skipping
        return data

    try:
        preferences = core_algos.compute_onlinedpo_pref(token_level_rewards=rewards_tensor, response_mask=mask_tensor)
        # Store the result
        data.batch["preferences"] = preferences

    except AttributeError:
        print("ERROR: Function 'compute_online_dpo_preference' not found in core_algos.py!")
        # Assign dummy value or raise error
        data.batch["preferences"] = None  # Indicate failure
    except Exception as e_pref:
        print(f"ERROR during core_algos.compute_online_dpo_preference: {e_pref}")
        import traceback

        traceback.print_exc()
        data.batch["preferences"] = None  # Indicate failure

    # print(f"---- [DEBUG] Exiting compute_onlineDPO_pref ----")
    return data


@contextmanager
def _timer(name: str, timing_raw: dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RaySPINTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        # assert get_torch_device().is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(role_worker_mapping)
        self.use_rm = need_reward_model(role_worker_mapping)
        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.async_rollout_mode = False
        self.device_name = device_name if device_name else self.config.trainer.device

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        print(f"DEBUG: Data sources shape: {data_sources.shape}")  # Added Print
        print(f"DEBUG: reward_extra_infos_dict keys before processing: {reward_extra_infos_dict.keys()}")  # Added Print

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        print(
            f"DEBUG: Output of process_validation_metrics (data_src2var2metric2val): {data_src2var2metric2val}"
        )  # Added Print
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different
        # parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to
        # different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and "
                "max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit_dpo(self):  # Renamed for clarity as standard PPO loop
        """
        The training loop of Online DPO using a periodically updated reference model.
        The driver process calls worker groups for computation.
        Advantage computation is replaced by DPO logic.
        """
        import traceback  # Ensure traceback is imported

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # Initialize logger
        logger = None
        try:
            logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False),
            )
        except Exception as e:
            print(f"Warning: Failed to initialize logger: {e}")

        self.global_steps = 0
        # Load checkpoint before doing anything
        loaded_step = self._load_checkpoint()
        self.global_steps = loaded_step + 1 if loaded_step is not None and loaded_step > 0 else 1
        print(
            f"Starting Online DPO training from global step {self.global_steps}. "
            f"Total steps: {self.total_training_steps}"
        )
        print(f"Reference model update frequency: {self.config.trainer.get('ref_update_freq', 'Not Set')}")

        # Check if reference policy is configured correctly for this mode
        if not self.use_reference_policy:
            print(
                "WARNING: 'use_reference_policy' is False. Periodic reference model update requires a "
                "reference policy worker. DPO updates might fail or use incorrect logic."
            )
            # Consider raising an error if strict adherence is required:
            # raise ValueError("Periodic reference model update requires 'use_reference_policy' to be True "
            #                  "and a configured reference worker.")

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            print("Running validation before Online DPO training...")
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            if logger and val_metrics:
                logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            if self.config.trainer.get("val_only", False):
                print("Validation only mode enabled. Exiting training.")
                if logger and hasattr(logger, "finish"):
                    logger.finish()
                return

        # Add tqdm progress bar
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Online DPO Training Progress",
            position=0,
            leave=True,
        )

        last_val_metrics = None
        should_stop = False

        for epoch in range(self.config.trainer.total_epochs):
            if should_stop:
                break
            print(f"--- Starting Online DPO Epoch {epoch} ---")
            try:
                train_iterator = iter(self.train_dataloader)
            except TypeError:
                print("Warning: Dataloader is not iterable.")
                train_iterator = self.train_dataloader  # Fallback attempt

            for batch_idx, batch_dict in enumerate(train_iterator):
                if self.global_steps > self.total_training_steps:
                    should_stop = True
                    break

                metrics = {}
                timing_raw = {}
                step_timer = Timer(logger=None)
                ref_log_prob_computed = False  # Flag to track if ref log probs were computed

                try:  # Outer try-except for the whole step
                    step_timer.start()
                    with _timer("step", timing_raw):
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        current_batch_size = batch.batch.batch_size[0]
                        print(
                            f"\n[Step {self.global_steps}, Batch {batch_idx}] Processing batch size: "
                            f"{current_batch_size}"
                        )

                        # --- Reference Model Update ---
                        ref_update_freq = self.config.trainer.get("ref_update_freq", -1)
                        if (
                            self.use_reference_policy
                            and ref_update_freq > 0
                            and self.global_steps % ref_update_freq == 0
                        ):
                            print(f"\n[Step {self.global_steps}] Updating Reference Model Weights from Actor...")
                            try:
                                # --- This requires careful implementation with FSDP ---
                                # 1. Save actor state dict (potentially to CPU memory or disk)
                                #    This needs to be done collectively across actor worker ranks.
                                #    The checkpoint_manager might be adaptable, or use FSDP APIs directly.
                                #    Example placeholder using a conceptual save/load mechanism:
                                actor_state_path = "/tmp/actor_state_mid"  # Temporary path
                                self.actor_rollout_wg.save_checkpoint(actor_state_path)  # Adapt save logic

                                # 2. Load the state dict onto the reference model worker group
                                #    This also needs collective loading on the ref worker ranks.
                                self.ref_policy_wg.load_checkpoint(actor_state_path, None, True)  # Adapt load logic

                                print(f"[Step {self.global_steps}] Reference Model Weights Updated.")
                                # Optionally remove the temporary state file
                                # os.remove(actor_state_path) # Needs rank-aware removal or shared storage

                            except Exception as sync_e:
                                print(f"ERROR during reference model sync at step {self.global_steps}: {sync_e}")
                                traceback.print_exc()

                        # Pop keys for generation
                        pop_batch_keys = ["input_ids", "attention_mask"]
                        if "position_ids" in batch.batch:
                            pop_batch_keys.append("position_ids")
                        pop_non_tensor_keys = ["raw_prompt_ids"] if "raw_prompt_ids" in batch.non_tensor_batch else []
                        if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                            pop_non_tensor_keys.extend(["multi_modal_data", "multi_modal_inputs"])
                        original_non_tensor_data = batch.non_tensor_batch
                        gen_batch = batch.pop(
                            batch_keys=pop_batch_keys,
                            non_tensor_batch_keys=pop_non_tensor_keys,
                        )
                        gen_batch = gen_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        # (Add Debug prints for gen_batch if needed)

                        # Generate sequences (chosen/rejected pairs)
                        with _timer("gen", timing_raw):
                            try:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                # (Add Debug prints for gen_batch_output if needed)
                            except Exception as gen_e:
                                print(f"\n!!!!!!!! ERROR DURING GENERATION (Step {self.global_steps}) !!!!!!!!")
                                print(gen_e)
                                traceback.print_exc()
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                step_timer.stop()
                                continue

                        # Combine original prompts with generated sequences
                        batch.non_tensor_batch = original_non_tensor_data  # Restore non-tensor data
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(current_batch_size)], dtype=object
                        )
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                        # (Add Debug prints after union if needed)

                        # Compute response mask (needed for ref logprob calc and DPO prep)
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        # --- Compute Log Probs for the CURRENT policy (used for KL if enabled, or ActorAsRef
                        # fallback) ---
                        # Note: For pure DPO with external ref, this 'old_log_probs' might not be strictly needed
                        #       unless used for other metrics or a fallback. Keep it for now.
                        with _timer("policy_log_prob", timing_raw):
                            policy_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(policy_log_prob_output)  # Adds 'old_log_probs'
                            # (Debug prints for old_log_probs)

                        # --- Compute Log Probs using the EXTERNAL Reference Model ---
                        if self.use_reference_policy:
                            with _timer("ref_log_prob_dpo", timing_raw):
                                # print(f"---- [Step {self.global_steps}] DEBUG DPO: Calling compute_ref_log_prob ----")
                                try:
                                    # 'batch' contains interleaved chosen/rejected sequences
                                    ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(
                                        batch
                                    )  # Returns DataProto with 'ref_log_prob'
                                    batch = batch.union(
                                        ref_log_prob_output
                                    )  # Adds 'ref_log_prob' key [batch_size * n, seq_len]
                                    ref_log_prob_computed = True  # Mark success
                                    # print(f"---- [Step {self.global_steps}] DEBUG DPO: ref_log_prob tensor shape: "
                                    #       f"{batch.batch['ref_log_prob'].shape} ----")
                                except Exception as ref_e:
                                    print(f"ERROR computing reference log probs at step {self.global_steps}: {ref_e}")
                                    traceback.print_exc()
                                    batch.batch["ref_log_prob"] = None  # Mark as failed
                                    ref_log_prob_computed = False
                        else:
                            print(
                                "Warning: Skipping external reference log prob calculation as use_reference_policy "
                                "is False."
                            )
                            # DPO update will likely fail unless ActorAsRef logic is re-enabled in dp_actor

                        # --- Compute Rewards/Scores (used to determine preference) ---
                        with _timer("reward_calc", timing_raw):
                            # (Reward calculation logic using RM or reward_fn as before)
                            # ... Ensure this calculates 'token_level_rewards' or similar ...
                            if self.use_rm:
                                reward_tensor_rm = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor_rm)  # Adds 'rm_scores'

                            reward_extra_infos_dict = {}
                            try:
                                if self.reward_fn is None:
                                    #  print(f"---- [DEBUG Step {self.global_steps}] ERROR: self.reward_fn is None! "
                                    #        f"Using dummy rewards. ----")
                                    # Use rm_scores if available, otherwise zeros
                                    reward_tensor = batch.batch.get(
                                        "rm_scores", torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                                    )
                                else:
                                    reward_result = self.reward_fn(batch, return_dict=True)
                                    reward_tensor = reward_result["reward_tensor"]  # Final combined reward
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})

                            except Exception:
                                # print(f'---- [DEBUG Step {self.global_steps}] Error in reward_fn call: {e}. '
                                #       f'Using dummy rewards. ----')
                                traceback.print_exc()
                                reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                                reward_extra_infos_dict = {}

                            # Use 'token_level_rewards' as the key for preference calculation
                            batch.batch["token_level_rewards"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                        # --- Determine Preferences ---
                        # Uses 'token_level_rewards' to determine chosen/rejected based on score
                        batch = compute_onlineDPO_pref(batch)  # Adds 'preferences' key

                        # --- Prepare DPO Batch ---
                        dpo_update_batch_proto = None  # Initialize
                        with _timer("prepare_dpo_batch", timing_raw):
                            try:
                                if "preferences" not in batch.batch or batch.batch["preferences"] is None:
                                    raise ValueError("'preferences' key missing or None after compute_onlineDPO_pref.")

                                # Check if reference log probs were computed successfully (if needed)
                                if self.use_reference_policy and not ref_log_prob_computed:
                                    raise ValueError("Reference log probs required but failed to compute.")

                                # Check required base keys
                                required_keys = ["input_ids", "attention_mask", "response_mask"]
                                for rk in required_keys:
                                    if rk not in batch.batch or batch.batch[rk] is None:
                                        raise KeyError(f"Required key '{rk}' missing from batch for DPO prep.")

                                preferences_mask = batch.batch["preferences"]  # Shape [batch_size * n]
                                not_preferences_mask = ~preferences_mask

                                # Gather Chosen/Rejected Base Tensors
                                chosen_input_ids = batch.batch["input_ids"][preferences_mask]
                                chosen_attention_mask = batch.batch["attention_mask"][preferences_mask]
                                rejected_input_ids = batch.batch["input_ids"][not_preferences_mask]
                                rejected_attention_mask = batch.batch["attention_mask"][not_preferences_mask]
                                chosen_position_ids = (
                                    batch.batch.get("position_ids")[preferences_mask]
                                    if "position_ids" in batch.batch
                                    else None
                                )
                                rejected_position_ids = (
                                    batch.batch.get("position_ids")[not_preferences_mask]
                                    if "position_ids" in batch.batch
                                    else None
                                )

                                # Create Labels
                                print("WARNING: Creating DPO labels using configured max_prompt_length...")
                                prompt_len = self.config.data.max_prompt_length
                                chosen_labels = chosen_input_ids.clone()
                                chosen_labels[:, :prompt_len] = -100
                                rejected_labels = rejected_input_ids.clone()
                                rejected_labels[:, :prompt_len] = -100

                                # Calculate and Gather Reference Log Probs (Sequence Level)
                                if self.use_reference_policy:
                                    ref_log_prob_tensor = batch.batch["ref_log_prob"]  # Token level [bsz * n, seq_len]
                                    response_mask_full = batch.batch[
                                        "response_mask"
                                    ]  # Response mask [bsz * n, seq_len]
                                    ref_sequence_logps = (ref_log_prob_tensor * response_mask_full).sum(
                                        dim=-1
                                    )  # Sequence level [bsz * n]
                                    reference_chosen_logps = ref_sequence_logps[preferences_mask]
                                    reference_rejected_logps = ref_sequence_logps[not_preferences_mask]
                                else:
                                    # If not using external ref, DPO needs ActorAsRef logic in dp_actor
                                    # We won't add the keys here, dp_actor will handle it (or fail if not modified)
                                    print(
                                        "Info: Not adding explicit reference logps to DPO batch "
                                        "(use_reference_policy=False)."
                                    )
                                    reference_chosen_logps = None  # Explicitly None
                                    reference_rejected_logps = None

                                # Package Tensors
                                dpo_tensors = {
                                    "chosen_input_ids": chosen_input_ids,
                                    "chosen_attention_mask": chosen_attention_mask,
                                    "chosen_labels": chosen_labels,
                                    "rejected_input_ids": rejected_input_ids,
                                    "rejected_attention_mask": rejected_attention_mask,
                                    "rejected_labels": rejected_labels,
                                }
                                # Conditionally add reference logps if computed
                                if reference_chosen_logps is not None:
                                    dpo_tensors["reference_chosen_logps"] = reference_chosen_logps
                                if reference_rejected_logps is not None:
                                    dpo_tensors["reference_rejected_logps"] = reference_rejected_logps
                                # Add position ids if they exist
                                if chosen_position_ids is not None:
                                    dpo_tensors["chosen_position_ids"] = chosen_position_ids
                                if rejected_position_ids is not None:
                                    dpo_tensors["rejected_position_ids"] = rejected_position_ids

                                # Prepare Meta Info
                                dpo_meta = {
                                    "dpo_beta": OmegaConf.select(self.config.algorithm, "dpo_beta", default=0.1),
                                    "dpo_loss_type": OmegaConf.select(
                                        self.config.algorithm, "dpo_loss_type", default="sigmoid"
                                    ),
                                    "dpo_label_smoothing": OmegaConf.select(
                                        self.config.algorithm, "dpo_label_smoothing", default=0.0
                                    ),
                                    "use_reference_policy": self.use_reference_policy,
                                    "reference_free": not self.use_reference_policy,  # False if using external ref
                                    "global_step": self.global_steps,
                                }

                                dpo_update_batch_proto = DataProto.from_dict(tensors=dpo_tensors, meta_info=dpo_meta)
                                # print(f"---- [Step {self.global_steps}] DEBUG DPO: Prepared DPO Update Batch ----")
                                # print(f"  Keys: {list(dpo_update_batch_proto.batch.keys())}")
                                # print(f"  Meta Info: {dpo_meta}")

                            except Exception as e_prep:
                                print(f"ERROR preparing DPO batch at step {self.global_steps}: {e_prep}")
                                traceback.print_exc()
                                dpo_update_batch_proto = None  # Skip update on error

                        # --- Actor Update Step ---
                        actor_output = None
                        if self.config.trainer.critic_warmup <= self.global_steps and dpo_update_batch_proto:
                            with _timer("update_actor", timing_raw):
                                # Pass the batch containing reference log probs (if computed)
                                # The modified update_actor_dpo expects them if reference_free=False
                                actor_output = self.actor_rollout_wg.update_actor_dpo(dpo_update_batch_proto)
                            if actor_output and "metrics" in actor_output.meta_info:
                                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                        elif dpo_update_batch_proto is None:
                            print(
                                f"Skipping actor update at step {self.global_steps} due to DPO batch preparation error."
                            )

                        # --- Validation and Saving ---
                        test_freq = OmegaConf.select(self.config.trainer, "test_freq", default=-1)
                        is_last_step = self.global_steps >= self.total_training_steps
                        if (
                            self.val_reward_fn is not None
                            and test_freq > 0
                            and (is_last_step or self.global_steps % test_freq == 0)
                        ):
                            print(f"\nRunning DPO validation at step {self.global_steps}...")
                            val_timing_raw = {}
                            with _timer("testing", val_timing_raw):
                                val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            if val_metrics:
                                metrics["time/validation_run"] = val_timing_raw.get("testing", 0)
                                metrics.update(val_metrics)
                            else:
                                print("Validation skipped or returned no metrics.")

                        save_freq = OmegaConf.select(self.config.trainer, "save_freq", default=-1)
                        if save_freq > 0 and (is_last_step or self.global_steps % save_freq == 0):
                            print(f"\nSaving DPO checkpoint at step {self.global_steps}...")
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()  # Saves actor (and potentially critic if used elsewhere)
                            metrics["time/save_checkpoint"] = timing_raw.get("save_checkpoint", 0)

                    # --- End main step timer context ---

                    # --- Metrics calculation AFTER the 'step' timer block ---
                    metrics.update(compute_dpo_data_metrics(batch=batch))  # Use DPO-specific metrics
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    if "step" in timing_raw:
                        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    else:
                        print(
                            f"Warning: 'step' key missing from timing_raw at step {self.global_steps}. "
                            f"Skipping throughput."
                        )

                    step_timer.stop()
                    metrics["time/step"] = step_timer.last

                    # Log metrics
                    log_freq = OmegaConf.select(self.config.trainer, "log_freq", default=1)
                    if logger and self.global_steps % log_freq == 0:
                        log_payload = metrics.copy()
                        # Add learning rate to log payload
                        if actor_output and "actor/lr" in metrics:
                            log_payload["actor/lr"] = metrics["actor/lr"]

                        print(f"[Step {self.global_steps} DPO] Logging Step Payload Keys: {list(log_payload.keys())}")
                        try:
                            logger.log(data=log_payload, step=self.global_steps)
                        except Exception as e:
                            print(f"Logging failed at step {self.global_steps}: {e}")

                    # Update progress bar
                    postfix_metrics = {
                        k: f"{v:.3f}" if isinstance(v, float) else v
                        for k, v in metrics.items()
                        if isinstance(v, int | float)
                    }
                    progress_bar.set_postfix(postfix_metrics)

                except Exception as step_e:
                    print(f"\n!!!!!!!! ERROR DURING DPO Step {self.global_steps} !!!!!!!!")
                    print(f"Caught Exception: {step_e}")
                    traceback.print_exc()
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    step_timer.stop()
                    should_stop = True
                    break

                if is_last_step or should_stop:
                    print(f"Stopping DPO training at step {self.global_steps}.")
                    break

                self.global_steps += 1
                progress_bar.update(1)

            # End of epoch handling
            if hasattr(self.train_dataloader, "reset"):
                try:
                    self.train_dataloader.reset()
                except Exception as e:
                    print(f"Warning: Failed to reset train dataloader state: {e}")
            if should_stop:
                break

        # --- Final cleanup and logging ---
        progress_bar.close()
        final_step = max(0, self.global_steps - 1)
        print(f"Online DPO Training finished at step {final_step}.")
        # Save final checkpoint
        save_freq = OmegaConf.select(self.config.trainer, "save_freq", default=-1)
        if not self.config.trainer.get("val_only", False) and (save_freq <= 0 or final_step % save_freq != 0):
            print(f"Saving final DPO checkpoint at step {final_step}...")
            self._save_checkpoint()

        # Final validation run
        if self.val_reward_fn and last_val_metrics is None and not self.config.trainer.get("val_only", False):
            print("Running final validation...")
            last_val_metrics = self._validate()
            if last_val_metrics and logger:
                last_val_metrics["final_validation"] = True
                try:
                    logger.log(data=last_val_metrics, step=final_step)
                except Exception as e:
                    print(f"[Final Val Metrics Log Error]: {e}")

        pprint(f"Final validation metrics: {last_val_metrics}")
        if logger and hasattr(logger, "finish"):
            logger.finish()
        print("Online DPO Training Run Complete.")
