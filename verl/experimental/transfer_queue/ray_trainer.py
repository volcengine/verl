# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import math
import numpy as np
import os
import ray
import tensordict
import torch
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from omegaconf import OmegaConf, open_dict
from packaging.version import parse as parse_version
from pprint import pprint
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import (
    BatchMeta,
    SimpleStorageUnit,
    TransferQueueController,
    get_placement_group,
    process_zmq_server_info,
)
from typing import Any, Optional

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.transferqueue_utils import (
    create_transferqueue_client,
    get_transferqueue_client,
    tqbridge,
)
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
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
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


@tqbridge(put_data=False)
def compute_reward_decorated(data, reward_fn):
    return compute_reward(data, reward_fn)


@tqbridge(put_data=False)
def compute_reward_async_decorated(data, reward_fn):
    return compute_reward_async.remote(data, reward_fn)


@tqbridge(put_data=False)
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

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

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return token_level_rewards, metrics


@tqbridge(put_data=True)
def compute_advantage(
        data: DataProto,
        adv_estimator: AdvantageEstimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit

    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )

        if config.get("use_pf_ppo", False):
            # the below code will resample the full data, for TQ adaption, we will return the resampled index
            # which will be read and used for resample later in fit func
            pf_ppo_reweight_idx = core_algos.compute_pf_ppo_reweight_data_tq(
                data.batch["token_level_rewards"],
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )

            advantages_td = TensorDict(
                {"advantages": advantages, "returns": returns,
                 }, batch_size=advantages.size(0)
            )
            non_tensor_batch = {"pf_ppo_reweight_idx": pf_ppo_reweight_idx}
            return DataProto(batch=advantages_td, non_tensor_batch=non_tensor_batch)
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)

    advantages_td = TensorDict(
        {"advantages": advantages, "returns": returns}, batch_size=advantages.size(0)
    )
    return DataProto(batch=advantages_td)


@tqbridge(put_data=False)
def compute_data_metrics_decorated(batch, use_critic: bool = True):
    return compute_data_metrics(batch, use_critic)


@tqbridge(put_data=False)
def compute_timing_metrics_decorated(batch, timing_raw: dict[str, float]) -> dict[str, Any]:
    return compute_timing_metrics(batch, timing_raw)


@tqbridge(put_data=False)
def compute_throughout_metrics_decorated(batch, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    return compute_throughout_metrics(batch, timing_raw, n_gpus)


@tqbridge(put_data=False)
def calculate_debug_metrics_decorated(data):
    from verl.utils.debug.metrics import calculate_debug_metrics

    return calculate_debug_metrics(data)


@tqbridge(put_data=False)
def compute_val_reward_decorated(reward_fn, data, return_dict):
    return reward_fn(data, return_dict)


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
            self,
            config,
            tokenizer,
            role_worker_mapping: dict[Role, WorkerType],
            resource_pool_manager: ResourcePoolManager,
            ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
            processor=None,
            reward_fn=None,
            val_reward_fn=None,
            train_dataset: Optional[Dataset] = None,
            val_dataset: Optional[Dataset] = None,
            collate_fn=None,
            train_sampler: Optional[Sampler] = None,
            device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.tq_client = self._initialize_transferqueue()

    def _initialize_transferqueue(self):
        # 1. initialize TransferQueueStorage
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            train_data_size = (
                    self.config.data.train_batch_size
                    * self.config.transfer_queue.num_global_batch
                    * self.config.actor_rollout_ref.rollout.n
            )
            val_data_size = self.val_dataset_size * self.config.actor_rollout_ref.rollout.val_kwargs.n

            total_storage_size = train_data_size + val_data_size
            self.data_system_storage_units = {}
            storage_placement_group = get_placement_group(
                self.config.transfer_queue.num_data_storage_units, num_cpus_per_actor=1
            )
            for storage_unit_rank in range(self.config.transfer_queue.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.config.transfer_queue.num_data_storage_units)
                )
                self.data_system_storage_units[storage_unit_rank] = storage_node
                logging.info(f"SimpleStorageUnit #{storage_unit_rank} has been created.")
        else:
            raise NotImplementedError("Currently only support AsyncSimpleStorageManager backend in TransferQueue")

        # 2. Initialize TransferQueueController (single controller only)

        # Sampler usage instructions:
        # For GRPO grouped sampling, you can initialize the controller with GRPOGroupNSampler:
        # Option 1: Pass sampler class (will be instantiated automatically)
        # self.data_system_controller = TransferQueueController.remote(sampler=GRPOGroupNSampler)

        # Option 2: Pass sampler instance (if you need custom configuration)
        # grpo_sampler = GRPOGroupNSampler()
        # self.data_system_controller = TransferQueueController.remote(sampler=grpo_sampler)

        # Then use sampling_config in get_meta calls:
        # sampling_config={"n_samples_per_prompt": 4}
        self.data_system_controller = TransferQueueController.remote()
        logging.info("TransferQueueController has been created.")

        # 3. register controller & storage and prepare necessary information
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        # Note: Need to generate a new DictConfig with allow_objects=True to preserve ZMQServerInfo instances
        # (which contain socket connection details). Without this flag, OmegaConf would flatten these objects to dicts,
        # breaking the transfer queue client initialization.
        tq_config = OmegaConf.create({"transfer_queue": {}}, flags={"allow_objects": True})
        tq_config.transfer_queue.controller_info = self.data_system_controller_info

        # TODO(TQ): should it be AsyncSimpleStorageManager or SyncSimpleStorageManager?
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            tq_config.transfer_queue.storage_unit_infos = self.data_system_storage_unit_infos

        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. create client
        create_transferqueue_client(
            client_id="Trainer",
            config=self.config.transfer_queue,
            sync=True
        )
        tq_client = get_transferqueue_client()
        return tq_client

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
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

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
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
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
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

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    @tqbridge(put_data=False)
    def _log_rollout_data(
            self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

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

    # TODO (TQ): @chenhao:
    #          1. unify return_dict branch (PR to main branch)
    #          2. in TQ ray_trainer.py, we only adapt if "rm_scores" in batch.batch.keys()
    #          3. modify RewardLoop so that it compatables with TQ BatchMeta (PR to main branch)
    # TODO (TQ): for compute_advantage and apply_kl_penalty, we do not dispatch now. Just preserve the current logic
    #            and use @tqbridge to decorate
    @tqbridge(put_data=False)  # this function is used by both fit&val, therefore cannot set put_data = True
    def _compute_or_extract_reward(
            self,
            batch: DataProto,
            reward_fn=None,
            return_dict: bool = False,
            sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            # Note: this compute_reward used here is NOT decorated, input: DataProto
            # TODO(TQ): later we will reorganize reward-related functions
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch_fields(self, non_tensor_batch_keys: set) -> set:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & non_tensor_batch_keys

        # pop those keys for generation
        tensor_batch_keys_to_pop = set()
        non_tensor_batch_keys_to_pop = non_tensor_batch_keys - reward_model_keys
        gen_batch_field_names = tensor_batch_keys_to_pop | non_tensor_batch_keys_to_pop

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch_field_names = gen_batch_field_names | non_tensor_batch_keys

        return gen_batch_field_names

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            if "uid" not in test_data.keys():
                test_data["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_data["raw_prompt"]))], dtype=object
                )

            # repeat test data
            repeated_test_data = self.repeat_dict(
                test_data, repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            test_batch: TensorDict = tu.dict_to_tensordict(repeated_test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0]["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            test_batch_meta = self.tq_client.put(data=test_batch, partition_id=f"val_{self.global_steps - 1}")

            # TODO (TQ): code error here!
            test_gen_fields = self._get_gen_batch_fields(tu.get_non_tensor_keys(test_batch))
            del test_batch  # should not use it later
            test_gen_meta = test_batch_meta.select_fields(list(test_gen_fields))
            test_gen_meta.update_extra_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_meta.extra_info}")

            # Note(TQ): we do not adapt pad & unpad processing in TQ version trainer

            if not self.async_rollout_mode:
                test_output_gen_meta = self.actor_rollout_wg.generate_sequences(test_gen_meta)
            else:
                test_output_gen_meta = self.async_rollout_manager.generate_sequences(test_gen_meta)

            test_batch_meta = test_batch_meta.union(test_output_gen_meta)

            print("validation generation end")

            # Store generated outputs
            test_response_meta = test_output_gen_meta.select_fields(["prompts", "uid", "reward_model", "responses"])
            data = self.tq_client.get_data(test_response_meta)
            output_ids = data["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            input_ids = data["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(data["uid"])

            ground_truths = [item.get("ground_truth", None) for item in data.get("reward_model", {})]
            sample_gts.extend(ground_truths)

            # evaluate using reward_function
            compute_reward_fields = [
                "responses",
                "prompts",
                "attention_mask",
                "reward_model",
                "data_source",
            ]
            # TODO(TQ): code error! why delete the below two line?
            if "rm_scores" in batch_meta.field_names:
                compute_reward_fields = ["rm_scores"]
            val_reward_meta = test_batch_meta.select_fields(compute_reward_fields)

            # evaluate using reward_function
            result = self._compute_or_extract_reward(val_reward_meta, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch_meta.field_names:
                num_turns_meta = test_batch_meta.select_fields(["__num_turns__"])
                data = self.tq_client.get_data(num_turns_meta)
                sample_turns.append(data["__num_turns__"])

            data_source = ["unknown"] * reward_tensor.shape[0]
            if "data_source" in test_batch_meta.field_names:
                data_source_meta = test_batch_meta.select_fields(["data_source"])
                data = self.tq_client.get_data(data_source_meta)
                data_source = data["data_source"]

            data_source_lst.append(data_source)

            self.tq_client.clear_samples(test_batch_meta)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
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

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
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

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                        OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                        is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
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

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

        # TODO (TQ): initialize tq during worker init when enable TQ switch is stable
        self.async_rollout_manager.create_transferqueue_client_for_workers()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

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
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
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
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
                hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
                and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
                "async_save" in self.config.actor_rollout_ref.actor.checkpoint
                and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
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
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
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

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    @tqbridge(put_data=False)
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size: (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        # when enable TQ just return index instead of reorder real data here
        global_idx = [j for partition in global_partition_lst for j in partition]
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return global_idx

    @classmethod
    def repeat_dict(
            cls, batch_dict: dict[str, torch.Tensor | np.ndarray], repeat_times=2, interleave=True
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Repeat the batch dict a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            dict: A new dict with repeated data.
        """
        if repeat_times == 1:
            return batch_dict

        repeated_batch_dict = {}
        if batch_dict:
            if interleave:
                # Interleave the data
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        repeated_batch_dict[key] = val.repeat_interleave(repeat_times, dim=0)
                    elif isinstance(val, np.ndarray):
                        repeated_batch_dict[key] = np.repeat(val, repeat_times, axis=0)
                    else:
                        raise ValueError(f"Unsupported type in data {type(val)}")
            else:
                # Stack the data
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        repeated_batch_dict[key] = (
                            val.unsqueeze(0).expand(repeat_times, *val.shape).reshape(-1, *val.shape[1:])
                        )
                    elif isinstance(val, np.ndarray):
                        repeated_batch_dict[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))
                    else:
                        raise ValueError(f"Unsupported type in data {type(val)}")
        return repeated_batch_dict

    # Note(TQ): here we skip the pad and unpad processing for dataproto type data, compared with current verl main version
    # as tq put/get tensordict and engine workers receives tensordict directly, later verl will deprecate dataproto too
    # therefore we adapt this function in this way: it receives batch meta and just pass it to inner function
    # however, in order to pass tensordict via tqbridge, the tqbridge needs to be modified
    def _compute_values(self, batch_meta: BatchMeta) -> BatchMeta:
        if self.use_legacy_worker_impl == "disable":
            batch_meta.set_extra_info("compute_loss", False)
            values_meta = self.critic_wg.infer_batch(batch_meta)
        else:
            values_meta = self.critic_wg.compute_values(batch_meta)
        return values_meta

    def _compute_ref_log_prob(self, batch_meta: BatchMeta) -> BatchMeta:
        if self.use_legacy_worker_impl == "disable":
            batch_meta.set_extra_info("compute_loss", False)
            batch_meta.set_extra_info("calculate_entropy", False)
            if self.ref_in_actor:
                batch_meta.set_extra_info("no_lora_adapter", True)

            if self.ref_in_actor:
                # output contains log_probs and ref_log_prob
                output_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            else:
                output_meta = self.ref_policy_wg.compute_ref_log_prob(batch_meta)
        else:
            output_meta = self.ref_policy_wg.compute_ref_log_prob(batch_meta)
        return output_meta

    def _compute_old_log_prob(self, batch_meta: BatchMeta) -> Tuple[BatchMeta, Any]:
        if self.use_legacy_worker_impl == "disable":
            batch_meta.set_extra_info("compute_loss", False)
            batch_meta.set_extra_info("calculate_entropy", True)
            output_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            # metrics originally is saved in tensordict as a NonTensorData
            # after tqbridge, metrics should be turned into batchmeta extra_info
            old_log_prob_mfu = output_meta.extra_info.get("metrics")["mfu"]
        else:
            output_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            old_log_prob_mfu = 0
        return output_meta, old_log_prob_mfu

    def _update_actor(self, batch_meta: BatchMeta) -> BatchMeta:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch_meta.set_extra_info("multi_turn", rollout_config.multi_turn.enable)
        # TODO: Make "temperature" single source of truth from generation.
        batch_meta.set_extra_info("temperature", rollout_config.temperature)
        # update actor
        if self.use_legacy_worker_impl == "disable":
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            extra_meta = {
                "calculate_entropy": calculate_entropy,
                "global_batch_size": ppo_mini_batch_size,
                "mini_batch_size": ppo_mini_batch_size,
                "epochs": ppo_epochs,
                "seed": seed,
                "dataloader_kwargs": {"shuffle": shuffle}
            }
            batch_meta.update_extra_info(extra_meta)
            actor_output_meta = self.actor_rollout_wg.update_actor(batch_meta)
            actor_output = actor_output_meta.extra_info.get("metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output_meta.set_extra_info("metrics", actor_output)
        else:
            actor_output_meta = self.actor_rollout_wg.update_actor(batch)
        return actor_output_meta

    def _update_critic(self, batch_meta: BatchMeta) -> BatchMeta:
        if self.use_legacy_worker_impl == "disable":
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            extra_meta = {
                "global_batch_size": ppo_mini_batch_size,
                "mini_batch_size": ppo_mini_batch_size,
                "epochs": ppo_epochs,
                "seed": seed,
                "dataloader_kwargs": {"shuffle": shuffle},
            }
            batch_meta.update_extra_info(extra_meta)
            critic_output_meta = self.critic_wg.train_mini_batch(batch_meta)
            output = critic_output_meta.extra_info.get("metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output_meta.set_extra_info("metrics", output)
        else:
            critic_output_meta = self.critic_wg.update_critic(batch)
        return critic_output_meta

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                base_get_meta_kwargs = dict(
                    batch_size=self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n,
                    partition_id=f"train_{self.global_steps - 1}",  # self.global_steps starts from 1
                )

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # add uid to batch
                batch_dict["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
                )
                # When n > 1, repeat input data before putting to data system, simulating DataProto repeat.
                repeated_batch_dict = self.repeat_dict(
                    batch_dict, repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                batch: TensorDict = tu.dict_to_tensordict(repeated_batch_dict)

                batch_meta = self.tq_client.put(data=batch, partition_id=f"train_{self.global_steps - 1}")
                batch_meta.set_extra_info("temperature", self.config.actor_rollout_ref.rollout.temperature)

                gen_batch_fields = self._get_gen_batch_fields(tu.get_non_tensor_keys(batch))
                gen_meta = batch_meta.select_fields(list(gen_batch_fields))

                # pass global_steps to trace
                gen_meta.set_extra_info("global_steps", self.global_steps)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_output_meta = self.actor_rollout_wg.generate_sequences(gen_meta)
                        else:
                            gen_output_meta = self.async_rollout_manager.generate_sequences(gen_meta)
                        timing_raw.update(gen_output_meta.extra_info["timing"])
                        gen_output_meta.extra_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_meta = deepcopy(gen_meta)
                            gen_baseline_meta.set_extra_info("do_sample", False)
                            if not self.async_rollout_mode:
                                gen_baseline_output_meta = self.actor_rollout_wg.generate_sequences(gen_baseline_meta)
                            else:
                                gen_baseline_output_meta = self.async_rollout_manager.generate_sequences(
                                    gen_baseline_meta)
                            batch_meta = batch_meta.union(gen_baseline_output_meta)
                            # compute reward model score on batch
                            rm_scores_output_meta = None
                            if self.use_rm and "rm_scores" not in batch_meta.field_names:
                                if not self.use_reward_loop:
                                    rm_scores_output_meta = self.rm_wg.compute_rm_score(batch_meta)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores_output_meta = self.reward_loop_manager.compute_rm_score(batch_meta)
                                batch_meta = batch_meta.union(rm_scores_output_meta)

                            compute_reward_fields = [
                                "responses",
                                "prompts",
                                "attention_mask",
                                "reward_model",
                                "data_source",
                                "rm_scores"
                            ]
                            compute_reward_meta = batch_meta.select_fields(compute_reward_fields)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                compute_reward_meta, reward_fn=self.reward_fn, sum_reward=True
                            )

                            reward_baseline_tensor_td = TensorDict({"reward_baselines": reward_baseline_tensor},
                                                                   batch_size=reward_baseline_tensor.size(0))
                            batch_meta = self.tq_client.put(data=reward_baseline_tensor_td, metadata=batch_meta)

                            keys_to_pop = set(gen_baseline_output_meta.field_names)
                            if rm_scores_output_meta is not None:
                                keys_to_pop.update(rm_scores_output_meta.field_names)
                            keys_to_retain = list(set(batch_meta.field_names) - keys_to_pop)
                            batch_meta = batch_meta.select_fields(keys_to_retain)
                            del rm_scores_output_meta, gen_baseline_meta, gen_baseline_output_meta

                    batch_meta = batch_meta.union(gen_output_meta)

                    # follow wuxibin's opinion and delete all response_mask related extra operations
                    # cuz response_mask should be put in TQ when generating sequences
                    assert "response_mask" in batch_meta.field_names

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        attention_mask_meta = batch_meta.select_fields(["attention_mask"])
                        balanced_idx = self._balance_batch(attention_mask_meta, metrics=metrics)
                        batch_meta.reorder(balanced_idx)

                    # compute global_valid tokens
                    data = self.tq_client.get_data(attention_mask_meta)
                    batch_meta.extra_info["global_token_num"] = torch.sum(data["attention_mask"], dim=-1).tolist()

                    # get images_seqlens
                    images_seqlens_all = []
                    images_seqlens_meta = batch_meta.select_fields(["multi_modal_inputs"])
                    data = self.tq_client.get_data(images_seqlens_meta)  # non tensor
                    for multi_modal_input in data:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch_meta.extra_info["images_seqlens"] = images_seqlens_all

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch_meta.field_names:
                            if not self.use_reward_loop:
                                reward_tensor_meta = self.rm_wg.compute_rm_score(batch_meta)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor_meta = self.reward_loop_manager.compute_rm_score(batch_meta)
                            batch_meta = batch_meta.union(reward_tensor_meta)

                        compute_reward_fields = [
                            "responses",
                            "prompts",
                            "attention_mask",
                            "reward_model",
                            "data_source",
                        ]
                        if "rm_scores" in batch_meta.field_names:
                            compute_reward_fields.append("rm_scores")

                        compute_reward_meta = batch_meta.select_fields(compute_reward_fields)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async_decorated(
                                data=compute_reward_meta, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                compute_reward_meta, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
                        old_log_prob_bypass_meta = batch_meta.select_fields(["rollout_log_probs"])

                        old_log_prob_output_meta = apply_bypass_mode(
                            batch=old_log_prob_bypass_meta,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                        batch_meta = batch_meta.union(old_log_prob_output_meta)
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob_meta_fields = [
                                "input_ids",
                                "attention_mask",
                                "position_ids",
                                "prompts",
                                "responses",
                                "response_mask",
                                "data_source",
                                "reward_model",
                                "extra_info",
                                "uid",
                                "index",
                                "tools_kwargs",
                                "interaction_kwargs",
                                "ability",
                            ]
                            old_log_prob_meta = batch_meta.select_fields(old_log_prob_meta_fields)
                            old_log_prob_output_meta, old_log_prob_mfu = self._compute_old_log_prob(old_log_prob_meta)

                            data = self.tq_client.get_data(old_log_prob_output_meta.select_fields["log_probs"])
                            old_log_probs = TensorDict(
                                {"old_log_probs": data["log_probs"]},
                                batch_size=data["log_probs"].size(0),
                            )
                            old_log_prob_output_meta = self.tq_client.put(data=old_log_probs,
                                                                          metadata=old_log_prob_output_meta)
                            old_log_prob_output_fields = ["response_mask", "old_log_probs", "entropys"]
                            data = self.tq_client.get_data(batch_meta.select_fields(old_log_prob_output_fields))
                            entropys = data["entropys"]
                            response_masks = data["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob_output_meta = old_log_prob_output_meta.select_fields(["old_log_probs"])
                            batch_meta = batch_meta.union(old_log_prob_output_meta)

                            if "rollout_log_probs" in batch_meta.field_names:
                                # TODO: we may want to add diff of probs too.
                                calculate_debug_metrics_fields = ["rollout_log_probs", "old_log_probs", "responses"]

                                if "response_mask" in batch_meta.field_names:
                                    calculate_debug_metrics_fields.append("response_mask")
                                if "attention_mask" in batch_meta.field_names:
                                    calculate_debug_metrics_fields.append("attention_mask")

                                calculate_debug_metrics_meta = batch_meta.select_fields(calculate_debug_metrics_fields)
                                metrics.update(calculate_debug_metrics_decorated(calculate_debug_metrics_meta))

                    assert "old_log_probs" in batch_meta.field_names, f'"old_log_probs" not in {batch_meta.field_names=}'
                    if self.use_reference_policy:
                        # compute reference log_prob
                        ref_log_prob_fields = [
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                            "prompts",
                            "responses",
                            "response_mask",
                            "old_log_probs",
                            "data_source",
                            "reward_model",
                            "extra_info",
                            "uid",
                            "index",
                            "tools_kwargs",
                            "interaction_kwargs",
                            "ability",
                        ]
                        ref_log_prob_meta = batch_meta.select_fields(ref_log_prob_fields)
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob_output_meta = self._compute_ref_log_prob(ref_log_prob_meta)
                            data = self.tq_client.get_data(ref_log_prob_output_meta.select_fields["log_probs"])
                            ref_log_probs = TensorDict(
                                {"ref_log_prob": data["log_probs"]},
                                batch_size=data["log_probs"].size(0),
                            )
                            ref_log_prob_output_meta = self.tq_client.put(data=ref_log_probs,
                                                                          metadata=ref_log_prob_output_meta)
                            batch_meta = batch_meta.union(ref_log_prob_output_meta)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values_meta = self._compute_values(batch_meta)
                            batch_meta = batch_meta.union(values_meta)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        reward_td = TensorDict({"token_level_scores": reward_tensor}, batch_size=reward_tensor.size(0))
                        batch_meta = self.tq_client.put(data=reward_td, metadata=batch_meta)

                        if reward_extra_infos_dict:
                            reward_extra_infos_dict_new = {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            reward_extra_infos_td = tu.dict_to_tensordict(reward_extra_infos_dict_new)
                            batch_meta = self.tq_client.put(data=reward_extra_infos_td, metadata=batch_meta)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            apply_kl_penalty_fields = [
                                "response_mask",
                                "token_level_scores",
                                "old_log_probs",
                                "ref_log_prob",
                            ]
                            apply_kl_penalty_meta = batch_meta.select_fields(apply_kl_penalty_fields)
                            token_level_rewards, kl_metrics = apply_kl_penalty(
                                apply_kl_penalty_meta,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty
                            )
                            token_level_rewards_td = TensorDict(
                                {"token_level_rewards": token_level_rewards}, batch_size=token_level_rewards.size(0)
                            )
                            apply_kl_penalty_meta = self.tq_client.put(data=token_level_rewards_td,
                                                                       metadata=apply_kl_penalty_meta)

                            metrics.update(kl_metrics)
                            batch_meta = batch_meta.union(apply_kl_penalty_meta)
                        else:
                            token_level_scores_meta = batch_meta.select_fields(["token_level_scores"])
                            data = self.tq_client.get_data(token_level_scores_meta)
                            token_level_rewards_td = TensorDict(
                                {"token_level_rewards": data["token_level_scores"]},
                                batch_size=data["token_level_scores"].size(0),
                            )
                            token_level_scores_meta = self.tq_client.put(data=token_level_rewards_td,
                                                                         metadata=token_level_scores_meta)
                            batch_meta = batch_meta.union(token_level_scores_meta)

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch_meta.field_names
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            rollout_correction_meta = ["old_log_probs",
                                                       "rollout_log_probs",
                                                       "response_mask"
                                                       ]
                            data, is_metrics = compute_rollout_correction_and_add_to_batch(
                                rollout_correction_meta, rollout_corr_config)  # data is a dataproto
                            correction_td = TensorDict(
                                {
                                    "response_mask": data.batch["response_mask"],
                                    "rollout_is_weights": data.batch["rollout_is_weights"]
                                },
                                batch_size=data.batch["rollout_is_weights"].size(0)
                            )
                            batch_meta = self.tq_client.put(data=correction_td, metadata=batch_meta)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        assert "response_mask" in batch_meta.field_names

                        compute_advantage_fields = [
                            "response_mask",
                            "token_level_rewards",
                        ]
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
                            compute_advantage_fields.append("values")
                        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                            compute_advantage_fields.append("uid")
                        else:
                            if "uid" in batch_meta.field_names:
                                compute_advantage_fields.append("uid")
                            if "reward_baselines" in batch_meta.field_names:
                                compute_advantage_fields.append("reward_baselines")

                        compute_advantage_meta = batch_meta.select_fields(compute_advantage_fields)
                        compute_advantage_meta = compute_advantage(
                            compute_advantage_meta,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        batch_meta = batch_meta.union(compute_advantage_meta)

                        if "resampled_idx" in batch_meta.field_names and self.config.transferqueue.enable:
                            resample_idx_meta = batch_meta.select_fields(["pf_ppo_reweight_idx"])
                            resampled_idx = self.tq_client.get_data(resample_idx_meta)  # list of int
                            full_batch_meta = copy(batch_meta)
                            batch_meta = full_batch_meta.select_samples(resampled_idx)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output_meta = self._update_critic(batch_meta)
                            batch_meta = batch_meta.union(critic_output_meta)
                        critic_output_metrics = reduce_metrics(critic_output_meta.extra_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            update_actor_fields = [
                                "input_ids",
                                "attention_mask",
                                "position_ids",
                                "prompts",
                                "responses",
                                "response_mask",
                                "old_log_probs",
                                "ref_log_prob",
                                "advantages",
                                "returns",
                                "token_level_rewards",
                                "token_level_scores",
                                "data_source",
                                "reward_model",
                                "extra_info",
                                "uid",
                                "index",
                                "tools_kwargs",
                                "interaction_kwargs",
                                "ability",
                            ]
                            update_actor_meta = batch_meta.select_fields(update_actor_fields)
                            actor_output_meta = self._update_actor(update_actor_meta)
                            # batch_meta = batch_meta.union(actor_output_meta)
                        actor_output_metrics = reduce_metrics(actor_output_meta.extra_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        log_rollout_fields = ["prompts", "responses", "token_level_scores", "reward_model"]
                        if "request_id" in batch_meta.field_names:
                            log_rollout_fields.append("request_id")
                        log_rollout_meta = batch_meta.select_fields(log_rollout_fields)
                        self._log_rollout_data(log_rollout_meta, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                compute_data_metrics_fields = [
                    "token_level_rewards",
                    "token_level_scores",
                    "advantages",
                    "returns",
                    "responses",
                    "attention_mask",
                    "response_mask",
                ]
                if "__num_turns__" in batch_meta.field_names:
                    compute_data_metrics_fields.append("__num_turns__")
                if "tool_call_counts" in batch_meta.field_names:
                    compute_data_metrics_fields.append("tool_call_counts")
                if self.use_critic:
                    compute_data_metrics_fields.append("values")

                compute_data_metrics_meta = batch_meta.select_fields(compute_data_metrics_fields)
                metrics.update(
                    compute_data_metrics_decorated(batch=compute_data_metrics_meta, use_critic=self.use_critic)
                )

                compute_timing_metrics_fields = ["responses", "attention_mask"]

                compute_timing_metrics_meta = batch_meta.select_fields(compute_timing_metrics_fields)
                metrics.update(
                    compute_timing_metrics_decorated(batch=compute_timing_metrics_meta, timing_raw=timing_raw)
                )

                compute_throughout_metrics_meta = BatchMeta(
                    samples=[],
                    extra_info={"global_token_num": batch_meta.get_extra_info("global_token_num")},
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics_decorated(
                        batch=compute_throughout_metrics_meta, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    # TODO (TQ) :support transfer queue
                    self.train_dataloader.sampler.update(batch=batch)

                self.tq_client.clear_samples(full_batch_meta)
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    # TODO (TQ): support transfer queue
                    self.train_dataset.on_batch_end(batch=batch)
