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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import asyncio
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import (
    TaskRunner as MainTaskRunner,
)
from verl.trainer.main_ppo import (
    create_rl_dataset,
    create_rl_sampler,
)
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_ascend_device_name, is_cuda_available

from .ray_trainer import RayPPOTrainer

import torch
import tensordict
import numpy as np
from tensordict import TensorDict
from packaging.version import parse as parse_version


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_ascend_device_name(config)

    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner(MainTaskRunner):
    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_batch_sampler(config.data, train_dataset, drop_last=True)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=tq_collect_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()





def repeat_dict(
    batch_dict: dict[str, torch.Tensor | np.ndarray], repeat_times=2, interleave=True
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


def dict_to_tensordict(data: dict[str, torch.Tensor | np.ndarray]) -> TensorDict:
        """
        Create a TensorDict from a dict of tensors and non_tensors.
        Note that this requires tensordict version at least 0.10
        """
        assert parse_version(tensordict.__version__) >= parse_version("0.10"), (
            "Storing non-tensor data in TensorDict at least requires tensordict version 0.10"
        )
        tensors_batch = {}
        batch_size = None

        for key, val in data.items():
            if isinstance(val, torch.Tensor | np.ndarray):
                tensors_batch[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

            if batch_size is None:
                batch_size = len(val)
            else:
                assert len(val) == batch_size

        if batch_size is None:
            batch_size = []
        else:
            batch_size = [batch_size]

        return TensorDict(tensors_batch, batch_size=batch_size)


class BatchSamplerWithId(BatchSampler):
    def __iter__(self):
        for bid, batch in enumerate(super().__iter__()):
            yield [(bid, idx) for idx in batch]


def create_rl_batch_sampler(data_config, dataset, drop_last):
    from verl.trainer.main_ppo import create_rl_sampler

    base_sampler = create_rl_sampler(data_config, dataset)
    batch_sampler = BatchSamplerWithId(
        sampler=base_sampler,
        batch_size=data_config.get("gen_batch_size", data_config.train_batch_size),
        drop_last=drop_last,
    )

    return batch_sampler


def tq_collect_fn(batch, config, prefix="train_"):
    import uuid

    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.transferqueue_utils import (
        create_transferqueue_client,
        get_transferqueue_client,
    )

    create_transferqueue_client(
        client_id="data_process",
        config=config.transfer_queue,
        enforce=True
    )
    tq_client = get_transferqueue_client()

    batch_dict = collate_fn(batch)
    partition_id = batch_dict.pop("batch_id")[0]
    
    batch_dict["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch_dict["input_ids"]))], dtype=object
    )

    batch_dict = repeat_dict(
        batch_dict, repeat_times=config.actor_rollout_ref.rollout.n, interleave=True
    )
    batch_dict: TensorDict = dict_to_tensordict(batch_dict)
    asyncio.run(tq_client.async_put(data=batch_dict, partition_id=f"{prefix}{partition_id}"))

    return list(batch_dict.keys())

if __name__ == "__main__":
    main()
