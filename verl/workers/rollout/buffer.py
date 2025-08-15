import copy
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import ray
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset


class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


class Buffer:
    def __init__(self, config):
        # init_wandb_secondary(args, wandb_run_id)
        self.config = config

        # 数据源相关属性
        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        # 初始化tokenizer和processor
        local_path = self.config.actor_rollout_ref.model.path
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=True, use_fast=True)

        # 加载RLHF数据集
        rldataset = RLHFDataset(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.dataset = []
        for item in tqdm(rldataset, desc="Loading RLHF dataset", total=len(rldataset)):
            self.dataset.append(item)

        self.n_samples_per_prompt = self.config.actor_rollout_ref.rollout.n

        self.buffer: List[List[Dict]] = []

    def get_num_rollout_per_epoch(self):
        return len(self.dataset) //self.config.actor_rollout_ref.rollout.rollout_batch_size

    def _get_samples_from_data_source(self, num_samples: int) -> List[List[Dict]]:
        """从数据源获取样本，整合了原RolloutDataSource.get_samples的逻辑"""
        samples = []
        # TODO unify the two branches
        if self.sample_offset + num_samples <= len(self.dataset):
            prompt_samples = self.dataset[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            prompt_samples = self.dataset[self.sample_offset :]
            num_samples -= len(prompt_samples)
            # self.epoch_id += 1
            # if self.args.rollout_shuffle:
            #     self.dataset.shuffle(self.epoch_id)
            prompt_samples += self.dataset[:num_samples]
            self.sample_offset = num_samples
        # self.sample_offset = 0
        

        for prompt_sample in prompt_samples:
            group = []
            prompt_sample["status"] = Status.PENDING
            prompt_sample["response_length"] = 0
            prompt_sample["response"] = []
            prompt_sample["uid"] = str(uuid.uuid4())

            for _ in range(self.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                group.append(sample)
            samples.append(group)

        return samples

    # TODO simplify remaining logic
    def get_samples(self, num_samples: int) -> List[List[Dict]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += self._get_samples_from_data_source(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> List[List[Dict]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []
        num_to_pop = min(len(self.buffer), num_samples)
        samples = self.buffer[:num_to_pop]
        del self.buffer[:num_to_pop]
        return samples

    def add_samples(self, samples: List[List[Dict]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert len(samples[i]) == self.n_samples_per_prompt, (
                f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.n_samples_per_prompt}"
            )
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    def get_buffer_length(self):
        return len(self.buffer)
