#!/usr/bin/env python3

# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import sys
import time
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import torch
from tensordict import TensorDict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from recipe.fully_async_policy.detach_utils import RolloutSample, assemble_batch_from_rollout_samples
from verl import DataProto


@dataclass
class MockAgentLoopMetrics:
    """Mock AgentLoopMetrics for testing"""

    generate_sequences: float = 0.5
    tool_calls: float = 0.0


@dataclass
class MockAgentLoopOutput:
    """Mock AgentLoopOutput for testing"""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int = 1
    metrics: MockAgentLoopMetrics = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = MockAgentLoopMetrics()


class MockConfig:
    """Mock configuration object"""

    def __init__(self):
        self.trainer = MockTrainerConfig()


class MockTrainerConfig:
    """Mock trainer configuration"""

    def __init__(self):
        self.balance_batch = False


class TestBatchUtils(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.tokenizer = MagicMock()
        self.config = MockConfig()

        # Mock postprocess_agent_loop_outputs function
        self.mock_postprocess = MagicMock()

        # Patch the postprocess function
        import recipe.fully_async_policy.detach_utils as detach_utils_module

        self.original_postprocess = detach_utils_module.postprocess_agent_loop_outputs
        detach_utils_module.postprocess_agent_loop_outputs = self.mock_postprocess

        # Mock compute_response_mask function
        self.original_compute_response_mask = detach_utils_module.compute_response_mask
        detach_utils_module.compute_response_mask = MagicMock(return_value=torch.ones(2, 128, dtype=torch.int64))

    def tearDown(self):
        """清理测试环境"""
        import recipe.fully_async_policy.detach_utils as detach_utils_module

        detach_utils_module.postprocess_agent_loop_outputs = self.original_postprocess
        detach_utils_module.compute_response_mask = self.original_compute_response_mask

    def create_mock_rollout_sample(self, sample_id: str, param_version: int = 1) -> RolloutSample:
        """创建测试用的 RolloutSample"""
        # 创建 mock AgentLoopOutput
        agent_loop_output = MockAgentLoopOutput(
            prompt_ids=torch.randint(0, 32000, (175,)).tolist(),
            response_ids=torch.randint(0, 32000, (175,)).tolist(),
            response_mask=[1] * 175,  # 真实的response长度
            num_turns=2,
            metrics=MockAgentLoopMetrics(generate_sequences=1.6468379497528076, tool_calls=0.0),
        )

        # 创建mock _gen_data
        mock_gen_data = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array(
                    [
                        [
                            {
                                "content": "Tom receives a $12 allowance per month.",
                                "role": "user",
                            }
                        ]
                    ],
                    dtype=object,
                ),
                "tools_kwargs": np.array([{}], dtype=object),
                "interaction_kwargs": np.array([{}], dtype=object),
                "index": np.array([4570], dtype=object),
            },
            meta_info={"global_steps": 1},
        )

        return RolloutSample(
            full_batch=mock_gen_data,
            agent_loop_output=agent_loop_output,
            sample_id=sample_id,
            epoch=0,
            rollout_n_index=0,
            original_sample_index=0,
            processing_time=1.6468379497528076,
            generation_timestamp=time.time(),
            param_version=param_version,
        )

    # def test_assemble_batch_empty_input(self):
    #     """测试空输入的情况"""
    #     with self.assertRaises(ValueError) as context:
    #         assemble_batch_from_rollout_samples([], self.tokenizer, self.config)
    #
    #     self.assertIn("Empty rollout_samples", str(context.exception))
    #
    # def test_assemble_batch_single_sample(self):
    #     """测试单个样本的批次组装"""
    #     # 设置mock返回值 - 使用正确的TensorDict格式
    #     mock_gen_batch = DataProto(
    #         batch=TensorDict({
    #             "input_ids": torch.randint(0, 1000, (1, 256)),
    #             "attention_mask": torch.ones(1, 256, dtype=torch.int64),
    #             "position_ids": torch.arange(256).unsqueeze(0),
    #             "prompts": torch.randint(0, 1000, (1, 128)),
    #             "responses": torch.randint(0, 1000, (1, 128)),
    #             "response_mask": torch.ones(1, 128, dtype=torch.int64),
    #         }, batch_size=1),
    #         non_tensor_batch={"__test_key": np.array(["test_value"], dtype=object)},
    #         meta_info={"test_meta": "test_value"}
    #     )
    #     self.mock_postprocess.return_value = mock_gen_batch
    #
    #     # 创建测试样本
    #     rollout_samples = [self.create_mock_rollout_sample("sample_1")]
    #
    #     # 调用函数
    #     result = assemble_batch_from_rollout_samples(
    #         rollout_samples=rollout_samples,
    #         tokenizer=self.tokenizer,
    #         config=self.config
    #     )
    #
    #     # 验证结果
    #     self.assertIsInstance(result, DataProto)
    #     self.assertIn("uid", result.non_tensor_batch)
    #     self.assertEqual(result.non_tensor_batch["uid"][0], "uid_sample_1")
    #
    #     # 验证meta_info包含预期字段
    #     expected_fields = [
    #         "rollout_param_versions", "sample_timestamps", "avg_processing_time",
    #         "max_processing_time", "param_version_diversity", "avg_sample_age", "assembly_time"
    #     ]
    #     for field in expected_fields:
    #         self.assertIn(field, result.meta_info)
    #
    #     # 验证统计信息
    #     self.assertEqual(result.meta_info["rollout_param_versions"], [1])
    #     self.assertAlmostEqual(result.meta_info["avg_processing_time"], 1.6468379497528076, places=5)
    #     self.assertEqual(result.meta_info["param_version_diversity"], 1)

    def test_assemble_batch_multiple_samples(self):
        """测试多个样本的批次组装"""
        # 设置mock返回值 - 使用正确的TensorDict格式
        mock_gen_batch = DataProto(
            batch=TensorDict(
                {
                    "input_ids": torch.randint(0, 1000, (2, 256)),
                    "attention_mask": torch.ones(2, 256, dtype=torch.int64),
                    "position_ids": torch.arange(256).unsqueeze(0).repeat(2, 1),
                    "prompts": torch.randint(0, 1000, (2, 128)),
                    "responses": torch.randint(0, 1000, (2, 128)),
                    "response_mask": torch.ones(2, 128, dtype=torch.int64),
                },
                batch_size=2,
            ),
            non_tensor_batch={"__test_key": np.array(["test_value1", "test_value2"], dtype=object)},
            meta_info={"test_meta": "test_value"},
        )
        self.mock_postprocess.return_value = mock_gen_batch

        # 创建测试样本
        rollout_samples = [
            self.create_mock_rollout_sample("sample_1", param_version=1),
            self.create_mock_rollout_sample("sample_2", param_version=2),
        ]

        print(rollout_samples)

        # 调用函数
        result = assemble_batch_from_rollout_samples(
            rollout_samples=rollout_samples, tokenizer=self.tokenizer, config=self.config
        )

        # 验证结果
        self.assertIsInstance(result, DataProto)
        self.assertEqual(len(result.non_tensor_batch["uid"]), 2)
        self.assertListEqual(list(result.non_tensor_batch["uid"]), ["uid_sample_1", "uid_sample_2"])

        # 验证多样本统计
        self.assertEqual(result.meta_info["rollout_param_versions"], [1, 2])
        self.assertEqual(result.meta_info["param_version_diversity"], 2)  # 两个不同版本
        self.assertAlmostEqual(result.meta_info["avg_processing_time"], 1.6468379497528076, places=5)

    # def test_assemble_batch_with_balance_batch_flag(self):
    #     """测试启用balance_batch标志的情况"""
    #     # 设置mock返回值 - 使用正确的TensorDict格式
    #     mock_gen_batch = DataProto(
    #         batch=TensorDict({
    #             "input_ids": torch.randint(0, 1000, (1, 256)),
    #             "attention_mask": torch.ones(1, 256, dtype=torch.int64),
    #             "position_ids": torch.arange(256).unsqueeze(0),
    #             "prompts": torch.randint(0, 1000, (1, 128)),
    #             "responses": torch.randint(0, 1000, (1, 128)),
    #             "response_mask": torch.ones(1, 128, dtype=torch.int64),
    #         }, batch_size=1),
    #         non_tensor_batch={"__test_key": np.array(["test_value"], dtype=object)},
    #         meta_info={"test_meta": "test_value"}
    #     )
    #     self.mock_postprocess.return_value = mock_gen_batch
    #
    #     # 设置config启用balance_batch
    #     self.config.trainer.balance_batch = True
    #
    #     # 创建测试样本
    #     rollout_samples = [self.create_mock_rollout_sample("sample_1")]
    #
    #     # 调用函数
    #     result = assemble_batch_from_rollout_samples(
    #         rollout_samples=rollout_samples,
    #         tokenizer=self.tokenizer,
    #         config=self.config,
    #         balance_batch=True
    #     )
    #
    #     # 验证结果（主要验证没有抛出异常）
    #     self.assertIsInstance(result, DataProto)
    #
    # def test_assemble_batch_attention_mask_processing(self):
    #     """测试attention_mask处理逻辑"""
    #     # 设置mock返回值 - 使用正确的TensorDict格式
    #     mock_gen_batch = DataProto(
    #         batch=TensorDict({
    #             "input_ids": torch.randint(0, 1000, (2, 256)),
    #             "attention_mask": torch.ones(2, 256, dtype=torch.int64),
    #             "position_ids": torch.arange(256).unsqueeze(0).repeat(2, 1),
    #             "prompts": torch.randint(0, 1000, (2, 128)),
    #             "responses": torch.randint(0, 1000, (2, 128)),
    #             "response_mask": torch.ones(2, 128, dtype=torch.int64),
    #         }, batch_size=2),
    #         non_tensor_batch={"__test_key": np.array(["test_value1", "test_value2"], dtype=object)},
    #         meta_info={"test_meta": "test_value"}
    #     )
    #     self.mock_postprocess.return_value = mock_gen_batch
    #
    #     # 创建测试样本
    #     rollout_samples = [
    #         self.create_mock_rollout_sample("sample_1"),
    #         self.create_mock_rollout_sample("sample_2"),
    #     ]
    #
    #     # 调用函数
    #     result = assemble_batch_from_rollout_samples(
    #         rollout_samples=rollout_samples,
    #         tokenizer=self.tokenizer,
    #         config=self.config
    #     )
    #
    #     # 验证global_token_num被正确计算
    #     self.assertIn("global_token_num", result.meta_info)
    #     self.assertIsInstance(result.meta_info["global_token_num"], list)
    #
    # def test_mock_postprocess_called_correctly(self):
    #     """测试postprocess_agent_loop_outputs被正确调用"""
    #     # 设置mock返回值 - 使用正确的TensorDict格式
    #     mock_gen_batch = DataProto(
    #         batch=TensorDict({
    #             "input_ids": torch.randint(0, 1000, (1, 256)),
    #             "attention_mask": torch.ones(1, 256, dtype=torch.int64),
    #             "position_ids": torch.arange(256).unsqueeze(0),
    #             "prompts": torch.randint(0, 1000, (1, 128)),
    #             "responses": torch.randint(0, 1000, (1, 128)),
    #             "response_mask": torch.ones(1, 128, dtype=torch.int64),
    #         }, batch_size=1),
    #         non_tensor_batch={"__test_key": np.array(["test_value"], dtype=object)},
    #         meta_info={"test_meta": "test_value"}
    #     )
    #     self.mock_postprocess.return_value = mock_gen_batch
    #
    #     # 创建测试样本
    #     rollout_samples = [self.create_mock_rollout_sample("sample_1")]
    #
    #     # 调用函数
    #     result = assemble_batch_from_rollout_samples(
    #         rollout_samples=rollout_samples,
    #         tokenizer=self.tokenizer,
    #         config=self.config
    #     )
    #
    #     # 验证postprocess_agent_loop_outputs被调用
    #     self.mock_postprocess.assert_called_once()
    #     call_args = self.mock_postprocess.call_args
    #
    #     # 验证调用参数
    #     agent_loop_outputs, tokenizer, config = call_args[0]
    #     self.assertEqual(len(agent_loop_outputs), 1)
    #     self.assertEqual(tokenizer, self.tokenizer)
    #     self.assertEqual(config, self.config)
    #


if __name__ == "__main__":
    unittest.main()
