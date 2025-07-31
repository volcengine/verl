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
import json
import os
import requests
from typing import Any, Optional
from PIL import Image
from uuid import uuid4

import numpy as np
import pytest
import ray
from omegaconf import DictConfig
from transformers.utils import get_json_schema

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.experimental.agent_loop.agent_loop import get_trajectory_info
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.utils import hf_processor


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.custom_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{{- messages[0]['content'] }}{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}{% for message in messages %}{% if message['role'] != 'system' or loop.first == false %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endif %}{% endfor %}{%- else %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.agent.num_workers = 2

    return config


def test_single_turn(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    agent_loop_manager = init_agent_loop_manager(init_config)

    raw_prompts = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is the animal in the image?"},
                ],
            }            
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }            
        ],
    ]
    multi_modal_data = [
        {'image': [Image.open(requests.get('https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg', stream=True).raw)]},
        {'image': [Image.open(requests.get('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg', stream=True).raw)]}
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "multi_modal_data": np.array(multi_modal_data),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()


class VerifyTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "verify_object",
                "description": "A tool for verifying the detected object in the image",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "description": "The object detected in the image.",
                        },
                    },
                    "required": ["object"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def verify_object(self, instance_id: str):
        """Verify the detected object.

        Args:
            object: The object detected in the image.

        Returns:
            the verify, which is True if the object is the same as the ground truth, otherwise False.
        """
        print(f"[DEBUG] verify_object: {object}")
        return {
            "verify": self._instance_dict[instance_id]["response"] == self._instance_dict[instance_id]["ground_truth"],
        }

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        try:
            answer = parameters.get("object", "")
            if not isinstance(answer, str):
                answer = str(answer)
            self._instance_dict[instance_id]["response"] = answer
            result = self.verify_object(instance_id)
            return json.dumps(result), int(result['verify']), {}
        except Exception as e:
            return str(e), 0, {}

def test_tool_agent(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    tool_config = {
        "tools": [
            {
                "class_name": "tests.experimental.agent_loop.test_basic_multimodal_agent_loop.VerifyTool",
                "config": {"type": "native"},
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": "verify_object",
                        "description": "A tool for verifying the object detected in the image.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "object": {
                                    "type": "string",
                                    "description": "The object detected in the image."
                                }
                            },
                            "required": ["object"]
                        }
                    }
                }
            },
        ]
    }
    tool_config_path = "/tmp/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 2
    agent_loop_manager = init_agent_loop_manager(init_config)

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are a helpful assistant to detect objects in images. "
                    "After detecting the object, you MUST use `verify_object` tool to verify your answer. "
                )}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is the animal in the image?"},
                ],
            }
        ],
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are a helpful assistant to detect objects in images. "
                    "After detecting the object, you MUST use `verify_object` tool to verify your answer. "
                )}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is the main object in the image?"},
                ],
            }
        ],
    ]
    multi_modal_data = [
        {'image': [Image.open(requests.get('https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg', stream=True).raw)]},
        {'image': [Image.open(requests.get('https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg', stream=True).raw)]}
    ]

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "multi_modal_data": np.array(multi_modal_data),
            'tools_kwargs': np.array([{'verify_object': {'create_kwargs': {'ground_truth': 'rabbit'}}},
                                       {'verify_object': {'create_kwargs': {'ground_truth': 'cat'}}}]),
            "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    processor = hf_processor(init_config.actor_rollout_ref.model.path)
    custom_chat_template = init_config.actor_rollout_ref.model.get("custom_chat_template", None)
    if custom_chat_template is not None:
        processor.chat_template = custom_chat_template
        processor.tokenizer.chat_template = custom_chat_template
    responses = result.batch["responses"]

    # Check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    # At least one of the responses calls the tool.
    # [user, assistant, tool, assistant]
    assert 4 in num_turns

    # Check response_mask
    processor = hf_processor(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"
    response_length = response_mask.size(1)

    for i in range(len(responses)):
        # response with tool response
        valid_tokens = responses[i][attention_mask[i][-response_length:].bool()]
        response_with_obs = processor.decode(valid_tokens)

        # response without tool response
        valid_tokens = responses[i][response_mask[i].bool()]
        response_without_obs = processor.decode(valid_tokens)

        assert "<tool_response>" not in response_without_obs, (
            f"found <tool_response> in response: {response_without_obs}"
        )
        assert "</tool_response>" not in response_without_obs, (
            f"found </tool_response> in response: {response_without_obs}"
        )
        print("=========================")
        print(response_with_obs)
        print("---")
        print(response_without_obs)

    print("Test passed!")
    ray.shutdown()


@pytest.mark.asyncio
async def test_get_trajectory_info():
    """Tests the get_trajectory_info method."""
    # Initialize the class to set up class-level attributes
    step = 10
    index = [1, 1, 3, 3]
    expected_info = [
        {"step": step, "sample_index": 1, "rollout_n": 0, "validate": False},
        {"step": step, "sample_index": 1, "rollout_n": 1, "validate": False},
        {"step": step, "sample_index": 3, "rollout_n": 0, "validate": False},
        {"step": step, "sample_index": 3, "rollout_n": 1, "validate": False},
    ]

    trajectory_info = await get_trajectory_info(step, index, validate=False)

    assert trajectory_info == expected_info
