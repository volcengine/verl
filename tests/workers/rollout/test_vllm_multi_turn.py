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
import asyncio
import json
from typing import Any, Dict

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse, ErrorResponse

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto


def init_config() -> DictConfig:
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "Qwen/Qwen2-7B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_vllm_multi_turn(config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])
    async_rollout_manager = init_async_rollout_manager(config)

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    async_chat_scheduler = async_rollout_manager.chat_scheduler

    # =========================== 2. Multi turn rollout  ===========================
    async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
        assert exception is None, f"exception: {exception}"
        messages, round = info["messages"], info["round"]
        message = completions.choices[0].message
        messages.append({"role": message.role, "content": message.content})
        print(f"[round={round}] role: {message.role}, content: {message.content}")

        extra_headers = {"x-request-id": completions.id}
        if round == 0:
            messages.append({"role": "user", "content": "What is your name?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 1},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        elif round == 1:
            messages.append({"role": "user", "content": "What is your favorite color?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 2},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        else:
            print("Done!")

    messages = [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}]
    async_rollout_manager.submit_chat_completions(
        callback=callback,
        callback_additional_info={"messages": messages, "round": 0},
        model=model_name,
        messages=messages,
    )
    assert len(messages) == 6
    for round, message in enumerate(messages):
        if round % 2 == 0:
            assert message["role"] == "user"
        else:
            assert message["role"] == "assistant"

    # =========================== 3. Generate sequences  ===========================
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
        },
    )
    result = async_rollout_manager.generate_sequences(prompts=batch)
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert len(result) == 2
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    ray.shutdown()


async def test_vllm_streaming_response(config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])
    async_rollout_manager = init_async_rollout_manager(config)
    async_llm_server = async_rollout_manager.async_llm_servers[0]

    # non-streaming request
    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "What is your name?"}],
        stream=False,
    )
    generator = async_llm_server.chat_completion_generator.remote(request)
    async for ref in generator:
        status_code, data = await ref
        print(f">>>> status_code: {status_code}, {data}")
        data = data[len("data: ") :].rstrip()
        if status_code != 200:
            response = ErrorResponse(**json.loads(data))
        else:
            response = ChatCompletionResponse(**json.loads(data))
            assert response.choices[0].message.role == "assistant"
            assert response.choices[0].message.content is not None

    # streaming request
    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "How are you?"}],
        stream=True,
    )
    generator = async_llm_server.chat_completion_generator.remote(request)
    async for ref in generator:
        status_code, data = await ref
        print(f">>>> status_code: {status_code}, {data}")
        data = data[len("data: ") :].rstrip()
        if status_code != 200:
            response = ErrorResponse(**json.loads(data))
        elif data == "[DONE]":
            break
        else:
            response = ChatCompletionStreamResponse(**json.loads(data))
            assert response.choices[0].delta.role is None or response.choices[0].delta.role == "assistant"
            assert response.choices[0].delta.content is not None

    ray.shutdown()


if __name__ == "__main__":
    config = init_config()
    test_vllm_multi_turn(config)
    asyncio.run(test_vllm_streaming_response(config))
