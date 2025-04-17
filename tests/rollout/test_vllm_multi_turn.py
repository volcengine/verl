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
from typing import Any, Dict

import ray
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import Worker, create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_async_workers import AsyncActorRolloutRefWorker, AsyncLLMManager
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler


async def test_vllm_multi_turn():
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "Qwen/Qwen2-7B-Instruct"
    model_name = "/".join(model_path.split("/")[-2:])
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096

    # =========================== 1. Create hybrid ActorRollout workers ===========================
    ray.init(
        runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                'VLLM_USE_V1': '1',
            }
        })
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout],
                                             config=config.actor_rollout_ref,
                                             role='actor_rollout')
    resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

    all_wg = {}
    wg_dicts = []
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict, worker_cls=Worker)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
        wg_dicts.append(wg_dict)
    actor_rollout_wg = all_wg['actor_rollout']
    actor_rollout_wg.init_model()

    # =========================== 2. Create AsyncLLMManager&ChatScheduler  ===========================
    async_rollout_manager = AsyncLLMManager(
        config=config.actor_rollout_ref,
        worker_group=actor_rollout_wg,
    )

    async_chat_scheduler = ChatCompletionScheduler(
        config=config.actor_rollout_ref.rollout,
        model_path=config.actor_rollout_ref.model.path,
        server_addresses=async_rollout_manager.server_addresses,
    )

    # =========================== 3. Multi turn rollout  ===========================
    async def callback(completions: ChatCompletion, info: Dict[str, Any]):
        messages, round = info["messages"], info["round"]
        message = completions.choices[0].message
        messages.append({"role": message.role, "content": message.content})
        print(f"[round={round}] role: {message.role}, content: {message.content}")

        extra_headers = {"x-request-id": completions.id}
        if round == 0:
            messages.append({"role": "user", "content": "What is your name?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "messages": messages,
                    "round": 1
                },
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        elif round == 1:
            messages.append({"role": "user", "content": "What is your favorite color?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "messages": messages,
                    "round": 2
                },
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        else:
            print("Done!")

    messages = [{
        "role": "user",
        "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."
    }]
    await async_chat_scheduler.submit_chat_completions(
        callback=callback,
        callback_additional_info={
            "messages": messages,
            "round": 0
        },
        model=model_name,
        messages=messages,
    )
    assert len(messages) == 6
    for round, message in enumerate(messages):
        if round % 2 == 0:
            assert message["role"] == "user"
        else:
            assert message["role"] == "assistant"


if __name__ == "__main__":
    asyncio.run(test_vllm_multi_turn())
