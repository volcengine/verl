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
from typing import Any, Dict, List

from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler


class NaiveChatCompletionScheduler(ChatCompletionScheduler):

    def __init__(self, config: DictConfig, model_path: str, server_addresses: List[str], max_cache_size: int = 10000):
        super().__init__(config, model_path, server_addresses, max_cache_size)

    async def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any]):
            info["all_completions"][info["index"]] = completions

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        tasks, all_completions = [], [None] * len(prompts)
        for i, prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "all_completions": all_completions,
                            "index": i
                        },
                        model=self.model_name,
                        messages=prompt,
                        **kwargs,
                    )))
        await asyncio.gather(*tasks)

        print("[NaiveChatCompletionScheduler] generate_sequences done")
        # TODO: completions => DataProto
        return all_completions
