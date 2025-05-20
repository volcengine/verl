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
import os
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse

from examples.ppo_trainer.naive_chat_scheduler import NaiveChatCompletionScheduler
from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code.

    WARNING: This class is for testing purpose only, do not use it in production.
    Please use a sandbox with strong isolation and security restrictions instead.
    """

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]
        print(f"execute code:\n{code}")

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            stdout, stderr = await process.communicate()

            return JSONResponse(content={"stdout": stdout.decode(), "stderr": stderr.decode(), "returncode": process.returncode})
        finally:
            try:
                os.unlink(temp_file)
            except:  # noqa: E722
                pass

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print("FastAPI startup")
            self.server_ready.set()
            yield

            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/code/execution", self.code_execution, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"


class ToolChatCompletionScheduler(NaiveChatCompletionScheduler):
    """This is a demo chat completion scheduler that supports sandbox code execution
    described in ReTool paper: https://arxiv.org/pdf/2504.11536
    """

    def __init__(self, config, model_path, server_addresses, sandbox_address, system_prompt, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.sandbox_address = sandbox_address
        self.system_prompt = system_prompt

    async def sandbox_code_execution(self, code: str) -> Dict[str, Any]:
        """Execute python code in sandbox."""
        try:
            session = aiohttp.ClientSession()
            async with session.post(
                url=f"http://{self.sandbox_address}/code/execution",
                json={"code": code},
            ) as resp:
                return await resp.json()
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body={
                "include_stop_str_in_output": True,
                "stop": ["</answer>", "</code>"],
            },
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        max_turns = 3

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            batch_conversations, batch_index, turn = (
                info["batch_conversations"],
                info["batch_index"],
                info["turn"],
            )
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            batch_conversations[batch_index].append({"role": role, "content": content})

            # STEP 0: check if we reach max turns
            if turn == max_turns:
                print(f"[id={completions.id},turn={turn}] Reach max turns {max_turns}, done!")
                return

            # STEP 1: check if we got answer
            matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if matches:
                print(f"[id={completions.id},turn={turn}] Got answer: {matches[0]}, done!")
                return

            # STEP 2: check if we got code block
            matches = re.findall(r"<code>\s*```python(.*?)```\s*</code>", content, re.DOTALL)
            if not matches:
                print(f"[id={completions.id},turn={turn}] No code block found, done!")
                return

            # STEP 3: execute code block in sandbox
            code = matches[0].strip()
            result = await self.sandbox_code_execution(code)
            stdout, stderr = result["stdout"], result["stderr"]
            batch_conversations[batch_index].append({"role": "tool", "content": f"{stdout}{stderr}"})
            print(f"[id={completions.id},turn={turn}] Code block executed, continue...")

            # STEP 4: resubmit chat completions with code block output
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_index": batch_index,
                    "turn": turn + 1,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                extra_headers=extra_headers,
                **kwargs,
            )

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = [{"role": "system", "content": self.system_prompt}] + list(conversation)
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "turn": 1,
                        },
                        model=self.model_name,
                        messages=batch_conversations[batch_index],
                        **kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        # _postprocess assumes n>=1
        batch_conversations = [[conversation] for conversation in batch_conversations]
        return self._postprocess(batch, batch_conversations, kwargs["n"])


system_prompt = """
You are a helpful assistant. Let's solve math problem in following steps:
1. Write a python code first and return the code to user, the code must be in following format:

<code>
```python
import os

print(...)
```
</code>

The code must explictly print necessary output to stdout. Remember stop generation at </code> immediately and return the code.
2. User will send the python code to a external sandbox to execute and get output from stdout.
3. User will send the output in format <interpreter>output</interpreter> to you, and you should use the output to answer the question.
The answer format must be: <answer>\\boxed{'The final answer goes here.'}</answer>
"""


def test_vllm_tool_calling():
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

    # Load config
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.model.path = "Qwen/Qwen2-7B-Instruct"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "tests.workers.rollout.test_vllm_tool_calling.ToolChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 8192
    config.actor_rollout_ref.rollout.response_length = 8192

    # Init sandbox and async rollout manager
    sandbox = Sandbox.options(num_cpus=1).remote()
    sandbox_address = ray.get(sandbox.get_server_address.remote())
    async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"sandbox_address": sandbox_address, "system_prompt": system_prompt})

    # Build dataset
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([[{"role": "user", "content": problem}] for problem in dataset["Problem"]])})

    result = async_rollout_manager.generate_sequences(prompts=prompts)
    assert len(result) == len(dataset)


if __name__ == "__main__":
    test_vllm_tool_calling()
