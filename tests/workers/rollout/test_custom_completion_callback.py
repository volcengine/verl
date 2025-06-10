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
import concurrent.futures
import os
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, ToolCompletionCallback


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

            response = {
                "status": "Success" if process.returncode == 0 else "Failed",
                "run_result": {
                    "status": "Finished",
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": process.returncode,
                },
            }
            return JSONResponse(content=response)
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
        app.router.add_api_route("/run_code", self.code_execution, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"


class CustomCompletionCallback(ToolCompletionCallback):
    def __init__(self, config: DictConfig, scheduler: ChatCompletionScheduler):
        super().__init__(config, scheduler)

        self.max_turns = 16
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        self.code_pattern = re.compile(r"<code>\s*```python(.*?)```\s*</code>", re.DOTALL)

        self.sandbox_fusion_url = config.reward_model.sandbox_fusion.url
        self.default_timeout = 10
        # TODO: support asyncio executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5))

    async def sandbox_code_execution(self, code: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        result_status, metadata = await loop.run_in_executor(
            self.executor,
            _process_single_case,
            0,  # case_index,
            None,  # stdin_data,
            None,  # expected_output,
            self.sandbox_fusion_url,  # sandbox_fusion_url
            code,  # generation
            self.default_timeout,  # timeout
            "python",  # language
        )

        return metadata

    @property
    def extra_body(self):
        extra = {
            "include_stop_str_in_output": True,
            "stop": ["</answer>", "</code>"],
        }
        return extra

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        role, content, finish_reason = completions.choices[0].message.role, completions.choices[0].message.content, completions.choices[0].finish_reason
        messages.append({"role": role, "content": content})
        turn = len(messages)

        # STEP 0: check if we reach max turns
        if len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if we reach max tokens
        if finish_reason == "length":
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Reach max tokens, done!")
            return

        # STEP 2: check if we got answer
        matches = self.answer_pattern.findall(content)
        if matches:
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Got answer: {matches[0]}, done!")
            return

        # STEP 3: check if we got code block
        matches = self.code_pattern.findall(content)
        if not matches:
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] No code block found, done!")
            return

        # STEP 4: execute code block in sandbox
        code = matches[0].strip()
        metadata = await self.sandbox_code_execution(code)
        if metadata["run_status"] != "Finished":
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Code block execution failed: {metadata}, done!")
            return

        stdout, stderr = metadata["stdout"], metadata["stderr"]
        messages.append({"role": "tool", "content": f"<interpreter>{stdout}{stderr}</interpreter>"})
        print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Code block executed, continue...")

        # STEP 5: resubmit chat completions with code block output
        self.scheduler.submit_chat_completions(
            messages=messages,
            request_id=completions.id,
            info=info,
        )


user_prompt_template = """
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

*user question:*
{question}
"""


if __name__ == "__main__":
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
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.multi_turn.completion_callback = "tests.workers.rollout.test_custom_completion_callback.CustomCompletionCallback"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4

    # Init sandbox and async rollout manager
    sandbox = Sandbox.options(num_cpus=1).remote()
    sandbox_address = ray.get(sandbox.get_server_address.remote())
    sandbox_fusion_url = f"http://{sandbox_address}/run_code"
    config.reward_model.sandbox_fusion.url = sandbox_fusion_url
    async_rollout_manager = init_async_rollout_manager(config)

    # Build dataset
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    prompts = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([[{"role": "user", "content": user_prompt_template.replace("{question}", problem)}] for problem in dataset["Problem"]]),
        },
    )

    result = async_rollout_manager.generate_sequences(prompts=prompts)
    assert len(result) == len(dataset) * config.actor_rollout_ref.rollout.n

    # Check max turns that sandbox is called
    num_turns = result.non_tensor_batch["__num_turns__"]
    print(f"num_turns: {num_turns}")
    assert np.max(num_turns) > 2, f"max turns: {np.max(num_turns)}"

    # Check response_mask
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"

    # Decode responses with response_mask
    for i in range(len(responses)):
        valid_tokens = responses[i][response_mask[i].bool()]
        response_str = tokenizer.decode(valid_tokens)
        assert "<tool_response>" not in response_str, f"found <tool_response> in response: {response_str}"
        assert "</tool_response>" not in response_str, f"found </tool_response> in response: {response_str}"
        print(f"response: {response_str}")

    print("Test passed!")
