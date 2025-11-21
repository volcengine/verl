# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

import pytest

from recipe.agent_lightning_like.example.agent_client import AgentClient
from recipe.agent_lightning_like.example.agent_server import _get_host_ip
from recipe.agent_lightning_like.notify import notify_llm_server_address

from .utils import agent_server_addr  # noqa: F401


@pytest.fixture
def standalone_llm_server_addr():
    from sglang.utils import terminate_process, wait_for_server

    try:
        from sglang.test.doc_patch import launch_server_cmd
    except ImportError:
        from sglang.utils import launch_server_cmd

    model_path = "Qwen/Qwen2.5-3B-Instruct"
    server_proc, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --tool-call-parser qwen25 --log-level warning"  # noqa: E501
    )
    localhost = _get_host_ip()
    server_addr = f"{localhost}:{port}"
    wait_for_server(f"http://{server_addr}")

    yield server_addr

    terminate_process(server_proc)


@pytest.mark.asyncio
async def test_agent_client_server(agent_server_addr, standalone_llm_server_addr):  # noqa: F811
    assert agent_server_addr
    assert standalone_llm_server_addr
    notify_llm_server_address(standalone_llm_server_addr)
    assert os.path.isfile(os.environ["LLM_SERVER_NOTIFY_FILE"])

    from datasets import load_dataset

    samples = load_dataset("parquet", data_files=os.path.expanduser("~/data/gsm8k/test.parquet"), split="train")
    sample = samples[0]
    raw_prompt = sample.pop("prompt")
    sample["raw_prompt"] = raw_prompt
    print(f"prompt: {raw_prompt}")
    sampling_params = {}
    try:
        client = AgentClient(server_address=agent_server_addr)
        response = await client.chat(trace_id="test_1234", sampling_params=sampling_params, max_turns=5, **sample)
        print(response)
    except Exception as e:
        import traceback

        print(f"Error in client.chat: {e}")
        traceback.print_exc()
