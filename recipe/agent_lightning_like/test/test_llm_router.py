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

import ray
import requests

from recipe.agent_lightning_like.trajectory import Trajectory

from .utils import (  # noqa: F401
    HEADER_TRACE_ID,
    agent_loop_mgr,
    agent_server_addr,
    init_config,
    llm_router_and_addr,
    tokenizer,
)


def test_llm_router(init_config, agent_loop_mgr, llm_router_and_addr):  # noqa: F811
    agent_loop_mgr.wake_up()
    llm_router, address = llm_router_and_addr

    url = f"http://{address}/v1/models"
    response = requests.get(url)

    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    resp_json = response.json()
    print("Response:", resp_json)
    assert "data" in resp_json and len(resp_json["data"]) > 0, "No models found"
    model = resp_json["data"][0]
    assert "id" in model and isinstance(model["id"], str)
    assert "object" in model and model["object"] == "model"
    assert "owned_by" in model and isinstance(model["owned_by"], str)
    assert "max_model_len" in model and model["max_model_len"] > 0

    url = f"http://{address}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        HEADER_TRACE_ID: "test_1234",
    }
    data = {
        "model": init_config.lightning_trainer.model_name,
        "messages": [
            {"role": "user", "content": "1+2="},
        ],
        "max_tokens": 50,
        "n": 1,
        "temperature": 0.6,
        "top_p": 0.9,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    resp_json = response.json()
    print("Response:", resp_json)
    assert "choices" in resp_json and len(resp_json["choices"]) > 0, "Empty choices"
    assert "message" in resp_json["choices"][0], "Choice does not contain message"
    message = resp_json["choices"][0]["message"]
    assert "role" in message and message["role"] == "assistant", "Message role is not assistant"
    assert "content" in message and isinstance(message["content"], str), "Message does not contain string content"
    assert "3" in message["content"], "Response does not contain the correct answer"

    trajectory: Trajectory = ray.get(llm_router.retrieve_trajectory.remote(data["model"], headers[HEADER_TRACE_ID]))
    assert trajectory is not None, "Trajectory is None"
    assert len(trajectory.items) == 1, "Trajectory should contain 1 items"
    item = trajectory.items[0]
    assert item.prompt_ids and len(item.prompt_ids) > 0, "Prompt ids should not be empty"
    assert item.response_ids and len(item.response_ids) > 0, "Response ids should not be empty"
    assert len(item.messages) == 2, "Trajectory item should contain 2 messages"
    assert item.messages[0]["role"] == "user" and item.messages[0]["content"] == data["messages"][0]["content"]
    assert item.messages[1]["role"] == "assistant", "Second message role is not assistant"
    assert item.messages[1]["content"] == message["content"]

    print("Test passed!")
