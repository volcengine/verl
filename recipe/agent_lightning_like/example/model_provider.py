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

from agents import Model, ModelProvider, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

from recipe.agent_lightning_like.notify import get_llm_server_address

DEFAULT_MODEL_NAME = "Default"

set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        model_configs = get_model_configs()
        model_name = model_name or DEFAULT_MODEL_NAME
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not found in model configs: {model_configs.keys()}")
        config = model_configs[model_name]
        base_url = config["base_url"]
        api_key = config.get("api_key", "")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def get_model_configs():
    """Demo: get model configurations from LLM_SERVER_NOTIFY_FILE."""
    server_address = get_llm_server_address()
    base_url = f"http://{server_address}/v1"
    model_configs = {
        DEFAULT_MODEL_NAME: {
            "base_url": base_url,
            "api_key": "",
        },
    }
    return model_configs
