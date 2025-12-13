# Agent-Lightning-like RL training Example

Agent-Lightning-like is a RL training recipe inspire by Agent Lightning (https://arxiv.org/abs/2508.03680). You can train almost **ANY** agent by writing a few lines of codes. More important, the agent can run in an independent Python environment or even on a separate machine, as a service. That makes the training simpler, especially when you have a complex agent system. 

Here is a tiny example to demonstrate how to use this recipe. The example uses OpenAI agent-sdk, but this recipe does not restrict on which framework you use to write the agent.

## Prepare agent server

Wrap the agent as a http service, if you don't have one. As an example, `agent_server.py` demonstrates how to set up a `/chat` API endpoint, which features an integrated `calc_gsm8k_reward` tool.

We need to inject two elements into the Agent, the LLM service url and additional request headers.

The LLM service url is provided after veRL training started. The agent gets it by calling `get_llm_server_address` defined in `recipe/agent_lightning_like/notify.py`:

```python
# model_provider.py
DEFAULT_MODEL_NAME = "Default"

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
```

The `LLM_SERVER_NOTIFY_FILE` env set the file that pass the llm endpoint from the trainer to the agent server.

An additional request header with a name "trace_id" is included in the request context, and we need to pass it to the LLM server. We do it by setting `extra_headers` in `model_settings`.

```python
# agent_server.py

@app.post("/chat")
async def chat(request: Annotated[ChatRequest, fastapi.Body()]):
    """A demo chat function."""
    context = request.context
    model_provider = CustomModelProvider()
    extra_headers = request.extra_headers or {}
    extra_headers.update({HEADER_TRACE_ID: context.trace_id})
    model_settings = ModelSettings(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        extra_headers=extra_headers,	# inject trace_id here
        extra_body=request.extra_body or {},
    )
    agent = Agent[UserContext](
        name="Assistant",
        instructions=request.system_prompt or "You are a helpful assistant.",
        tools=[calc_gsm8k_reward],
    )
    # ......
```

## Write agent client

Trainer uses a client to send prompts to the agent server, that starts the rollout. The client shall implement an async `chat` method, like the demo `agent_client.py`. The `chat` method is expected to throw no exceptions.

```python
# agent_client.py

class AgentClient(AgentClientBase):

    async def chat(self, trace_id: str, sampling_params: dict[str, Any], **kwargs) -> Any:
        # kwargs include "max_turns" and non-tensor fields of a data sample from RLHFDataset
        # ...
        # async send request to agent server
```

## Prepare dataset
Let's prepare two small datasets for training and evaluation:
```bash
python examples/data_preprocess/gsm8k_tool_agent_loop.py
```

We use a simple `CustomDataset` class defined in `dataset.py` to adapt the "agent_name" field in the generated dataset with the one we define in `agent_loop.yaml`.

```python
# dataset.py
from verl.utils.dataset import RLHFDataset

class CustomDataset(RLHFDataset):
    """A custom dataset for the agent-lightning-like example."""
    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        row_dict["agent_name"] = "lightning_demo"  # must match the name in agent_loop.yaml
        row_dict.pop("tools_kwargs", None)  # remove tools_kwargs if exists, tools defined in agent server side
        return row_dict
```

## Training

Prepare these yaml config file if you train your own agent:  `agent_loop.yaml`, `agent_client.yaml`, `recipe/agent_lightning_like/config/lightning_ppo_trainer.yaml`, and write a start script.

Run this demo example:

```bash
bash recipe/agent_lightning_like/example/run_qwen2.5_7b.sh 2>&1 | tee run.log
```

You probably need a 8-GPU node for this example, or choose a smaller model.

The validation score is expected to reach about 93.6/100 after training one epoch.

## Testing

There are some CI tests in `recipe/agent_lightning_like/test`.

Run a test:

```bash
PYTHONPATH=$(pwd) pytest -s recipe/agent_lightning_like/test/test_xxx.py
```

