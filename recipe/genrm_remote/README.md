# Generative Reward Model

## Scripts

### Step 1: Launch a vLLM Server (Optional)

Deploy the pretrained GenRM model using vLLM. Skip this step if you want to use an external api service.

```bash 
VLLM_SERVER_DEV_MODE=1 vllm serve verl-team/GenRM-CI-Test-1.5B --served-model-name genrm-demo --enable-sleep-mode --dtype float32
```
Note that the wake_up and sleep operations for managing CUDA memory in vLLM are only available when both `VLLM_SERVER_DEV_MODE=1` and `enable_sleep_mode` are set. This capability is particularly beneficial when the model server shares resources with a training workload on the same machine. It allows the reward model service to be temporarily offloaded (to free up GPU memory) during intensive training sessions and reloaded when the service is required again. The relevant vllm code implementation can be found below:

[VLLM_SERVER_DEV_MODE](https://github.com/vllm-project/vllm/blob/5a19a6c6705fe83db2e3517a2d2f473586901743/vllm/entrypoints/openai/api_server.py#L971)

[sleep and wake_up mode](https://github.com/vllm-project/vllm/blob/5a19a6c6705fe83db2e3517a2d2f473586901743/vllm/entrypoints/openai/api_server.py#L994-L1003)

When the backend is configured as `SERVER_BACKEND`="VLLM", the `USE_OFFLOAD` flag can be toggled between True and False.(see `reward_function.py`)


### Step 2: Perform RL using GenRM

```bash
bash recipe/api-genrm/run_genrm_remote.sh
```

The implementation works by passing a customized reward function (see `reward_function.py`)

For convenience, we run both the RL training and server on the same machine. To use an external server, configure the `BASE_URL` and `API_KEY` in `reward_function.py` first.

## Advanced: Customizing Your GenRM

You can use sglang server with data parallel for faster inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4
```

Note that you should modify the `BASE_URL` in `reward_function.py` to match your SGLang Server address.

You can also create your own customized GenRM by implementing a custom reward function. Here are some tips for customizing your own GenRM based on `reward_function.py`:

- Design appropriate prompts for your GenRM
- Convert GenRM responses into RL rewards
- ...

Since these aspects are highly flexible, we only provide a demo implementation. The actual design and implementation of GenRM is left to the user's discretion.
