# Generative Reward Model

## Scripts

### Step 1: Launch a vLLM Server (Optional)

Deploy the pretrained GenRM model using vLLM. Skip this step if you want to use an external api service.

```bash 
vllm serve dyyyyyyyy/Qwen2.5-1.5B-GenRM-QueryOnly --served-model-name genrm-demo
```

### Step 2: Perform RL using GenRM

```bash
bash recipe/api-genrm/run_genrm_remote.sh
```

The implementation works by passing a customized reward function (see `reward_function.py`)

For convenience, we run both the RL training and server on the same machine. To use an external server, configure the `BASE_URL` and `API_KEY` in `reward_function.py` first.

## Advanced: Customizing Your GenRM

You can use sglang server with data parallel for faster inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path dyyyyyyyy/Qwen2.5-1.5B-GenRM-QueryOnly --dp-size 4
```

Note that you should modify the `BASE_URL` in `reward_function.py` to match your SGLang Server address.

You can also create your own customized GenRM by implementing a custom reward function. Here are some tips for customizing your own GenRM based on `reward_function.py`:

- Design appropriate prompts for your GenRM
- Convert GenRM responses into RL rewards
- ...

Since these aspects are highly flexible, we only provide a demo implementation. The actual design and implementation of GenRM is left to the user's discretion.
