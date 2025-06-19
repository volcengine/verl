# Generative Reward Model

## Scripts

### Step 1: Launch a vLLM Server (Optional)

Deploy the pretrained GenRM model using vLLM. Skip this step if you want to use an external api service.

```
vllm serve dyyyyyyyy/Qwen2.5-1.5B-GenRM-QueryOnly --served-model-name genrm-demo
```

### Step 2: Perform RL using GenRM

```
bash recipe/api-genrm/train_grpo.sh
```

The implementation works by passing a customized reward function (see `reward_function.py`)

For convenience, we run both the RL training and server on the same machine. To use an external server, configure the `BASE_URL` and `API_KEY` in `reward_function.py` first.


## Advanced: Customizing Your GenRM

<!-- You can also customize your own GenRM by 实现你自己的custom reward function，以下是一些给予`reward_function.py`的修改的tips： -->
You can create your own customized GenRM by implementing a custom reward function. Here are some tips for customizing your own GenRM based on `reward_function.py`:

- Design appropriate prompts for your GenRM
- Convert GenRM responses into RL rewards
- Decide whether to provide ground truth to GenRM (when available)
- ...

Since these aspects are highly flexible, we only provide a demo implementation. The actual design and implementation of GenRM is left to the user's discretion.

## Results

