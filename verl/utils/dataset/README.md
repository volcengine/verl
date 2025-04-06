# Dataset Format
## RLHF dataset
We combine all the data sources into a single parquet files. We directly organize the prompt into the chat format so that multi-turn chats can be easily incorporated. In the prompt, we may add instruction following texts to guide the model output the answers in a particular format so that we can extract the answers.

Math problems
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```

## RLAgent dataset

The dataset is designed for RL training with the LLM as the agent. 
It is built on top of the design in https://github.com/HMJiangGatech/verl_agent_env_examples.

The dataset is organized as follows:
```json
{
    "env_name": "verl_env/sokoban-v0",
    "seed": 0,
    "env_kwargs": null,
}
```
The `env_name` is the name of the environment.
The `seed` is the seed for the environment.
The `env_kwargs` is the kwargs for the environment.
They are used to initialize the environment, `initialize_env` in `https://github.com/HMJiangGatech/verl_agent_env_examples/blob/master/src/verl_agent_env/app.py`.

