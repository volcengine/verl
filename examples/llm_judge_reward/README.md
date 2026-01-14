# LLM-as-Judge Reward Scoring

This directory contains an implementation of LLM-as-Judge reward scoring for verl training.

## Overview

The LLM-as-Judge reward system uses a large language model to evaluate the quality of generated responses. This is particularly useful for:
- Open-ended questions without a single correct answer
- Tasks requiring subjective evaluation (e.g., explanation quality, reasoning)
- Domains where rule-based scoring is difficult

## Files

| File | Description |
|------|-------------|
| `llm_judge.py` | Core implementation of LLM-as-Judge reward scoring (located in `verl/utils/reward_score/`) |
| `example_llm_judge.py` | Standalone script to test LLM-as-Judge scoring |
| `run_llm_judge_ppo.sh` | Example script for PPO training with LLM-as-Judge |
| `sample_dataset.jsonl` | Sample dataset for testing |
| `README.md` | This file |

## Quick Start

### 1. Test LLM-as-Judge Standalone

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run with example questions in batch mode
python example_llm_judge.py --batch-mode

# Run with a custom question
python example_llm_judge.py \
    --question "What is 7 * 8?" \
    --response "The answer is 56." \
    --reference "56"

# Use Azure OpenAI
AZURE_OPENAI_KEY="..." python example_llm_judge.py --batch-mode \
    --api-url "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-05-15" \
    --model gpt-4
```

### 2. Use in PPO Training

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run training with LLM-as-Judge reward
bash run_llm_judge_ppo.sh
```

## Configuration Options

### Reward Function Parameters

| Parameter | Type | Default | Description |
|-----------|-------|----------|-------------|
| `api_key` | str | `OPENAI_API_KEY` env var | API key for the LLM service |
| `base_url` | str | `https://api.openai.com/v1/chat/completions` | API endpoint URL |
| `model` | str | `gpt-4` | Model name to use for judgment |
| `max_tokens` | int | `100` | Maximum tokens for judge response |
| `temperature` | float | `0.1` | Sampling temperature (lower = more consistent) |
| `timeout` | int | `60` | Request timeout in seconds |
| `max_concurrent` | int | `10` | Maximum concurrent API requests |
| `judge_prompt_template` | str | `DEFAULT_JUDGE_PROMPT` | Custom prompt template with `{question}`, `{reference}`, `{response}` placeholders |

### Using Custom Judge Prompts

You can provide a custom judge prompt in two ways:

#### 1. Via extra_info in dataset

Add `"judge_prompt"` field to your dataset:

```jsonl
{
    "prompt": "Your question here",
    "answer": "Reference answer",
    "extra_info": {
        "question": "Your question here",
        "judge_prompt": "Evaluate the following: {question}\\nResponse: {response}\\nScore 0-1 based on creativity."
    }
}
```

#### 2. Via configuration

Set `judge_prompt_template` in the reward kwargs:

```bash
trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.judge_prompt_template="..."
```

## API Provider Support

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
# Uses default: https://api.openai.com/v1/chat/completions
```

### Azure OpenAI

```bash
export AZURE_OPENAI_KEY="..."
python example_llm_judge.py --batch-mode \
    --api-url "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-05-15" \
    --model gpt-4
```

### Other OpenAI-Compatible APIs

Any API that follows the OpenAI chat completions format can be used:

```bash
python example_llm_judge.py --batch-mode \
    --api-url "https://your-api-endpoint.com/v1/chat/completions" \
    --model "your-model-name"
```

## Integration with Training

### Method 1: Via custom_reward_function (Recommended)

This is the easiest way to integrate:

```yaml
# In your training config
trainer:
  ppo_train:
    reward_manager:
      source: custom
      custom_reward_function:
        path: /path/to/llm_judge.py
        name: compute_score_async
        reward_kwargs:
          api_key: ${oc.env:OPENAI_API_KEY}
          base_url: "https://api.openai.com/v1/chat/completions"
          model: "gpt-4"
          max_concurrent: 10
```

### Method 2: Direct import in custom reward file

Create your own reward file:

```python
from verl.utils.reward_score.llm_judge import compute_score_async

def my_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # Pre-process or add custom logic
    # ...
    # Call LLM-as-Judge
    return compute_score_async(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs
    )
```

## Performance Considerations

### Concurrency

- Use `max_concurrent` to control API rate limits
- Higher values = faster processing but may hit rate limits
- Start with `10` and adjust based on your API limits

### Timeout

- Set appropriate timeout for your model (60s is reasonable for GPT-4)
- Longer timeouts for larger responses

### Retries

The implementation includes automatic retry logic (default: 2 retries) with exponential backoff.

## Example Results

Running the batch mode example produces output like:

```
============================================
LLM-as-Judge: Batch Evaluation with Examples
============================================
Model: gpt-4
API URL: https://api.openai.com/v1/chat/completions
Max Concurrent: 10
--------------------------------------------------------------------------------

[math]
  Question: What is 15 * 7?
  Score: 1.00

[science]
  Question: What is the chemical formula for water?
  Score: 0.95

[history]
  Question: When did World War II end?
  Score: 0.90

[coding]
  Question: Write a function to reverse a string in Python.
  Score: 0.85

[incorrect]
  Question: What is the capital of France?
  Score: 0.00

[partial]
  Question: What are the three primary colors?
  Score: 0.50

[excellent]
  Question: Explain what a black hole is.
  Score: 1.00
--------------------------------------------------------------------------------
Average Score: 0.86 / 1.0
============================================
```

## Troubleshooting

### API Key Error

```
Error: API key is required.
```
Solution: Set `OPENAI_API_KEY` environment variable or use `--api-key` argument.

### Timeout Error

```
HTTP request timed out
```
Solution: Increase `--timeout` or reduce `max_tokens` to speed up responses.

### Rate Limit Error

```
HTTP 429: Rate limit exceeded
```
Solution: Decrease `--max-concurrent` to reduce concurrent requests.

### Score Parsing Error

```
Could not parse score from response: ...
```
Solution: The judge returned text that couldn't be parsed as a number. This usually happens with custom prompts. Ensure your prompt instructs the model to output only a number.

## License

Apache License 2.0 - See parent LICENSE file for details.
