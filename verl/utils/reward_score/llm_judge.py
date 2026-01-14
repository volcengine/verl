# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""LLM-as-Judge reward scoring using async API calls."""

import asyncio
import logging
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Default judge prompt template
DEFAULT_JUDGE_PROMPT = """You are an impartial judge evaluating the quality of a generated response.

Question:
{question}

Reference Answer:
{reference}

Generated Response:
{response}

Evaluate the response on a scale of 0 to 1:
- 0: Completely incorrect, irrelevant, or nonsense
- 0.25: Poor - has some relevant content but mostly incorrect
- 0.5: Acceptable - partially correct but with significant errors or missing information
- 0.75: Good - mostly correct with minor issues
- 1.0: Excellent - fully correct, well-reasoned, and addresses all aspects of the question

Consider the following criteria:
- Factual correctness
- Relevance to the question
- Clarity and coherence
- Completeness of the answer

Output ONLY a single number between 0 and 1 (e.g., "0.75"). Do not include any additional text."""


async def _call_llm_judge(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.1,
    timeout: int = 60,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """
    Make an async call to LLM API for judgment.

    Args:
        url: The API endpoint URL (e.g., "https://api.openai.com/v1/chat/completions")
        api_key: The API key for authentication
        model: The model name to use (e.g., "gpt-4", "gpt-3.5-turbo")
        prompt: The prompt to send to the LLM
        max_tokens: Maximum tokens for the response
        temperature: Sampling temperature (lower for more deterministic output)
        timeout: Request timeout in seconds
        semaphore: Optional semaphore for concurrent request limiting

    Returns:
        Dict containing 'score', 'raw_response', and optional 'error' field
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        async with semaphore or asyncio.Semaphore(1):
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"API returned status {resp.status}: {error_text}")
                        return {
                            "score": 0.0,
                            "raw_response": "",
                            "error": f"HTTP {resp.status}: {error_text}",
                        }

                    result = await resp.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()

                    # Parse score from response
                    # Look for a number in formats like: 0.75, 0.8, 1.0, etc.
                    match = re.search(r"[0-9]*\.?[0-9]+", raw_response)
                    if match:
                        score = float(match.group())
                        # Clamp score to [0, 1]
                        score = max(0.0, min(1.0, score))
                    else:
                        logger.warning(f"Could not parse score from response: {raw_response}")
                        score = 0.0

                    return {
                        "score": score,
                        "raw_response": raw_response,
                        "model_used": model,
                    }

    except asyncio.TimeoutError:
        logger.error(f"API request timed out after {timeout}s")
        return {"score": 0.0, "raw_response": "", "error": "Timeout"}
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error: {e}")
        return {"score": 0.0, "raw_response": "", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in _call_llm_judge: {e}")
        return {"score": 0.0, "raw_response": "", "error": str(e)}


def _build_judge_prompt(
    question: str,
    reference: str,
    response: str,
    custom_prompt_template: str | None = None,
) -> str:
    """
    Build the judge prompt from question, reference, and response.

    Args:
        question: The original question/prompt
        reference: The reference/ground truth answer
        response: The generated response to evaluate
        custom_prompt_template: Optional custom prompt template with {question}, {reference}, {response} placeholders

    Returns:
        The formatted prompt string
    """
    if custom_prompt_template:
        return custom_prompt_template.format(
            question=question,
            reference=reference,
            response=response,
        )
    return DEFAULT_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        response=response,
    )


async def compute_score_async(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    api_key: str,
    base_url: str = "https://api.openai.com/v1/chat/completions",
    model: str = "gpt-4",
    max_tokens: int = 100,
    temperature: float = 0.1,
    timeout: int = 60,
    max_concurrent: int = 10,
    judge_prompt_template: str | None = None,
    enable_retries: bool = True,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Async compute reward score using LLM-as-Judge via API calls.

    This function is designed to be called as part of a batch processing workflow.
    It makes asynchronous HTTP requests to the LLM API for scoring.

    Args:
        data_source: Dataset identifier (e.g., "gsm8k", "human_eval")
        solution_str: The generated response text from the model
        ground_truth: The reference/expected answer
        extra_info: Dict containing additional context. Expected keys:
            - "question": The original question/prompt (required if using default prompt)
            - "prompt": The original prompt (alternative to "question")
            - "judge_prompt": Optional custom judge prompt (overrides judge_prompt_template)
        api_key: API key for the LLM service
        base_url: Base URL for the LLM API endpoint
        model: Model name to use for judgment
        max_tokens: Maximum tokens for the judge's response
        temperature: Sampling temperature (0.1-0.3 recommended for scoring)
        timeout: Request timeout in seconds
        max_concurrent: Maximum concurrent API requests (for rate limiting)
        judge_prompt_template: Custom prompt template with {question}, {reference}, {response} placeholders
        enable_retries: Whether to retry on failure
        max_retries: Maximum number of retries on failure

    Returns:
        Dict with the following structure:
        {
            "score": float,  # The computed score (0.0 - 1.0)
            "question": str,  # The original question (from extra_info)
            "reference": str,  # The reference answer
            "response": str,  # The generated response (solution_str)
            "raw_response": str,  # The raw response from the LLM judge
            "model_used": str,  # The model used for judgment
            "error": str | None,  # Error message if scoring failed
        }

    Example usage in custom_reward_function config:
        custom_reward_function:
          path: /path/to/llm_judge.py
          name: compute_score_async
          reward_kwargs:
            api_key: ${oc.env:OPENAI_API_KEY}
            base_url: "https://api.openai.com/v1/chat/completions"
            model: "gpt-4"
            max_concurrent: 10
    """
    # Extract question from extra_info
    question = extra_info.get("question") or extra_info.get("prompt", "")

    # Use custom prompt from extra_info if provided
    custom_prompt = extra_info.get("judge_prompt") or judge_prompt_template

    # Build the judge prompt
    judge_prompt = _build_judge_prompt(
        question=question,
        reference=ground_truth,
        response=solution_str,
        custom_prompt_template=custom_prompt,
    )

    # Create semaphore for concurrent request limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Make API call with retries
    for attempt in range(max_retries + 1):
        result = await _call_llm_judge(
            url=base_url,
            api_key=api_key,
            model=model,
            prompt=judge_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            semaphore=semaphore,
        )

        # If successful or retries disabled, return result
        if result.get("error") is None or not enable_retries:
            return {
                "score": result.get("score", 0.0),
                "question": question,
                "reference": ground_truth,
                "response": solution_str,
                "raw_response": result.get("raw_response", ""),
                "model_used": model,
                "error": result.get("error"),
            }

        # Exponential backoff for retries
        if attempt < max_retries:
            wait_time = 2**attempt  # 1, 2, 4 seconds
            logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s due to: {result.get('error')}")
            await asyncio.sleep(wait_time)

    # All retries exhausted
    return {
        "score": 0.0,
        "question": question,
        "reference": ground_truth,
        "response": solution_str,
        "raw_response": "",
        "model_used": model,
        "error": f"Max retries ({max_retries}) exceeded",
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> dict[str, Any]:
    """
    Synchronous wrapper for compute_score_async.

    This function allows the async implementation to be used in sync contexts.
    It creates a new event loop to run the async function.

    Args:
        Same as compute_score_async, passed via kwargs

    Returns:
        Same as compute_score_async
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            compute_score_async(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
        )
    finally:
        loop.close()


async def batch_compute_scores(
    items: list[dict[str, Any]],
    api_key: str,
    base_url: str = "https://api.openai.com/v1/chat/completions",
    model: str = "gpt-4",
    max_tokens: int = 100,
    temperature: float = 0.1,
    timeout: int = 60,
    max_concurrent: int = 10,
    judge_prompt_template: str | None = None,
) -> list[dict[str, Any]]:
    """
    Batch compute scores for multiple items concurrently.

    This is useful for testing or batch evaluation outside of the training pipeline.

    Args:
        items: List of dicts, each containing:
            - "solution_str": The generated response
            - "ground_truth": The reference answer
            - "extra_info": Dict with "question" and optional "judge_prompt"
        api_key: API key for the LLM service
        base_url: Base URL for the LLM API endpoint
        model: Model name to use
        max_tokens: Maximum tokens for responses
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        max_concurrent: Maximum concurrent requests
        judge_prompt_template: Custom prompt template

    Returns:
        List of result dicts, one per input item

    Example:
        items = [
            {
                "solution_str": "The answer is 42.",
                "ground_truth": "42",
                "extra_info": {"question": "What is 6 * 7?"}
            },
            ...
        ]
        results = await batch_compute_scores(items, api_key="sk-...")
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for item in items:
        question = item.get("extra_info", {}).get("question", "")
        custom_prompt = item.get("extra_info", {}).get("judge_prompt") or judge_prompt_template

        judge_prompt = _build_judge_prompt(
            question=question,
            reference=item["ground_truth"],
            response=item["solution_str"],
            custom_prompt_template=custom_prompt,
        )

        task = _call_llm_judge(
            url=base_url,
            api_key=api_key,
            model=model,
            prompt=judge_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            semaphore=semaphore,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and add context
    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception in batch_compute_scores for item {i}: {result}")
            output.append({
                "score": 0.0,
                "question": items[i].get("extra_info", {}).get("question", ""),
                "reference": items[i]["ground_truth"],
                "response": items[i]["solution_str"],
                "raw_response": "",
                "error": str(result),
            })
        else:
            output.append({
                "score": result.get("score", 0.0),
                "question": items[i].get("extra_info", {}).get("question", ""),
                "reference": items[i]["ground_truth"],
                "response": items[i]["solution_str"],
                "raw_response": result.get("raw_response", ""),
                "error": result.get("error"),
            })

    return output
