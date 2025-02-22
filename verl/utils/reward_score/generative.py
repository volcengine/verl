import re
import asyncio
import numpy as np
import torch
from typing import List, Dict, Any
from openai import AsyncOpenAI
from transformers import AutoTokenizer


def extract_output(solution_text: str):
    # Match everything inside the last \boxed{} in the solution text
    boxed_pattern = r'\\bold{(.*)}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


async def _query_openai_with_semaphore(semaphore: asyncio.Semaphore, client: AsyncOpenAI, sequence_str: str,
                                       config: Dict[str, Any]) -> float:
    """
    Request method with semaphore.
    """
    async with semaphore:
        return await _query_openai_async(client, sequence_str, config)


async def _query_openai_async(client: AsyncOpenAI, sequence_str: str, config) -> float:
    """
    Query OpenAI API asynchronously.
    """
    max_retries = config.max_retries  # Maximum number of retries
    retry_count = 0
    scoring_prompt = open(config.scoring_prompt, "r").read()
    min_score = config.min_score  # Minimum valid score
    max_score = config.max_score  # Maximum valid score
    while retry_count < max_retries:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": scoring_prompt + '\n' + sequence_str
            },
        ]
        if config.tokenizer and config.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            messages = tokenizer.apply_chat_template(messages, tokenize=False)
        try:
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                n=config.num_samples,
            )
            if config.num_samples == 1:
                # Handle single sample case
                try:
                    scores = response.choices[0].message.content.strip()
                    score = float(extract_output(scores))

                    # Check if the score is within the valid range
                    if min_score <= score <= max_score:
                        return score
                    else:
                        print(f"Score {score} out of range [{min_score}, {max_score}]. Retrying...")
                        retry_count += 1
                        if retry_count >= max_retries:
                            print("Max retries reached. Returning default score.")
                            return config.default_score
                        continue  # Retry the request
                except Exception as e:
                    print(f"Processing error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print("Max retries reached. Returning default score.")
                        return config.default_score
                    continue  # Retry the request
            else:
                # Handle multiple samples case
                raw_scores = [choice.message.content.strip() for choice in response.choices]
                valid_scores = []
                for score in raw_scores:
                    try:
                        extracted_score = float(extract_output(score))
                        # Check if the score is within the valid range
                        if min_score <= extracted_score <= max_score:
                            valid_scores.append(extracted_score)
                        else:
                            print(f"Score {extracted_score} out of range [{min_score}, {max_score}]. Skipping...")
                    except Exception as e:
                        print(f"Processing error: {e}")
                if valid_scores:  # If there are any valid scores
                    if config.sc_mode == "mean":
                        return float(np.mean(valid_scores))
                    elif config.sc_mode == "median":
                        return float(np.median(valid_scores))
                    elif config.sc_mode == "majority":
                        return float(np.round(np.mean(valid_scores)))
                    else:
                        raise ValueError(f"Unknown consistency mode: {config.sc_mode}")
                else:
                    # No valid scores, retry the request
                    retry_count += 1
                    print("No valid scores found. Retrying...")
                    if retry_count >= max_retries:
                        print("Max retries reached. Returning default score.")
                        return config.default_score
                    continue  # Retry the request
        except Exception as e:
            print(f"Error querying OpenAI API: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached. Returning default score.")
                return config.default_score
            continue  # Retry the request


async def process_data_async(data_source: List[str], solution_str: List[str], ground_truth: List[str],
                             extra_info: List[Dict[str, Any]], config) -> torch.Tensor:
    """
    Process data asynchronously using OpenAI API.
    """
    reward_tensor = torch.zeros(len(solution_str), dtype=torch.float32)
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.server_url)

    remaining_tasks = list(range(len(solution_str)))
    while remaining_tasks:
        # Dynamic semaphore creation
        semaphore = asyncio.Semaphore(config.initial_concurrency)
        current_batch = remaining_tasks[:config.initial_concurrency]
        remaining_tasks = remaining_tasks[config.initial_concurrency:]
        tasks = []

        for i in current_batch:
            prompt = solution_str[i]
            response = ground_truth[i]
            if response is None:
                sequence_str = prompt
            else:
                sequence_str = f"{prompt}\nReference:\n{response}"

            task = _query_openai_with_semaphore(semaphore, client, sequence_str, config)
            tasks.append((i, sequence_str, task, data_source[i]))

        # Execute tasks in parallel
        results = await asyncio.gather(*[task for _, _, task, _ in tasks])

        # Adjust concurrency based on success rate
        success_rate = sum(1 for r in results if r != config.default_score) / len(results)
        if success_rate > 0.6:
            config.initial_concurrency = min(config.max_concurrency, int(config.initial_concurrency * 2))
        else:
            config.initial_concurrency = max(1, int(config.initial_concurrency / 2))

        # Update reward tensor
        for idx, (i, _, _, _) in enumerate(tasks):
            reward_tensor[i] = results[idx]

    return reward_tensor
