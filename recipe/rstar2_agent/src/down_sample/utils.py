# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from verl.protocol import DataProto, DataProtoItem


def filter_by_mask(batch: DataProto, mask: torch.Tensor, num_trainer_replicas: int) -> DataProto:
    # Filter batch to keep only valid samples
    batch = batch[mask]
    # Round down to the nearest multiple of world size
    max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
    if not max_batch_size:
        # give up, you got everything either all wrong or right.
        return None

    size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
    size_mask[:max_batch_size] = True
    batch = batch[size_mask]
    return batch


def decode_prompt_response_str(data: DataProto, tokenizer) -> tuple[list[str], list[str]]:
    """
    Decode the prompt and response strings from a DataProto object using the provided tokenizer.

    Args:
        data (DataProto): The DataProto object containing the data.
        tokenizer: The tokenizer to decode the IDs into strings.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - List of decoded prompt strings.
            - List of decoded response strings.
    """
    prompts = []
    responses = []

    for item in data:
        # Decode prompt IDs
        if "prompt_text" in item.non_tensor_batch and item.non_tensor_batch['prompt_text'] is not None:
            prompt_str = item.non_tensor_batch['prompt_text']
        else:
            prompt_ids = item.batch['prompts']
            valid_prompt_length = item.batch['attention_mask'][:prompt_ids.shape[-1]].sum()
            prompt_str = tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False)
        prompts.append(prompt_str)

        # Decode response IDs
        if "response_text" in item.non_tensor_batch and item.non_tensor_batch['response_text'] is not None:
            response_str = item.non_tensor_batch['response_text']
        else:
            response_ids = item.batch['responses']
            valid_response_length = item.batch['attention_mask'][item.batch['prompts'].shape[-1]:].sum()
            response_str = tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)
        responses.append(response_str)

    return prompts, responses
