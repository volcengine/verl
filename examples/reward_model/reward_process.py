from typing import List, Dict, Tuple, Optional
from verl import DataProto
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
from torch import Tensor
from tensordict import TensorDict
import re
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def reward_preprocess(
        data: DataProto,
        tokenizer: PreTrainedTokenizer,
    ) -> Tuple[List[Dict[str, str]], Optional[DataProto]]:
    """
    Preprocesses data to create prompts for a generative reward model.

    This function constructs a detailed prompt that asks a Causal LLM to act as a judge,
    compare a model's generated response to a ground truth solution, and provide a score.

    args:
        data: DataProto containing batch information, including raw prompts and responses.
        tokenizer: The tokenizer used for decoding the model's response.

    return:
        - chats: A list of chat histories, where each history is a prompt for the reward model.
        - rm_info: A list of pre-calculated "exact match" scores, which can be used for
                   hybrid scoring or as a fallback in the postprocessing step.
    """
    chats = []
    extra_scores = []

    for i in range(data.batch.batch_size[0]):
        # extract raw prompt
        if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
            original_chat: list = data.non_tensor_batch["raw_prompt"][i]
        else:
            original_chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()
        original_chat = [c for c in original_chat if c['role'] != 'system']

        # extract response
        response_ids = data.batch["responses"][i]
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # question = original_chat[-1]['content']
        # decode
        response = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data.non_tensor_batch['reward_model'][i]['ground_truth']

        system_prompt = (
            "You are a meticulous and expert mathematical judge. Your task is to evaluate a student's "
            "solution to a math word problem based on a provided ground truth solution. "
        )
        user_content = (
            f'<response> \n {response} \n </response>\n\n'
            f'<ground_truth> \n {ground_truth} \n </ground_truth>\n\n'
            """Analyze the semantic similarity between 'response' and 'ground_truth' in the current context. 
And Evaluate the Reasoning and Mathematical Accuracy of the student's solution.
Provide a numerical score between 0 and 5 as a 5-point similarity rating. 
Note that you must provide the shortest possible answer, only one number with your score.
example:  \\boxed{5}"""
        )
        chat = [{'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_content}]
        chats.append(chat)

        from verl.utils.reward_score.gsm8k import compute_score
        score = compute_score(response, ground_truth)
        extra_scores.append(score)

    device = data.batch["responses"].device
    scores_tensor = torch.tensor(extra_scores, dtype=torch.float32, device=device)
    extra_dataproto = DataProto.from_dict({"scores": scores_tensor})

    return chats, extra_dataproto


def reward_postprocess(
        output_data: DataProto,
        tokenizer: PreTrainedTokenizer,
        extra_data: Optional[TensorDict] = None,
    ) -> Tensor:
    """
    Parses the generated text from a Causal LLM to extract a numerical score.

    This function expects the score to be in a "\\boxed{score}" format. It handles
    cases where the score is missing, malformed, or out of range.

    args:
        output_data: DataProto containing the generated sequences from the reward model.
        tokenizer: The tokenizer used for decoding.
        rm_info: Optional list of pre-calculated scores from the preprocessing step.
                 This can be used for more complex scoring logic, e.g., as a fallback.

    return:
        A torch.Tensor of float32 scores for the batch.
    """
    # Extract generated sequences from the DataProto
    generated_sequences = output_data.batch['responses']
    
    # Decode the batch of sequences into text
    generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
    if extra_data is not None:
        extra_scores = extra_data['scores']
    else:
        extra_scores = torch.zeros(len(generated_texts), dtype=torch.float32)
    
    scores = []
    for i, text in enumerate(generated_texts):
        # Use regex to find all occurrences of \boxed{...}
        matches = re.findall(r'\\boxed\{(\d+)\}', text)
        
        try:
            # If matches are found, take the last one as it's most likely the final answer
            last_match_str = matches[-1]
            # Convert the extracted string to a float
            score = float(last_match_str)
            score = torch.clamp(torch.tensor(score), min=0.0, max=5.0).item()
        except Exception:
            logger.warning(
                f"Could not parse valid float score in generated text: '{text}'. "
                "Defaulting to 0.0."
            )
            score = 0.0
            
        scores.append( (score + extra_scores[i].item()*5)/2 )

    # Convert the list of scores to a PyTorch tensor on the correct device
    device = generated_sequences.device
    rm_score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    
    return rm_score_tensor

