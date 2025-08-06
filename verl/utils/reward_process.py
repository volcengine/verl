from typing import List, Dict, Tuple, Optional
from verl import DataProto
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
from torch import Tensor
from tensordict import TensorDict
import re
import logging
import os

from verl.utils.reward_score.gsm8k import compute_score

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
def reward_preprocess(
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
    ) -> Tuple[List[List[Dict[str, str]]], Optional[Tensor]]:
    """
    Preprocesses data to create prompts for a generative reward model while also computing auxiliary scores.

    This function serves a dual purpose:
    1. Constructs chat-style prompts for the generative reward model (GenRM)
    2. Computes auxiliary scores that can be used by traditional reward functions

    The auxiliary scores allow users to combine both GenRM and traditional reward functions in their pipeline.
    args:
        questions: List of question strings
        answers: List of answer strings to be evaluated
        ground_truths: List of ground truth strings
    return:
        - chats: A list of chat histories formatted as prompts for the generative reward model
        - extra_scores: A tensor of pre-computed auxiliary scores (e.g., exact match scores) or None
    """
    chats = []
    extra_scores = []

    for question, answer, ground_truth in zip(questions, answers, ground_truths):
        system_prompt = (
            "You are a meticulous and expert mathematical judge. Your task is to evaluate a student's "
            "solution to a math word problem based on a provided ground truth solution. "
        )
        user_content = (
            # f'<question> \n {question} \n </question>\n\n'
            f'<response> \n {answer} \n </response>\n\n'
            f'<ground_truth> \n {ground_truth} \n </ground_truth>\n\n'
            """Analyze the semantic similarity between 'response' and 'ground_truth' in the current context.
And Evaluate the Reasoning and Mathematical Accuracy of the student's solution.
Provide a numerical score between 0 and 5 as a 5-point similarity rating.
Note that you must provide the shortest possible answer, only one number with your score.
example:  \\boxed{5}"""
        )
        chat = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content}]
        chats.append(chat)

        score = compute_score(answer, ground_truth)
        extra_scores.append(score)

    scores_tensor = torch.tensor(extra_scores, dtype=torch.float32)
    return chats, scores_tensor

def reward_postprocess(
        gen_rm_responses: list[str],
        scores_tensor: Optional[Tensor] = None,
    ) -> Tensor:
    """
    Parses the generated text from a Causal LLM to extract a numerical score.

    This function expects the score to be in a "\\boxed{score}" format. It handles
    cases where the score is missing, malformed, or out of range.

    args:
        gen_rm_responses: List of generated response strings from the reward model.
        scores_tensor: Optional tensor of pre-calculated scores from the preprocessing step.
                 This can be used for more complex scoring logic, e.g., as a fallback.

    return:
        A torch.Tensor of float32 scores for the batch.
    """
    scores = []
    for text in gen_rm_responses:
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
            
        scores.append(score)
    # Convert the list of scores to a PyTorch tensor
    rm_score_tensor = torch.tensor(scores, dtype=torch.float32)
    
    if scores_tensor is not None:
        rm_score_tensor = rm_score_tensor.to(scores_tensor.device)
        rm_score_tensor = (rm_score_tensor + scores_tensor * 5) / 2

    return rm_score_tensor
