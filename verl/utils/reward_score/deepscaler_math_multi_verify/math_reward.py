"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from verl.utils.reward_score.deepscaler_math_multi_verify.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
from verl.utils.reward_score.deepscaler_math_multi_verify.utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd, extract_k_boxed_answers
from verl.utils.reward_score.deepscaler_math_multi_verify.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType


ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
        
        # Check if this is a multi-answer problem
        if len(ground_truths) == 1 and ',' in str(ground_truths[0]):
            # Multi-answer problem with comma-separated answers
            expected_answers = str(ground_truths[0]).split(',')
            expected_answers = [ans.strip() for ans in expected_answers]
            num_problems = len(expected_answers)
            
            # Extract k boxed answers from model response
            model_answers = extract_k_boxed_answers(model_solution, num_problems, strict_order=False)
            
            # Check if we found the expected number of answers
            if len(model_answers) != num_problems:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            
            # Check each answer pair
            for i, (model_ans, expected_ans) in enumerate(zip(model_answers, expected_answers)):
                # Process expected answer if it contains \boxed
                if "\\boxed" in expected_ans:
                    processed_expected = extract_answer(expected_ans)
                    if processed_expected is not None:
                        expected_ans = processed_expected
                
                # Check if this answer is correct
                is_correct = grade_answer_mathd(model_ans, expected_ans) or grade_answer_sympy(model_ans, expected_ans)
                if not is_correct:
                    return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
            
            # All answers are correct
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        
        else:
            # Single answer problem (original logic)
            model_answer = extract_answer(model_solution)
            if model_answer is None:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            
            # Process each ground truth
            processed_ground_truths = []
            for truth in ground_truths:
                truth = str(truth)
                if "\\boxed" in truth:
                    processed_truth = extract_answer(truth)
                    if processed_truth is not None:
                        processed_ground_truths.append(processed_truth)
                else:
                    processed_ground_truths.append(truth)
            
            if not processed_ground_truths:
                return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
            
            # Check against all possible correct answers
            for ground_truth in processed_ground_truths:
                is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
                if is_correct:
                    return RewardOutput(reward=self.config.correct_reward, is_correct=True)
            
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)



def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
