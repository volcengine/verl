# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import re
import torch
from vllm import LLM, SamplingParams
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.import_utils import import_external_libs
from verl import DataProto
from tensordict import TensorDict


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(1)
    return None

def extract_last_final_answer(text):
    """
    find the contents after the last "Final Answer:"
    """

    pattern1 = r'Final Answer:((?:[^<]|<[^<])*?)\n'
    pattern2 = f'The answer is:((?:[^<]|<[^<])*?)\n'
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    return None

    
def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        # we cannot parse it
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return None #extract_last_final_answer(model_output)

def extract_question(solution_str):
    question = solution_str.split('Please reason step by step')[0].strip().split('<|im_start|>user')[-1].strip()
    return question

class RewardModelWorker(Worker):

    def __init__(self, config):
        print('RewardModelWorker init')
        super().__init__()
        self.config = config
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        self.template = """User: ### Question: {question}\n\n### Ground Truth Answer: {ground_truth}\n\n### Student Answer: {student_answer}\n\nFor the above question, please verify if the student's answer is equivalent to the ground truth answer.\nDo Not solve the question by yourself, just check if the student's answer is equivalent to the ground truth answer.\nIf the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get('external_lib', None))
        self.llm = LLM(model=self.config.model.path)
        self.tokenizer = hf_tokenizer(self.config.model.input_tokenizer, trust_remote_code=self.config.model.get('trust_remote_code', False))
        # self.llm.sleep()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        # self.llm.wake_up()
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        # data = data.to('cuda')
        # recover sequence_str and ground_truth
        sequence_strs = []
        ground_truths = []
        valid_response_lengths = []
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            sequence_strs.append(sequences_str)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            ground_truths.append(ground_truth)
            valid_response_lengths.append(valid_response_length)
            data_source = data_item.non_tensor_batch['data_source']

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < 1:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # extract question
        questions = [extract_question(sequence_str) for sequence_str in sequence_strs]

        # extract solution
        solutions = [extract_solution(sequence_str) for sequence_str in sequence_strs]

        # format message
        messages = [self.template.format(question=question, ground_truth=ground_truth, student_answer=solution) for question, ground_truth, solution in zip(questions, ground_truths, solutions)]

        # generate
        outputs = self.llm.generate(messages, self.sampling_params)

        # extract response
        responses = [output.outputs[0].text.strip() for output in outputs]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i, (ground_truth, solution, verification, valid_response_length) in enumerate(zip(ground_truths, solutions, responses, valid_response_lengths)):
            score = 0
            # extracted successfully
            if solution is None:
                score -= 0.5
            # if solution is None. change to string None to avoid error
            if not solution:
                solution = 'No Answer'
            # pass the verification
            if 'Final Decision: Yes' in verification:
                score += 1
                #penalize the length difference after tokenization
                tokenized_solution = self.tokenizer.encode(solution)
                tokenized_ground_truth = self.tokenizer.encode(ground_truth)
                difference = abs(len(tokenized_solution) - len(tokenized_ground_truth))
                # clip the difference to 10
                difference = min(difference, 10)
                score -= difference * 0.05
            reward_tensor[i, valid_response_length - 1] = score
            print('Valid Response Length:', valid_response_length)
            print('Reward:', score)
            print('Solution:', solution)

        # self.llm.sleep()

        # reward_tensor = reward_tensor.to('cpu')
        # torch.cuda.empty_cache()
        batch = TensorDict({'rm_scores': reward_tensor}, batch_size=reward_tensor.shape[0])

        return DataProto(batch=batch)