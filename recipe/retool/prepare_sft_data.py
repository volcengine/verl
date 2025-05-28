import datasets
import pandas as pd

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

sft_dataset = datasets.load_dataset('JoeYing/ReTool-SFT')['train']

ci_user_prompt_template_v3 = '''Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. 
Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.
The last part of your response should be in the following format:
<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>

*user question:*
{question}

Remember to place the final answer in the last part using the format: 
<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>'''


def make_map_fn(split):
    def process_fn(example, idx):
        messages = example.pop("messages")

        prompt = messages[0]['content']
        response = messages[1]['content']

        # rewrite prompt to follow DAPO reward manager

        # step 1: extract raw problem
        raw_problem = prompt.split('*user question:*')[1].split('Remember to place the final answer in the last part using the format: \n<answer>')[0]

        removed_instruction = '''The last part of your response should be in the following format:
<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>'''

        prompt = prompt.replace(removed_instruction, '')

        removed_instruction = '''Remember to place the final answer in the last part using the format: 
<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>'''

        prompt = prompt.replace(removed_instruction, '')

        prompt = prompt + "Remember to put your answer on its own line after \"Answer:\"."

        # extract last box from response
        last_box = last_boxed_only_string(response)
        answer = remove_boxed(last_box)
        
        raw_solution = response.split('<answer>')[0]

        new_solution = raw_solution + f'\nAnswer: {answer}'

        question = question_raw + " " + instruction_following

        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data

    return process_fn


