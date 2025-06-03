import datasets
import pandas as pd

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

sft_dataset = datasets.load_dataset('JoeYing/ReTool-SFT')['train']

from transformers.utils import get_json_schema

import sys
from io import StringIO

class CaptureOutput:
    def __enter__(self):
        # Save the original streams and create new ones for capture
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_capture = StringIO()
        self.stderr_capture = StringIO()
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Get the captured content
        self.stdout = self.stdout_capture.getvalue()
        self.stderr = self.stderr_capture.getvalue()
        # Close the capture streams
        self.stdout_capture.close()
        self.stderr_capture.close()
        return False  # Do not suppress exceptions


def execute_python_script(script: str):
    """
    Execute a python script and return the stdout and stderr
    
    Args:
        script: the python script to execute
    """
    with CaptureOutput() as captured:
        exec(script)
    output = captured.stdout + '\n' + captured.stderr
    return output
    
execute_python_script_schema = get_json_schema(execute_python_script)


TOOLS = [
    {
        "type": "function",
        "function": execute_python_script_schema
    },
]


def process_single_response(response, new_messages):
    if '\n</code>\n' not in response:
        new_messages.append({
            'role': 'assistant',
            'content': response
        })
    else:
        assert '\n</code>\n' in response
        code, intepreter_response = response.split('\n</code>\n')
        new_messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'execute_python_script', 'arguments': {'script': '{}'.format(code)}}},
        ]},)

        intepreter_response = intepreter_response.split('<interpreter>\n')[1]

        if '\n</interpreter>\n\n' in intepreter_response:
            intepreter_output, response = intepreter_response.split('\n</interpreter>\n\n')
        elif '\n</interpreter>\n' in intepreter_response:
            intepreter_output, response = intepreter_response.split('\n</interpreter>\n')
        else:
            raise ValueError(intepreter_response)

        new_messages.append({'role': 'tool', 'name': 'execute_python_script', 'content': '{}'.format(intepreter_output)},)

        if len(response) > 0:
            # extract <code></code> and <interpreter></interpreter>
            # extract last box from response

            if '<answer>' in response:
                last_box = last_boxed_only_string(response)
                answer = remove_boxed(last_box)

                response = response.split('<answer>')[0]
                response += f'Answer: {answer}'

            new_messages.append({'role': 'assistant', 'content': response},)
        else:
            # consecutive function calls
            pass
        # rewrite answer to follow DAPO reward manager


def make_map_fn(split):
    def process_fn(example, idx):
        messages = example.pop("messages")

        prompt = messages[0]['content']
        response = messages[1]['content']

        # rewrite prompt to follow DAPO reward manager

        new_messages = []

        # step 1: extract raw problem
        raw_problem = prompt.split('*user question:*\n')[1].split('Remember to place the final answer in the last part using the format: \n<answer>')[0]
        prompt = raw_problem + "\nRemember to put your answer on its own line after \"Answer:\"."

        new_messages.append({
            'role': 'user',
            'content': prompt
        })

        responses = response.split('\n<code>\n')

        try:
            for response in responses:
                process_single_response(response, new_messages)

        except Exception as e:
            new_messages = []

        return {
            'messages': new_messages,
            'tools': TOOLS,
        }

    return process_fn


sft_dataset = sft_dataset.map(function=make_map_fn("train"), with_indices=True)
sft_dataset = sft_dataset.filter(lambda x: len(x['messages']) > 0)

sft_dataset.to_parquet('/mnt/hdfs/zhangchi.usc1992_ssd_hldy/public_exp/data/retool_sft_dataset.parquet')


from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from transformers import AutoTokenizer


config = {
    'max_length': 8192,
    'truncation': 'error',
    'multiturn': {
        'messages_key': 'messages',
        'tool_key': 'tools',
    }
}

tokenizer = AutoTokenizer.from_pretrained('/mnt/hdfs/zhangchi.usc1992_lf_lq/models/Qwen3-8B')
dataset = MultiTurnSFTDataset(parquet_files=['/mnt/hdfs/zhangchi.usc1992_ssd_hldy/public_exp/data/retool_sft_dataset.parquet'], 
                              config=config, tokenizer=tokenizer)

