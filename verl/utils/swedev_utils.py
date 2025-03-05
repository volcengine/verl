import requests
import hashlib
import random
import torch
from tensordict import TensorDict

OBS_ERROR_MSG = "obs too long"

def is_stop(response):
    is_finish = "<function=finish>" in response
    return is_finish

def initialize_runtime(idx, instance_id):
    url = get_api(type="start")
    payload = {"instance_hash": instance_id}
    try:
        api_response = requests.post(url, json=payload)
        return idx, api_response.json()
    except Exception as e:
        print(f"Initializing - API call failed: {e}")
        return idx, None

def call_observation_api(sid, text: str):
    """Call external API to get observation for a given text"""
    if type(sid) == torch.Tensor:
        sid = sid.item()
    url = get_api(type="action")
    payload = {
        "sid": sid,
        "content": text,
    }
    try:
        api_response = requests.post(url, json=payload)
        return api_response.json()
    except Exception as e:
        print(f"Observation - API call failed: {e}")
        return None    

def calc_reward(reward_json):
    # patch_is_None
    # patch_exists
    # patch_succesfully_applied
    # resolved
    # f2p_count
    # f2p_total
    # p2p_count
    # p2p_total
    
    if 'reward' in reward_json:
        return reward_json['reward']
    else:
        try:
            return reward_json['f2p_count'] / reward_json['f2p_total']
        except:
            return 0.0
    
def get_generation_tokens(role, model="qwen"):
    assert role in ["user", "assistant"], f"role {role} not supported"
    if model == "qwen":
        if role == "user":
            return [151645, 198, 151644, 872, 198]
        elif role == "assistant":
            return [151645, 198, 151644, 77091, 198]
    else:
        raise NotImplementedError
    
def expand_tensor_dict(td, repeats: int):
    # Access the TensorDict inside DataProto's 'batch' attribute
    assert repeats >= 1
    duplicated_values = []
    for key in sorted(td.keys()):
        for _ in range(repeats - 1):
            duplicated_values.append(td[key])
    return duplicated_values
  
def get_api(type):
    base_url = random.sample([
        "http://60.165.239.98:5000",
        "http://60.165.239.99:5000"
    ], 1)[0]
    # TODO: only support shouyun1 now
    base_url = "http://60.165.239.98:5000"
    assert type in ["reward", "action", "start", "postprocess"]
    if type == "reward":
        return f"{base_url}/compute_reward"
    elif type == "action":
        return f"{base_url}/process_action"
    elif type == "start":
        return f"{base_url}/start_instance"
    elif type == "postprocess":
        return f"{base_url}/postprocess"

def remove_trailing_pad_tokens(tensor, pad_token_id):
    tensor = tensor.flip(dims=[0])
    non_pad_indices = (tensor != pad_token_id).nonzero(as_tuple=True)[0]
    if len(non_pad_indices) > 0:
        first_non_pad_index = non_pad_indices[0].item()
        tensor = tensor[first_non_pad_index:]
    tensor = tensor.flip(dims=[0])
    return tensor

def get_system_prompt():
    return "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\n<IMPORTANT>\n* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.\n* When configuring git credentials, use \"openhands\" as the user.name and \"openhands@all-hands.dev\" as the user.email by default, unless explicitly instructed otherwise.\n* The assistant MUST NOT include comments in the code unless they are necessary to describe non-obvious behavior.\n</IMPORTANT>\nYou have access to the following functions:\n\n---- BEGIN FUNCTION #1: execute_bash ----\nDescription: Execute a bash command in the terminal.\n* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.\n* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.\n* Timeout: If a command execution result says \"Command timed out. Sending SIGINT to the process\", the assistant should retry running the command in the background.\n\nParameters:\n  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.\n---- END FUNCTION #1 ----\n\n---- BEGIN FUNCTION #2: finish ----\nDescription: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.\nNo parameters are required for this function.\n---- END FUNCTION #2 ----\n\n---- BEGIN FUNCTION #3: str_replace_editor ----\nDescription: Custom editing tool for viewing, creating and editing files\n* State is persistent across command calls and discussions with the user\n* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* The `create` command cannot be used if the specified `path` already exists as a file\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n* The `undo_edit` command will revert the last edit made to the file at `path`\n\nNotes for using the `str_replace` command:\n* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique\n* The `new_str` parameter should contain the edited lines that should replace the `old_str`\n\nParameters:\n  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.\nAllowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]\n  (2) path (string, required): Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.\n  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.\n  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.\n  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.\n  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.\n  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.\n---- END FUNCTION #3 ----\n\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<function=example_function_name>\n<parameter=example_parameter_1>value_1</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format, start with <function= and end with </function>\n- Required parameters MUST be specified\n- Only call one function at a time\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"

def _get_swebench_workspace_dir_name(instance: dict) -> str:
    if isinstance(instance["version"], int):
        instance["version"] = f'{instance["version"]}.0'
    return f'{instance["repo"]}__{instance["version"]}'.replace('/', '__')

IN_CONTEXT_LEARNING_EXAMPLE_PREFIX = """
Here's a running example of how to perform a task with the provided tools.

--------------------- START OF EXAMPLE ---------------------

USER: Create a list of numbers from 1 to 10, and display them in a web page at port 5000.

ASSISTANT: Sure! Let me first check the current directory:
<function=execute_bash>
<parameter=command>
pwd && ls
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
/workspace
openhands@runtime:~/workspace$

ASSISTANT: There is no `app.py` file in the current directory. Let me create a Python file `app.py`:
<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/workspace/app.py</parameter>
<parameter=file_text>
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    numbers = list(range(1, 11))
    return str(numbers)

if __name__ == '__main__':
    app.run(port=5000)
</parameter>
</function>

USER: EXECUTION RESULT of [str_replace_editor]:
File created successfully at: /workspace/app.py

ASSISTANT: I have created a Python file `app.py` that will display a list of numbers from 1 to 10 when you run it. Let me run the Python file for you:
<function=execute_bash>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
[1] 121
[1]+  Exit 1                  python3 app.py > server.log 2>&1

ASSISTANT: Looks like the server is running with PID 121 then crashed. Let me check the server log:
<function=execute_bash>
<parameter=command>
cat server.log
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
Traceback (most recent call last):
  File "/workspace/app.py", line 2, in <module>
    from flask import Flask
ModuleNotFoundError: No module named 'flask'

ASSISTANT: Looks like the server crashed because the `flask` module is not installed. Let me install the `flask` module for you:
<function=execute_bash>
<parameter=command>
pip3 install flask
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
Defaulting to user installation because normal site-packages is not writeable
Collecting flask
  Using cached flask-3.0.3-py3-none-any.whl (101 kB)
Collecting blinker>=1.6.2
  Using cached blinker-1.7.0-py3-none-any.whl (13 kB)
Collecting Werkzeug>=3.0.0
  Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)
Collecting click>=8.1.3
  Using cached click-8.1.7-py3-none-any.whl (97 kB)
Collecting itsdangerous>=2.1.2
  Using cached itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Requirement already satisfied: Jinja2>=3.1.2 in /home/openhands/.local/lib/python3.10/site-packages (from flask) (3.1.3)
Requirement already satisfied: MarkupSafe>=2.0 in /home/openhands/.local/lib/python3.10/site-packages (from Jinja2>=3.1.2->flask) (2.1.5)
Installing collected packages: Werkzeug, itsdangerous, click, blinker, flask
Successfully installed Werkzeug-3.0.2 blinker-1.7.0 click-8.1.7 flask-3.0.3 itsdangerous-2.2.0

ASSISTANT: The `flask` module is installed successfully. Let me run the Python file again:
<function=execute_bash>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
[1] 124

ASSISTANT:
Let me check the server log again:
<function=execute_bash>
<parameter=command>
cat server.log
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
* Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit

ASSISTANT:
The server is running on port 5000 with PID 124. You can access the list of numbers by visiting http://127.0.0.1:5000. If you have any further questions, feel free to ask!

USER: Now kill the server, make it display the numbers in a table format.

ASSISTANT:
Sure! Let me stop the server first:
<function=execute_bash>
<parameter=command>
kill 124
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
[1]+  Terminated              python3 app.py > server.log 2>&1

ASSISTANT:
Now let me display the numbers in a table format:
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/workspace/app.py</parameter>
<parameter=old_str>return str(numbers)</parameter>
<parameter=new_str>return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'</parameter>
</function>

USER: EXECUTION RESULT of [str_replace_editor]:
The file /workspace/app.py has been edited. Here's the result of running `cat -n` on a snippet of /workspace/app.py:
     3
     4  @app.route('/')
     5  def index():
     6      numbers = list(range(1, 11))
     7      return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'
     8
     9  if __name__ == '__main__':
    10      app.run(port=5000)
Review the changes and make sure they are as expected. Edit the file again if necessary.

ASSISTANT:
Running the updated file:
<function=execute_bash>
<parameter=command>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
[1] 126

ASSISTANT:
The server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000. Let me know if you have any further requests!
<function=finish>
</function>

--------------------- END OF EXAMPLE ---------------------

Do NOT assume the environment is the same as in the example above.

--------------------- NEW TASK DESCRIPTION ---------------------
""".lstrip()

IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX = """
--------------------- END OF NEW TASK DESCRIPTION ---------------------

PLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.
""".lstrip()

def get_instruction(instance: dict):
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    instruction = (
        '<uploaded_files>\n'
        f'/workspace/{workspace_dir_name}\n'
        '</uploaded_files>\n'
        f"I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following PR description:\n\n"
        f'<pr_description>\n'
        f'{instance["problem_statement"]}\n'
        '</pr_description>\n\n'
        'Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\n'
        "I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        'Your task is to make the minimal changes to non-tests files in the /workspace directory to ensure the <pr_description> is satisfied.\n'
        'Follow these steps to resolve the issue:\n'
        '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
        '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error\n'
        '3. Edit the sourcecode of the repo to resolve the issue\n'
        '4. Rerun your reproduce script and confirm that the error is fixed!\n'
        '5. Think about edgecases and make sure your fix handles them as well\n'
        "Your thinking should be thorough and so it's fine if it's very long.\n"
    )
    content = (
        IN_CONTEXT_LEARNING_EXAMPLE_PREFIX
        + instruction
        + IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX
    )    
    return content

def split_array(array, split_size, dim=0):
    """
    Split the array according to the specified split_size along the given dimension.
    
    Args:
        array: List where each element may be of shape [response_count, 2] with varying response_count
        split_size: int or list of ints, specifying the size of each chunk
        dim: dimension along which to split, default is 0 (batch dimension)
    
    Returns:
        List of split arrays
    """
    # Ensure dim is 0, as we only handle splitting on batch dimension
    if dim != 0:
        raise ValueError("Only splitting on batch dimension (dim=0) is supported")
    
    batch_size = len(array)
    split_sizes = []
    
    # Calculate split points
    if isinstance(split_size, int):
        idx0 = 0
        while idx0 < batch_size:
            idx1 = min(batch_size, idx0 + split_size)
            split_sizes.append((idx0, idx1))
            idx0 = idx1
    elif isinstance(split_size, (list, tuple)):
        if len(split_size) == 0:
            raise RuntimeError("Insufficient number of elements in split_size.")
        
        idx0 = 0
        for size in split_size:
            if not isinstance(size, int):
                raise TypeError("split(): argument 'split_size' must be int or list of ints")
            idx1 = idx0 + size
            if idx1 > batch_size:
                idx1 = batch_size
            split_sizes.append((idx0, idx1))
            idx0 = idx1
        
        if idx0 < batch_size:
            raise RuntimeError(f"Split method expects split_size to sum exactly to {batch_size} (batch size), but got split_size={split_size}")
    else:
        raise TypeError("split(): argument 'split_size' must be int or list of ints")
    
    # Split the array based on calculated split points
    result = []
    for start, end in split_sizes:
        result.append(array[start:end])
    
    return result

def string_to_hash_tensor(s: str) -> torch.Tensor:
    hash_object = hashlib.md5(s.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    hash_int = hash_int % (2**63)
    return torch.tensor(hash_int, dtype=torch.int64)

def clear_swe_non_batch(data, is_dict=False):
    keys = ['repo', 'base_commit', 'patch', 'test_patch', 'problem_statement', 'hints_text', 'created_at', 'version', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit', 'description']
    keys.append("__index_level_0__") # pandas specific
    keys.append("index") # verl specific
    for key in keys:
        if not is_dict:
            if key in data.non_tensor_batch.keys():
                del data.non_tensor_batch[key]
        else:
            if key in data.keys():
                del data[key]