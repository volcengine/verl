# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import aiohttp
import asyncio
import traceback
import os
import datetime

from typing import Dict, List, Literal, Callable, Optional

# Global variable to store the path for failed submissions
_failed_submissions_path = os.path.expanduser("~")


def set_failed_submissions_path(path: str):
    """
    Set the path where failed submissions will be saved.

    Args:
        path: The directory path to save failed submissions
    """
    global _failed_submissions_path
    _failed_submissions_path = os.path.expanduser(path)
    # Create directory if it doesn't exist
    os.makedirs(_failed_submissions_path, exist_ok=True)
    print(f"Failed submissions will be saved to: {_failed_submissions_path}")


def get_failed_submissions_path() -> str:
    """
    Get the current path where failed submissions will be saved.

    Returns:
        The current path for saving failed submissions
    """
    return _failed_submissions_path


async def call_long_batch(
                        url: str,
                        submissions: List[Dict],
                        session: aiohttp.ClientSession,
                        max_retries: int = 4,
                        backoff_factor: float = 0.5):

    sub_num = len(submissions)
    results = [None] * sub_num
    sub_ids = list(range(sub_num))
    attempt_count = 0
    while submissions and attempt_count < max_retries:
        attempt_count += 1
        try:
            data = {
                "type": "batch",
                "submissions": submissions
            }
            queue_timeouts = []
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()
                for sub_id, result in zip(sub_ids, response_json['results']):
                    if result['reason'] != 'queue_timeout':
                        results[sub_id] = result
                    else:
                        queue_timeouts.append((sub_id, submissions[sub_id]))
            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]
        except aiohttp.ClientResponseError as e:
            print(f"Attempt {attempt_count}: Server responded with {e.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Attempt {attempt_count}: Caught {type(e).__name__}: {repr(e)}")
        except Exception as e:
            print(f"run_tool_calls_on_server_async Error: {e}")
            traceback.print_exc()
        finally:
            await asyncio.sleep(backoff_factor * (2 ** (attempt_count - 1)))

    # Save failed submissions to file if any remain after max retries
    if submissions:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_file = os.path.join(_failed_submissions_path, f"failed_submissions_{timestamp}.json")

        failed_data = {
            "timestamp": timestamp,
            "url": url,
            "max_retries": max_retries,
            "failed_submissions": []
        }

        for sub_id, submission in zip(sub_ids, submissions):
            failed_data["failed_submissions"].append({
                "original_index": sub_id,
                "submission": submission
            })

        try:
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(submissions)} failed submissions to: {failed_file}")
        except Exception as e:
            print(f"Failed to save failed submissions: {e}")

    return results


async def run_tool_calls_on_server_async(
                                    tool_calls: List,
                                    session: aiohttp.ClientSession,
                                    language: Literal["python", "cpp"] = "python",
                                    max_retries: int = 4,
                                    backoff_factor: float = 0.5,
                                    generate_tool_call_code: Callable = None,
                                    generate_tool_call_input: Callable = None,
                                    host_addr: str = "localhost",
                                    host_port: str = "8088"):
    submissions = []
    for tool_call in tool_calls:
        submissions.append({
            "type": language,
            "solution": generate_tool_call_code(tool_call),
            "input": generate_tool_call_input(tool_call),
        })

    url = f"http://{host_addr}:{host_port}/run/long-batch"
    results = await call_long_batch(url, submissions, session, max_retries, backoff_factor)

    if None in results:
        failed_indices = [i for i, result in enumerate(results) if result is None]
        # throw an error if any tool call failed after max retries
        if len(failed_indices) > 0:
            raise RuntimeError(f"run_tool_calls_on_server_async failed for {len(failed_indices)} tool calls after {max_retries} attempts.")
        
    for i in range(len(results)):
        if results[i]['run_success'] and results[i]['success']:
            output_parts = []
            output_parts.append('Tool call success')
            if results[i]["stdout"]:
                output_parts.append(f'stdout: {results[i]["stdout"]}')
            if results[i]["stderr"]:
                output_parts.append(f'stderr: {results[i]["stderr"]}')
            output_parts.append(f'execution time: {results[i]["cost"]:.2f}s')
            results[i] = '\n'.join(output_parts)
        else:
            output_parts = []
            output_parts.append('Tool call failure')
            output_parts.append(f'reason: {results[i]["reason"]}')
            if results[i]["stdout"]:
                output_parts.append(f'stdout: {results[i]["stdout"]}')
            if results[i]["stderr"]:
                output_parts.append(f'stderr: {results[i]["stderr"]}')
            output_parts.append(f'execution time: {results[i]["cost"]:.2f}s')
            results[i] = '\n'.join(output_parts)

    return results


### Generate tool call code

code_template_setup = '''
import os
import base64
import sys
import ast
import traceback
from typing import Optional, Any
import linecache
from types import CodeType
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

class CodeExecutionError(Exception):
    """Custom exception for code execution errors with line information"""
    def __init__(self, original_error: Exception, code: str, line_offset: int = 0):
        self.original_error = original_error
        self.code = code
        self.line_offset = line_offset
        
        # Get error line number
        if hasattr(original_error, 'lineno'):
            self.lineno = original_error.lineno
        else:
            tb = getattr(original_error, '__traceback__', None)
            if tb:
                while tb.tb_next:
                    tb = tb.tb_next
                self.lineno = tb.tb_lineno
            else:
                self.lineno = -1
        
        # Adjust line number for code segment
        if self.lineno != -1:
            self.lineno += line_offset
        
        # Format error message
        error_type = type(original_error).__name__
        error_msg = str(original_error)
        
        if self.lineno != -1:
            # Get the problematic line
            lines = code.splitlines()
            if 0 <= self.lineno - 1 < len(lines):
                error_line = lines[self.lineno - 1]
                # Create error message with line information
                super().__init__(f"{error_type} at line {self.lineno}: {error_msg}\\n  {error_line}")
                return
        
        super().__init__(f"{error_type}: {error_msg}")

class PersistentExecutor:
    def __init__(self):
        self.exec_globals = {
            '__name__': '__main__',
            '__file__': '<string>',
            '__builtins__': __builtins__
        }

    def split_code(self, code: str) -> tuple[str, Optional[str]]:
        """
        Intelligently split code into main body and last expression
        
        Args:
            code: The source code string
            
        Returns:
            tuple[str, Optional[str]]: (main code body, last expression if exists)
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            if not tree.body:
                return code, None
            
            # Check if the last node is a pure expression (not a call)
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                # Get the line range of the last expression
                last_expr_start = last_node.lineno
                last_expr_end = last_node.end_lineno if hasattr(last_node, 'end_lineno') else last_node.lineno
                
                # Split the code
                lines = code.splitlines()
                main_code = '\\n'.join(lines[:last_expr_start-1])
                last_expr = '\\n'.join(lines[last_expr_start-1:last_expr_end])
                return main_code, last_expr
        except SyntaxError as e:
            raise CodeExecutionError(e, code)
        return code, None

    def execute_code(self, code: str, replay_history_code: bool) -> None:
        """
        Execute code while maintaining persistent environment state.
        If the last line is an expression, its value will be printed to stdout.
        
        Args:
            code: The source code string to execute
            replay_history_code: If True, suppress stdout and stderr output
        """
        try:
            # Split code intelligently
            main_code, last_expr = self.split_code(code)
            
            # Set up output redirection if replay_history_code is True
            if replay_history_code:
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                stdout_context = redirect_stdout(stdout_capture)
                stderr_context = redirect_stderr(stderr_capture)
            else:
                stdout_context = redirect_stdout(sys.stdout)
                stderr_context = redirect_stderr(sys.stderr)
            
            # Execute main code body
            if main_code:
                try:
                    # Compile code to get better error line numbers
                    compiled_code = compile(main_code, '<string>', 'exec')
                    with stdout_context, stderr_context:
                        exec(compiled_code, self.exec_globals)
                except Exception as e:
                    raise CodeExecutionError(e, main_code)
            
            # If there's a last expression, try to evaluate and print it
            if last_expr:
                try:
                    # Compile expression to get better error line numbers
                    compiled_expr = compile(last_expr, '<string>', 'eval')
                    with stdout_context, stderr_context:
                        last_value = eval(compiled_expr, self.exec_globals)
                    
                    # Only print the result if not in replay mode
                    if last_value is not None and not replay_history_code:
                        print(repr(last_value), file=sys.stdout)
                except Exception as e:
                    # Try executing as statement if evaluation fails
                    try:
                        compiled_stmt = compile(last_expr, '<string>', 'exec')
                        with stdout_context, stderr_context:
                            exec(compiled_stmt, self.exec_globals)
                    except Exception as e:
                        # Calculate line offset for the last expression
                        line_offset = len(main_code.splitlines()) if main_code else 0
                        raise CodeExecutionError(e, last_expr, line_offset)
                    
        except Exception as e:
            if replay_history_code:
                return
            if isinstance(e, CodeExecutionError):
                print(str(e), file=sys.stderr)
            else:
                traceback.print_exc(file=sys.stderr)
            os._exit(1)
            return

persistent_executor = PersistentExecutor()
'''

code_template_exec = '''
code_to_execute = base64.b64decode("{}".encode()).decode()
persistent_executor.execute_code(code_to_execute, replay_history_code={})
'''

def combine_code_template(code_to_execute: str, history_code_to_execute: Optional[List[str]] = None) -> str:
    history_code_to_execute = history_code_to_execute or []
    final_code = code_template_setup
    for history_code in history_code_to_execute:
        final_code += code_template_exec.format(history_code, "True")
    final_code += code_template_exec.format(code_to_execute, "False")
    return final_code


def generate_tool_call_code(tool_call: Dict) -> str:
    import base64

    def jupyter_code_gencode(json_format_data: Dict) -> str:
        code_to_execute = base64.b64encode(json_format_data["arguments"]["code"].encode()).decode()
        history_code_to_execute = [
            base64.b64encode(tool_call_json["arguments"]["code"].encode()).decode()
            for tool_call_json in json_format_data.get("history_tool_calls", []) if tool_call_json["name"] == "jupyter_code"
        ]
        return combine_code_template(code_to_execute, history_code_to_execute)

    def python_code_with_standard_io_gencode(json_format_data: Dict) -> str:
        code_to_execute = base64.b64encode(json_format_data["arguments"]["code"].encode()).decode()
        return combine_code_template(code_to_execute)

    if tool_call["name"] == "jupyter_code":
        return jupyter_code_gencode(tool_call)
    elif tool_call["name"] == "python_code_with_standard_io":
        return python_code_with_standard_io_gencode(tool_call)
    else:
        raise ValueError(f"Unsupported tool call name: {tool_call['name']}")


def generate_tool_call_input(tool_call: Dict) -> str:
    if tool_call["name"] == "jupyter_code":
        return None
    elif tool_call["name"] == "python_code_with_standard_io":
        return tool_call["arguments"]["input"]
    else:
        raise ValueError(f"Unsupported tool call name: {tool_call['name']}")
