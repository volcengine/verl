# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import concurrent.futures  # <-- Import concurrent.futures
import functools
import inspect
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Optional

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)

# Sandbox runners by URL scheme, which register_sandbox_runner fills (see its docstring)
_SANDBOX_RUNNERS: dict[str, Callable[..., tuple[Optional[dict[str, Any]], Optional[str]]]] = {}
_runner_failures_logged: set[tuple[str, type]] = set()
# Schemes whose missing or conflicting runner was logged: once, rather than for every case
_runner_lookups_logged: set[str] = set()
_NO_RUNNER = object()
# A URL scheme: a letter or digit, then letters, digits and _ + . -
_SCHEME = re.compile(r"[a-z0-9][a-z0-9_+.-]*")
_runner_logged_lock = threading.Lock()
# Schemes two packages registered runners for, with both; looking one up fails, naming them, since a
# failure at registration would go unseen (verl logs a plugin that fails to load at DEBUG level).
_sandbox_runner_conflicts: dict[str, list[str]] = {}

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


def register_sandbox_runner(scheme: str, runner: Any = _NO_RUNNER) -> Callable[..., Any]:
    """Registers the sandbox runner for ``scheme://`` URLs, for a sandbox another package provides.

    With ``reward.sandbox_fusion.url=<scheme>://...``, each test case goes to the runner instead of
    to a Sandbox Fusion service. A runner takes ``call_sandbox_api``'s arguments by name (and should
    accept ``**kwargs``, for any verl adds), runs the code itself, and returns ``call_sandbox_api``'s
    ``(response, error)``, the response shaped as Sandbox Fusion's ``/run_code`` reply. It is called
    from many threads at once, and enforces ``run_timeout`` itself. A package registers its runner
    when verl imports it as a ``verl.plugins`` entry point.

    Args:
        scheme: The URL scheme (e.g. "mybox"), any case.
        runner: The runner; without it, returns a decorator that registers the function it decorates.

    Returns:
        The runner, or the decorator.
    """

    if not isinstance(scheme, str):
        # @register_sandbox_runner without its scheme
        raise TypeError(f"register_sandbox_runner takes the URL scheme first, not {scheme!r}")
    name = scheme.lower()
    # The rule call_sandbox_api looks a scheme up by, so that what is registered can be reached
    if sandbox_runner_scheme(f"{scheme}://") != name:
        raise ValueError(f"{scheme!r} is not a sandbox runner's scheme: http(s) URLs go to Sandbox Fusion's service")

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(fn):
            raise TypeError(f"The sandbox runner for {name}:// URLs must be callable, not {fn!r}")
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(type(fn).__call__):
            # Called from worker threads, which have no event loop to await it in
            raise TypeError(f"The sandbox runner for {name}:// URLs must be a plain function, not a coroutine function")
        current = _SANDBOX_RUNNERS.get(name)
        # One from the same module replaces it: a re-import or reload of that module
        if (
            current is not None
            and _unwrap(current) is not _unwrap(fn)
            and (_runner_module(fn) is None or _runner_module(current) != _runner_module(fn))
        ):
            runners = _sandbox_runner_conflicts.setdefault(name, [_qualified_name(current)])
            new = _qualified_name(fn)
            if new not in runners:
                runners.append(new)
            logger.warning("Two sandbox runners for %s:// URLs: %s", name, " and ".join(runners))
            return fn
        _SANDBOX_RUNNERS[name] = fn
        return fn

    return register if runner is _NO_RUNNER else register(runner)


def _unwrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn


def _runner_module(fn: Callable[..., Any]) -> Optional[str]:
    # A callable instance's is its class's; None when unknown, which matches no other
    return getattr(_unwrap(fn), "__module__", None)


def _qualified_name(fn: Callable[..., Any]) -> str:
    fn = _unwrap(fn)
    return f"{_runner_module(fn) or '?'}.{getattr(fn, '__qualname__', type(fn).__qualname__)}"


def _check_runner_result(scheme: str, result: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The runner's (response, error), or an error saying what it returned instead."""
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and (result[0] is None or isinstance(result[0], dict))
        and (result[1] is None or isinstance(result[1], str))
        # verl reads both by truth: an empty error is none, and an empty response no answer
        and (result[0] or result[1])
    ):
        return result[0] or None, result[1] or None
    shape = type(result).__name__
    if isinstance(result, tuple):
        shape = f"({', '.join(type(r).__name__ for r in result)})"
    return None, f"The sandbox runner for {scheme}:// returned {shape}, not a response or an error"


def sandbox_runner_scheme(url: str) -> Optional[str]:
    """Returns the scheme of a URL that names a sandbox runner, or None for Sandbox Fusion's service.

    Args:
        url: The sandbox URL, e.g. reward.sandbox_fusion.url.

    Returns:
        The scheme in lower case, or None for an http or https URL (http+unix:// and the like
        included), or one without a scheme.
    """
    if not isinstance(url, str):
        return None
    # Not urlsplit, which drops a scheme that is not RFC 3986's (my_box)
    scheme, separator, _ = url.strip().partition("://")
    scheme = scheme.lower()
    if not separator or not _SCHEME.fullmatch(scheme) or scheme.split("+", 1)[0] in ("http", "https"):
        return None
    return scheme


def get_sandbox_runner(scheme: str) -> Callable[..., tuple[Optional[dict[str, Any]], Optional[str]]]:
    """Returns the sandbox runner registered for ``scheme``.

    Args:
        scheme: The URL scheme (e.g. "mybox"), any case.

    Returns:
        The runner, called as ``call_sandbox_api`` is.

    Raises:
        ValueError: When no package has registered one in this process, or two have.
    """
    name = scheme.lower()
    if name in _sandbox_runner_conflicts:
        raise ValueError(
            f"Two sandbox runners are registered for {name}:// URLs: {' and '.join(_sandbox_runner_conflicts[name])}; "
            "uninstall one of the packages"
        )
    runner = _SANDBOX_RUNNERS.get(name)
    if runner is None:
        raise ValueError(
            f"No sandbox runner is registered for {name}:// URLs (registered: "
            f"{', '.join(sorted(_SANDBOX_RUNNERS)) or 'none'}): the package that provides it registers it when verl "
            "loads its plugins, so it must be installed wherever the reward workers run, and allowed by "
            "VERL_USE_EXTERNAL_PLUGINS, or named in VERL_USE_EXTERNAL_MODULES (a plugin that fails to load is logged "
            "at DEBUG level)"
        )
    return runner


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    A URL whose scheme is not http or https goes to the sandbox runner a package registers for
    that scheme (see ``register_sandbox_runner``).

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API, or of another sandbox by its scheme.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    scheme = sandbox_runner_scheme(sandbox_fusion_url)
    if scheme is not None:
        try:
            runner = get_sandbox_runner(scheme)
        except ValueError as e:
            # The whole message once a scheme: it is the same for every case
            with _runner_logged_lock:
                first = scheme not in _runner_lookups_logged
                _runner_lookups_logged.add(scheme)
            if first:
                logger.error(f"{log_prefix}{e}")
            return None, str(e) if first else f"No usable sandbox runner for {scheme}:// URLs (logged above)"
        try:
            # By name, so that the runner's contract does not hang on this function's parameter order
            result = runner(
                sandbox_fusion_url=sandbox_fusion_url.strip(),
                code=code,
                stdin=stdin,
                compile_timeout=compile_timeout,
                run_timeout=run_timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
        except Exception as e:
            error_msg = f"The sandbox runner for {scheme}:// failed: {e!r}"
            # The traceback once a scheme and kind of failure: a runner that fails does so for every
            # case, from many threads
            with _runner_logged_lock:
                first = (scheme, type(e)) not in _runner_failures_logged
                _runner_failures_logged.add((scheme, type(e)))
            logger.error(f"{log_prefix}{error_msg}", exc_info=first)
            return None, error_msg
        result = _check_runner_result(scheme, result)
        if result[1]:
            logger.error(f"{log_prefix}{result[1]}")
        return result

    # A sandbox runner takes the languages it supports; these are the service's
    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}"
            )  # <-- Use internal log_prefix
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for Gateway Timeout (504) specifically for retrying
            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )  # <-- Use internal log_prefix
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(
                f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
            )  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> tuple[int, dict[str, Any]]:
    """Helper function to process a single test case."""
    api_response = None
    error_msg = None
    logger.info(f"Processing test case {case_index + 1}.")

    current_generation_code = generation

    if fn_name and language == "python":
        # Wrapper assumes stdin_data is a JSON string for function arguments.
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1)
"""
        current_generation_code = wrapper_code

    stdin = None if stdin_data is None else str(stdin_data)
    try:
        if concurrent_semaphore:
            # logger.debug(f"Case {case_index + 1}: Attempting to acquire semaphore.")
            with concurrent_semaphore:
                # logger.debug(f"Case {case_index + 1}: Semaphore acquired. Calling API.")
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=current_generation_code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
            # logger.debug(f"Case {case_index + 1}: Semaphore released.")
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                code=current_generation_code,
                stdin=stdin,
                compile_timeout=timeout,
                run_timeout=timeout,
                memory_limit_mb=memory_limit_mb,
                language=language,
            )
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_msg}")
        traceback.print_exc()

    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = -1  # Default error: API request error or unknown sandbox error

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1  # API request itself failed (includes timeout after retries)
        logger.error(f"Case {case_index}: API error occurred: {error_msg}")
        # Log code and input only on error for brevity
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        logger.error(f"Case {case_index}: code: {generation_to_log}")
        logger.error(f"Case {case_index}: input: {stdin}")
    elif api_response:
        # --- Add debug logging ---
        logger.debug(f"Case {case_index}: API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        # Extract compile information
        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        # Extract run information
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        # --- Determine status based on API response ---
        api_status = metadata["api_status"]

        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1  # Internal sandbox error
        elif api_status == "Failed":
            # --- Add debug logging ---
            logger.debug(f"API returned Failed status. Response: {api_response}")
            logger.debug(f"Compile Result: {compile_result}")
            logger.debug(f"Run Result: {run_result}")
            # --- Check the logic here ---
            # Compile failed or timed out
            is_compile_error = compile_result and (
                metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
            )
            if is_compile_error:
                # Differentiate between compile_error and compile_timeout based on specific status
                if metadata["compile_status"] == "TimeLimitExceeded":
                    metadata["status"] = "compile_timeout"
                else:  # Includes Error and Finished but return_code != 0 cases
                    metadata["status"] = "compile_error"
                result_status = -4
            # Run failed or timed out
            elif run_result:
                # Modified condition: Check for TimeLimitExceeded OR (Finished with non-zero exit code) OR Error status
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                )
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"  # Runtime timeout
                        result_status = -3
                    else:  # Includes Error and Finished with non-zero return_code
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    # Other Failed status with run_result, classify as unknown failure
                    logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                    metadata["status"] = "unknown_failure"
                    result_status = -1  # Default to -1
            else:
                # Status is Failed but neither a clear compile error nor run_result exists
                logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                metadata["status"] = "unknown_failure_state"
                result_status = -1  # Default to -1
        elif api_status == "Success":
            # Run completed successfully, now check the answer
            if run_result and metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                # Note: Output might contain trailing newlines, need normalization
                if expected_output is None or str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                    result_status = True
                    metadata["status"] = "success"
                else:
                    result_status = False
                    metadata["status"] = "wrong_answer"
            else:
                # Status is Success but run_result status is not Finished, this is unexpected
                metadata["status"] = "unexpected_success_state"
                result_status = -1  # Classify as unknown error
        else:
            # API returned an unknown top-level status
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1  # Default to -1
    else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    return result_status, metadata


def check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API, or ``<scheme>://...`` for a sandbox runner
            (see ``register_sandbox_runner``).
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        language: The programming language of the code.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    logger.info("Starting correctness check for generation.")

    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    assert_cases = in_outs.get("assert_case", [""] * num_cases)  # Default to empty strings if not provided
    results = [None] * num_cases  # Initialize with placeholders
    metadata_list = [None] * num_cases  # Initialize with placeholders

    if num_cases == 0:
        logger.warning("Empty inputs provided.")
        return [], []

    if len(inputs) != len(expected_outputs):
        logger.warning(f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)}).")
        # Return error based on the number of inputs provided
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    # If assert_cases is provided, it overrides inputs and outputs
    if len(assert_cases) != num_cases:
        logger.warning(
            f"Mismatch between number of assert cases ({len(assert_cases)}) and inputs/outputs ({num_cases})."
        )
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    first_compile_error_index = -1

    # max_workers is limited by sandbox_fusion_max_concurrent from concurrent_semaphore
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, os.cpu_count() * 5)) as executor:
        # Submit all tasks, passing the concurrent_semaphore to _process_single_case
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation + "\n\n" + assert_cases[i],  # Append assert case to generation
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata

                # Check for compile error (-4)
                if result_status == -4:
                    if first_compile_error_index == -1 or index < first_compile_error_index:
                        first_compile_error_index = index
                    # Optimization: could potentially cancel futures for index > first_compile_error_index
                    # However, cancellation is not guaranteed. Post-processing is safer.

            except Exception as exc:
                logger.error(f"Test case {index} generated an exception: {exc}")
                traceback.print_exc()
                results[index] = -1  # Mark as API/internal error
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]) if expected_outputs[index] else None,
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    # Post-processing for compile errors
    if first_compile_error_index != -1:
        logger.warning(
            f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors."
        )
        for i in range(first_compile_error_index + 1, num_cases):
            # Only update if not already processed (though it should be None or have a result)
            if results[i] != -4:  # Avoid overwriting if it somehow already got -4
                results[i] = -4
                # Update or create metadata for skipped cases due to compile error
                if metadata_list[i] is None:  # If future failed before returning metadata
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]) if expected_outputs[i] else None,
                        "api_request_error": None,
                        "status": "compile_error_skipped",  # Indicate skipped due to prior compile error
                    }
                else:  # If future completed but result is overridden
                    metadata_list[i]["status"] = "compile_error_skipped"

    logger.info(f"Correctness check finished. Results: {results}")
    return results, metadata_list
