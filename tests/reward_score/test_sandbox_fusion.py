import pytest
import os # For possible configurations, e.g., API URL or authentication
from unittest.mock import patch, MagicMock, call # 导入 mock
import time # 导入 time 用于模拟延迟

# Import the function to be tested
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness, call_sandbox_api # 导入 call_sandbox_api 以便 mock

# Get SANDBOX_URL from environment variable
SANDBOX_URL = os.environ.get("SANDBOX_FUSION_URL")
# Define skip condition and reason
skip_reason = "SANDBOX_FUSION_URL environment variable not set"
skip_condition = not SANDBOX_URL

# --- Test code (for real API calls) ---
CODE_SUCCESS = """
import sys
data = sys.stdin.read()
if data == 'input1':
    print('output1\\n', end='')
elif data == 'input2':
    print('output2\\n', end='')
else:
    print('unexpected input', end='')
"""

CODE_WRONG_OUTPUT = """
print('wrong_output\\n', end='')
"""

CODE_COMPILE_ERROR = """
a=b
"""

CODE_RUNTIME_ERROR = """
import sys
print("About to raise error", file=sys.stderr)
raise ValueError("This is a runtime error")
"""

CODE_TIMEOUT = """
import time
import sys
print("Sleeping...", file=sys.stderr)
time.sleep(10) # Sleep time should be longer than the timeout set in the test
print("Finished sleeping", file=sys.stderr)
"""

# --- Test input/output data ---
INPUT_OUTPUT_VALID = {
    "inputs": ["input1", "input2"],
    "outputs": ["output1\n", "output2\n"]
}

INPUT_OUTPUT_SINGLE = {
    "inputs": ["input1"],
    "outputs": ["output1\n"]
}

INPUT_OUTPUT_MISMATCH = {
    "inputs": ["input1"],
    "outputs": ["output1\n", "output2\n"]
}

INPUT_OUTPUT_INVALID_MISSING_KEY = {"inputs": ["input1"]}

# --- Integration test cases (calling real API) ---

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_correct():
    """Integration test: Code is correct, output is correct"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_SUCCESS)
    assert results == [True, True]
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[0]["stdout"] == "output1\n"
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[1]["stdout"] == "output2\n"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_wrong_output():
    """Integration test: Code runs successfully, but output is wrong"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_WRONG_OUTPUT)
    assert results == [False, False]
    assert metadata_list[0]["status"] == "wrong_answer"
    assert metadata_list[0]["stdout"] == "wrong_output\n"
    assert metadata_list[1]["status"] == "wrong_answer"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_compile_error():
    """Integration test: Code causes compile error"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_COMPILE_ERROR, language="cpp")
    assert results == [-4, -4]
    assert metadata_list[0]["status"] == "compile_error"
    assert metadata_list[1]["status"] == "compile_error"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_error():
    """Integration test: Code causes runtime error"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_RUNTIME_ERROR)
    assert results == [-2]
    assert metadata_list[0]["status"] == "runtime_error"
    # More assertions can be added based on the actual API response, e.g., exit_code, stderr

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_timeout():
    """Integration test: Code causes runtime timeout"""
    test_timeout = 5 # Set a timeout shorter than the sleep time in CODE_TIMEOUT
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_TIMEOUT, timeout=test_timeout)
    assert results == [-3]
    assert metadata_list[0]["status"] == "timeout"
    # More assertions can be added based on the actual API response, e.g., run_status

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_concurrency_high_load():
    """Integration test: High concurrency (100 cases) against real API with mixed results (success, wrong answer, timeout)"""
    concurrency_level = 100
    # Indices for different expected outcomes
    wrong_answer_indices = {10, 25, 50}
    timeout_indices = {5, 30, 60, 90} # Indices where we expect a timeout

    # Generate 100 input/output pairs and code
    high_load_inputs = []
    high_load_outputs = []
    expected_results_map = {} # Store expected result for each index

    for i in range(concurrency_level):
        if i in timeout_indices:
            # Use a special input to trigger timeout in the code
            high_load_inputs.append(f"input_timeout_{i}")
            # Output doesn't matter for timeout, but keep it consistent
            high_load_outputs.append(f"output_{i}\n")
            expected_results_map[i] = -3 # Expect timeout
        elif i in wrong_answer_indices:
            high_load_inputs.append(f"input_{i}")
            # Intentionally set wrong expected output
            high_load_outputs.append(f"wrong_output_{i}\n")
            expected_results_map[i] = False # Expect wrong answer
        else:
            high_load_inputs.append(f"input_{i}")
            # Correct expected output
            high_load_outputs.append(f"output_{i}\n")
            expected_results_map[i] = True # Expect success

    high_load_in_outs = {"inputs": high_load_inputs, "outputs": high_load_outputs}

    # Code that handles normal inputs, and sleeps on specific "timeout" inputs
    code_mixed_concurrent = """
import sys
import time
data = sys.stdin.read()
if data.startswith('input_timeout_'):
    time.sleep(20) # Sleep longer than the test timeout
    print(f"output_{data.split('_')[-1]}\\n", end='') # Still print something in case it finishes early
elif data.startswith('input_'):
    print(f"output_{data.split('_')[-1]}\\n", end='')
else:
    print("unknown_input\\n", end='')
"""
    # Set a reasonable timeout per case (must be less than the sleep time in the code)
    test_timeout = 15 # Allow slightly more time due to potential API load, but less than 20s sleep

    start_time = time.time()
    results, metadata_list = check_correctness(
        SANDBOX_URL,
        high_load_in_outs,
        code_mixed_concurrent, # Use the new code
        timeout=test_timeout
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nHigh concurrency test ({concurrency_level} cases with {len(wrong_answer_indices)} wrong answers, {len(timeout_indices)} timeouts) duration: {duration:.2f} seconds")

    # Verify results against the expected map
    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"

    correct_count = 0
    wrong_count = 0
    timeout_count = 0
    unexpected_results = []
    for i, r in enumerate(results):
        expected = expected_results_map[i]
        if r == expected:
            if expected is True:
                correct_count += 1
            elif expected is False:
                wrong_count += 1
            elif expected == -3:
                timeout_count += 1
        else:
            unexpected_results.append((i, r, f"Expected {expected}"))

    print(f"Correct results (True): {correct_count}/{concurrency_level - len(wrong_answer_indices) - len(timeout_indices)}")
    print(f"Expected wrong answers (False, correctly identified): {wrong_count}/{len(wrong_answer_indices)}")
    print(f"Expected timeouts (-3, correctly identified): {timeout_count}/{len(timeout_indices)}")

    if unexpected_results:
        print("Unexpected results found:")
        for idx, res, expected_str in unexpected_results[:10]: # Print first 10 unexpected
             print(f"  Index {idx}: Got {res}, {expected_str}. Metadata: {metadata_list[idx]}")
        assert False, f"Found {len(unexpected_results)} unexpected results."

    assert correct_count == concurrency_level - len(wrong_answer_indices) - len(timeout_indices), "Incorrect number of successful results"
    assert wrong_count == len(wrong_answer_indices), "Incorrect number of identified wrong answers"
    assert timeout_count == len(timeout_indices), "Incorrect number of identified timeouts"

    # Verify metadata count and basic status of one of each type
    assert len(metadata_list) == concurrency_level
    # Find the first correct index
    first_correct_index = next(i for i in range(concurrency_level) if i not in wrong_answer_indices and i not in timeout_indices)
    assert metadata_list[first_correct_index]["status"] == "success"
    assert metadata_list[first_correct_index]["stdout"] == f"output_{first_correct_index}\n"

    # Check the status of the first intentionally wrong case
    first_wrong_index = min(wrong_answer_indices)
    assert metadata_list[first_wrong_index]["status"] == "wrong_answer"
    assert metadata_list[first_wrong_index]["stdout"] == f"output_{first_wrong_index}\n"
    assert metadata_list[first_wrong_index]["expected_output"] == f"wrong_output_{first_wrong_index}\n"

    # Check the status of the first intentionally timeout case
    first_timeout_index = min(timeout_indices)
    assert metadata_list[first_timeout_index]["status"] == "timeout"
    # For timeout, stdout might be None or empty depending on when the timeout occurred
    # assert metadata_list[first_timeout_index]["stdout"] is None or metadata_list[first_timeout_index]["stdout"] == ""


# --- Unit test cases (using mock) ---


@patch('verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api')
def test_unit_concurrency_order(mock_call_sandbox_api):
    """单元测试：验证并发执行时结果顺序是否正确"""
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {
        "inputs": ["input1", "input2", "input3"],
        "outputs": ["output1", "output2", "output3"]
    }

    # 模拟 call_sandbox_api 的行为，故意让第二个调用延迟返回
    def side_effect(*args, **kwargs):
        stdin = kwargs.get('stdin')
        if stdin == 'input1':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}}, None)
        elif stdin == 'input2':
            time.sleep(0.1) # 模拟延迟
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output2", "return_code": 0}}, None)
        elif stdin == 'input3':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}}, None)
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    # 验证结果列表的顺序是否与输入顺序一致
    assert results == [True, True, True]
    # 验证元数据列表的顺序和内容
    assert len(metadata_list) == 3
    assert metadata_list[0]["case_index"] == 0
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["case_index"] == 1
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[2]["case_index"] == 2
    assert metadata_list[2]["status"] == "success"
    # 验证 mock 被调用了三次
    assert mock_call_sandbox_api.call_count == 3

@patch('verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api')
def test_unit_api_timeout_error_concurrent(mock_call_sandbox_api):
    """单元测试：验证并发执行中某个 API 调用超时失败"""
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {
        "inputs": ["input1", "input2_timeout", "input3"],
        "outputs": ["output1", "output2", "output3"]
    }

    # 模拟 call_sandbox_api 的行为，让第二个调用返回 API 错误
    api_error_message = "API Call Failed: Gateway Timeout (504) on attempt 3/3"
    def side_effect(*args, **kwargs):
        stdin = kwargs.get('stdin')
        if stdin == 'input1':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}}, None)
        elif stdin == 'input2_timeout':
            return (None, api_error_message) # 模拟 API 调用失败
        elif stdin == 'input3':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}}, None)
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    # 验证结果列表，API 超时应用例返回 -1
    assert results == [True, -1, True]
    # 验证元数据列表
    assert len(metadata_list) == 3
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["status"] == "api_error"
    assert metadata_list[1]["api_request_error"] == api_error_message
    assert metadata_list[2]["status"] == "success"
    # 验证 mock 被调用了三次
    assert mock_call_sandbox_api.call_count == 3


# --- 保持现有的集成测试用例 ---

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_correct():
    """Integration test: Code is correct, output is correct"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_SUCCESS)
    assert results == [True, True]
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[0]["stdout"] == "output1\n"
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[1]["stdout"] == "output2\n"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_wrong_output():
    """Integration test: Code runs successfully, but output is wrong"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_WRONG_OUTPUT)
    assert results == [False, False]
    assert metadata_list[0]["status"] == "wrong_answer"
    assert metadata_list[0]["stdout"] == "wrong_output\n"
    assert metadata_list[1]["status"] == "wrong_answer"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_compile_error():
    """Integration test: Code causes compile error"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_COMPILE_ERROR, language="cpp")
    assert results == [-4, -4]
    assert metadata_list[0]["status"] == "compile_error"
    assert metadata_list[1]["status"] == "compile_error"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_error():
    """Integration test: Code causes runtime error"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_RUNTIME_ERROR)
    assert results == [-2]
    assert metadata_list[0]["status"] == "runtime_error"
    # More assertions can be added based on the actual API response, e.g., exit_code, stderr

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_timeout():
    """Integration test: Code causes runtime timeout"""
    test_timeout = 5 # Set a timeout shorter than the sleep time in CODE_TIMEOUT
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_TIMEOUT, timeout=test_timeout)
    assert results == [-3]
    assert metadata_list[0]["status"] == "timeout"
    # More assertions can be added based on the actual API response, e.g., run_status

# --- Unit test cases (do not depend on external API, but function signature has changed) ---

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_unit_invalid_input_format():
    """Unit test: Invalid in_outs format passed"""
    results, metadata_list = check_correctness(SANDBOX_URL, None, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

    results, metadata_list = check_correctness(SANDBOX_URL, {}, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_INVALID_MISSING_KEY, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_unit_input_output_mismatch():
    """Unit test: Mismatch between the number of inputs and outputs"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_MISMATCH, CODE_SUCCESS)
    assert results == [-1]
    assert len(metadata_list) == 1
    assert metadata_list[0]["error"] == "Input/output count mismatch"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_concurrency_all_timeout():
    """Integration test: High concurrency (100 cases) against real API, all causing timeout"""
    concurrency_level = 100
    code_infinite_loop = '''
def knight_moves(X, Y):
    MOD = 10**9 + 7
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    dp[0][0] = 1
    for i in range(1, X + 1):
        for j in range(1, Y + 1):
            dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % MOD
    return dp[X][Y]

def solve():
    X, Y = map(int, input().split())
    print(knight_moves(X, Y))

if __name__ == "__main__":
    solve()
    '''

    # Generate 100 simple input/output pairs (content doesn't matter)
    timeout_inputs = [f"324 384429" for i in range(concurrency_level)]
    timeout_outputs = [f"output_{i}\n" for i in range(concurrency_level)]
    timeout_in_outs = {"inputs": timeout_inputs, "outputs": timeout_outputs}

    # Set a timeout for the test cases
    test_timeout = 10 # Set a timeout value

    start_time = time.time()
    results, metadata_list = check_correctness(
        SANDBOX_URL,
        timeout_in_outs,
        code_infinite_loop,
        timeout=test_timeout
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nHigh concurrency all timeout test ({concurrency_level} cases) duration: {duration:.2f} seconds")

    # Verify all results are -3 (timeout)
    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"
    all_timed_out = all(r == -3 for r in results)
    if not all_timed_out:
        non_timeout_indices = [i for i, r in enumerate(results) if r != -3]
        print(f"Indices that did not time out: {non_timeout_indices}")
        # Print metadata for the first few non-timeout cases for debugging
        for i in non_timeout_indices[:5]:
            print(f"Metadata for non-timeout case {i}: {metadata_list[i]}")
    assert all_timed_out, f"Not all {concurrency_level} concurrent tests resulted in timeout (-3). Results: {results}"

    # Verify metadata count and status of the first case
    assert len(metadata_list) == concurrency_level
    assert metadata_list[0]["status"] == "timeout"


# --- Unit test cases (using mock) ---


@patch('verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api')
def test_unit_concurrency_order(mock_call_sandbox_api):
    """单元测试：验证并发执行时结果顺序是否正确"""
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {
        "inputs": ["input1", "input2", "input3"],
        "outputs": ["output1", "output2", "output3"]
    }

    # 模拟 call_sandbox_api 的行为，故意让第二个调用延迟返回
    def side_effect(*args, **kwargs):
        stdin = kwargs.get('stdin')
        if stdin == 'input1':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}}, None)
        elif stdin == 'input2':
            time.sleep(0.1) # 模拟延迟
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output2", "return_code": 0}}, None)
        elif stdin == 'input3':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}}, None)
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    # 验证结果列表的顺序是否与输入顺序一致
    assert results == [True, True, True]
    # 验证元数据列表的顺序和内容
    assert len(metadata_list) == 3
    assert metadata_list[0]["case_index"] == 0
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["case_index"] == 1
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[2]["case_index"] == 2
    assert metadata_list[2]["status"] == "success"
    # 验证 mock 被调用了三次
    assert mock_call_sandbox_api.call_count == 3

@patch('verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api')
def test_unit_api_timeout_error_concurrent(mock_call_sandbox_api):
    """单元测试：验证并发执行中某个 API 调用超时失败"""
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {
        "inputs": ["input1", "input2_timeout", "input3"],
        "outputs": ["output1", "output2", "output3"]
    }

    # 模拟 call_sandbox_api 的行为，让第二个调用返回 API 错误
    api_error_message = "API Call Failed: Gateway Timeout (504) on attempt 3/3"
    def side_effect(*args, **kwargs):
        stdin = kwargs.get('stdin')
        if stdin == 'input1':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}}, None)
        elif stdin == 'input2_timeout':
            return (None, api_error_message) # 模拟 API 调用失败
        elif stdin == 'input3':
            return ({"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}}, None)
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    # 验证结果列表，API 超时应用例返回 -1
    assert results == [True, -1, True]
    # 验证元数据列表
    assert len(metadata_list) == 3
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["status"] == "api_error"
    assert metadata_list[1]["api_request_error"] == api_error_message
    assert metadata_list[2]["status"] == "success"
    # 验证 mock 被调用了三次
    assert mock_call_sandbox_api.call_count == 3

