import pytest

import verl.tools.sandbox_fusion_tools as sandbox_fusion_tools
from verl.tools.sandbox_fusion_tools import ExecutionWorker, SandboxFusionTool
from verl.tools.schemas import ToolResponse


def test_execution_worker_execute_without_rate_limiter():
    worker = ExecutionWorker(enable_global_rate_limit=False)
    assert worker.rate_limit_worker is None
    assert worker.execute(lambda: "ok") == "ok"


@pytest.mark.parametrize(
    "metadata, expected_substrings",
    [
        (
            {"run_status": "Finished", "exit_code": 0, "stdout": "hello\n", "stderr": ""},
            ["hello\n"],
        ),
        (
            {"run_status": "Finished", "exit_code": 1, "stdout": "out\n", "stderr": "err\n"},
            ["Code execution failed with status: runtime_error", "exit_code=1", "out\nerr\n"],
        ),
        (
            {"run_status": "CompileError", "status": "compile_error", "compile_stderr": "bad syntax"},
            ["Code execution failed with status: compile_error", "bad syntax"],
        ),
    ],
)
def test_execute_code_returns_plain_text(monkeypatch, metadata, expected_substrings):
    def fake_process_single_case(*args, **kwargs):
        return None, metadata

    monkeypatch.setattr(sandbox_fusion_tools, "_process_single_case", fake_process_single_case)

    tool = SandboxFusionTool.__new__(SandboxFusionTool)
    tool.sandbox_fusion_url = "http://dummy"
    tool.memory_limit_mb = 1024

    result = SandboxFusionTool.execute_code(tool, "instance-1", "print('hi')", timeout=1, language="python")
    assert isinstance(result, str)
    assert not isinstance(result, ToolResponse)
    for substring in expected_substrings:
        assert substring in result

