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

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op


@pytest.fixture(autouse=True)
def reset_rollout_trace_config_singleton():
    """Fixture to reset the RolloutTraceConfig singleton before each test."""
    RolloutTraceConfig.reset()


@pytest.fixture
def mock_weave_client():
    """Mocks the weave module and its client, yielding the mock client."""
    mock_weave = MagicMock()
    mock_client = MagicMock()
    mock_call = MagicMock()
    mock_client.create_call.return_value = mock_call
    mock_weave.init.return_value = mock_client

    # Also mock the call_context if it's used internally by the decorator
    mock_weave.trace.context.call_context.return_value = MagicMock()

    with patch.dict(sys.modules, {"weave": mock_weave, "weave.trace.context": mock_weave.trace.context}):
        yield mock_client


class TracedClass:
    @rollout_trace_op
    # @weave.op
    async def my_method(self, a, b="default"):
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @weave.op
    async def middle_method(self, a, b="default"):
        await self.my_method("test_a1", b="test_b1")
        return f"result: {a}, {b}"

    @rollout_trace_op
    async def my_method_with_exception(self):
        raise ValueError("Test Exception")

    async def upper_method(self):
        await self.my_method("test_a0", b="test_b0")
        await self.middle_method("test_a2", b="test_b2")


class UntracedClass:
    @rollout_trace_op
    async def my_method(self, x):
        return x * 2


async def test_rollout_trace_on_untraced_class():
    """Tests that the decorator works correctly when no backend is configured."""
    instance = UntracedClass()
    assert await instance.my_method(10) == 20


async def test_rollout_trace_with_tracer(mock_weave_client):
    """Tests that the decorator calls the tracer's methods correctly."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()
    assert RolloutTraceConfig.get_client() is mock_weave_client

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_weave_client.create_call.assert_called_once()
    call_kwargs = mock_weave_client.create_call.call_args.kwargs
    assert call_kwargs["op"] == "TracedClass.my_method"
    expected_inputs = {"a": "test_a", "b": "test_b"}
    assert call_kwargs["inputs"] == expected_inputs

    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once_with(mock_call, output=result)


async def test_rollout_trace_with_exception(mock_weave_client):
    """Tests that `finish` is called with the exception when one is raised."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    mock_weave_client.create_call.assert_called_once()
    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once()

    # Check that finish_call was called with the exception
    args, kwargs = mock_weave_client.finish_call.call_args
    assert args[0] == mock_call
    assert "exception" in kwargs
    assert isinstance(kwargs["exception"], ValueError)


async def test_rollout_trace_with_dummy_backend(mock_weave_client):
    """Tests that the tracer is not called when the backend is 'dummy'."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="dummy")
    instance = TracedClass()

    await instance.my_method("test_a")

    mock_weave_client.create_call.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("RUN_WEAVE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping weave integration test. Set RUN_WEAVE_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_weave_backend():
    """Integration test with a real weave backend."""

    # This assumes that the weave environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.upper_method()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    print("\nWeave integration test ran successfully. Check your weave project for the trace.")
