"""Complete unit tests for HTTP Server Engine Adapters.

This module contains comprehensive unit tests for both HttpServerEngineAdapter
and AsyncHttpServerEngineAdapter classes, covering all public methods,
error handling scenarios, edge cases, and boundary conditions using pytest and mock frameworks.
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import requests

# Now import the module under test
from verl.workers.rollout.sglang_rollout.http_server_engine import (
    AsyncHttpServerEngineAdapter,
    HttpServerEngineAdapter,
    launch_server_process,
)

# Mock SGLang dependencies to make tests independent
sys.modules["sglang"] = Mock()
sys.modules["sglang.srt"] = Mock()
sys.modules["sglang.srt.server_args"] = Mock()
sys.modules["sglang.srt.entrypoints"] = Mock()
sys.modules["sglang.srt.entrypoints.EngineBase"] = Mock()
sys.modules["sglang.srt.entrypoints.http_server"] = Mock()
sys.modules["sglang.srt.utils"] = Mock()


# Create mock ServerArgs class
class MockServerArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Set defaults
        if not hasattr(self, "host"):
            self.host = "localhost"
        if not hasattr(self, "port"):
            self.port = 8000
        if not hasattr(self, "node_rank"):
            self.node_rank = 0
        if not hasattr(self, "api_key"):
            self.api_key = None

    def url(self):
        return f"http://{self.host}:{self.port}"


# Replace the real ServerArgs with our mock
sys.modules["sglang.srt.server_args"].ServerArgs = MockServerArgs


class TestLaunchServerProcess:
    """Test cases for launch_server_process function."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session")
    def test_launch_server_process_success(self, mock_session_class, mock_process_class):
        """Test successful server process launch and health check."""
        # Mock process
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        # Mock session and responses
        mock_session = Mock()
        mock_session_class.return_value.__enter__.return_value = mock_session

        # Mock successful health check responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        # Create server args
        server_args = MockServerArgs(host="localhost", port=8000, node_rank=0, api_key="test-key")

        # Test
        result = launch_server_process(server_args)

        # Assertions
        assert result == mock_process
        mock_process.start.assert_called_once()
        assert mock_session.get.call_count >= 2  # health_generate and flush_cache

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process")
    def test_launch_server_process_non_master(self, mock_process_class):
        """Test server launch for non-master nodes (should return immediately)."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        server_args = MockServerArgs(host="localhost", port=8000, node_rank=1)
        result = launch_server_process(server_args)

        assert result == mock_process
        mock_process.start.assert_called_once()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session")
    def test_launch_server_process_timeout(self, mock_session_class, mock_process_class):
        """Test timeout during server health check."""
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        mock_session = Mock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.get.side_effect = requests.RequestException("Connection failed")

        server_args = MockServerArgs(host="localhost", port=8000, node_rank=0)

        with patch("time.time", side_effect=[0, 400]):  # Simulate timeout
            with pytest.raises(TimeoutError):
                launch_server_process(server_args)

        mock_process.terminate.assert_called_once()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session")
    def test_launch_server_process_died(self, mock_session_class, mock_process_class):
        """Test server process dies during startup."""
        mock_process = Mock()
        mock_process.is_alive.return_value = False
        mock_process_class.return_value = mock_process

        server_args = MockServerArgs(host="localhost", port=8000, node_rank=0)

        with pytest.raises(RuntimeError, match="Server process terminated unexpectedly"):
            launch_server_process(server_args)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestHttpServerEngineAdapter:
    """Test cases for HttpServerEngineAdapter class."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    def test_init_with_router_registration(self, mock_post, mock_launch):
        """Test initialization with router registration."""
        mock_process = Mock()
        mock_launch.return_value = mock_process

        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = HttpServerEngineAdapter(
            router_ip="192.168.1.1",
            router_port=8080,
            host="localhost",
            port=8000,
            node_rank=0,
        )

        assert adapter.router_ip == "192.168.1.1"
        assert adapter.router_port == 8080
        assert adapter.process == mock_process
        mock_post.assert_called_once()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_init_without_router(self, mock_launch):
        """Test initialization without router registration."""
        mock_process = Mock()
        mock_launch.return_value = mock_process

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        assert adapter.router_ip is None
        assert adapter.router_port is None
        assert adapter.process == mock_process

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    def test_register_with_router_failure(self, mock_post, mock_launch):
        """Test router registration failure handling."""
        mock_process = Mock()
        mock_launch.return_value = mock_process

        mock_post.side_effect = requests.RequestException("Connection failed")

        # Should not raise exception, just log error
        adapter = HttpServerEngineAdapter(
            router_ip="192.168.1.1",
            router_port=8080,
            host="localhost",
            port=8000,
            node_rank=0,
        )

        assert adapter.router_ip == "192.168.1.1"
        mock_post.assert_called_once()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    def test_make_request_success(self, mock_post, mock_launch):
        """Test successful HTTP request."""
        mock_launch.return_value = Mock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)
        result = adapter._make_request("test_endpoint", {"param": "value"})

        assert result == {"status": "success"}
        mock_post.assert_called_with(
            "http://localhost:8000/test_endpoint",
            json={"param": "value"},
            timeout=adapter.timeout,
        )

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get")
    def test_make_request_get_method(self, mock_get, mock_launch):
        """Test HTTP GET request."""
        mock_launch.return_value = Mock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)
        result = adapter._make_request("test_endpoint", method="GET")

        assert result == {"data": "test"}
        mock_get.assert_called_with("http://localhost:8000/test_endpoint", timeout=adapter.timeout)

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_make_request_non_master(self, mock_launch):
        """Test request from non-master node returns empty dict."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=1)
        result = adapter._make_request("test_endpoint")

        assert result == {}

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    @patch("time.sleep")
    def test_make_request_retry_logic(self, mock_sleep, mock_post, mock_launch):
        """Test retry logic for failed requests."""
        mock_launch.return_value = Mock()

        # First two calls fail, third succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.ConnectionError(),
            Mock(status_code=200, json=lambda: {"success": True}),
        ]

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0, max_retries=2)
        result = adapter._make_request("test_endpoint")

        assert result == {"success": True}
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    def test_make_request_http_error(self, mock_post, mock_launch):
        """Test HTTP error handling."""
        mock_launch.return_value = Mock()

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = mock_response

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with pytest.raises(requests.exceptions.HTTPError):
            adapter._make_request("test_endpoint")

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    @patch("time.sleep")
    def test_make_request_max_retries_exceeded(self, mock_sleep, mock_post, mock_launch):
        """Test max retries exceeded."""
        mock_launch.return_value = Mock()
        mock_post.side_effect = requests.exceptions.Timeout()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0, max_retries=1)

        with pytest.raises(RuntimeError, match="Failed to complete request"):
            adapter._make_request("test_endpoint")

        assert mock_post.call_count == 2  # Initial + 1 retry

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_update_weights_from_tensor(self, mock_launch):
        """Test update_weights_from_tensor method."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "updated"}

            result = adapter.update_weights_from_tensor(
                ["tensor1", "tensor2"], load_format="safetensors", flush_cache=True
            )

            assert result == {"status": "updated"}
            mock_request.assert_called_once_with(
                "update_weights_from_tensor",
                {
                    "serialized_named_tensors": ["tensor1", "tensor2"],
                    "load_format": "safetensors",
                    "flush_cache": True,
                },
            )

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_generate(self, mock_launch):
        """Test generate method."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"text": "Generated text"}

            result = adapter.generate(
                prompt="Hello world",
                sampling_params={"temperature": 0.7},
                return_logprob=True,
            )

            assert result == {"text": "Generated text"}
            mock_request.assert_called_once_with(
                "generate",
                {
                    "text": "Hello world",
                    "sampling_params": {"temperature": 0.7},
                    "return_logprob": True,
                },
                only_master=False,
            )

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get")
    @patch("time.sleep")
    def test_flush_cache(self, mock_sleep, mock_get, mock_launch):
        """Test flush_cache method."""
        mock_launch.return_value = Mock()

        # First call fails, second succeeds
        mock_responses = [
            Mock(status_code=503),  # Service unavailable
            Mock(status_code=200, json=lambda: {"cache_flushed": True}),
        ]
        mock_get.side_effect = mock_responses

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)
        result = adapter.flush_cache()

        assert result == {"cache_flushed": True}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_flush_cache_non_master(self, mock_launch):
        """Test flush_cache for non-master node."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=1)
        result = adapter.flush_cache()

        assert result == {}

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_memory_management_methods(self, mock_launch):
        """Test memory release and resume methods."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test release_memory_occupation
            result = adapter.release_memory_occupation(["weights", "kv_cache"])
            assert result == {"status": "success"}
            mock_request.assert_called_with("release_memory_occupation", {"tags": ["weights", "kv_cache"]})

            # Test resume_memory_occupation
            result = adapter.resume_memory_occupation(["weights"])
            assert result == {"status": "success"}
            mock_request.assert_called_with("resume_memory_occupation", {"tags": ["weights"]})

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_distributed_weights_methods(self, mock_launch):
        """Test distributed weights update methods."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "initialized"}

            # Test init_weights_update_group
            result = adapter.init_weights_update_group(
                master_address="localhost",
                master_port=29500,
                rank_offset=0,
                world_size=4,
                group_name="test_group",
                backend="nccl",
            )

            assert result == {"status": "initialized"}
            mock_request.assert_called_with(
                "init_weights_update_group",
                {
                    "master_address": "localhost",
                    "master_port": 29500,
                    "rank_offset": 0,
                    "world_size": 4,
                    "group_name": "test_group",
                    "backend": "nccl",
                },
            )

            # Test update_weights_from_distributed
            # Mock torch for dtype testing
            class MockTorchDtype:
                def __init__(self, name):
                    self.name = name

                def __str__(self):
                    return f"torch.{self.name}"

            mock_dtypes = [MockTorchDtype("float32"), MockTorchDtype("float16")]

            mock_request.return_value = {"status": "updated"}
            result = adapter.update_weights_from_distributed(
                names=["layer1.weight", "layer2.bias"],
                dtypes=mock_dtypes,
                shapes=[(1024, 512), (512,)],
                group_name="test_group",
                flush_cache=True,
            )

            assert result == {"status": "updated"}
            mock_request.assert_called_with(
                "update_weights_from_distributed",
                {
                    "names": ["layer1.weight", "layer2.bias"],
                    "dtypes": ["float32", "float16"],
                    "shapes": [(1024, 512), (512,)],
                    "group_name": "test_group",
                    "flush_cache": True,
                },
            )

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_generation_control_methods(self, mock_launch):
        """Test pause and continue generation methods."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "paused"}

            # Test pause_generation
            result = adapter.pause_generation()
            assert result == {"status": "paused"}
            mock_request.assert_called_with("pause_generation", {})

            # Test continue_generation
            mock_request.return_value = {"status": "continued"}
            result = adapter.continue_generation()
            assert result == {"status": "continued"}
            mock_request.assert_called_with("continue_generation", {})

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.kill_process_tree")
    def test_shutdown(self, mock_kill, mock_post, mock_launch):
        """Test shutdown method."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_launch.return_value = mock_process

        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = HttpServerEngineAdapter(
            router_ip="192.168.1.1",
            router_port=8080,
            host="localhost",
            port=8000,
            node_rank=0,
        )

        adapter.shutdown()

        # Should unregister from router
        assert mock_post.call_count == 2  # Once for registration, once for unregistration
        # Should kill process
        mock_kill.assert_called_once_with(12345)

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.kill_process_tree")
    def test_shutdown_with_errors(self, mock_kill, mock_post, mock_launch):
        """Test shutdown method with errors."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_launch.return_value = mock_process

        # Mock registration success but unregistration failure
        mock_post.side_effect = [
            Mock(status_code=200),  # Registration success
            requests.RequestException("Unregistration failed"),  # Unregistration failure
        ]

        # Mock process kill failure
        mock_kill.side_effect = Exception("Kill failed")

        adapter = HttpServerEngineAdapter(
            router_ip="192.168.1.1",
            router_port=8080,
            host="localhost",
            port=8000,
            node_rank=0,
        )

        # Should not raise exceptions
        adapter.shutdown()

        assert mock_post.call_count == 2
        mock_kill.assert_called_once_with(12345)

    # Edge cases for HttpServerEngineAdapter
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_empty_and_none_parameters(self, mock_launch):
        """Test handling of empty and None parameters."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test generate with all None parameters
            result = adapter.generate()
            assert result == {"status": "success"}

            # Test with empty lists
            result = adapter.update_weights_from_tensor([])
            assert result == {"status": "success"}

            # Test with None tags
            result = adapter.release_memory_occupation(None)
            assert result == {"status": "success"}

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_large_payload_handling(self, mock_launch):
        """Test handling of large payloads."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test with large tensor list
            large_tensor_list = [f"tensor_{i}" for i in range(1000)]
            result = adapter.update_weights_from_tensor(large_tensor_list)
            assert result == {"status": "success"}

            # Test with large prompt
            large_prompt = "A" * 10000
            result = adapter.generate(prompt=large_prompt)
            assert result == {"status": "success"}

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_unicode_and_special_characters(self, mock_launch):
        """Test handling of unicode and special characters."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"text": "生成的中文文本 🚀"}

            # Test with unicode prompt
            result = adapter.generate(prompt="你好世界 🌍")
            assert result == {"text": "生成的中文文本 🚀"}

            # Test with special characters in group name
            result = adapter.init_weights_update_group(
                master_address="localhost",
                master_port=29500,
                rank_offset=0,
                world_size=4,
                group_name="test-group_123",
                backend="nccl",
            )

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_malformed_responses(self, mock_launch):
        """Test handling of malformed server responses."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            # Test response without JSON
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_post.return_value = mock_response

            with pytest.raises(json.JSONDecodeError):
                adapter._make_request("test_endpoint")

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_timeout_edge_cases(self, mock_launch):
        """Test various timeout scenarios."""
        mock_launch.return_value = Mock()

        # Test with very small timeout
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0, timeout=0.001)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()

            with pytest.raises(RuntimeError, match="Failed to complete request"):
                adapter._make_request("test_endpoint")

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_extreme_configuration_values(self, mock_launch):
        """Test extreme configuration values."""
        mock_launch.return_value = Mock()

        # Test with extreme values
        adapter = HttpServerEngineAdapter(
            host="localhost",
            port=8000,
            node_rank=0,
            timeout=0.001,  # Very small
            max_retries=100,  # Very large
            retry_delay=0.001,  # Very small
        )

        assert adapter.timeout == 0.001
        assert adapter.max_retries == 100
        assert adapter.retry_delay == 0.001


class TestAsyncHttpServerEngineAdapter:
    """Test cases for AsyncHttpServerEngineAdapter class."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_init(self, mock_launch):
        """Test async adapter initialization."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(
            host="localhost",
            port=8000,
            node_rank=0,
            max_connections=50,
        )

        assert adapter.max_connections == 50
        assert adapter._need_reload is True
        assert adapter._session is None

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_get_session(self, mock_launch):
        """Test session creation and management."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Test session creation
        async with adapter._get_session() as session:
            assert isinstance(session, aiohttp.ClientSession)
            assert not session.closed

        # Session should be reused
        async with adapter._get_session() as session2:
            assert session2 is adapter._session

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_get_session_error_handling(self, mock_launch):
        """Test session error handling and recreation."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Simulate error during session usage
        try:
            async with adapter._get_session():
                raise Exception("Test error")
        except Exception:
            pass

        # Session should be None after error
        assert adapter._session is None

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_make_async_request_success(self, mock_launch):
        """Test successful async HTTP request."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Mock aiohttp session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_response.raise_for_status = Mock()

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.closed = False

        with patch.object(adapter, "_get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            result = await adapter._make_async_request("test_endpoint", {"param": "value"})

            assert result == {"status": "success"}
            mock_session.post.assert_called_once()

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_make_async_request_get_method(self, mock_launch):
        """Test async GET request."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.raise_for_status = Mock()

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.closed = False

        with patch.object(adapter, "_get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            result = await adapter._make_async_request("test_endpoint", method="GET")

            assert result == {"data": "test"}
            mock_session.get.assert_called_once()

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_make_async_request_non_master(self, mock_launch):
        """Test async request from non-master node."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=1)
        result = await adapter._make_async_request("test_endpoint")

        assert result == {}

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_async_generate(self, mock_launch):
        """Test async generate method."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_async_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"text": "Generated text"}

            result = await adapter.generate(
                prompt="Hello world",
                sampling_params={"temperature": 0.7},
                return_logprob=True,
            )

            assert result == {"text": "Generated text"}
            mock_request.assert_called_once()

            # Test async_generate alias
            result2 = await adapter.async_generate(prompt="Test")
            assert result2 == {"text": "Generated text"}

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_async_memory_management(self, mock_launch):
        """Test async memory management methods."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_async_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test release_memory_occupation
            result = await adapter.release_memory_occupation(["weights"])
            assert result == {"status": "success"}

            # Test resume_memory_occupation (should handle _need_reload)
            adapter._need_reload = True
            result = await adapter.resume_memory_occupation(["weights"])
            assert result == {"status": "success"}
            assert adapter._need_reload is False
            assert mock_request.call_count == 3  # release + release (for reload) + resume

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_launch):
        """Test async context manager support."""
        mock_launch.return_value = Mock()

        async with AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0) as adapter:
            assert isinstance(adapter, AsyncHttpServerEngineAdapter)
            assert adapter._session is None  # Not created yet

        # Session should be closed after context exit
        if adapter._session:
            assert adapter._session.closed

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_close_method(self, mock_launch):
        """Test close method."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Create a session
        async with adapter._get_session() as session:
            assert not session.closed

        # Close should close the session
        await adapter.close()
        assert adapter._session is None

        # Multiple close calls should be safe
        await adapter.close()

    # Edge cases for AsyncHttpServerEngineAdapter
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_session_recreation_under_load(self, mock_launch):
        """Test session recreation under high load."""
        mock_launch.return_value = Mock()
        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Simulate session errors that trigger recreation
        session_creation_count = 0

        async def mock_get_session():
            nonlocal session_creation_count
            session_creation_count += 1

            mock_session = AsyncMock()
            mock_session.closed = False

            if session_creation_count <= 2:
                # First two sessions fail
                mock_session.post.side_effect = aiohttp.ClientError("Session error")
            else:
                # Third session succeeds
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"success": True})
                mock_response.raise_for_status = Mock()
                mock_session.post.return_value.__aenter__.return_value = mock_response

            return mock_session

        with patch.object(adapter, "_get_session") as mock_get_session_method:
            mock_get_session_method.side_effect = lambda: mock_get_session()

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await adapter._make_async_request("test_endpoint")
                assert result == {"success": True}
                assert session_creation_count == 3

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self, mock_launch):
        """Test truly concurrent async requests."""
        mock_launch.return_value = Mock()
        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        request_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return {"request_id": request_count}

        with patch.object(adapter, "_make_async_request", side_effect=mock_request):
            # Run concurrent requests
            tasks = [adapter.generate(prompt=f"Prompt {i}") for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all("request_id" in result for result in results)

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_async_context_manager_errors(self, mock_launch):
        """Test async context manager error handling."""
        mock_launch.return_value = Mock()

        # Test exception during context manager usage
        try:
            async with AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0):
                # Simulate error during usage
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # Test that cleanup still happened
        # (We can't easily verify this without more complex mocking)

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_async_session_timeout_scenarios(self, mock_launch):
        """Test various async timeout scenarios."""
        mock_launch.return_value = Mock()
        adapter = AsyncHttpServerEngineAdapter(
            host="localhost",
            port=8000,
            node_rank=0,
            timeout=0.001,  # Very short timeout
        )

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.post.side_effect = asyncio.TimeoutError()

        with patch.object(adapter, "_get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session

            with pytest.raises(RuntimeError, match="Failed to complete async request"):
                await adapter._make_async_request("test_endpoint")

        await adapter.close()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_async_extreme_configuration(self, mock_launch):
        """Test async adapter with extreme configuration values."""
        mock_launch.return_value = Mock()

        adapter = AsyncHttpServerEngineAdapter(
            host="localhost",
            port=8000,
            node_rank=0,
            max_connections=1,  # Very small
            timeout=0.001,
        )

        assert adapter.max_connections == 1
        assert adapter.timeout == 0.001


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get")
    def test_flush_cache_recovery(self, mock_get, mock_launch):
        """Test flush cache recovery from failures."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0, max_retries=1)

        # Simulate multiple failures then success
        mock_get.side_effect = [
            requests.exceptions.ConnectionError(),
            requests.exceptions.Timeout(),
            Mock(status_code=503),  # Service unavailable
            Mock(status_code=200, json=lambda: {"cache_flushed": True}),
        ]

        with patch("time.sleep"):
            result = adapter.flush_cache()
            assert result == {"cache_flushed": True}

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get")
    def test_flush_cache_max_retries(self, mock_get, mock_launch):
        """Test flush cache max retries exceeded."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0, max_retries=1)

        # All attempts fail
        mock_get.side_effect = requests.exceptions.ConnectionError()

        with patch("time.sleep"):
            result = adapter.flush_cache()
            assert result == {}  # Should return empty dict on failure

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_network_partition_recovery(self, mock_launch):
        """Test recovery from network partition scenarios."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0, max_retries=3)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            # Simulate network partition then recovery
            mock_post.side_effect = [
                requests.exceptions.ConnectionError("Network unreachable"),
                requests.exceptions.ConnectionError("Network unreachable"),
                Mock(status_code=200, json=lambda: {"recovered": True}),
            ]

            with patch("time.sleep"):
                result = adapter._make_request("test_endpoint")
                assert result == {"recovered": True}


class TestResourceManagement:
    """Test resource management and cleanup."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.kill_process_tree")
    def test_resource_cleanup_on_exception(self, mock_kill, mock_launch):
        """Test resource cleanup when exceptions occur."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_launch.return_value = mock_process

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Simulate exception during operation
        with patch.object(adapter, "_make_request", side_effect=Exception("Test error")):
            try:
                adapter.generate(prompt="test")
            except Exception:
                pass

        # Cleanup should still work
        adapter.shutdown()
        mock_kill.assert_called_once_with(12345)

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_async_resource_cleanup_on_exception(self, mock_launch):
        """Test async resource cleanup when exceptions occur."""
        mock_launch.return_value = Mock()
        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Create a session
        async with adapter._get_session():
            pass

        # Simulate exception
        try:
            async with adapter._get_session():
                raise Exception("Test error")
        except Exception:
            pass

        # Cleanup should still work
        await adapter.close()
        assert adapter._session is None

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_multiple_shutdown_calls(self, mock_launch):
        """Test multiple shutdown calls are safe."""
        mock_process = Mock()
        mock_launch.return_value = mock_process

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Multiple shutdown calls should be safe
        adapter.shutdown()
        adapter.shutdown()
        adapter.shutdown()

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @pytest.mark.asyncio
    async def test_multiple_async_close_calls(self, mock_launch):
        """Test multiple async close calls are safe."""
        mock_launch.return_value = Mock()
        adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Multiple close calls should be safe
        await adapter.close()
        await adapter.close()
        await adapter.close()


class TestDataTypeHandling:
    """Test handling of various data types."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_torch_dtype_handling(self, mock_launch):
        """Test handling of PyTorch data types."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test with various torch dtypes
            import torch

            dtypes = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
            shapes = [(1024, 512), (512,), (256, 256), (128, 128, 64), (1,)]
            names = [f"layer_{i}" for i in range(len(dtypes))]

            result = adapter.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name="test_group",
            )

            assert result == {"status": "success"}

            # Verify dtype conversion
            call_args = mock_request.call_args[0][1]
            expected_dtypes = ["float32", "float16", "int32", "int64", "bool"]
            assert call_args["dtypes"] == expected_dtypes

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_complex_data_structures(self, mock_launch):
        """Test handling of complex data structures."""
        mock_launch.return_value = Mock()
        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test with complex sampling params
            complex_sampling_params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "stop_sequences": ["</s>", "\n\n"],
                "max_tokens": 100,
                "logit_bias": {"token_123": 0.5, "token_456": -0.5},
                "nested_config": {
                    "beam_search": True,
                    "num_beams": 4,
                    "early_stopping": True,
                },
            }

            result = adapter.generate(
                prompt="Test prompt",
                sampling_params=complex_sampling_params,
            )

            assert result == {"status": "success"}
            # Verify the complex structure was passed through
            call_args = mock_request.call_args[0][1]
            assert call_args["sampling_params"] == complex_sampling_params


class TestServerArgsConfiguration:
    """Test ServerArgs configuration edge cases."""

    def test_server_args_edge_cases(self):
        """Test ServerArgs with edge case values."""
        # Test with minimal required args
        args = MockServerArgs(host="localhost", port=8000)
        assert args.host == "localhost"
        assert args.port == 8000

        # Test with all possible args (this depends on ServerArgs implementation)
        # We're just testing that it doesn't crash
        try:
            args = MockServerArgs(
                host="0.0.0.0",
                port=65535,  # Max port
                node_rank=999,
                api_key="very-long-api-key-" + "x" * 1000,
            )
        except Exception:
            # If ServerArgs has validation, that's fine
            pass


class TestIntegration:
    """Integration tests for both adapters."""

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post")
    def test_sync_async_compatibility(self, mock_post, mock_launch):
        """Test that sync and async adapters have compatible interfaces."""
        mock_launch.return_value = Mock()
        mock_post.return_value = Mock(status_code=200)

        sync_adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)
        async_adapter = AsyncHttpServerEngineAdapter(host="localhost", port=8001, node_rank=0)

        # Both should have the same public methods
        sync_methods = {
            name for name in dir(sync_adapter) if not name.startswith("_") and callable(getattr(sync_adapter, name))
        }
        async_methods = {
            name for name in dir(async_adapter) if not name.startswith("_") and callable(getattr(async_adapter, name))
        }

        # Async adapter should have all sync methods plus async-specific ones
        expected_extra_methods = {"async_generate", "close", "__aenter__", "__aexit__"}
        assert sync_methods.issubset(async_methods)
        assert expected_extra_methods.issubset(async_methods)

    @patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process")
    def test_error_scenarios(self, mock_launch):
        """Test various error scenarios."""
        mock_launch.return_value = Mock()

        adapter = HttpServerEngineAdapter(host="localhost", port=8000, node_rank=0)

        # Test with None payload
        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {}
            result = adapter.generate()
            assert result == {}

        # Test with empty parameters
        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {}
            result = adapter.update_weights_from_tensor([])
            assert result == {}
