"""Complete unit tests for HTTP Server Engine Adapters.

This module contains comprehensive unit tests for both HttpServerEngineAdapter
and AsyncHttpServerEngineAdapter classes, covering all public methods,
error handling scenarios, edge cases, and boundary conditions using pytest and mock frameworks.

Tests use real SGLang modules for integration testing while mocking external dependencies.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import requests

# Import the module under test
from verl.workers.rollout.sglang_rollout.http_server_engine import (
    AsyncHttpServerEngineAdapter,
    HttpServerEngineAdapter,
    launch_server_process,
)


@pytest.mark.real_sglang
class TestLaunchServerProcess:
    """Test cases for launch_server_process function."""

    def test_launch_server_process_success(
        self, mock_multiprocessing_process, mock_requests_session, real_adapter_kwargs
    ):
        """Test successful server process launch and health check."""
        # Import real SGLang ServerArgs
        from sglang.srt.server_args import ServerArgs

        # Create server args using real ServerArgs
        server_args = ServerArgs(**real_adapter_kwargs)

        # Test
        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process_class.return_value = mock_multiprocessing_process
            with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session") as mock_session_class:
                mock_session_class.return_value.__enter__.return_value = mock_requests_session

                result = launch_server_process(server_args)

                # Assertions
                assert result == mock_multiprocessing_process
                mock_multiprocessing_process.start.assert_called_once()
                assert mock_requests_session.get.call_count >= 2  # health_generate and flush_cache

    def test_launch_server_process_non_master(self, mock_multiprocessing_process, non_master_adapter_kwargs):
        """Test server launch for non-master nodes (should return immediately)."""
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(**non_master_adapter_kwargs)

        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process_class.return_value = mock_multiprocessing_process
            result = launch_server_process(server_args)

            assert result == mock_multiprocessing_process
            mock_multiprocessing_process.start.assert_called_once()

    def test_launch_server_process_timeout(self, mock_multiprocessing_process, real_adapter_kwargs):
        """Test timeout during server health check."""
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(**real_adapter_kwargs)

        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process_class.return_value = mock_multiprocessing_process
            with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session") as mock_session_class:
                mock_session = Mock()
                mock_session.get.side_effect = requests.RequestException("Connection failed")
                mock_session_class.return_value.__enter__.return_value = mock_session

                with patch("time.time", side_effect=[0, 400]):  # Simulate timeout
                    with pytest.raises(TimeoutError):
                        launch_server_process(server_args)

                mock_multiprocessing_process.terminate.assert_called_once()

    def test_launch_server_process_died(self, real_adapter_kwargs):
        """Test server process dies during startup."""
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(**real_adapter_kwargs)

        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process = Mock()
            mock_process.is_alive.return_value = False
            mock_process_class.return_value = mock_process

            with pytest.raises(RuntimeError, match="Server process terminated unexpectedly"):
                launch_server_process(server_args)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.real_sglang
class TestHttpServerEngineAdapter:
    """Test cases for HttpServerEngineAdapter class."""

    def test_init_with_router_registration(self, mock_launch_server_process, mock_requests_post, router_adapter_kwargs):
        """Test initialization with router registration."""
        adapter = HttpServerEngineAdapter(**router_adapter_kwargs)

        assert adapter.router_ip == "192.168.1.1"
        assert adapter.router_port == 8080
        assert adapter.process == mock_launch_server_process.return_value
        mock_requests_post.assert_called_once()

    def test_init_without_router(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test initialization without router registration."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        assert adapter.router_ip is None
        assert adapter.router_port is None
        assert adapter.process == mock_launch_server_process.return_value

    def test_register_with_router_failure(self, mock_launch_server_process, router_adapter_kwargs):
        """Test router registration failure handling."""
        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Connection failed")

            # Should not raise exception, just log error
            adapter = HttpServerEngineAdapter(**router_adapter_kwargs)

            assert adapter.router_ip == "192.168.1.1"
            mock_post.assert_called_once()

    @pytest.mark.mock_only
    def test_make_request_success(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test successful HTTP request."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "success"}
            mock_post.return_value = mock_response

            result = adapter._make_request("test_endpoint", {"param": "value"})

            assert result == {"status": "success"}
            mock_post.assert_called_with(
                "http://localhost:8000/test_endpoint",
                json={"param": "value"},
                timeout=adapter.timeout,
            )

    @pytest.mark.mock_only
    def test_make_request_get_method(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test HTTP GET request."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_get.return_value = mock_response

            result = adapter._make_request("test_endpoint", method="GET")

            assert result == {"data": "test"}
            mock_get.assert_called_with("http://localhost:8000/test_endpoint", timeout=adapter.timeout)

    def test_make_request_non_master(self, mock_launch_server_process):
        """Test request from non-master node returns empty dict."""
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 1, "model_path": "/tmp/test_model"}
        adapter = HttpServerEngineAdapter(**kwargs)
        result = adapter._make_request("test_endpoint")

        assert result == {}

    @pytest.mark.mock_only
    def test_make_request_retry_logic(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test retry logic for failed requests."""
        adapter = HttpServerEngineAdapter(max_retries=2, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            with patch("time.sleep") as mock_sleep:
                # First two calls fail, third succeeds
                mock_post.side_effect = [
                    requests.exceptions.Timeout(),
                    requests.exceptions.ConnectionError(),
                    Mock(status_code=200, json=lambda: {"success": True}),
                ]

                result = adapter._make_request("test_endpoint")

                assert result == {"success": True}
                assert mock_post.call_count == 3
                assert mock_sleep.call_count == 2

    @pytest.mark.mock_only
    def test_make_request_http_error(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test HTTP error handling."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
            mock_post.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                adapter._make_request("test_endpoint")

    @pytest.mark.mock_only
    def test_make_request_max_retries_exceeded(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test max retries exceeded."""
        adapter = HttpServerEngineAdapter(max_retries=1, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            with patch("time.sleep"):
                mock_post.side_effect = requests.exceptions.Timeout()

                with pytest.raises(RuntimeError, match="Failed to complete request"):
                    adapter._make_request("test_endpoint")

                assert mock_post.call_count == 2  # Initial + 1 retry

    @pytest.mark.mock_only
    def test_update_weights_from_tensor(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test update_weights_from_tensor method."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.mock_only
    def test_generate(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test generate method."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.mock_only
    def test_flush_cache(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test flush_cache method."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            with patch("time.sleep") as mock_sleep:
                # First call fails, second succeeds
                mock_responses = [
                    Mock(status_code=503),  # Service unavailable
                    Mock(status_code=200, json=lambda: {"cache_flushed": True}),
                ]
                mock_get.side_effect = mock_responses

                result = adapter.flush_cache()

                assert result == {"cache_flushed": True}
                assert mock_get.call_count == 2
                mock_sleep.assert_called_once()

    def test_flush_cache_non_master(self, mock_launch_server_process):
        """Test flush_cache for non-master node."""
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 1, "model_path": "/tmp/test_model"}
        adapter = HttpServerEngineAdapter(**kwargs)
        result = adapter.flush_cache()

        assert result == {}

    @pytest.mark.mock_only
    def test_memory_management_methods(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test memory release and resume methods."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.mock_only
    def test_distributed_weights_methods(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test distributed weights update methods."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.mock_only
    def test_generation_control_methods(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test pause and continue generation methods."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.mock_only
    def test_shutdown(self, mock_launch_server_process, mock_kill_process_tree, router_adapter_kwargs):
        """Test shutdown method."""
        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            adapter = HttpServerEngineAdapter(**router_adapter_kwargs)

            adapter.shutdown()

            # Should unregister from router
            assert mock_post.call_count == 2  # Once for registration, once for unregistration
            # Should kill process
            mock_kill_process_tree.assert_called_once_with(mock_launch_server_process.return_value.pid)

    def test_shutdown_with_errors(self, mock_launch_server_process, mock_kill_process_tree, router_adapter_kwargs):
        """Test shutdown method with errors."""
        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            # Mock registration success but unregistration failure
            mock_post.side_effect = [
                Mock(status_code=200),  # Registration success
                requests.RequestException("Unregistration failed"),  # Unregistration failure
            ]

            # Mock process kill failure
            mock_kill_process_tree.side_effect = Exception("Kill failed")

            adapter = HttpServerEngineAdapter(**router_adapter_kwargs)

            # Should not raise exceptions
            adapter.shutdown()

            assert mock_post.call_count == 2
            mock_kill_process_tree.assert_called_once_with(mock_launch_server_process.return_value.pid)

    # Edge cases for HttpServerEngineAdapter
    def test_empty_and_none_parameters(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of empty and None parameters."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    def test_large_payload_handling(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of large payloads."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    def test_unicode_and_special_characters(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of unicode and special characters."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    def test_malformed_responses(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of malformed server responses."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            # Test response without JSON
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_post.return_value = mock_response

            with pytest.raises(json.JSONDecodeError):
                adapter._make_request("test_endpoint")

    def test_timeout_edge_cases(self, mock_launch_server_process):
        """Test various timeout scenarios."""
        # Test with very small timeout
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 0, "model_path": "/tmp/test_model", "timeout": 0.001}
        adapter = HttpServerEngineAdapter(**kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()

            with pytest.raises(RuntimeError, match="Failed to complete request"):
                adapter._make_request("test_endpoint")

    def test_extreme_configuration_values(self, mock_launch_server_process):
        """Test extreme configuration values."""
        # Test with extreme values
        kwargs = {
            "host": "localhost",
            "port": 8000,
            "node_rank": 0,
            "model_path": "/tmp/test_model",
            "timeout": 0.001,  # Very small
            "max_retries": 100,  # Very large
            "retry_delay": 0.001,  # Very small
        }
        adapter = HttpServerEngineAdapter(**kwargs)

        assert adapter.timeout == 0.001
        assert adapter.max_retries == 100
        assert adapter.retry_delay == 0.001


class TestAsyncHttpServerEngineAdapter:
    """Test cases for AsyncHttpServerEngineAdapter class."""

    def test_init(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async adapter initialization."""
        adapter = AsyncHttpServerEngineAdapter(max_connections=50, **basic_adapter_kwargs)

        assert adapter.max_connections == 50
        assert adapter._need_reload is True
        assert adapter._session is None

    @pytest.mark.asyncio
    async def test_get_session(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test session creation and management."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

        # Test session creation
        async with adapter._get_session() as session:
            assert isinstance(session, aiohttp.ClientSession)
            assert not session.closed

        # Session should be reused
        async with adapter._get_session() as session2:
            assert session2 is adapter._session

        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_session_error_handling(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test session error handling and recreation."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

        # Simulate error during session usage
        try:
            async with adapter._get_session():
                raise Exception("Test error")
        except Exception:
            pass

        # Session should be None after error
        assert adapter._session is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_make_async_request_success(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test successful async HTTP request."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.asyncio
    async def test_make_async_request_get_method(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async GET request."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.asyncio
    async def test_make_async_request_non_master(self, mock_launch_server_process):
        """Test async request from non-master node."""
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 1, "model_path": "/tmp/test_model"}
        adapter = AsyncHttpServerEngineAdapter(**kwargs)
        result = await adapter._make_async_request("test_endpoint")

        assert result == {}

    @pytest.mark.asyncio
    async def test_async_generate(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async generate method."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.asyncio
    async def test_async_memory_management(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async memory management methods."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

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

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async context manager support."""
        async with AsyncHttpServerEngineAdapter(**basic_adapter_kwargs) as adapter:
            assert isinstance(adapter, AsyncHttpServerEngineAdapter)
            assert adapter._session is None  # Not created yet

        # Session should be closed after context exit
        if adapter._session:
            assert adapter._session.closed

    @pytest.mark.asyncio
    async def test_close_method(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test close method."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

        # Create a session
        async with adapter._get_session() as session:
            assert not session.closed

        # Close should close the session
        await adapter.close()
        assert adapter._session is None

        # Multiple close calls should be safe
        await adapter.close()


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_flush_cache_recovery(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test flush cache recovery from failures."""
        adapter = HttpServerEngineAdapter(max_retries=1, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
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

    def test_flush_cache_max_retries(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test flush cache max retries exceeded."""
        adapter = HttpServerEngineAdapter(max_retries=1, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            # All attempts fail
            mock_get.side_effect = requests.exceptions.ConnectionError()

            with patch("time.sleep"):
                result = adapter.flush_cache()
                assert result == {}  # Should return empty dict on failure

    def test_network_partition_recovery(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test recovery from network partition scenarios."""
        adapter = HttpServerEngineAdapter(max_retries=3, **basic_adapter_kwargs)

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

    def test_resource_cleanup_on_exception(
        self, mock_launch_server_process, mock_kill_process_tree, basic_adapter_kwargs
    ):
        """Test resource cleanup when exceptions occur."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        # Simulate exception during operation
        with patch.object(adapter, "_make_request", side_effect=Exception("Test error")):
            try:
                adapter.generate(prompt="test")
            except Exception:
                pass

        # Cleanup should still work
        adapter.shutdown()
        mock_kill_process_tree.assert_called_once_with(mock_launch_server_process.return_value.pid)

    @pytest.mark.asyncio
    async def test_async_resource_cleanup_on_exception(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async resource cleanup when exceptions occur."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

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

    def test_multiple_shutdown_calls(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test multiple shutdown calls are safe."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

        # Multiple shutdown calls should be safe
        adapter.shutdown()
        adapter.shutdown()
        adapter.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_async_close_calls(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test multiple async close calls are safe."""
        adapter = AsyncHttpServerEngineAdapter(**basic_adapter_kwargs)

        # Multiple close calls should be safe
        await adapter.close()
        await adapter.close()
        await adapter.close()


class TestDataTypeHandling:
    """Test handling of various data types."""

    def test_torch_dtype_handling(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of PyTorch data types."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    def test_complex_data_structures(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of complex data structures."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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

    def test_server_args_edge_cases(self, mock_sglang_modules):
        """Test ServerArgs with edge case values."""
        MockServerArgs = mock_sglang_modules["MockServerArgs"]

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

    def test_sync_async_compatibility(self, mock_launch_server_process, mock_requests_post, basic_adapter_kwargs):
        """Test that sync and async adapters have compatible interfaces."""
        sync_adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)
        async_kwargs = {**basic_adapter_kwargs, "port": 8001}
        async_adapter = AsyncHttpServerEngineAdapter(**async_kwargs)

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

    def test_error_scenarios(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test various error scenarios."""
        adapter = HttpServerEngineAdapter(**basic_adapter_kwargs)

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
