"""Pytest configuration for SGLang rollout tests.

This module provides shared fixtures and configuration specifically for SGLang-based
rollout worker tests. Tests use real SGLang modules for integration testing.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def basic_adapter_kwargs():
    """Provide basic kwargs for creating HTTP server adapters."""
    return {
        "host": "localhost",
        "port": 8000,
        "node_rank": 0,
        "model_path": "/tmp/test_model",
    }


@pytest.fixture
def router_adapter_kwargs():
    """Provide kwargs for creating adapters with router configuration."""
    return {
        "router_ip": "192.168.1.1",
        "router_port": 8080,
        "host": "localhost",
        "port": 8000,
        "node_rank": 0,
        "model_path": "/tmp/test_model",
    }


@pytest.fixture
def non_master_adapter_kwargs():
    """Provide kwargs for creating non-master node adapters."""
    return {
        "host": "localhost",
        "port": 8000,
        "node_rank": 1,  # Non-master
        "model_path": "/tmp/test_model",
    }


@pytest.fixture
def mock_launch_server_process():
    """Mock the launch_server_process function for testing without actual server startup."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process") as mock_launch:
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345
        mock_launch.return_value = mock_process
        yield mock_launch


@pytest.fixture
def mock_multiprocessing_process():
    """Create mock multiprocessing.Process for testing without actual process creation."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process") as mock_process_class:
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process
        yield mock_process


@pytest.fixture
def mock_requests_session():
    """Create mock requests.Session for testing HTTP interactions."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session") as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for testing HTTP POST requests."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for testing HTTP GET requests."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_aiohttp_session():
    """Create mock aiohttp.ClientSession for testing async HTTP interactions."""
    mock_session = AsyncMock()
    mock_session.closed = False

    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"status": "success"})
    mock_response.raise_for_status = Mock()

    # Mock context managers
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response

    return mock_session


@pytest.fixture
def mock_kill_process_tree():
    """Mock kill_process_tree function for testing cleanup without actual process termination."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.kill_process_tree") as mock_kill:
        yield mock_kill


# Test environment fixtures for real SGLang testing
@pytest.fixture(scope="session")
def sglang_test_model_path():
    """Provide a test model path for SGLang tests.

    This can be overridden by environment variable SGLANG_TEST_MODEL_PATH
    for tests that need a real model.
    """
    import os

    return os.getenv("SGLANG_TEST_MODEL_PATH", "/tmp/test_model")


@pytest.fixture
def real_adapter_kwargs(sglang_test_model_path):
    """Provide kwargs for creating adapters with real SGLang integration."""
    return {
        "host": "localhost",
        "port": 8000,
        "node_rank": 0,
        "model_path": sglang_test_model_path,
    }


# Pytest configuration for SGLang tests
def pytest_configure(config):
    """Configure pytest with custom markers for SGLang tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "sglang: mark test as SGLang-specific")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "real_sglang: mark test as requiring real SGLang installation")
    config.addinivalue_line("markers", "mock_only: mark test as using mocked dependencies only")


# Async test configuration
pytest_plugins = ("pytest_asyncio",)
