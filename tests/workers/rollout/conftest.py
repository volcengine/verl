"""Pytest configuration for rollout tests.

This module provides shared fixtures and configuration for rollout worker tests.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_server_args():
    """Create mock ServerArgs for testing."""
    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.ServerArgs") as mock_args:
        mock_instance = Mock()
        mock_instance.host = "localhost"
        mock_instance.port = 8000
        mock_instance.node_rank = 0
        mock_instance.api_key = "test-key"
        mock_instance.url.return_value = "http://localhost:8000"
        mock_args.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_multiprocessing_process():
    """Create mock multiprocessing.Process for testing."""
    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process") as mock_process_class:
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process
        yield mock_process


@pytest.fixture
def mock_requests_session():
    """Create mock requests.Session for testing."""
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
def mock_aiohttp_session():
    """Create mock aiohttp.ClientSession for testing."""
    mock_session = Mock()
    mock_session.closed = False

    # Mock response
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json = Mock(return_value={"status": "success"})
    mock_response.raise_for_status = Mock()

    # Mock context managers
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response

    return mock_session


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Async test configuration
pytest_plugins = ("pytest_asyncio",)
