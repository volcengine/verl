Of course, here is the English translation of the provided document.

# SGLang Rollout Tests

This directory contains tests specifically for the SGLang backend's rollout worker.

## 📁 Directory Structure

```
tests/workers/rollout/rollout_sglang/
├── conftest.py                # Pytest configuration and fixtures specific to SGLang
├── test_http_server_engine.py   # Tests for HTTP Server Engine Adapters
├── run_tests.py                 # Test execution script
└── README.md                    # This document
```

## 🎯 Test Scope

### HTTP Server Engine Adapters

  - `HttpServerEngineAdapter` - Synchronous HTTP adapter
  - `AsyncHttpServerEngineAdapter` - Asynchronous HTTP adapter
  - `launch_server_process` - Server process launch function

### Features Covered

  - Server initialization and configuration
  - HTTP request handling (GET/POST)
  - Asynchronous operation support
  - Error handling and retry mechanisms
  - Memory management
  - Distributed weight updates
  - Router registration and unregistration
  - Resource cleanup

## 🔧 Test Environment Setup

### SGLang Dependencies

The tests now use the **real SGLang module** for integration testing, instead of mock objects.

#### Installation Requirements

Ensure SGLang is installed:

```bash
pip install sglang[all]
```

#### Environment Variables

  - `SGLANG_TEST_MODEL_PATH`: Path to the test model (default: `/tmp/test_model`)

<!-- end list -->

```bash
export SGLANG_TEST_MODEL_PATH="/path/to/your/test/model"
```

### Test Types

  - **Integration Tests**: Use the real SGLang module, marked with `@pytest.mark.real_sglang`
  - **Unit Tests**: Mock only external dependencies (HTTP requests, process management), marked with `@pytest.mark.mock_only`

## 🚀 Running Tests

### Basic Execution

```bash
# Navigate to the test directory
cd tests/workers/rollout/rollout_sglang

# Run all tests
python run_tests.py

# Or use pytest directly
python -m pytest
```

### Running by Test Type

```bash
# Run only mock unit tests (does not require a real SGLang model)
python run_tests.py -m "mock_only"

# Run only real SGLang integration tests
python run_tests.py -m "real_sglang"

# Exclude slow tests
python run_tests.py -m "not slow"
```

### Running with Options

```bash
# Verbose output
python run_tests.py -v

# With coverage report
python run_tests.py -c

# Generate HTML coverage report
python run_tests.py -c --html

# Run tests in parallel (requires pytest-xdist)
python run_tests.py -p

# Run a specific test
python run_tests.py -k "test_init"

# Combining options
python run_tests.py -v -c --html -x
```

### Using Pytest Directly

```bash
# Basic run
pytest

# Verbose output
pytest -v -s

# With coverage
pytest --cov=verl.workers.rollout.sglang_rollout --cov-report=term-missing

# Asyncio mode
pytest --asyncio-mode=auto

# Run a specific test class
pytest test_http_server_engine.py::TestHttpServerEngineAdapter

# Run a specific test method
pytest test_http_server_engine.py::TestHttpServerEngineAdapter::test_init_with_router_registration
```

## 🔧 Test Configuration

### Real SGLang Integration

  - **Real Modules**: Tests use the actual `sglang` module and `ServerArgs` class.
  - **Model Requirement**: Some tests may require real model files.
  - **Environment Configuration**: Test parameters are configured via environment variables.

### Fixtures

  - `basic_adapter_kwargs` - Basic adapter arguments
  - `router_adapter_kwargs` - Arguments with router configuration
  - `non_master_adapter_kwargs` - Non-master node arguments
  - `real_adapter_kwargs` - Arguments for real SGLang integration
  - `sglang_test_model_path` - Test model path
  - `mock_launch_server_process` - Mock for server process launch
  - `mock_requests_*` - Mocks for HTTP requests
  - `mock_aiohttp_session` - Mock for async HTTP session

### Markers

  - `@pytest.mark.asyncio` - Asynchronous test
  - `@pytest.mark.sglang` - SGLang-specific test
  - `@pytest.mark.integration` - Integration test
  - `@pytest.mark.slow` - Slow test
  - `@pytest.mark.real_sglang` - Test requiring a real SGLang installation
  - `@pytest.mark.mock_only` - Test using only mock dependencies

## 📊 Test Statistics

  - **Total Test Cases**: 50+
  - **Test Classes**: 8 main test classes
  - **Methods Covered**: All public methods
  - **Integration Level**: Real SGLang module + mocked external dependencies

## 🐛 Troubleshooting

### Common Issues

1.  **SGLang Import Error**

    ```
    ModuleNotFoundError: No module named 'sglang'
    ```

      - **Solution**: Install SGLang
        ```bash
        pip install sglang[all]
        ```

2.  **Model Path Error**

    ```
    FileNotFoundError: Model path not found
    ```

      - **Solution**: Set the correct model path
        ```bash
        export SGLANG_TEST_MODEL_PATH="/path/to/valid/model"
        ```

3.  **Async Test Failure**

    ```
    RuntimeError: This event loop is already running
    ```

      - Ensure you are using `pytest --asyncio-mode=auto`

4.  **Coverage Report Issues**

    ```
    Coverage.py warning: No data was collected
    ```

      - Ensure the module path is correct: `verl.workers.rollout.sglang_rollout`

### Debugging Tests

```bash
# Run a single test with verbose output
pytest test_http_server_engine.py::TestHttpServerEngineAdapter::test_init_with_router_registration -v -s

# Enter the debugger on test failure
pytest test_http_server_engine.py --pdb

# Show the slowest tests
pytest test_http_server_engine.py --durations=10

# Run only fast mock tests
pytest -m "mock_only" -v
```

### Performance Testing

```bash
# Run all integration tests (can be slow)
pytest -m "real_sglang" -v

# Skip slow tests
pytest -m "not slow" -v
```

## 🔗 Related Documents

  - [Main Rollout Tests](https://www.google.com/search?q=../README_tests.md)
  - [HTTP Server Engine Implementation](../../../../verl/workers/rollout/sglang_rollout/http_server_engine.py)
  - [SGLang Official Documentation](https://github.com/sgl-project/sglang)

## 📝 Contribution Guidelines

### Adding New Tests

1.  Add a new method to the appropriate test class.
2.  Use a descriptive test method name.
3.  Include a detailed docstring.
4.  Use appropriate fixtures.
5.  Add appropriate test markers:
      - `@pytest.mark.real_sglang` - if it requires a real SGLang installation.
      - `@pytest.mark.mock_only` - if it only requires mocks.
      - `@pytest.mark.slow` - if the test is slow.

### Test Naming Conventions

  - Test methods should start with `test_`.
  - Use descriptive names, e.g., `test_init_with_router_registration`.
  - Test classes should start with `Test`.
  - Edge case tests should include a description of the specific scenario.

### Mock Usage Guide

  - **Selective Mocking**: Only mock external dependencies (e.g., HTTP requests, process management).
  - **Keep it Real**: Use the actual SGLang module for core logic testing.
  - Prioritize using existing fixtures.
  - Create new fixtures for new external dependencies.
  - Verify the call counts and arguments of mock objects.