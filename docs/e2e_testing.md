# End-to-End Testing Configuration System

This document explains how the end-to-end testing system for VERL has been reorganized to support multiple jobs with shared configurations.

## Configuration Structure

The configuration system uses a hierarchical approach:

1. **Base Configuration** (`gsm8k_base.yaml`): Contains common parameters shared across all GSM8K tests.
2. **Job-Specific Configurations** (e.g., `gsm8k_function_rm.yaml`): Extend the base configuration with job-specific settings.

## Running Tests

Tests can be run using the `run_test.sh` script, which accepts a configuration name:

```bash
bash tests/e2e/run_test.sh gsm8k_function_rm
```

## GitHub Workflow

The GitHub workflow has been reorganized into multiple jobs:

1. **prepare_gsm8k**: Prepares the dataset and caches it for other jobs.
2. **function_rm_tests**: Runs function-based reward model tests.
3. **model_rm_tests**: Runs model-based reward model tests.
4. **advanced_tests**: Runs specialized configurations like ReMax and Ulysses.

## Adding New Test Configurations

To add a new test configuration:

1. Create a new YAML file in `verl/trainer/config/` that extends the base configuration.
2. Include the specific parameters required for your test.
3. Add the test to the appropriate job in `.github/workflows/e2e_gsm8k.yml`.

## Benefits

This approach provides several benefits:

1. **Reduced Duplication**: Common configuration parameters are defined only once.
2. **Parallel Execution**: Different test types can run in parallel as separate jobs.
3. **Better Organization**: Tests are grouped by type for clearer reporting.
4. **Faster CI**: Jobs can be cached and reused, reducing overall CI time. 