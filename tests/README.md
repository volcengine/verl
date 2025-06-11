# Tests layout

Each folder under tests/ corresponds to a test category for a sub-namespace in verl. For instance:
- `tests/trainer` for testing functionality related to `verl/trainer`
- `tests/models` for testing functionality related to `verl/models`
- ...

There are a few folders with `special_` prefix, created for special purposes:
- `special_distributed`: unit tests that must run with multiple GPUs
- `special_e2e`: end-to-end tests with training/generation scripts
- `special_npu`: tests for NPUs
- `special_sanity`: a suite of quick sanity tests
- `special_standalone`: a set of test that are designed to run in dedicated environments

Accelerators for tests 
- By default tests are run with GPU available, except for the ones under `special_npu`, and any test script whose name ends with `on_cpu.py`.
- For test scripts with `on_cpu.py` name suffix would be tested on CPU resources in linux environment.