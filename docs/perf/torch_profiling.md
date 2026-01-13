# PyTorch Profiling in verl

Last updated: 01/13/2026.

This guide explains how to use the native [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for profiling verl training runs.

## Configuration

Profiling in verl can be configured through parameters in the trainer configuration file (e.g., `ppo_trainer.yaml`).

### Global Profiling Control

In `global_profiler`, you can control when and how profiling occurs globally:

* **`global_profiler.steps`**: List of step numbers to profile. E.g., `[1, 2, 5]` profiles steps 1, 2, and 5. Set to `null` to disable.
* **`global_profiler.save_path`**: Directory to save the profiling results. Default is `outputs/profile`.

### Role Profiling Control

Each RL role (Actor, Critic, etc.) has its own `profiler` configuration:

* **`enable`**: Whether to enable profiling for this role.
* **`all_ranks`**: If `True`, profiles all ranks.
* **`ranks`**: List of specific ranks to profile if `all_ranks` is `False`.
* **`tool_config.torch`**: Configuration specific to the PyTorch Profiler.

#### PyTorch Profiler Options (`tool_config.torch`)

You can customize the PyTorch Profiler behavior using the following fields under `tool_config.torch`:

* **`activities`**: List of activities to profile. Options: `cpu`, `cuda`. Default: `[cpu, cuda]`.
* **`with_stack`**: Record source code file and line number. Default: `false`.
* **`record_shapes`**: Record shapes of operator inputs. Default: `false`.
* **`profile_memory`**: Track tensor memory allocation/free. Default: `false`.
* **`schedule`**: (Advanced) configuration for `wait`, `warmup`, `active`, `repeat` cycles.

## Examples

### End-to-End Collection

Enable profiling for specific steps and roles (e.g., Actor).

```yaml
global_profiler:
  steps: [1, 2, 5]
  save_path: ./outputs/profile

actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: True
      tool_config:
        torch:
          discrete: False
          activities: [cpu, cuda]
```

### Discrete Mode Collection

```yaml
global_profiler:
  steps: [1, 2, 5]
  save_path: ./outputs/profile

actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: False
      ranks: [0] # Collect specific global rank
      tool_config:
        torch:
          discrete: True
          activities: [cpu, cuda]
  rollout:
    profiler:
      enable: True
      all_ranks: False
      ranks: [0] # Replica Rank of the inference instance in Agent Loop mode
      tool_config:
        torch:
          discrete: True # Discrete mode must be enabled in Agent Loop mode
```

**Agent Loop Scenario Description**:

When Rollout runs in `Agent Loop <../advance/agent_loop.rst>`_ mode, performance data for the Rollout phase **must be collected using discrete mode**. At this time, the Profiler is triggered by the inference engine backend, and export paths and other parameters **must be set via environment variables**:

*   **vLLM Engine**
    *   Reference: [vLLM Profiling](https://github.com/vllm-project/vllm/blob/v0.12.0/docs/contributing/profiling.md)
    *   `VLLM_TORCH_PROFILER_DIR`: Sets the save path (**Required**).
    *   `VLLM_TORCH_PROFILER_WITH_STACK`: Controls stack tracing (1: on, 0: off, default: on).

*   **SGLang Engine**
    *   Reference: [SGLang Profiling](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)
    *   `SGLANG_TORCH_PROFILER_DIR`: Sets the save path (**Required**).
    *   `SGLANG_PROFILE_WITH_STACK`: Controls stack tracing (1: on, 0: off, default: on).

## Visualization

Collected trace files (usually `.json` or `.json.gz`) are stored in the configured `save_path`.

You can visualize them using:

1.  **Chrome Tracing**: Open `chrome://tracing` in a Chrome browser and load the JSON file.
2.  **Perfetto**: Open [ui.perfetto.dev](https://ui.perfetto.dev/) and load the file (recommended for large traces).
3.  **TensorBoard**: If using the TensorBoard plugin for PyTorch Profiler.
