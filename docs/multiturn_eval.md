Lightweight Multi-Turn Tool Evaluation
======================================

This guide explains how to run multi-turn tool-call evaluation with ``main_multiturn_eval``. It reuses the training AgentLoop rollout without running PPO optimization, so you can validate tool-use flows with minimal overhead.

Goals
~~~~~

- Add a missing “single-run multi-turn tool eval” path by reusing the training rollout/AgentLoop pipeline.
- Decouple evaluation from training scripts—only keep the rollout stage instead of bundling eval with PPO runs.
- Prioritize vLLM support. sglang currently fails when ``tp=1`` (and has not been adapted for multi-TP); vLLM coverage is sufficient for the current use case.

Key Files
~~~~~~~~~

- ``verl/trainer/main_multiturn_eval.py``: Entry point. Extracts AgentLoop from ``main_ppo`` to do Ray init, dataset/sampler creation, checkpoint loading, generation, and reward collection.
- ``verl/trainer/config/multiturn_eval.yaml``: Hydra config mirroring the training actor/rollout/reward layout, plus ``evaluation`` and ``output`` sections.
- ``examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn_eval.sh``: Example script for vLLM async multi-turn tool evaluation; use directly or as a template.

How to Run
~~~~~~~~~~

.. code-block:: bash

   # Run from the project root
   bash examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn_eval.sh \
     actor_rollout_ref.model.path=/path/to/model \
     checkpoint_dir=/path/to/fsdp_or_fsdp2_checkpoint_parent_dir \
     data.eval_files=/path/to/eval.parquet \
     output.path=./eval_results \
     output.scores_path=evaluation_scores.json

Notes:

- ``checkpoint_dir`` can point to a specific ``global_step_*`` or its parent; the script will pick the latest checkpoint automatically.
- Defaults: ``rollout.mode=async``, ``rollout.name=vllm``, ``multi_turn.enable=true``. Add ``multi_turn.tool_config_path`` and ``multi_turn.interaction_config_path`` if tools/interaction are needed.
- Outputs go to ``output.path`` (``evaluation_scores.*`` and ``evaluation_summary.json``).

FSDP Checkpoint Loading
~~~~~~~~~~~~~~~~~~~~~~~

- Verified loading weights from FSDP/FSDP2 training checkpoints.
- ``main_multiturn_eval.py`` includes a ``DeviceMesh`` compatibility patch and process-group registration to avoid Torch 2.8+ FSDP sharded-load errors.
- To skip checkpoint restore and just use the base model, set ``checkpoint_dir=null``.

Validated Scenarios
~~~~~~~~~~~~~~~~~~~

All succeeded when loading checkpoints saved from FSDP training:

- GSM8K multi-turn tool use: ``async + vllm + tp1 -> agentloop``; ``async + vllm + tp2 -> agentloop``.
- Geo3K multi-turn tool use: ``async + vllm + tp1 -> agentloop``; ``async + vllm + tp2 -> agentloop``.
- Multimodal multi-turn tool use: ``async + vllm + multimodal + tp1 -> agentloop (multimodal)``; ``async + vllm + multimodal + tp2 -> agentloop (multimodal)``.

Lightweight Flow & Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Evaluation keeps only the AgentLoop rollout from ``main_ppo`` and removes optimization, making startup and resource usage much lighter than training.
- Default summaries live in ``aggregate_summary``; per-sample records are built in ``collect_sample_records``. To add new metrics or formats, extend these functions or emit custom metrics from AgentLoop (they will be stored under ``agent_metrics``).

Limitations & Next Steps
~~~~~~~~~~~~~~~~~~~~~~~~

- Currently validated only with vLLM; sglang still breaks when ``tp=1`` (and multi-TP is untested). Next step: fix sglang TP compatibility.
- For other tasks or tool configs, reuse the provided script template and swap ``data.*`` and tool config paths.