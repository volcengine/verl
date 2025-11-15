EP MCP Frozen Lake – VERL Integration Repro Notes

Project Context & Objectives
- Objective: Integrate Eval Protocol (EP) agentic evaluations (via MCP environments) with VERL’s multi-turn rollout/agent loop to enable RL training (PPO) against interactive environments. Use the Frozen Lake MCP environment as the canonical Phase 1 example.
- Why: VERL provides scalable training (FSDP/Megatron, Ray orchestration, vLLM/SGLang rollout); EP provides environment abstractions (MCP servers, tool calling), robust rollout handling, and standardized evaluation flows. Bridging the two enables multi-turn RL on realistic tool-augmented tasks.
- Scope (Phase 1):
  - Reuse VERL’s `ToolAgentLoop` to parse tool calls and call EP MCP tools using `fastmcp` (no python-sdk code changes needed).
  - Provide VERL recipe/config to point the agent loop at EP MCP servers, and a small dataset demonstrating multi-turn Frozen Lake interaction.
  - Run PPO with vLLM backend on Qwen/Qwen3-30B-A3B-Instruct-2507 across 8x H100.
- Non-goals (Phase 1):
  - Deep adapter inside EP to consume VERL async server manager (policy adapter is a Phase 2 idea on the VERL side).
  - Reward model integration beyond a minimal setup; Phase 1 can run with `reward_model.enable=false` or a simple reducer.
- Success Criteria:
  - End-to-end run in VERL successfully starts vLLM rollout, the `ToolAgentLoop` invokes EP MCP tools, and PPO trainer iterates without config assertions.
  - Reproducible instructions and configs checked into the repo.

Summary of Architecture
- Rollout Engine: vLLM, orchestrated by VERL’s `AgentLoopManager` and `AsyncLLMServerManager`.
- Agent Loop: VERL `ToolAgentLoop` parses tool calls from model outputs, executes MCP tools via `MCPBaseTool`/`ClientManager`, then feeds tool responses back into the conversation.
- Tools: Exposed by EP MCP server (`frozen_lake_mcp.py`), discovered at runtime through `verl/recipe/ep_agent/mcp_servers.json`.
- Dataset: RLHF-style dataset with prompts as chat turns; for this integration we use a system-only message and rely on the environment observation next turns. Converted to Parquet for VERL’s `RLHFDataset`.
- Trainer: `RayPPOTrainer` handles actor/critic/reward-manager workers; we use 8 GPUs with FSDP-based actor, vLLM for rollout, and a disabled reward model for a smoke test.


Context
- Goal: Run multi-turn PPO in VERL using Eval Protocol (EP) MCP Frozen Lake env, vLLM engine, Qwen/Qwen3-30B-A3B-Instruct-2507 on 8x H100.
- VERL recipe/config added:
  - Agent loop registry: `verl/recipe/ep_agent/agent_loop.yaml`
  - MCP tools config: `verl/recipe/ep_agent/tools_config.yaml`
  - MCP server list: `verl/recipe/ep_agent/mcp_servers.json` (points to `http://localhost:8000/mcp/`)
- Example dataset + converter:
  - JSONL: `verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.jsonl`
  - Parquet generator: `verl/examples/ep_mcp/frozen_lake/convert_jsonl_to_parquet.py`

Host Environment
- OS: Linux (from session)
- GPUs: 8x NVIDIA H100 (verified inside container via `nvidia-smi`)
- Docker image used: `verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2`
- NVIDIA Driver/CUDA as reported in container logs:
  - NVIDIA-SMI 535.129.03
  - CUDA 12.6 (Forward Compatibility enabled)

MCP Server (EP) – Start
1) Start EP MCP Frozen Lake server (host shell):
```bash
cd python-sdk/examples/frozen_lake_mcp
python server.py --port 8000 --seed 42 \
  > /home/bchen/home/eval_protocol/python-sdk/examples/frozen_lake_mcp/server_run.log 2>&1 &
```
2) Verify logs contain control-plane endpoints and startup:
```text
✅ Registered 4 session-aware control plane endpoints
🚀 Starting FrozenLake MCP server on port 8000
```

Container Provisioning
1) Create & start container (host shell):
```bash
docker create --gpus all --net=host --shm-size="10g" \\
  --cap-add=SYS_ADMIN \\
  -v /home/bchen/home/eval_protocol:/workspace/verl \\
  --name verl \\
  verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 sleep infinity
docker start verl
```
2) Verify GPUs inside container:
```bash
docker exec verl nvidia-smi
```
3) Install VERL (container):
```bash
docker exec verl bash -lc "cd /workspace/verl/verl && pip install --no-deps -e ."
```

Dataset Preparation (host)
1) Update/edit JSONL rows to ensure >= 8 rows for 8 GPUs (example path):
`verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.jsonl`

2) Generate Parquet (host). Note: host had NumPy 2.x warnings, but Parquet still wrote successfully:
```bash
python verl/examples/ep_mcp/frozen_lake/convert_jsonl_to_parquet.py
```
Expected message:
```text
Wrote 8 rows to /home/bchen/home/eval_protocol/verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.parquet
```

PPO Run Command (container)
The following Hydra overrides worked to get past config validations; the run later failed on dataloader constraints when rows < GPUs, so ensure 8+ rows and batch size divisible by 8.

Final command used:
```bash
docker exec -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HYDRA_FULL_ERROR=1 \
  verl bash -lc '
cd /workspace/verl/verl && python -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/workspace/verl/verl/recipe/ep_agent/agent_loop.yaml \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/workspace/verl/verl/recipe/ep_agent/tools_config.yaml \
  actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=512 \
  data.train_files=/workspace/verl/verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.parquet \
  data.val_files=/workspace/verl/verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.parquet \
  data.prompt_key=prompt data.return_raw_chat=true data.max_prompt_length=4096 \
  data.train_batch_size=8 data.shuffle=false data.dataloader_num_workers=0 \
  algorithm.adv_estimator=grpo critic.enable=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.logger=[console] \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  reward_model.enable=false
'
```

Paper Trail of Errors and Fixes
1) Missing actor micro-batch (FSDPActorConfig):
```text
AssertionError: [actor] Please set at least one of 'actor.ppo_micro_batch_size' or 'actor.ppo_micro_batch_size_per_gpu'
```
Fix: add `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`.

2) Missing rollout log-prob micro-batch:
```text
ValueError: [actor_rollout_ref.rollout] Please set at least one of 'actor_rollout_ref.rollout.log_prob_micro_batch_size' or '..._per_gpu'.
```
Fix: add `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`.

3) Critic override key mismatch:
```text
ConfigAttributeError: Key 'critic.micro_batch_size_per_gpu' is not in struct
```
Correct key is `critic.ppo_micro_batch_size_per_gpu=1` (per dataclass `FSDPCriticConfig`).

4) Train dataloader empty with small dataset:
```text
AssertionError: Train dataloader is empty!
```
Root cause: `drop_last=True` and dataset too small vs batch & DP/GPU constraints. Later, another constraint fired:
```text
AssertionError: real_train_batch_size (3) must be divisible by minimal possible batch size (8)
```
Fix attempted: increase dataset rows to 8 and set `data.train_batch_size=8`.

5) Mini-batch size default too large for small runs:
```text
ValueError: train_batch_size (8) must be >= actor.ppo_mini_batch_size (256)
```
Root cause: `actor.ppo_mini_batch_size` defaults to 256 in VERL. For small datasets/smoke tests, this must be lowered.
Fix: add `actor_rollout_ref.actor.ppo_mini_batch_size=8` (≤ `data.train_batch_size`).

6) Critic model path invalid for smoke test:
```text
OSError: Can't load the configuration of '~/models/deepseek-llm-7b-chat' ...
```
Fix for smoke test: disable critic and switch to an estimator that doesn't require values.
Add `algorithm.adv_estimator=grpo critic.enable=false`.

Current Status
- Config validations pass with overrides above.
- With 8 rows, `data.train_batch_size=8`, and `actor_rollout_ref.actor.ppo_mini_batch_size=8`, the run should proceed. If dataloader is still empty:
  - Verify Parquet in container exists at `/workspace/verl/verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.parquet` and contains 8 rows.
  - Ensure `data.shuffle=false` and no curriculum sampler is selected.
  - Check that `train_batch_size` equals 8 and `drop_last=True` leaves at least 1 batch.
  - If DP size or sampler divides data further, you may need `data.train_batch_size=8 * dp_size` (here dp_size defaults to 1 for Ray driver dataloader; DP is used inside workers).

Useful Checks (container)
```bash
python - <<'PY'
import pyarrow.parquet as pq
tbl = pq.read_table('/workspace/verl/verl/examples/ep_mcp/frozen_lake/frozen_lake_dataset.parquet')
print('rows:', tbl.num_rows)
PY
```

Log Tail
```bash
docker logs -f --tail=200 verl
```

MCP Server Logs
```bash
tail -n 200 /home/bchen/home/eval_protocol/python-sdk/examples/frozen_lake_mcp/server_run.log
```

Notes / Next Steps
- If dataloader remains empty even with 8 rows and batch 8:
  - Try `data.train_batch_size=8` and `trainer.n_gpus_per_node=1` just to validate end-to-end, then scale up.
  - Or increase dataset rows to a larger multiple (e.g., 32) and keep `train_batch_size` a multiple of 8.
  - If you prefer to avoid the DP batch constraints, run single-GPU first to validate MCP integration, then scale.
- Ensure Hugging Face auth is available for `Qwen/Qwen3-30B-A3B-Instruct-2507` if pull fails: set `HUGGING_FACE_HUB_TOKEN` in container env.

Contact Handoff
- All commands above are exact reproductions of what we ran.
- Key files to inspect: 
  - `verl/recipe/ep_agent/*.yaml,json`
  - `verl/examples/ep_mcp/frozen_lake/*`
  - `python-sdk/examples/frozen_lake_mcp/*`
- The run is very close; remaining issue centers on dataloader constraints when dataset size and batch size do not satisfy divisibility rules across GPUs.


