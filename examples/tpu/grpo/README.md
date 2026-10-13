# GRPO RL Training on Google Cloud TPU with `verl` + `verl-hardware-plugin`

This directory contains reference scripts for running **GRPO (Group Relative Policy Optimization) RL training** on Google Cloud TPU v6e using `verl` core together with [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin):
- `run_qwen3_0_6b_torchtitan.sh`: Fast 5-step smoke test (`Qwen3-0.6B`, `train_batch_size=4`, `total_training_steps=5`, `ray-plasma-store` weight sync) to validate that the TPU stack comes up cleanly.
- `run_qwen3_4b_torchtitan.sh`: Realistic 100-step GRPO training recipe (`Qwen3-4B`, `train_batch_size=32`, `rollout.n=8`, `SimpleFSDP` + `TPUSplashAttention` + `torch.compile`, and peer-to-peer `tpu-sync` weight sync).

The training setup uses:
- **Actor Engine**: TorchTitan (`model_engine=torchtitan`, registered by `verl-hardware-plugin`)
- **Rollout Engine**: vLLM on TPU (`actor_rollout_ref.rollout.name=vllm`)
- **Weight Sync**: `actor_rollout_ref.rollout.checkpoint_engine.backend`, either `ray-plasma-store` (through the Ray object store) or `tpu-sync` (peer to peer between the TPU slices), both provided by `verl-hardware-plugin`
- **Trainer Mode**: V1 disaggregated async overlap (`trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`)
- **Placement Strategy**: Non-colocated multi-slice execution (Slice 0: 8-chip TorchTitan trainer; Slice 1: 8-chip TP=8 vLLM rollout server)

---

## Step-by-Step Instructions

### Step 1. Clone `verl` and `verl-hardware-plugin`

On the machine from which you submit Ray jobs, clone both repositories. Both are shipped to the cluster with every job, so code changes need no image rebuild or `pip install` on the cluster:

```bash
# 1. Clone verl
git clone https://github.com/verl-project/verl
cd verl

# 2. Clone verl-hardware-plugin alongside verl
git clone https://github.com/verl-project/verl-hardware-plugin.git ../verl-hardware-plugin
export PLUGIN_REPO="$(cd ../verl-hardware-plugin && pwd)"
```

---

### Step 2. Build the TPU Image (optional)

[`Dockerfile`](Dockerfile) builds the image the Ray pods run: Ray 2.56, PyTorch 2.13 (CPU) + `torch_tpu`, JAX + `libtpu`, `tpu-sync`, `google-pytorch/torchtitan` and vLLM + `vllm-torchtpu`. `verl` and `verl-hardware-plugin` are not baked in; they are shipped with every job (Step 4). Skip this step if you already have an image, e.g. `us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261013-grpo-dockerfile`, which was built from this file.

The registry and GitHub tokens are passed as BuildKit secrets, so they are not stored in the image:

```bash
export AUTH_TOKEN="$(gcloud auth print-access-token)"  # read access to the TorchTPU Python registry
export GITHUB_TOKEN=<token>  # read access to google-pytorch/torchtitan and vllm-project/vllm-torchtpu
docker buildx build -f examples/tpu/grpo/Dockerfile \
  --secret id=auth_token,env=AUTH_TOKEN --secret id=github_token,env=GITHUB_TOKEN \
  -t <registry>/verl-tpu:<tag> --push examples/tpu/grpo
```

---

### Step 3. Connect to the GKE Ray TPU Cluster

Ensure your KubeRay cluster on TPU v6e (2 slices of `v6e-8` = 4 hosts $\times$ 4 chips = 16 TPU chips total, running the image from Step 2) is `Running`, and port-forward the Ray head dashboard service:

```bash
# Verify 1 head pod + 4 TPU worker pods are Running
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster

# Port-forward the Ray Dashboard / Job Submission API to localhost:23333
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 > /dev/null 2>&1 &
export RAY_ADDRESS="http://localhost:23333"
```

Ensure the model and dataset paths referenced by the scripts are mounted inside the Ray pods (or override `MODEL_PATH`, `TRAIN_FILE`, and `TEST_FILE` via environment variables):
- `MODEL_PATH`: `/data/jialei/assets/hf/Qwen3-0.6B` or `/data/jialei/assets/hf/Qwen3-4B`
- `TRAIN_FILE`: `/data/jialei/data/gsm8k/train.parquet`
- `TEST_FILE`: `/data/jialei/data/gsm8k/test.parquet`

---

### Step 4. Submit a GRPO Job

From the root of the `verl` repository, submit the job with `py_modules` pointing to `${PLUGIN_REPO}/verl_hardware_plugin` and `"VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"`. Ray packages both `verl` and `verl_hardware_plugin` from your local checkout and distributes them to the head pod and all TPU worker pods automatically.

#### Fast 5-Step Smoke Test (`Qwen3-0.6B`)

```bash
export RAY_ADDRESS="http://localhost:23333"
export PLUGIN_REPO="/path/to/verl-hardware-plugin"

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\", \".venv\", \"__pycache__\", \".ruff_cache\", \".mypy_cache\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"VERL_LOGGING_LEVEL\": \"INFO\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

#### Realistic GRPO Training Run (`Qwen3-4B`)

```bash
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "{
    \"py_modules\": [\"${PLUGIN_REPO}/verl_hardware_plugin\"],
    \"excludes\": [\".git\", \"logs\", \"*.log\", \"*.pt\", \"*.bin\", \".venv\", \"__pycache__\", \".ruff_cache\", \".mypy_cache\"],
    \"env_vars\": {
      \"PYTHONPATH\": \".\",
      \"PYTHONUNBUFFERED\": \"1\",
      \"VERL_PLATFORM\": \"tpu\",
      \"VERL_USE_EXTERNAL_MODULES\": \"verl_hardware_plugin\",
      \"VERL_LOGGING_LEVEL\": \"INFO\",
      \"RAY_memory_monitor_refresh_ms\": \"0\",
      \"RAY_memory_usage_threshold\": \"0.99\",
      \"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS\": \"1\",
      \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\"
    }
  }" \
  -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
```

---

### Step 5. Monitor Progress

```bash
# Check job status
ray job status --address "${RAY_ADDRESS}" <JOB_ID>

# Stream live logs
ray job logs --follow --address "${RAY_ADDRESS}" <JOB_ID>
```
