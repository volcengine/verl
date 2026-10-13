# GRPO on Cloud TPU with `verl` + `verl-hardware-plugin`

GRPO on GSM8K with Qwen3 on 2 TPU v6e-4 slices (8 chips): a TorchTitan trainer on one slice, a vLLM rollout on the other, and `tpu-sync` weight sync between them. TPU support comes from [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin).

| File | |
|---|---|
| [`run_qwen3_0_6b_torchtitan.sh`](run_qwen3_0_6b_torchtitan.sh) | 5-step smoke test (Qwen3-0.6B) |
| [`run_qwen3_4b_torchtitan.sh`](run_qwen3_4b_torchtitan.sh) | 100-step GRPO run (Qwen3-4B) |
| [`ray-tpu-v6e4-2slice.yaml`](ray-tpu-v6e4-2slice.yaml) | KubeRay `RayCluster`: 1 CPU head pod and 2 TPU v6e-4 slices |
| [`Dockerfile`](Dockerfile) | Image of the Ray pods |

## 1. Build the image

```bash
docker buildx build -t <IMAGE> --push examples/tpu/grpo
```

## 2. Upload the model and data

The scripts read the models and the dataset from `$RAY_DATA_HOME` in the pods, which defaults to `/data/jialei` (`gs://torchprime/jialei`, mounted at `/data`). To use your own bucket:

```bash
hf download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B
hf download Qwen/Qwen3-4B --local-dir Qwen3-4B
python examples/data_preprocess/gsm8k.py --local_save_dir gsm8k
gcloud storage cp -r Qwen3-0.6B Qwen3-4B gs://<BUCKET>/<DIR>/assets/hf/
gcloud storage cp -r gsm8k gs://<BUCKET>/<DIR>/data/
```

Then run the scripts with `RAY_DATA_HOME=/data/<DIR>` (step 4).

## 3. Create the Ray cluster

You need a GKE Standard cluster with the Ray operator, Cloud Storage FUSE and Workload Identity (`--addons=RayOperator,GcsFuseCsiDriver --workload-pool=<PROJECT>.svc.id.goog`). Add 2 `ct6e-standard-4t` nodes; each one is a v6e-4 slice:

```bash
gcloud container node-pools create <NODE_POOL> --cluster <CLUSTER> --region <REGION> --node-locations <ZONE> \
  --machine-type ct6e-standard-4t --num-nodes 2 --spot --disk-type hyperdisk-balanced \
  --workload-metadata GKE_METADATA --scopes cloud-platform
```

In [`ray-tpu-v6e4-2slice.yaml`](ray-tpu-v6e4-2slice.yaml), set the `image`, the `bucketName` and the Google service account of `ray-ksa`, which needs read access to the bucket and `roles/iam.workloadIdentityUser` for `serviceAccount:<PROJECT>.svc.id.goog[default/ray-ksa]`. Then:

```bash
kubectl apply -f examples/tpu/grpo/ray-tpu-v6e4-2slice.yaml
kubectl get pods -l ray.io/cluster=ray-tpu-v6e4-cluster  # wait for 3 pods at 2/2 Running
kubectl port-forward svc/ray-tpu-v6e4-cluster-head-svc 23333:8265 &
```

## 4. Run

From the root of `verl`, with `verl-hardware-plugin` cloned next to it. Ray ships both checkouts to the cluster with every job.

```bash
git clone https://github.com/verl-project/verl-hardware-plugin.git ../verl-hardware-plugin
export RAY_ADDRESS=http://localhost:23333
RUNTIME_ENV='{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"], "excludes": [".git", ".venv"],
  "env_vars": {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin", "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}'

ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
```

The scripts take their settings from environment variables (e.g. `-- RAY_DATA_HOME=/data/<DIR> TOTAL_TRAINING_STEPS=5 bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh`) and pass extra arguments to `verl.trainer.main_ppo` as Hydra overrides. Follow a job with `ray job logs -f <JOB_ID>`.

To use 2 v6e-8 slices (16 chips) instead, create two node pools with `--num-nodes 2 --tpu-topology 2x4`, set `numOfHosts: 2` and `cloud.google.com/gke-tpu-topology: 2x4` in the YAML, and run the scripts with `NNODES_TRAINER=2 NNODES_ROLLOUT=2`.
