# syntax=docker/dockerfile:1
# TPU image for the GRPO examples in this directory. verl and verl-hardware-plugin are not installed here: every job
# ships them through the Ray runtime_env (see README.md).
#
# Build from the repository root. The tokens are BuildKit secrets, so they are not stored in the image:
#   export AUTH_TOKEN="$(gcloud auth print-access-token)"  # TorchTPU Python registry
#   export GITHUB_TOKEN=<token>  # read access to google-pytorch/torchtitan and vllm-project/vllm-torchtpu
#   docker buildx build -f examples/tpu/grpo/Dockerfile \
#     --secret id=auth_token,env=AUTH_TOKEN --secret id=GIT_AUTH_TOKEN,env=GITHUB_TOKEN \
#     -t verl-tpu:latest examples/tpu/grpo

FROM rayproject/ray:2.56.0-py312

ARG TORCH_TPU_VERSION=0.1.2.dev20261006000518
ARG TPU_SYNC_VERSION=0.0.1.dev20261006025601
# Later google-pytorch/torchtitan commits re-lay out torchtitan.components, which verl's TorchTitan engine (written
# against torchtitan 0.3.0) does not support.
ARG TORCHTITAN_COMMIT=f880cc73725de77d892ecab888244e7718e5ad65
# Includes vllm-torchtpu#1160 (same-step prefix-hit fix).
ARG VLLM_TORCHTPU_COMMIT=9faafb177ca1f1fca39d404caa5a7fadbfdb9a06

USER root
RUN apt-get update && apt-get install -y gcc-12 g++-12 build-essential cmake ninja-build \
        libgoogle-perftools4 libopenblas-base libopenmpi-dev libomp-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

# Sources. BuildKit fetches the private repositories with the GIT_AUTH_TOKEN secret.
ADD https://github.com/google-pytorch/torchtitan.git#${TORCHTITAN_COMMIT} /torchtitan
ADD https://github.com/vllm-project/vllm-torchtpu.git#${VLLM_TORCHTPU_COMMIT} /vllm-torchtpu
ADD https://github.com/vllm-project/vllm.git#v0.29.0 /vllm
# vLLM's TPU requirements must not pull upstream tpu-inference (JAX/torchax) over vllm-torchtpu.
RUN sed -i '/tpu-inference/d' /vllm/requirements/tpu.txt && ln -sfnT /vllm-torchtpu /torchtpu-vllm

# Python packages, with the private TorchTPU registry as an extra index (PyPI stays the primary index). The pip calls
# run one after another on purpose: later calls may override pins of earlier ones, so the core packages are re-pinned
# at the end and checked.
ENV VLLM_TARGET_DEVICE=tpu
RUN --mount=type=secret,id=auth_token,env=AUTH_TOKEN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -e
export PIP_CACHE_DIR=/root/.cache/pip PIP_ROOT_USER_ACTION=ignore
tpu_index="https://oauth2accesstoken:${AUTH_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
export PIP_EXTRA_INDEX_URL="${tpu_index}"
libtpu_links=https://storage.googleapis.com/jax-releases/libtpu_releases.html
torch_cpu_index=https://download.pytorch.org/whl/cpu
torch=2.13.0 torchvision=0.28.0 jax=0.10.2 libtpu=0.0.47
transformers="transformers>=5.10.4,<5.11"

pip install --upgrade pip uv keyrings.google-artifactregistry-auth pandas==2.3.3
pip install "jax[tpu]==${jax}" -f "${libtpu_links}"
pip install tokamax==0.0.14
pip install "torch==${torch}" "torchvision==${torchvision}" --index-url "${torch_cpu_index}"
pip install "libtpu==${libtpu}" -f "${libtpu_links}"
pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}" --index-url "${tpu_index}"
pip install typeguard==2.13.3
pip install portpicker absl-py numpy gcsfs frozendict triton fairscale tamm spmd_types==0.2.1 "tyro>=1.0.5"
(cd /torchtitan && pip install -r requirements.txt)

# vLLM without device extensions, then vllm-torchtpu.
pip install --upgrade pip setuptools
pip install cmake ninja setuptools-scm setuptools-rust "pyzmq>=27"
cd /vllm-torchtpu
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 VLLM_TARGET_DEVICE=empty \
    pip install --no-build-isolation -e ../vllm
pip install tpu-info pathwaysutils numba==0.65.0 "${transformers}" "fastapi<0.137"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    pip install --no-build-isolation --pre --no-deps -e .
cd /

# verl dependencies.
pip install accelerate codetiming datasets dill hydra-core peft "pyarrow>=19.0.0" pybind11 pylatexenc \
    "tensordict>=0.10.0" torchdata "${transformers}" wandb "packaging>=20.0" uvicorn "fastapi<0.137" \
    latex2sympy2_extended math_verify tensorboard TransferQueue==0.1.8

# Re-pin the core packages after all resolvers ran, and fail the build if any of them drifted.
pip install --no-deps "torch==${torch}" "torchvision==${torchvision}" --index-url "${torch_cpu_index}"
pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}" --index-url "${tpu_index}"
pip install --no-deps "jax==${jax}" "jaxlib==${jax}" "libtpu==${libtpu}" numba==0.65.0 "${transformers}" "fastapi<0.137" \
    -f "${libtpu_links}"
python - <<PY
import importlib.metadata as m, sys
want = {"torch": "${torch}+cpu", "torch-tpu": "${TORCH_TPU_VERSION}", "tpu-sync-torch": "${TPU_SYNC_VERSION}",
        "jax": "${jax}", "jaxlib": "${jax}", "libtpu": "${libtpu}"}
got = {k: m.version(k) for k in want}
print("Pinned versions:", got)
sys.exit(0 if got == want else f"Version drift: want {want}, got {got}")
PY
EOF

WORKDIR /app
ENV PYTHONPATH="/vllm-torchtpu/src:/vllm-torchtpu:/torchtpu-vllm/src:/torchtpu-vllm:/app:/torchtitan:/torchtitan/torchtitan"
