# syntax=docker/dockerfile:1
# TPU image for the GRPO examples in this directory: Ray, PyTorch + torch_tpu, JAX + libtpu, tpu-sync,
# google-pytorch/torchtitan and vLLM + vllm-torchtpu. verl and verl-hardware-plugin are not installed here; every
# job ships them through the Ray runtime_env (see README.md).
#
# Build from the repository root. Both tokens are BuildKit secrets, so they are not stored in any image layer or in
# the image history:
#   export AUTH_TOKEN="$(gcloud auth print-access-token)"  # read access to the TorchTPU Python registry
#   export GITHUB_TOKEN=<token>  # read access to google-pytorch/torchtitan and vllm-project/vllm-torchtpu
#   docker buildx build -f examples/tpu/grpo/Dockerfile \
#     --secret id=auth_token,env=AUTH_TOKEN --secret id=github_token,env=GITHUB_TOKEN \
#     -t verl-tpu:latest examples/tpu/grpo

FROM rayproject/ray:2.56.0-py312

ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_TPU_VERSION=0.1.2.dev20261006000518
ARG TPU_SYNC_VERSION=0.0.1.dev20261006025601
ARG JAX_VERSION=0.10.2
ARG LIBTPU_VERSION=0.0.47
# google-pytorch/torchtitan (2026-09-18). Later commits re-lay out torchtitan.components (checkpointer/, data/,
# optimizer/) and drop spmd_backend="default", which verl's TorchTitan engine and verl-hardware-plugin do not support.
ARG TORCHTITAN_COMMIT=f880cc73725de77d892ecab888244e7718e5ad65
ARG VLLM_VERSION=0.29.0
# vllm-project/vllm-torchtpu, includes #1160 (same-step prefix-hit fix).
ARG VLLM_TORCHTPU_COMMIT=9faafb177ca1f1fca39d404caa5a7fadbfdb9a06
ARG TRANSFORMERS="transformers>=5.10.4,<5.11"
ARG FASTAPI="fastapi<0.137"
ARG NUMBA="numba==0.65.0"

ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG LIBTPU_FIND_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Private TorchTPU registry. pip uses it as an extra index, so PyPI stays the primary index for standard packages
# like JAX. The access token comes from the auth_token build secret.
ARG TORCH_TPU_INDEX=us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/

# Build-time pip settings (ARGs are not kept in the image environment). The pip cache lives in a BuildKit cache
# mount, so it speeds up rebuilds without being stored in the image.
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG PIP_ROOT_USER_ACTION=ignore
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

USER root
WORKDIR /

RUN apt-get update && apt-get install -y \
        gcc-12 g++-12 build-essential cmake ninja-build \
        libgoogle-perftools4 libopenblas-base libopenmpi-dev libomp-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

# JAX / libtpu, CPU PyTorch (TPU support comes from torch_tpu), torch_tpu, tpu-sync and the torchtitan TPU deps.
# The pip calls stay separate and in this order on purpose: later calls may override pins of earlier ones, and the
# last pip step below re-pins the core packages.
RUN --mount=type=secret,id=auth_token,env=AUTH_TOKEN --mount=type=cache,target=/root/.cache/pip \
    export PIP_EXTRA_INDEX_URL="https://oauth2accesstoken:${AUTH_TOKEN}@${TORCH_TPU_INDEX}" \
    && (pip uninstall -y torch torchvision || true) \
    && pip install --upgrade pip uv keyrings.google-artifactregistry-auth pandas==2.3.3 \
    && pip install "jax[tpu]==${JAX_VERSION}" -f "${LIBTPU_FIND_LINKS}" \
    && pip install "tokamax==0.0.14" \
    && pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_CPU_INDEX}" \
    && pip install "libtpu==${LIBTPU_VERSION}" -f "${LIBTPU_FIND_LINKS}" \
    && pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}" \
        --index-url "https://oauth2accesstoken:${AUTH_TOKEN}@${TORCH_TPU_INDEX}" \
    && pip install "typeguard==2.13.3" \
    && pip install portpicker absl-py numpy gcsfs frozendict triton fairscale tamm "spmd_types==0.2.1" "tyro>=1.0.5"

# Sources of torchtitan, vllm-torchtpu and vLLM at the pinned revisions. The GitHub token is sent through an
# http.extraheader that is not saved, so it never ends up in a remote URL or in .git/config. vLLM's TPU
# requirements are patched so they do not pull upstream tpu-inference (JAX/torchax) over vllm-torchtpu.
RUN --mount=type=secret,id=github_token,env=GITHUB_TOKEN \
    set -e; \
    gh_fetch() { \
        git init -q "$2"; \
        git -C "$2" remote add origin "https://github.com/$1.git"; \
        git -C "$2" -c http.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 -w0)" \
            fetch -q --depth 1 origin "$3"; \
        git -C "$2" checkout -q FETCH_HEAD; \
    }; \
    gh_fetch google-pytorch/torchtitan /torchtitan "${TORCHTITAN_COMMIT}"; \
    gh_fetch vllm-project/vllm-torchtpu /vllm-torchtpu "${VLLM_TORCHTPU_COMMIT}"; \
    ln -sfnT /vllm-torchtpu /torchtpu-vllm; \
    git clone -q --depth 1 --branch "v${VLLM_VERSION}" https://github.com/vllm-project/vllm.git /vllm; \
    sed -i '/tpu-inference/d' /vllm/requirements/tpu.txt

# torchtitan requirements, vLLM (no device extensions) and vllm-torchtpu.
ENV VLLM_TARGET_DEVICE="tpu"
RUN --mount=type=secret,id=auth_token,env=AUTH_TOKEN --mount=type=cache,target=/root/.cache/pip \
    export PIP_EXTRA_INDEX_URL="https://oauth2accesstoken:${AUTH_TOKEN}@${TORCH_TPU_INDEX}" \
    && cd /torchtitan && pip install -r requirements.txt \
    && pip install --upgrade pip setuptools \
    && pip install cmake ninja setuptools-scm setuptools-rust "pyzmq>=27" \
    && cd /vllm-torchtpu \
    && TORCH_DEVICE_BACKEND_AUTOLOAD=0 SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" VLLM_TARGET_DEVICE="empty" \
        pip install --no-build-isolation -e ../vllm \
    && pip install tpu-info pathwaysutils "${NUMBA}" "${TRANSFORMERS}" "${FASTAPI}" \
    && TORCH_DEVICE_BACKEND_AUTOLOAD=0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
        pip install --no-build-isolation --pre --no-deps -e .

# verl dependencies, then re-pin PyTorch / torch_tpu / tpu-sync / JAX / libtpu / Transformers after all resolvers ran.
RUN --mount=type=secret,id=auth_token,env=AUTH_TOKEN --mount=type=cache,target=/root/.cache/pip \
    export PIP_EXTRA_INDEX_URL="https://oauth2accesstoken:${AUTH_TOKEN}@${TORCH_TPU_INDEX}" \
    && pip install accelerate codetiming datasets dill hydra-core peft "pyarrow>=19.0.0" pybind11 pylatexenc \
        "tensordict>=0.10.0" torchdata "${TRANSFORMERS}" wandb "packaging>=20.0" uvicorn "${FASTAPI}" \
        latex2sympy2_extended math_verify tensorboard TransferQueue==0.1.8 \
    && pip install --no-deps "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_CPU_INDEX}" \
    && pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}" \
        --index-url "https://oauth2accesstoken:${AUTH_TOKEN}@${TORCH_TPU_INDEX}" \
    && pip install --no-deps "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" "libtpu==${LIBTPU_VERSION}" \
        "${NUMBA}" "${TRANSFORMERS}" "${FASTAPI}" -f "${LIBTPU_FIND_LINKS}"

# Fail the build if a pinned package drifted.
RUN python -c "import importlib.metadata as m, sys; \
want = {'torch': '${TORCH_VERSION}+cpu', 'torch-tpu': '${TORCH_TPU_VERSION}', 'tpu-sync-torch': '${TPU_SYNC_VERSION}', \
        'jax': '${JAX_VERSION}', 'jaxlib': '${JAX_VERSION}', 'libtpu': '${LIBTPU_VERSION}'}; \
got = {k: m.version(k) for k in want}; \
print('Pinned versions:', got); \
sys.exit(0 if got == want else f'Version drift: want {want}, got {got}')"

WORKDIR /app
ENV PYTHONPATH="/vllm-torchtpu/src:/vllm-torchtpu:/torchtpu-vllm/src:/torchtpu-vllm:/app:/torchtitan:/torchtitan/torchtitan"
