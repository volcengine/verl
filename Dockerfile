# RUN git clone https://github.com/vllm-project/vllm.git && cd vllm; pip install -r requirements/tpu.txt; sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev; VLLM_TARGET_DEVICE="tpu" python setup.py develop; cd ..

# syntax=docker/dockerfile:experimental
# To build the docker container, run:
# docker build --network=host --progress=auto -t verl_tpu -f Dockerfile .
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Add the Cloud Storage FUSE distribution URL as a package source
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-bullseye main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install the Google Cloud SDK and GCS fuse
RUN apt-get update && apt-get install -y google-cloud-sdk git fuse gcsfuse && gcsfuse -v

# Set the default Python version to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

WORKDIR /workspaces

# Install verl
# Optimization: we rerun `pip install -e .` only if `pyproject.toml` changes.
# Copy only the installation-related files first to make Docker cache them separately.
# WORKDIR /workspaces/torchprime
# COPY pyproject.toml /workspaces/torchprime/
# RUN pip install -e .

# Copy verl into the container
RUN echo "Cloning verl from GitHub";
RUN git clone --depth 1 https://github.com/Chrisytz/verl.git /workspaces/verl; pip uninstall -y libtpu torch_xla; pip install --no-cache-dir "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler pytest py-spy pyext pre-commit ruff 'torch_xla[tpu]~=2.6.0' -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html

RUN pip install "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

WORKDIR /workspaces/verl
RUN pip install --no-deps -e .

WORKDIR /workspaces/

# Install vLLM for TPU
RUN git clone https://github.com/vllm-project/vllm.git && cd vllm; pip install -r requirements/tpu.txt; VLLM_TARGET_DEVICE="tpu" python setup.py develop

