FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    build-essential \
    libibverbs-dev \
    libnuma-dev \
    numactl \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

RUN pip install vllm==0.12.0

RUN pip install pybind11

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn && \
    rm -rf /var/lib/apt/lists/*

RUN pip install nvidia-mathdx

RUN MAX_JOBS=128 pip install -v --disable-pip-version-check --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

RUN export NVTE_FRAMEWORK=pytorch && MAX_JOBS=128 NVTE_BUILD_THREADS_PER_JOB=4 pip3 install --resume-retries 999 --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10

RUN pip install --upgrade transformers tokenizers

RUN pip install codetiming tensordict mathruler pylatexenc qwen_vl_utils

RUN pip install flash_attn==2.8.1 --no-build-isolation

RUN NSIGHT_VERSION=2025.6.1_2025.6.1.190-1_$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/nsight-systems-${NSIGHT_VERSION}.deb && \
    apt-get update && apt-get install -y libxcb-cursor0 && \
    apt-get install -y ./nsight-systems-${NSIGHT_VERSION}.deb && \
    rm -rf /usr/local/cuda/bin/nsys && \
    ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys  /usr/local/cuda/bin/nsys && \
    rm -rf /usr/local/cuda/bin/nsys-ui && \
    ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys-ui /usr/local/cuda/bin/nsys-ui && \
    rm nsight-systems-${NSIGHT_VERSION}.deb && \
    rm -rf /var/lib/apt/lists/*

# =========================
# Install DeepEP
# =========================
# the dependency of IBGDA
# RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Clone and build deepep and deepep-nvshmem
WORKDIR /home/dpsk_a2a
RUN git clone -b v2.3.1 https://github.com/NVIDIA/gdrcopy.git && \
    git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout a84a248 && \
    cd /home/dpsk_a2a && \
    wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz && \
    tar -xvf nvshmem_src_3.2.5-1.txz && mv nvshmem_src deepep-nvshmem && \
    cd deepep-nvshmem && git apply /home/dpsk_a2a/DeepEP/third-party/nvshmem.patch && \
    sed -i '16i#include <getopt.h>' /home/dpsk_a2a/deepep-nvshmem/examples/moe_shuffle.cu

ENV CUDA_HOME=/usr/local/cuda
# Set MPI environment variables. Having errors when not set.
# ENV CPATH=/usr/local/mpi/include:$CPATH
# ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH
# ENV LD_LIBRARY_PATH=/usr/local/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV GDRCOPY_HOME=/home/dpsk_a2a/gdrcopy

# Build deepep-nvshmem
WORKDIR /home/dpsk_a2a/deepep-nvshmem
RUN NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/home/dpsk_a2a/deepep-nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 && cd build && make install -j

# Build deepep
WORKDIR /home/dpsk_a2a/DeepEP
ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/install
RUN python setup.py install

RUN pip3 install --no-deps trl

RUN pip3 install nvtx matplotlib liger_kernel

RUN pip install -U git+https://github.com/ISEEKYAN/mbridge.git

RUN pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.15.0

RUN pip install git+https://github.com/volcengine/verl.git@v0.6.0 && \
    pip uninstall -y verl

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*