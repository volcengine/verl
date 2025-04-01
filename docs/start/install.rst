Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

verl supports various backends. Currently, the following configurations are available:

- **FSDP** and **Megatron-LM** (optional) for training.
- **SGLang** (preview), **vLLM** and **TGI** for rollout generation.

Choices of Backend Engines
----------------------------

1. Training:

We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in :doc:`FSDP Workers<../workers/fsdp_workers>`.

For users who pursue better scalability, we recommend using **Megatron-LM** backend. Currently, we support Megatron-LM v0.11 [1]_. The guide for using Megatron-LM backend can be found in :doc:`Megatron-LM Workers<../workers/megatron_workers>`.

2. Inference:

For inference, the integration of both vllm v0.6.3 and v0.8.2 is stable. For huggingface TGI integration, it is usually used for debugging and single GPU exploration. Regarding sglang integration, it is blazing fast and under rapid development - we release it as a preview feature and please give us feedbacks.


Install from docker image
-------------------------

We provide pre-built Docker images for quick setup. For SGLang usage, please follow the later sections in this doc.

Image and tag: ``whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6``. See files under ``docker/`` for NGC-based image or if you want to build your own.

1. Launch the desired Docker image:

.. code:: bash

    docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v <image:tag>


2.	Inside the container, install verl:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .
    # or install from pypi via `pip3 install verl`

.. note::
    
    The Docker image is built with the following configurations:

    - **PyTorch**: 2.4.0+cu124
    - **CUDA**: 12.4
    - **Megatron-LM**: core_r0.11.0
    - **vLLM**: 0.6.3
    - **Ray**: 2.10.0
    - **TransformerEngine**: 2.0.0+754d2a0

    Now verl has been **compatible to Megatron-LM core_r0.11.0**, and there is **no need to apply patches** to Megatron-LM. Also, the image has integrated **Megatron-LM core_r0.11.0**, located at ``/opt/nvidia/Meagtron-LM``. One more thing, because verl only use ``megatron.core`` module for now, there is **no need to modify** ``PATH`` if you have installed Megatron-LM with this docker image.


Install verl-SGLang from scratch
---------------------------------------------

If you want to use SGLang instead of vllm for inference, please follow the instruction here. SGLang has largely support the rearch and inference workload at xAI. For verl-sglang installation, ignore the version conflicts reported by pip with vllm. And, SGLang support native API for RLHF, do not need to patch a single line of code.

The following steps are quick installation guide for verl-SGLang.

.. code:: bash

    # Create a virtual environment and use uv for quick installation
    python3 -m venv ~/.python/verl-sglang && source ~/.python/verl-sglang/bin/activate
    python3 -m pip install --upgrade pip && python3 -m pip install --upgrade uv

    # Install verl-SGLang
    git clone https://github.com/volcengine/verl verl-sglang && cd verl-sglang
    python3 -m uv pip install .
    
    # Install the latest stable version of sglang with verl support, currently, the latest version is 0.4.3.post3
    # For SGLang installation, you can also refer to https://docs.sglang.ai/start/install.html
    python3 -m uv pip install "sglang[all]==0.4.4.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python


Install from custom environment
---------------------------------------------

If you do not want to use the official docker image, here is how to start from your own environment. To manage environment, we recommend using conda:

.. code:: bash

   conda create -n verl python==3.10
   conda activate verl

For installing the latest version of verl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   # install verl together with some lightweight dependencies in setup.py
   pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
   pip3 install flash-attn --no-build-isolation
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip3 install -e .


Megatron is optional. It's dependencies can be setup as below:

.. code:: bash

   # apex
   pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
       git+https://github.com/NVIDIA/apex

   # transformer engine
   pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@stable
   # megatron core
   pip3 install megatron-core==0.11.0


Install with AMD GPUs - ROCM kernel support
------------------------------------------------------------------

When you run on AMD GPUs (MI300) with ROCM platform, you cannot use the previous quickstart to run verl. You should follow the following steps to build a docker and run it. 

If you encounter any issues in using AMD GPUs running verl, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.

Find the docker for AMD ROCm: `docker/Dockerfile.rocm <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.rocm>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    #  Build the docker in the repo dir:
    # docker build -f docker/Dockerfile.rocm -t verl-rocm:03.04.2015 .
    # docker images # you can find your built docker
    FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

    # Set working directory
    # WORKDIR $PWD/app

    # Set environment variables
    ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

    # Install vllm
    RUN pip uninstall -y vllm && \
        rm -rf vllm && \
        git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
        cd vllm && \
        MAX_JOBS=$(nproc) python3 setup.py install && \
        cd .. && \
        rm -rf vllm

    # Copy the entire project directory
    COPY . .

    # Install dependencies
    RUN pip install "tensordict<0.6" --no-deps && \
        pip install accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        liger-kernel \
        numpy \
        pandas \
        peft \
        "pyarrow>=15.0.0" \
        pylatexenc \
        "ray[data,train,tune,serve]" \
        torchdata \
        transformers \
        wandb \
        orjson \
        pybind11 && \
        pip install -e . --no-deps

Build the image:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker build -t verl-rocm .

Launch the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run --rm -it \
      --device /dev/dri \
      --device /dev/kfd \
      -p 8265:8265 \
      --group-add video \
      --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --privileged \
      -v $HOME/.ssh:/root/.ssh \
      -v $HOME:$HOME \
      --shm-size 128G \
      -w $PWD \
      verl-rocm \
      /bin/bash

(Optional): If you do not want to root mode and require assign yuorself as the user
Please add ``-e HOST_UID=$(id -u)`` and ``-e HOST_GID=$(id -g)`` into the above docker launch script. 

(Currently Support): Training Engine: FSDP; Inference Engine: vLLM - We will support Megatron and SGLang in the future.
