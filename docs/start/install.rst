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

.. note:: 

    We are announcing the direct support of megatron GPTModel, without need to implement your own model any more. Also it's easy to use TransformerEngine's support for even higher performance.
    The main branch of verl has enabled this feature, but may not stable enough. If you encounter issues, please feel free to report and try `0.3.x branch <https://github.com/volcengine/verl/tree/v0.3.x>`_ instead.

2. Inference:

For inference, the integration of both vllm v0.6.3 and v0.8.2 is stable. Please avoid vllm v0.7 since it has known bugs.

Regarding sglang integration, it is blazing fast and under rapid development - we release it as a preview feature and please give us feedbacks.

For huggingface TGI integration, it is usually used for debugging and single GPU exploration.


Install from docker image
-------------------------

We provide pre-built Docker images for quick setup. For SGLang usage, please follow the later sections in this doc.

Image and tag: ``whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6`` if you need both FSDP and Megatron support.

We highly recommend ``hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2-verl0.3.0.post1`` with vllm v0.8.2 for fastest rollout performance with FSDP.

See files under ``docker/`` for NGC-based image or if you want to build your own.

1. Launch the desired Docker image:

.. code:: bash

    docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v <image:tag>


2.	Inside the container, install latest verl:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .
    # or install from pypi via `pip3 install verl`

.. note::
    
    The Docker image ``whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6`` is built with the following configurations:

    - **PyTorch**: 2.4.0+cu124
    - **CUDA**: 12.4
    - **Megatron-LM**: core_r0.11.0
    - **vLLM**: 0.6.3
    - **Ray**: 2.10.0
    - **TransformerEngine**: 2.0.0+754d2a0

    Now verl has been **compatible to Megatron-LM core_r0.11.0**, and there is **no need to apply patches** to Megatron-LM. Also, the image has integrated **Megatron-LM core_r0.11.0**, located at ``/opt/nvidia/Meagtron-LM``. One more thing, because verl only use ``megatron.core`` module for now, there is **no need to modify** ``PATH`` if you have installed Megatron-LM with this docker image.


Install SGLang as rollout backend
---------------------------------------------

If you want to use SGLang instead of vllm for inference, please follow the instruction here. SGLang has largely support the research and inference workload at xAI. SGLang support native API for RLHF, do not need to patch a single line of code.

The following steps are quick installation guide for verl-SGLang.

1. Install from source

.. code:: bash
    # clone from github
    git clone https://github.com/volcengine/verl verl-sglang && cd verl-sglang

    # Create a virtual environment and install dependencies
    python3 -m venv .venv --upgrade-deps && source .venv/bin/activate
    python3 -m pip install .[sglang]
    python3 -m pip install .[gpu]

.. note::
    Chekc that you have the following dependencies installed:

    - **PyTorch**: 2.5.1+cu124
    - **CUDA**: 12.4
    - **SGLang**: 0.4.4.post4
    - **torch-memory-saver**: 0.0.5


2. use docker image

We also provide a pre-built imgae ``ocss884/verl-sglang:ngc-th2.5.1-cu126-sglang0.4.4.post4`` for SGLang backend.

.. note::
    As we are fast moving integrating SGLang into verl, sometimes we may use a specific commit from SGLang main branch for installation but not stable release from Pypi. If you encounter any issues, feel free to contact @ocss884.


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
        datasets \
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
