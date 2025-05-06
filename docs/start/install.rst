Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

verl supports various backends. Currently, the following configurations are available:

- **FSDP** and **Megatron-LM** (optional) for training.
- **SGLang**, **vLLM** and **TGI** for rollout generation.

Choices of Backend Engines
----------------------------

1. Training:

We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in :doc:`FSDP Workers<../workers/fsdp_workers>`.

For users who pursue better scalability, we recommend using **Megatron-LM** backend. Currently, we support `Megatron-LM v0.11 <https://github.com/NVIDIA/Megatron-LM/tree/v0.11.0>`_. The guide for using Megatron-LM backend can be found in :doc:`Megatron-LM Workers<../workers/megatron_workers>`.

.. note:: 

    verl directly supports megatron's `GPTModel` API on the main branch with mcore v0.11. For mcore v0.4 try `0.3.x branch <https://github.com/volcengine/verl/tree/v0.3.x>`_ instead.

2. Inference:

For inference, vllm 0.6.3 and 0.8.2 have been tested for stability. Avoid using vllm 0.7x due to reported issues with its functionality.

For SGLang, refer to the :doc:`SGLang Backend<../workers/sglang_worker>` for detailed installation and usage instructions. **SGLang offers better throughput and is under extensive development.** We encourage users to report any issues or provide feedback via the `SGLang Issue Tracker <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/106>`_.

For huggingface TGI integration, it is usually used for debugging and single GPU exploration.

Install from docker image
-------------------------

We provide pre-built Docker images for quick setup.

For vLLM with Megatron or FSDP, please use ``whatcanyousee/verl:ngc-cu124-vllm0.8.3-sglang0.4.5-mcore0.12.0-te2.2``.

For latest vLLM with FSDP, please refer to ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``.

For SGLang with FSDP, please use ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post1`` which is provided by SGLang RL Group.

See files under ``docker/`` for NGC-based image or if you want to build your own.

1. Launch the desired Docker image and attach into it:

.. code:: bash

    docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag>
    docker start verl
    docker exec -it verl bash


2.	Inside the container, install latest verl:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl
    # pick your choice of inference engine: vllm or sglang
    # pip3 install -e .[vllm]
    # pip3 install -e .[sglang]
    # or install from pypi instead of git via:
    # pip3 install verl[vllm]
    # pip3 install verl[sglang]

.. note::
    
    The Docker image ``whatcanyousee/verl:ngc-cu124-vllm0.8.3-sglang0.4.5-mcore0.12.0-te2.2`` is built with the following configurations:

    - **PyTorch**: 2.6.0+cu124
    - **CUDA**: 12.4
    - **cuDNN**: 9.8.0
    - **nvidia-cudnn-cu12**: 9.8.0.87, **important for the usage of Megatron FusedAttention with MLA Support**
    - **Flash Attenttion**: 2.7.4.post1
    - **Flash Infer**: 0.2.2.post1
    - **vLLM**: 0.8.3
    - **SGLang**: 0.4.5.post3
    - **Megatron-LM**: v0.11.0
    - **TransformerEngine**: 2.2.0
    - **Ray**: 2.44.1


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
   pip3 install torch torchvision
   pip3 install flash-attn --no-build-isolation
   git clone https://github.com/volcengine/verl.git
   cd verl
   # pick your choice of inference engine: vllm or sglang
   # pip3 install -e .[vllm]
   # pip3 install -e .[sglang]


Megatron is optional. It's dependencies can be setup as below:

.. code:: bash

   # apex
   pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
       git+https://github.com/NVIDIA/apex

   # transformer engine
   pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
   # megatron core
   pip3 install --no-deps megatron-core==0.12.0


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

(Currently Support): Training Engine: FSDP; Inference Engine: vLLM and SGLang - We will support Megatron in the future.
