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

For vLLM with Megatron or FSDP, please use the stable version of image ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3``.

For latest vLLM with FSDP, please refer to ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``.

For SGLang with FSDP, please use ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post4`` which is provided by SGLang RL Group.

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
    
    The Docker image ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3`` is built with the following configurations:

    - **PyTorch**: 2.6.0+cu124
    - **CUDA**: 12.4
    - **cuDNN**: 9.8.0
    - **nvidia-cudnn-cu12**: 9.8.0.87, **important for the usage of Megatron FusedAttention with MLA Support**
    - **Flash Attenttion**: 2.7.4.post1
    - **Flash Infer**: 0.2.2.post1
    - **vLLM**: 0.8.5
    - **SGLang**: 0.4.6.post4
    - **Megatron-LM**: core_v0.12.0
    - **TransformerEngine**: 2.3
    - **Ray**: 2.44.1


Install from custom environment
---------------------------------------------

We recommend to use docker images for convinience. However, if your environment is not compatible with the docker image, you can also install verl in a python environment.


Pre-requisites
::::::::::::::

For training and inference engines to utilize better and faster hardware support, CUDA/cuDNN and other dependencies are required,
and some of the dependencies are easy to be overrided when installing other packages,
so we put them in the :ref:`Post-installation` step.

We need to install the following pre-requisites:

- **CUDA**: Version >= 12.4
- **cuDNN**: Version >= 9.8.0
- **Apex**

CUDA above 12.4 is recommended to use as the docker image,
please refer to `NVIDIA's official website <https://developer.nvidia.com/cuda-toolkit-archive>`_ for other version of CUDA.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommanded
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda-toolkit-12-4
    update-alternatives --set cuda /usr/local/cuda-12.4


cuDNN can be installed via the following command,
please refer to `NVIDIA's official website <https://developer.nvidia.com/rdp/cudnn-archive>`_ for other version of cuDNN.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommanded
    wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn-cuda-12

NVIDIA Apex is required for Megatron-LM and FSDP training.
You can install it via the following command, but notice that this steps can take a very long time.
It is recommanded to set the ``MAX_JOBS`` environment variable to accelerate the installation process,
but do not set it too large, otherwise the memory will be overloaded and your machines may hang.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommanded
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


Install dependencies
::::::::::::::::::::

.. note::

    We recommend to use a fresh new conda environment to install verl and its dependencies.

    **Notice that the inference frameworks often strictly limit your pytorch version and will directly override your installed pytorch if not paying enough attention.**

    As a countermeasure, it is recommended to install inference frameworks first with the pytorch they needed. For vLLM, if you hope to use your existing pytorch,
    please follow their official instructions
    `Use an existing PyTorch installation <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source>`_ .


1. First of all, to manage environment, we recommend using conda:

.. code:: bash

   conda create -n verl python==3.10
   conda activate verl


2. Then, execute the ``install.sh`` script that we provided in verl:

.. code:: bash

    # Make sure you have activated verl conda env
    # If you need to run with megatron
    bash scripts/install_vllm_sglang_mcore.sh
    # Or if you simply need to run with FSDP
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh


If you encounter errors in this step, please check the script and manually follow the steps in the script.


Install verl
::::::::::::

For installing the latest version of verl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install --no-deps -e .


Post-installation
:::::::::::::::::

Please make sure that the installed packages are not overridden during the installation of other packages.

The packages worth checking are:

- **torch** and torch series
- **vLLM**
- **SGLang**
- **pyarrow**
- **tensordict**
- **nvidia-cudnn-cu12**: For Magetron backend

If you encounter issues about package versions during running verl, please update the outdated ones.


Install with AMD GPUs - ROCM kernel support
------------------------------------------------------------------

When you run on AMD GPUs (MI300) with ROCM platform, you cannot use the previous quickstart to run verl. You should follow the following steps to build a docker and run it. 
If you encounter any issues in using AMD GPUs running verl, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.

Find the docker for AMD ROCm: `docker/Dockerfile.rocm <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.rocm>`_
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

Build the image
::::::::::::::::::::::::

.. code-block:: bash

    docker build -t verl-rocm .

Launch the container
::::::::::::::::::::::::::::

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
