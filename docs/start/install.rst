Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

veRL supports various backends. Currently, the following configurations are available:

- **FSDP** and **Megatron-LM** (optional) for training.
- **vLLM** for rollout generation, **SGLang** support coming soon.

Install from docker image
-------------------------

We provide pre-built Docker images for quick setup.

Image and tag: ``verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3``. See files under ``docker/`` if you want to build your own image.

1. Launch the desired Docker image:

.. code:: bash

    docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v <image:tag>


2.	Inside the container, install veRL:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .
    # or install from pypi via `pip3 install verl`


3. Setup Megatron (optional)

If you want to enable training with Megatron, Megatron code must be added to PYTHONPATH:

.. code:: bash

    cd ..
    git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
    cp verl/patches/megatron_v4.patch Megatron-LM/
    cd Megatron-LM && git apply megatron_v4.patch
    pip3 install -e .
    export PYTHONPATH=$PYTHONPATH:$(pwd)


You can also get the Megatron code after verl's patch via

.. code:: bash

    git clone -b core_v0.4.0_verl https://github.com/eric-haibin-lin/Megatron-LM

Install from custom environment
---------------------------------

To manage environment, we recommend using conda:

.. code:: bash

   conda create -n verl python==3.9
   conda activate verl

For installing the latest version of veRL, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   # install verl together with some lightweight dependencies in setup.py
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip3 install -e .

You can also install veRL using ``pip3 install``

.. code:: bash

   # directly install from pypi
   pip3 install verl

Dependencies
------------

veRL requires Python >= 3.9 and CUDA >= 12.1.

veRL support various backend, we currently release FSDP and Megatron-LM
for actor training and vLLM for rollout generation.

The following dependencies are required for all backends, PyTorch FSDP and Megatron-LM.

The pros, cons and extension guide for using PyTorch FSDP backend can be
found in :doc:`FSDP Workers<../workers/fsdp_workers>`.

.. code:: bash

   # install torch [or you can skip this step and let vllm to install the correct version for you]
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

   # install vllm
   pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
   pip3 install ray

   # flash attention 2
   pip3 install flash-attn --no-build-isolation

For users who pursue better scalability, we recommend using Megatron-LM
backend. Please install the above dependencies first.

Currently, we support Megatron-LM\@core_v0.4.0 and we fix some internal
issues of Megatron-LM. Here's the additional installation guide.

The pros, cons and extension guide for using Megatron-LM backend can be
found in :doc:`Megatron-LM Workers<../workers/megatron_workers>`.

.. code:: bash

   # FOR Megatron-LM Backend
   # apex
   pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
            --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
            git+https://github.com/NVIDIA/apex

   # transformer engine
   pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

   # megatron core v0.4.0
   cd ..
   git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   cp ../verl/patches/megatron_v4.patch .
   git apply megatron_v4.patch
   pip3 install -e .
   export PYTHONPATH=$PYTHONPATH:$(pwd)
