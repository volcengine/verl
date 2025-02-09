Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

verl supports various backends. Currently, the following configurations are available:

- **FSDP** and **Megatron-LM** (optional) for training.
- **vLLM** adn **TGI** for rollout generation, **SGLang** support coming soon.

Training backends
------------------

We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in :doc:`FSDP Workers<../workers/fsdp_workers>`.

For users who pursue better scalability, we recommend using **Megatron-LM** backend. Currently, we support Megatron-LM v0.4 [1]_. The guide for using Megatron-LM backend can be found in :doc:`Megatron-LM Workers<../workers/megatron_workers>`.


Install from docker image
-------------------------

We provide pre-built Docker images for quick setup.

Image and tag: ``verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3``. See files under ``docker/`` for NGC-based image or if you want to build your own.

1. Launch the desired Docker image:

.. code:: bash

    docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v <image:tag>


2.	Inside the container, install verl:

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
    export PYTHONPATH=$PYTHONPATH:$(pwd)/Megatron-LM

Install from custom environment
---------------------------------

To manage environment, we recommend using conda:

.. code:: bash

   conda create -n verl python==3.9
   conda activate verl

For installing the latest version of verl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   # install verl together with some lightweight dependencies in setup.py
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip3 install -e .


Megatron is optional. It's dependencies can be setup as below:

.. code:: bash

   # apex
   pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
       git+https://github.com/NVIDIA/apex

   # transformer engine
   pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

   # megatron core v0.4.0: clone and apply the patch
   # You can also get the patched Megatron code patch via
   # git clone -b core_v0.4.0_verl https://github.com/eric-haibin-lin/Megatron-LM
   cd ..
   git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   cp ../verl/patches/megatron_v4.patch .
   git apply megatron_v4.patch
   pip3 install -e .
   export PYTHONPATH=$PYTHONPATH:$(pwd)


.. [1] Megatron v0.4 is supported with verl's patches to fix issues such as virtual pipeline hang. It will be soon updated with latest the version of upstream Megatron-LM without patches.