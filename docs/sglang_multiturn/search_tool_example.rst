=======================
Search Tool Integration
=======================
Introduction
------------
- We have added **search-tool** invocation capability to **veRL-sglang MultiTurnRL**, enabling the model to issue retrieval requests during Actor rollout and directly leverage the returned results for training.

Basic Configuration
-------------------

To enable multi-turn rollout, make sure to configure the following fields in your rollout configuration:

.. code-block:: yaml

    actor_rollout_ref: 
        rollout: 
            multi_turn: True
            name: "sglang_async"

These configuration activates the sglang_async engine for multi-turn interaction during rollout.

Custom Tool Configuration
-------------------------

In ``examples/sglang_multiturn/config/tool_config/search_tool_config.yaml``, specify ``retrieval_service_url`` and concurrency settings:

.. code-block:: yaml

    tools:
      - class_name: "verl.tools.search_tool.SearchTool"
        config: {
          "retrieval_service_url": "http://127.0.0.1:8000/retrieve",
          "num_workers": 120,
          "rate_limit": 150,
          "default_timeout": 30
        }

How to Use
----------

Environment Setup
~~~~~~~~~~~~~~~~~

**Create a new Docker container**

.. code-block:: bash

    docker run \
        -it \
        --shm-size 32g \
        --gpus all \
        -v /models/shared/.cache:/root/.cache \
        --ipc=host \
        --network=host \
        --privileged \
        --name sglang_{your-name} \
        lmsysorg/sglang:dev \
        /bin/zsh

If you need to restart after exiting the container:

.. code-block:: bash

    docker start -i sglang_{your-name}

**Update Python and use a virtual environment**

.. code-block:: bash

    apt update
    apt install -y python3.10 python3.10-venv

    # Create the virtual environment
    python3 -m venv ~/.python/veRL-multiturn-rollout

    # Activate the virtual environment
    source ~/.python/veRL-multiturn-rollout/bin/activate

    # Install uv
    python3 -m pip install uv

**Install veRL upstream**

.. code-block:: bash

    cd ~
    git clone https://github.com/volcengine/verl.git
    cd verl

    # Install verl
    python3 -m uv pip install .
    python3 -m uv pip install -r ./requirements_sglang.txt

    # Manually install flash-attn
    python3 -m uv pip install wheel
    python3 -m uv pip install packaging
    python3 -m uv pip install flash-attn --no-build-isolation --no-deps

**Set up your own local retrieval**

.. note::
    Skip this section if using your own service

* Here we choose the local dense retriever provided in the searchR1 example; see `searchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`_ for detailed documentation.

  * Requires GPU (approximately 5–7 GB GPU memory per card during operation), high accuracy, fast.
  * For a GPU-free version, refer to the `detailed documentation <https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md>`_ in searchR1.

.. important::
    It is recommended to use conda to install the environment for the retrieval service, as faiss-gpu installation often fails in venv.

.. note::
    In this configuration, the above venv environment is used for training; the retriever uses the conda environment.

.. code-block:: bash

    # Download the Miniconda installer script
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

    # Install to $HOME/miniconda3 in batch mode
    bash ~/miniconda.sh -b -p $HOME/miniconda3

    # Activate conda (only in the current shell)
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

    # (Optional) Add conda to your default shell startup script
    conda init

    # Reload shell configuration
    source ~/.bashrc

    # Create and activate the retriever environment
    conda create -n retriever python=3.10 -y
    conda activate retriever

    # Install PyTorch with GPU support
    conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

    # Install other Python packages
    pip install transformers datasets pyserini huggingface_hub

    # Install the GPU version of faiss
    conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

    # Install the API service framework
    pip install uvicorn fastapi

**Download indexing and corpus**

.. note::
    The download is about 60–70 GB (approximately 132 GB when uncompressed)

.. code-block:: bash

    conda activate retriever

    save_path=/the/path/to/save
    python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path
    cat $save_path/part_* > $save_path/e5_Flat.index
    gzip -d $save_path/wiki-18.jsonl.gz

**Start the local flat e5 retrieval server**

.. note::
    * The first startup will download the model and load the index. Normal startup time is 1–2 minutes.
    * After startup, each GPU uses about 5–7 GB of memory (RL training can be performed on the same node).

.. code-block:: bash

    conda activate retriever

    index_file=$save_path/e5_Flat.index
    corpus_file=$save_path/wiki-18.jsonl
    retriever_name=e5
    retriever_path=intfloat/e5-base-v2

    python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
      --index_path $index_file \
      --corpus_path $corpus_file \
      --topk 3 \
      --retriever_name $retriever_name \
      --retriever_model $retriever_path \
      --faiss_gpu

Testing on 8 × H20
------------------

**Set WANDB_API_KEY**

.. note::
    If you do not know how to get an API key, refer to `this guide <https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914>`_.

.. code-block:: bash

    export WANDB_API_KEY={YOUR_WANDB_API_KEY}

    # Define a timestamp function
    function now() {
        date '+%Y-%m-%d-%H-%M'
    }

**Preprocess the dataset**

.. note::
    The following data processing and training commands are executed in the veRL-multiturn-rollout venv environment

.. code-block:: bash

    # To define your own prompt, modify examples/data_preprocess/prompt.yaml
    # Default storage directory is ~/data/searchR1_processed_direct
    python3 examples/data_preprocess/preprocess_search_r1_dataset.py --config examples/data_preprocess/prompt.yaml

**Run tests**

.. code-block:: bash

    # Ensure now() is defined
    # Create log directory
    mkdir -p logs

    # Set GPUs and run, using an appropriate log path
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
      trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) \
      > logs/searchR1-like$(now).log 2>&1 &