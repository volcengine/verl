Welcome to verl's documentation!
================================================

.. _hf_arxiv: https://arxiv.org/pdf/2409.19256

verl is a flexible, efficient and production-ready RL training framework designed for large language models (LLMs) post-training. It is an open source implementation of the `HybridFlow <hf_arxiv>`_ paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping and parallelism**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 5
   :caption: Quickstart

   start/install
   start/quickstart

.. toctree::
   :maxdepth: 4
   :caption: Programming guide

   hybrid_flow

.. toctree::
   :maxdepth: 5
   :caption: Data Preparation

   preparation/prepare_data
   preparation/reward_function

.. toctree::
   :maxdepth: 5
   :caption: Configurations

   examples/config

.. toctree::
   :maxdepth: 2
   :caption: PPO Example

   examples/ppo_code_architecture
   examples/gsm8k_example

.. toctree:: 
   :maxdepth: 1
   :caption: PPO Trainer and Workers

   workers/ray_trainer
   workers/fsdp_workers
   workers/megatron_workers

.. toctree::
   :maxdepth: 1
   :caption: Performance Tuning Guide
   
   perf/perf_tuning

.. toctree::
   :maxdepth: 1
   :caption: Experimental Results

   experiment/ppo

.. toctree::
   :maxdepth: 1
   :caption: Advance Usage and Extension

   advance/placement
   advance/dpo_extension
   advance/fsdp_extension
   advance/megatron_extension

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq/faq

Contribution
-------------

verl is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/volcengine/verl>`_, `Slack <https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA>`_ and `Wechat <https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG>`_ for discussions.

Code formatting
^^^^^^^^^^^^^^^^^^^^^^^^
We use yapf (Google style) to enforce strict code formatting when reviewing MRs. Run yapf at the top level of verl repo:

.. code-block:: bash

   pip3 install yapf
   yapf -ir -vv --style ./.style.yapf verl examples tests
