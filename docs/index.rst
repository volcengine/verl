Welcome to veRL/HybridFlow's documentation!
================================================

veRL (HybridFlow) is a flexible, efficient and industrial-level RL(HF) training framework designed for large language models (LLMs) Post-Training. 

veRL is flexible and easy to use with:

- **Easy to support diverse RL(HF) algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular API design**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular Hugging Face models


veRL is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, veRL achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 5
   :caption: Quickstart
   :titlesonly:
   :numbered:

   start/install
   start/quickstart

.. toctree::
   :maxdepth: 5
   :caption: Data Preparation
   :titlesonly:
   :numbered:

   preparation/prepare_data
   preparation/reward_function

.. toctree::
   :maxdepth: 2
   :caption: PPO Example
   :titlesonly:
   :numbered:

   examples/ppo_code_architecture
   examples/config
   examples/gsm8k_example

.. toctree:: 
   :maxdepth: 1
   :caption: PPO Trainer and Workers

   workers/ray_trainer
   workers/fsdp_workers
   workers/megatron_workers

.. toctree::
   :maxdepth: 1
   :caption: Advance Usage and Extension

   advance/placement
   advance/dpo_extension
   advance/fsdp_extension
   advance/megatron_extension


Contribution
-------------

veRL is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/volcengine/verl>`_ .

.. and check out our
.. :doc:`contribution guidelines <contribute>`.
