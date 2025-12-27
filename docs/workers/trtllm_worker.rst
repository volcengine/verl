TensorRT-LLM Backend
=====================

Last updated: 12/24/2025.

**Authored By TensorRT-LLM Team**

Introduction
------------
`TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ is a high-performance inference engine for LLMs. Currently, verl fully supports using TensorRT-LLM as the inference engine during the rollout phase.

Installation
------------
The docker file `docker/Dockerfile.stable.trtllm` is a good reference for building your own docker image. In the future it will be based on TensorRT-LLM weekly release build.

The Dockerfile uses NGC's pre-built TensorRT-LLM release image as the base, which already includes TensorRT-LLM pre-installed. 
The default base image is ``nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6``, but you can specify a different version using the ``TRTLLM_BASE_IMAGE`` build argument. 
For available images, visit the `TensorRT-LLM container documentation <https://nvidia.github.io/TensorRT-LLM/installation/containers.html>`_.

If you need to install a specific TensorRT-LLM version after building the image, you can do so by adding installation commands at the end of the Dockerfile.
Refer to the `TensorRT-LLM installation guide <https://nvidia.github.io/TensorRT-LLM/installation/index.html>`_ for more either pip install or build from source.

Using TensorRT-LLM as the inference engine for GRPO
--------------------------------------------------

We provide a GRPO recipe script `examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh` for you to test the performance and accuracy curve of TensorRT-LLM as the inference engine for GRPO. You can run the script as follows:

.. code-block:: bash
    ## for fSDP training engine
    bash examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh
    ## for Megatron-Core training engine
    bash examples/grpo_trainer/run_qwen2-7b_math_megatron_trtllm.sh

Using TensorRT-LLM as the inference engine for DAPO
--------------------------------------------------

We provide a DAPO recipe script `recipe/dapo/test_dapo_7b_math_trtllm.sh`.

.. code-block:: bash
    ## for fSDP training engine
    bash recipe/dapo/test_dapo_7b_math_trtllm.sh
    ## for Megatron-Core training engine
    TRAIN_ENGINE=megatron bash recipe/dapo/test_dapo_7b_math_trtllm.sh
