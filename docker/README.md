# Dockerfiles of verl

We provide pre-built Docker images for quick setup. And from this version, we utilize a new image release hierarchy for productivity and stability.

The image types are divided into three large categories:

- **Base Image**: Without inference and training frameworks, only basic dependencies are installed. Can directly install vllm or SGLang on top of it, without need of reinstall torch or CUDA.
- **Application Image**: Stable version with inference and training frameworks installed.
- **Preview Image**: Unstable version with the latest frameworks and features.

The first two types of images are hosted on dockerhub [verlai/verl](https://hub.docker.com/r/verlai/verl) repository, while the preview images are hosted on community repository.

> The image versions are mapped with verl releases, for example, image with tag ``verl0.4`` is built for verl release ``v0.4.x``.

## Base Image

The stable base image is ``verlai/verl:base-verl0.4-cu124-cudnn9.8-torch2.6-fa2.7.4``. The installed package versions can be found from tags, and the Dockerfile can be found in ``verl[version]-[packages]/Dockerfile.base``.

The base images for preview are ``verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.8.0`` and ``verlai/verl:base-verl0.5-preview-cu128-cudnn9.8-torch2.7.1-fa2.8.0`` with different CUDA versions.

The update of base image is not frequent, and the app image can be built on top of it without reinstalling base packages.

## Application Image

From this version, we divide images built for vLLM and SGLang as the divergence of dependent packages like FlashInfer.

There are four types of application images available:

- **vLLM with FSDP and Megatron**: ``verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2``, with Deep-EP support: ``verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2-deepep``.
- **SGLang with FSDP and Megatron**: ``verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2`` (need vLLM support, but can have some package conflicts), with Deep-EP support: ``verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2-deepep``.
- **Preview version of SGLang with FSDP and Megatron, CUDA 12.6**: ``verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2``
- **Preview version of SGLang with FSDP and Megatron, CUDA 12.8**: ``verlai/verl:app-preview-verl0.5-sglang0.4.8-mcore0.12.2-te2.2``

For Megatron 0.13.0, we offer preview images, to use latest codes, just replace ``mcore0.12.2`` with ``mcore0.13.0-preview`` in the above image tag.

The latest vLLM support is coming soon.

Docker images with Megatron backends are runnable with large language model like ``Qwen/Qwen3-235B-A22B``, ``deepseek-ai/DeepSeek-V3-0324`` post-training. Refer to the :doc:`Large Language Model Post-Training documentation<../perf/dpsk>` for more details.

Application images can be updated frequently, and the Dockerfile can be found in ``docker/verl[version]-[packages]/Dockerfile.app.[frameworks]``. Based on the base image, it is easy to build your own application image with the desired inference and training frameworks.

## Community Image

For vLLM with FSDP, please refer to [hiyouga/verl](https://hub.docker.com/r/hiyouga/verl) repository and the latest version is ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``.

For SGLang with FSDP, please refer to [ocss884/verl-sglang](https://hub.docker.com/r/ocss884/verl-sglang) repository and the latest version is ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5`` which is provided by SGLang RL Group.

See files under ``docker/`` for NGC-based image or if you want to build your own.

Note that For aws instances with EFA net interface (Sagemaker AI Pod), you need to install EFA driver as shown in ``docker/Dockerfile.extenstion.awsefa``

## Installation from Docker

After pulling the desired Docker image and installing desired inference and training frameworks, you can run it with the following steps:

1. Launch the desired Docker image and attach into it:

```sh
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash
```

2.	If you use the images provided, you only need to install verl itself without dependencies:

```sh
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
```

[Optional] If you hope to switch between different frameworks, you can install verl with the following command:

```sh
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```
