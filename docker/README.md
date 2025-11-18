# Dockerfiles of verl

We provide pre-built Docker images for quick setup. And from this version, we utilize a new image release hierarchy for productivity and stability.

Start from v0.6.0, we use vllm and sglang release image as our base image.

## Base Image

- vLLM: https://hub.docker.com/r/vllm/vllm-openai
- SGLang: https://hub.docker.com/r/lmsysorg/sglang

## Application Image

Upon base image, the following packages are added:
- flash_attn
- Megatron-LM
- Apex
- TransformerEngine
- DeepEP

Latest docker file:
- [Dockerfile.stable.vllm](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.stable.vllm)
- [Dockerfile.stable.sglang](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.stable.sglang)

All pre-built images are available in dockerhub: https://hub.docker.com/r/verlai/verl. For example, `verlai/verl:sgl055.latest`, `verlai/verl:vllm011.latest`.

You can find the latest images used for development and ci in our github workflows:
- [.github/workflows/vllm.yml](https://github.com/volcengine/verl/blob/main/.github/workflows/vllm.yml)
- [.github/workflows/sgl.yml](https://github.com/volcengine/verl/blob/main/.github/workflows/sgl.yml)


## Installation from Docker

After pulling the desired Docker image and installing desired inference and training frameworks, you can run it with the following steps:

1. Launch the desired Docker image and attach into it:

```sh
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash
```

2. If you use the images provided, you only need to install verl itself without dependencies:

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
