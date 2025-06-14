#!/bin/bash

# Add user mode

# docker run --rm -it \
#   --device /dev/dri \
#   --device /dev/kfd \
#   -p 8265:8265 \
#   --group-add video \
#   --cap-add SYS_PTRACE \
#   --security-opt seccomp=unconfined \
#   --privileged \
#   -v $HOME/.ssh:/home/$(id -un)/.ssh \
#   -v $HOME:$HOME \
#   -v $(pwd)/docker-entrypoint.sh:/docker-entrypoint.sh \
#   --shm-size 128G \
#   --name verl_vllm_upstream \
#   -w $PWD \
#   --entrypoint /docker-entrypoint.sh \
#   rocm/sglang-staging:20250212 \
#   /bin/bash


# dockerhub Link: https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm

  # -p 8265:8265 \
# # assign port
docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8264:8264 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name verl_upstream_rocm \
  -w $PWD \
  verl_base \
  /bin/bash

  # rocm/megatron-lm-training-private:20250610_verl_training_base \
  
  # verl_training_base \


  # compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-6.4:94_ubuntu22.04_py3.10_pytorch_release-2.7_575e247 \

  # lmsysorg/sglang:v0.4.6.post1-rocm630 \
  # lmsysorg/sglang:v0.4.6.post2-rocm630 \
  # lmsysorg/sglang:v0.4.5-rocm630 \ # Hai modified
  # lmsysorg/sglang:v0.4.5.post1-rocm630 \

  # -v $HOME/.ssh:/root/.ssh \
  # -v $HOME:$HOME \

  # require very large memory
  # -v $HOME:/root \

  # -v $HOME/.ssh:/root/.ssh \
  # -v $HOME:$HOME \
  # -v $HOME/.cache/huggingface:/root/.cache/huggingface \


  # lmsysorg/sglang:v0.4.5-rocm630-srt \
  # lmsysorg/sglang:v0.4.4.post3-rocm630 \
  # lmsysorg/sglang:v0.4.4.post3-rocm630-srt \

# docker run -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) myimage


# # # use host
# docker run --rm -it \
#   --device /dev/dri \
#   --device /dev/kfd \
#   --network host \
#   --ipc host \
#   --group-add video \
#   --cap-add SYS_PTRACE \
#   --security-opt seccomp=unconfined \
#   --privileged \
#   -v $HOME/.ssh:/root/.ssh \
#   -v $HOME:$HOME \
#   --shm-size 128G \
#   --name verl_vllm_upstream \
#   -w $PWD \
#   lmsysorg/sglang:v0.4.4.post3-rocm630-srt \
#   /bin/bash


  # lmsysorg/sglang:v0.4.4.post3-rocm630-srt \
  # lmsysorg/sglang:v0.4.4.post3-rocm630 \