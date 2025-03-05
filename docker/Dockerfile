# Build instructions
# DOCKER_BUILDKIT=1 docker build . -f Dockerfile --progress=plain -t gitlab-master.nvidia.com/smajumdar/nemo_containers/verl:0.2.0

# docker push gitlab-master.nvidia.com/smajumdar/nemo_containers/verl:0.2.0

# docker run -it --rm --user 1001 --gpus all -v /PATH:/PATH gitlab-master.nvidia.com/smajumdar/nemo_containers/verl:0.2.0 /bin/bash

# FROM verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
FROM verlai/verl:ngc-th2.4.0-cu124-vllm0.6.3-te1.7-v0.0.4

# Set working directory
WORKDIR /opt

# Install verl
ENV VERL_COMMIT=merges
RUN git clone https://github.com/titu1994/verl.git && \
    cd verl && \
    git checkout $VERL_COMMIT && \
    pip3 install -e .

# Install vllm flash-attn
RUN pip3 uninstall -y vllm flash-attn
RUN pip3 install vllm>=0.7.0
RUN MAX_JOBS=4 NINJA_FLAGS="-j4" pip3 install --no-cache-dir --no-build-isolation flash-attn
RUN pip3 install -U pynvml

WORKDIR /workspace

RUN pip install math-verify[antlr4_9_3]

# COPY entrypoint.sh /app/entrypoint.sh
# RUN apt-get update && \
#     apt-get install -y gosu sudo && \
#     rm -rf /var/lib/apt/lists/* && \
#     gosu nobody true
# RUN sudo su -

# Patches
# COPY *.patch /workspace/
# RUN cd /workspace/ && patch /usr/local/lib/python3.10/dist-packages/vllm/distributed/parallel_state.py parallel_state.patch
# RUN cd /workspace/ && patch /usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py uniproc_executor.patch
# RUN cd /workspace/ && patch /usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/weight_utils.py weight_utils.patch
# RUN chmod +x /app/entrypoint.sh
# RUN chmod 777 -R /root
# RUN pip install ray[default] pylatexenc wandb
# ENTRYPOINT ["/app/entrypoint.sh"]
# CMD ["/usr/bin/bash"]