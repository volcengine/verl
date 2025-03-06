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
RUN pip3 install vllm>=0.7.3
RUN MAX_JOBS=4 NINJA_FLAGS="-j4" pip3 install --no-cache-dir --no-build-isolation flash-attn
RUN pip3 install -U pynvml

WORKDIR /workspace

# Patches
# COPY *.patch /workspace/
# RUN cd /workspace/ && patch -f /usr/local/lib/python3.10/dist-packages/vllm/distributed/parallel_state.py parallel_state.patch
# RUN cd /workspace/ && patch -f /usr/local/lib/python3.10/dist-packages/vllm/executor/uniproc_executor.py uniproc_executor.patch
# RUN cd /workspace/ && patch -f /usr/local/lib/python3.10/dist-packages/vllm/model_executor/model_loader/weight_utils.py weight_utils.patch

# Fix CV2
RUN pip install opencv-fixer==0.2.5 && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

# Run additional dependencies
RUN pip install math-verify[antlr4_9_3] ray[default] pylatexenc wandb

CMD ["/usr/bin/bash"]
