conda create -n verl python=3.10
conda activate verl

pip3 install -e .
pip3 install vllm==0.8.3
pip3 install flash-attn --no-build-isolation