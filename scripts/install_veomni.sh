#!/bin/bash



export MAX_JOBS=32
eval "$(conda shell.bash hook)"
conda create -n verlVeomni python=3.11 -y
conda activate verlVeomni

python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# FlashAttention will be installed automatically by VeOmni[gpu]
# The correct version (2.8.3+cu12torch2.8cxx11abiTRUE) is specified in VeOmni's pyproject.toml

# pip install --no-cache-dir flashinfer-python==0.3.1



echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


echo "6. Install veomni"

git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
uv pip install -e .[gpu]



echo "Successfully installed all packages"
