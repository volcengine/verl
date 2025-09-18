# Verl x CollabLLM

This repository implements [CollabLLM](https://arxiv.org/pdf/2502.00640) (ICML 2025) using the Verl framework. For the original implementation, see the [CollabLLM repository](https://github.com/Wuyxin/collabllm).

## Quick Start

### 1. Prepare Your Dataset

First, process your dataset using the provided script:

```bash
python process_dataset.py ... --dataset_type <sft or rl>
```

**Requirements:**
- Input: A Hugging Face multiturn dataset
- Example format: See [collabllm-multiturn-math-hard](https://huggingface.co/datasets/collabllm/collabllm-multiturn-math-hard)
- To generate your own dataset: Use [build_dataset.py](https://github.com/Wuyxin/collabllm/blob/main/scripts/engine/build_dataset.py) from the original CollabLLM repository

*Note: Check `process_dataset.py` for example commands and usage.*

### 2. Train Your Model

**For Supervised Fine-Tuning (SFT):**
```bash
bash train_sft_collabllm.sh
```

**For Reinforcement Learning (RL):**
```bash
bash train_rl_collabllm.sh
```

## What is CollabLLM?

CollabLLM is a method for training language models to collaborate effectively in multi-turn conversations. This implementation adapts the original imlpementation to work with the Verl training framework.

# Citation
If you find CollabLLM useful in your research, please cite the following:

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and 
            Gavin Li and Yao Dou and Weixin Cai and James Zou and 
            Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```
