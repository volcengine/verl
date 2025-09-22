# CollabLLM

This repository implements [CollabLLM](https://arxiv.org/pdf/2502.00640) (ICML 2025) using the verl framework. For the original implementation, see the [CollabLLM repository](https://github.com/Wuyxin/collabllm).


CollabLLM is a method for training language models to collaborate effectively in multi-turn conversations. This implementation adapts the original imlpementation to work with the Verl training framework.

## Quick start

### 0. Environment
Make sure the required packages for `verl` are installed. Additionally, install `litellm` and export the required API keys. The API model will be used for user simulators and, optionally, LLM Judges (see the Configuration section below).

### 1. Prepare Your Dataset

First, process your dataset using the provided script:

```bash
python process_dataset.py --dataset <> ... --dataset_type <sft or rl>
```


**Requirements:**
- Input: A Hugging Face multiturn dataset. Existing datasets: `collabllm/collabllm-multiturn-$DATASET`, with `DATASET` in one of [`math-hard(-large)`, `medium(-large)`, `bigcodebench(-large)`] (*-large are the datasets used in the CollabLLM paper)
- Example format: See [collabllm-multiturn-math-hard](https://huggingface.co/datasets/collabllm/collabllm-multiturn-math-hard)
- To generate your own dataset: Use [build_dataset.py](https://github.com/Wuyxin/collabllm/blob/main/scripts/engine/build_dataset.py) from the original CollabLLM repository

*Note: Check `process_dataset.py` for example commands and usage.*

### 2. Train Your Model

**(Optional) For Supervised Fine-Tuning (SFT):**
```bash
bash train_sft_collabllm.sh
```

**For Reinforcement Learning (RL):**

```bash
bash train_rl_collabllm.sh
```

The RL script shows an example to train CollabLLM on `math-hard-large`. 

- The config to sample future conversations are in `recipe/collabllm/config/collabllm_interaction_config.yaml`. 
- The Multiturn-aware Reward is aggregated from these three conversational-level rewards:

    ```
    +reward_model.reward_kwargs.metric_weights.accuracy=1 \
    +reward_model.reward_kwargs.metric_weights.interactivity=1 \
    +reward_model.reward_kwargs.metric_weights.token_amount=-0.0001 \
    ```

    You can remove, add, or modify the weights depending on your task. A list of implemented metrics you can already add are under `recipe/collabllm/metrics`. For example, on `medium-large`, you can replace `accuracy` with `bleu_score` via
    ```
    +reward_model.reward_kwargs.metric_weights.bleu_score=1 
    ```
    which will instead apply bleu score on the sampled future conversations. 

## Configuration 
Read [doc](https://verl.readthedocs.io/en/latest/) for detailed configurations.

## Citation
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
