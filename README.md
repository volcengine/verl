<h1 style="text-align: center;">veRL: Volcano Engine Reinforcement Learning for LLM</h1>

veRL is a flexible, efficient and production-ready RL training framework designed for large language models (LLMs). 

veRL is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

veRL is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


veRL is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, veRL achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

<p align="center">
| <a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2409.19256v2"><b>Paper</b></a> | <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><b>Slack</b></a> | <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><b>Wechat</b></a> | 

<!-- <a href=""><b>Slides</b></a> | -->
</p>

## News

- [2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024. [Slides](https://github.com/eric-haibin-lin/verl-data/tree/neurips) and [video](https://neurips.cc/Expo/Conferences/2024/workshop/100677) available.
- [2024/10] veRL is presented at Ray Summit. [Youtube video](https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37) available.
- [2024/08] HybridFlow (verl) is accepted to EuroSys 2025.

## Key Features

- **FSDP** and **Megatron-LM** for training.
- **vLLM** and **TGI** for rollout generation, **SGLang** support coming soon.
- huggingface models support
- Supervised fine-tuning
- Reinforcement learning from human feedback with [PPO](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer) and [GRPO](https://github.com/volcengine/verl/tree/main/examples/grpo_trainer)
  - Support model-based reward and function-based reward (verifiable reward)
- flash-attention integration, sequence packing, and long context support via DeepSpeed Ulysses
- scales up to 70B models and hundreds of GPUs
- experiment tracking with wandb and mlflow

## Upcoming Features
- Reward model training
- DPO training

## Getting Started

Checkout this [Jupyter Notebook](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer/verl_getting_started.ipynb) to get started with PPO training with a single 24GB L4 GPU (**FREE** GPU quota provided by [Lighting Studio](https://lightning.ai/hlin-verl/studios/verl-getting-started))!

**Quickstart:**
- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)

**Running an PPO example step-by-step:**
- Data and Reward Preparation
  - [Prepare Data (Parquet) for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**
- [PPO](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**
- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API Design Tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add models with the FSDP backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add models with the Megatron-LM backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)


## Citation and acknowledgement

If you find the project helpful, please cite:
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```tex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and supported by Anyscale, Bytedance, LMSys.org, Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, and University of Hong Kong.

## Publications Using veRL
- [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://arxiv.org/abs/2410.09302)
- [Flaming-hot Initiation with Regular Execution Sampling for Large Language Models](https://arxiv.org/abs/2410.21236)
- [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME/)

We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in MLSys/LLM reasoning/multimodal alignment.
