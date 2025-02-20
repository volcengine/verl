<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Readily integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

<p align="center">
| <a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2409.19256v2"><b>Paper</b></a> | <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><b>Slack</b></a> | <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><b>Wechat</b></a> | <a href="https://x.com/verl_project"><b>Twitter</b></a>

<!-- <a href=""><b>Slides</b></a> | -->
</p>

## News
- [2025/3] We will present verl(HybridFlow) at [EuroSys 2025](https://2025.eurosys.org/). See you in in Rotterdam!
- [2025/2] verl v0.2.0.post1 is released! See [release note](https://github.com/volcengine/verl/releases/) for details.
- [2025/2] We presented verl in the [Bytedance/NVIDIA/Anyscale Ray Meetup](https://lu.ma/ji7atxux). See you in San Jose!
- [2025/1] [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) is released with SOTA-level performance on LLM & VLM. The RL scaling preview model is trained using verl, reaching OpenAI O1-level performance on math benchmarks (70.0 pass@1 on AIME).
- [2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024. [Slides](https://github.com/eric-haibin-lin/verl-data/tree/neurips) and [video](https://neurips.cc/Expo/Conferences/2024/workshop/100677) available.
- [2024/12] verl is presented at Ray Forward 2024. Slides available [here](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/Ray_Forward_2024_%E5%B7%AB%E9%94%A1%E6%96%8C.pdf).
- [2024/10] verl is presented at Ray Summit. [Youtube video](https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37) available.
- [2024/08] HybridFlow (verl) is accepted to EuroSys 2025.

## Key Features

- **FSDP** and **Megatron-LM** for training.
- **vLLM** and **TGI** for rollout generation, **SGLang** support coming soon.
- huggingface models support
- Supervised fine-tuning
- Reinforcement learning from human feedback with [PPO](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer), [GRPO](https://github.com/volcengine/verl/tree/main/examples/grpo_trainer), [ReMax](https://github.com/volcengine/verl/tree/main/examples/remax_trainer), Reinforce++, etc
  - Support model-based reward and function-based reward (verifiable reward)
- flash-attention, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [long context](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh)
- scales up to 70B models and hundreds of GPUs
- experiment tracking with wandb, swanlab and mlflow

## Upcoming Features
- Reward model training
- DPO training
- DeepSeek integration with Megatron backend
- SGLang integration
- vision language model RL

## Getting Started

**Quickstart:**
- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

**Running a PPO example step-by-step:**
- Data and Reward Preparation
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**
- [PPO, GRPO, ReMax](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**
- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)

**Blogs from the community**
- [HybridFlow veRL 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升20倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

Checkout this [Jupyter Notebook](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer/verl_getting_started.ipynb) to get started with PPO training with a single 24GB L4 GPU (**FREE** GPU quota provided by [Lighting Studio](https://lightning.ai/hlin-verl/studios/verl-getting-started))!

## Performance Tuning Guide
The performance is essential for on-policy RL algorithm. We write a detailed performance tuning guide to allow people tune the performance. See [here](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for more details.

## vLLM v0.7 testing version
We have released a testing version of veRL that supports vLLM>=0.7.0. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.7.md) for installation guide and more information.

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

## Awesome work using verl
- [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://arxiv.org/abs/2410.09302)
- [Flaming-hot Initiation with Regular Execution Sampling for Large Language Models](https://arxiv.org/abs/2410.21236)
- [Process Reinforcement Through Implicit Rewards](https://github.com/PRIME-RL/PRIME/)
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of DeepSeek R1 Zero recipe for reasoning tasks
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning agent training framework
- [Logic R1](https://github.com/Unakar/Logic-RL): a reproduced DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset.
- [deepscaler](https://github.com/agentica-project/deepscaler): iterative context scaling with GRPO
- [critic-rl](https://github.com/HKUNLP/critic-rl): Teaching Language Models to Critique via Reinforcement Learning

## Contribution Guide
Contributions from the community are welcome!

### Code formatting
We use yapf (Google style) to enforce strict code formatting when reviewing PRs. To reformat you code locally, make sure you installed **latest** `yapf`
```bash
pip3 install yapf --upgrade
```
Then, make sure you are at top level of verl repo and run
```bash
bash scripts/format.sh
```
We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in MLSys/LLM reasoning/multimodal alignment.
