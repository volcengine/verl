# SpecRL: Accelerating RL Rollout with Speculative Decoding

## Introduction

In many scenarios, the RL training datasets are recycled across multiple epochs. 
Between adjacent epochs, responses to the same prompts often exhibit high similarity, particularly in structured tasks such as mathematics and code generation. 
SpecRL exploits this observation by leveraging speculative decoding to accelerate RL rollout.
It uses token segments from historical responses as draft sequences, achieving up to **2.1× speedup**.

SpecRL is the first effort to integrate speculative decoding into RL. 
As a model-free drafting approach, it offers distinct advantages over methods relying on smaller models (e.g., small LLMs or Eagle models):

1. **Low Drafting Cost**: No GPU inference is required for drafting, making it effective even with large rollout batch sizes.
2. **Training Stability**: No need to train draft models during RL, ensuring consistent performance and ease of deployment.
3. **High Flexibility**: Compatible with synchronous RL, multi-turn RL, and asynchronous RL.

SpecRL operates in conjunction with the **Suffix-Tree-based Distributed Draft Server**, which efficiently caches historical responses, distributes them to workers, and indexes them using suffix trees for fast retrieval.

## Evaluation Results

Our evaluations on Qwen2.5 and Qwen3 models demonstrate up to **2.1× speedup** in rollout and validation phases.

**Experiment results.** Qwen3-14B-Base trained with DAPO, temperature = 1, max response length = 8K, FSDP backend, 32 H100 GPU, batch size = 256, rollout.n = 16.

![SpecRL Performance on Qwen3-14B-Base (DAPO)](https://raw.githubusercontent.com/He-Jingkai/he-jingkai.github.io/cc25105fc7e30da6b01bb40bce14e713b9a64945/assets/img/specrl-results-qwen3-14B-dapo.png)

![specrl-results-qwen3-14B-dapo-score](https://raw.githubusercontent.com/He-Jingkai/he-jingkai.github.io/refs/heads/main/assets/img/dapo-1009-score.png)

## Installation

This recipe is based on verl commit `ccd7d93`. Please contact the authors for any adaptability issues.

```sh
# Install the Distributed Draft Server and its C++ dependencies
sudo apt install -y libprotobuf-dev protobuf-compiler libprotoc-dev \
    libgrpc-dev libgrpc++-dev protobuf-compiler-grpc \
    libxxhash-dev libboost-all-dev cmake

pip install verl@git+https://github.com/volcengine/verl.git@ccd7d934f91be98bb3732c78bd1870fa39c399ad
pip install git+https://github.com/He-Jingkai/specRL.git --no-build-isolation -v
# pip install git+ssh://git@code.byted.org/jingkai.he/specRL.git  --no-build-isolation -v
```

## Usage

Replace `verl.trainer.main_ppo` with `recipe.specRL.main_ppo` in your training scripts. Speculative decoding is enabled by default. To disable it, use `+actor_rollout_ref.rollout.enable_spec_decoding=False`.

## Contact

SpecRL is migrated from the internal environment. 
If you encounter any issues or have suggestions, please contact:
- Jingkai He: `jingkai.he@bytedance.com`
- Tianjian Li: `litianjian@bytedance.com`

## Acknowledgments

SpecRL leverages the vLLM patch implementation from Snowflake's [ArcticInference](https://github.com/snowflakedb/ArcticInference) as its code base.
