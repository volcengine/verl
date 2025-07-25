# CosyVoice2 LLM Reinforcement Learning Recipe

This recipe shows how to train the **CosyVoice2** large language model with reinforcement learning algorithms such as **GRPO** using the [veRL](https://github.com/volcengine/verl) framework. Our experiments show that applying GRPO reduces the character error rate (CER) on the Seed-TTS test_zh set from 1.81% to 1.06%.

We initialize the model from a Supervised Fine-Tuned (SFT) version of Qwen2-0.5B-Instruct and then continue training with reinforcement learning. Given an input sentence, the model predicts the corresponding CosyVoice2 speech tokens. For the SFT training recipe please refer to [PR #1887](https://github.com/k2-fsa/icefall/pull/1887).

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Reward Function & ASR Server](#reward-function--asr-server)
- [Training](#training)
- [Evaluation](#evaluation)
- [Single-Utterance Inference](#single-utterance-inference)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Environment Setup

Stage `-1` of `run.sh` installs all required dependencies:

```bash
bash run.sh -1 -1      # run only stage -1
```

The script performs the following tasks:

1. Clones and installs **veRL** (without Megatron).
2. Checks out the **CosyVoice** source code to `/workspace/CosyVoice` and installs the Python packages from `requirements-cosyvoice.txt`.
3. Downloads the TTS codec model `iic/CosyVoice2-0.5B` from **ModelScope** into `/workspace/CosyVoice2-0.5B`.
4. Installs **PytritonSensevoice** together with **Pytriton**.
5. Downloads the SFT-finetuned CosyVoice2-0.5B LLM whose vocabulary was extended on Emilia-Zh data.

> [!TIP]
> The **veRL** repository evolves quickly. To reproduce our results you can checkout this [specific commit](https://github.com/yuekaizhang/verl/tree/thread).

## Data Preparation

`prepare_data.py` expects a JSON/JSONL file with at least the following schema:

```jsonc
{
  "text": "An example sentence to be synthesized."
}
```
You can download the JSONL files from the metadata directory of the [SparkAudio/voxbox](https://huggingface.co/datasets/SparkAudio/voxbox/tree/main/metadata) dataset on Hugging Face.

Stage `0` converts raw JSONL files into the parquet format expected by veRL:

```bash
bash run.sh 0 0
```
Create two JSONL files – `train.jsonl` and `test.jsonl`.  
The script will generate two parquet files:

```
data/parquet_tiny/train.parquet
data/parquet_tiny/test.parquet
```

Each sample is automatically wrapped into a chat-style prompt with the special system token `<|SPEECH_GENERATION_START|>` so that the LLM learns to output CosyVoice2 speech tokens.

> [!TIP]
> For the `prompt_template` we recommend using the same configuration as during SFT training. See the corresponding setup [here](https://github.com/yuekaizhang/icefall/blob/emilia/egs/emilia/TTS/llasa_cosyvoice2_token/train.py#L84).

## Reward Function & ASR Server

To compute rewards we run a lightweight server that:

1. Converts generated speech tokens back to a 16 kHz waveform with the **CosyVoice2** pretrained U-Net model.
2. Transcribes the waveform with **SenseVoice** ASR.
3. Calculates the pinyin-level error rate against the ground-truth text and maps it to a score in the range \[0 … 1\].

Start the server (stage `1`) in a dedicated terminal / GPU:

```bash
bash run.sh 1 1
# Triton server listens on ports 8000/8001/8002
```

The custom reward implementation lives in [`reward_tts.py`](./reward_tts.py) and calls the server to obtain the reward score.

## Training

Run stage `2` to start GRPO training:

```bash
bash run.sh 2 2
```

Key CLI arguments passed to `verl.trainer.main_ppo`:

* `algorithm.adv_estimator=grpo` – use GRPO instead of PPO.
* `data.train_files=data/parquet_aishell3/train.parquet` and `data.val_files=data/parquet_aishell3/test.parquet`
* `actor_rollout_ref.model.path=/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000` – path to the pretrained CosyVoice2 LLM.
* `custom_reward_function.path=reward_tts.py` – custom reward function described above.
* `trainer.total_epochs=1` – number of training epochs (adjust as needed).

Tune `CUDA_VISIBLE_DEVICES`, batch sizes and other hyper-parameters according to your hardware.

## Evaluation

After training finishes we gather the sharded FSDP weights and export a HuggingFace-style checkpoint (stage `3`):

```bash
bash run.sh 3 3   # merges weights into $llm_path/merged_hf_model
```

We can then evaluate the model on the CosyVoice3 zero-shot Chinese test set (stage `4`):

```bash
bash run.sh 4 4
```

This command launches distributed inference via `infer_dist.py` and computes WER with `scripts/compute_wer.sh`.

> [!TIP]
> The script also supports the Seed-TTS test set by setting `dataset=test_zh`.

## Single-Utterance Inference

For a quick demo run stage `5`:

```bash
bash run.sh 5 5
```

The script synthesizes a tongue-twister using the merged checkpoint and prints the path of the generated audio file.

## Results

| Model                                                 | Seed-TTS `test_zh` CER | Cosyvoice3 `zero_shot_zh` |Comment                                                                        |
|-|------------------------------------------------------|------------------------|--------------------------------------------------------------------------------|
| Official CosyVoice2 LLM                               | 1.45 %             |4.08%| See the [paper](https://arxiv.org/abs/2412.10117)                              |
| SFT (initialized from Qwen2-0.5B-Instruct)            | 1.81 %                 |4.83%| See [PR #1887](https://github.com/k2-fsa/icefall/pull/1887)                    |
| GRPO (this work, trained on AIShell-3)                | **1.06 %**             |4.03%|                                                                                |

## Acknowledgement

This work is inspired by the implementation in  
https://github.com/channel-io/ch-tts-llasa-rl-grpo

