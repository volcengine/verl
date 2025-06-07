# Char Count
## Introduction
Char count is a simple NLP task. We create it for beginners to grasp the idea of RLVR. The task can be trained using a tiny model (e.g., https://huggingface.co/HuggingFaceTB/SmolLM2-135M) on a consumer GPU with only 8GB.

## Problem formulation
The prompt is: "How many {char} are there in {word}?". In order for LLM to better answer this question, we create SFT dataset with intermediate steps. For example,

```text
Question: How many n are there in n-i-n-e?
Answer:
n = n
i != n
n = n
e != n
\boxed{2}
```

Note that
- We add a dash between each individual char to make the task easier because each individual char will be tokenized to the same token by most tokenizer.
- In the SFT dataset, we create a CoT by listing all the individual chars and whether it equals to the target. In the end, it outputs the final answer inside the box.
- The task can be verified.
- The word is not always meaningful. Each char is sampled uniformly from a to z. We make the total length and the answer uniformly distributed within a range.

## Scripts
To create the dataset, run
```bash
python3 create_dataset.py
```
We create a train set and a val set. Both of them are used of SFT and RL. You can specify the total number of data, min/max length and data path.

To run the SFT
```bash
bash train_sft.sh
```
We train SFT for 3 epochs. After 3 epochs, the validation score is around 0.12.

To run GRPO
```bash
bash train_grpo.sh
```
We train GRPO for 2 epochs. After 2 epochs, the validation score is around 0.36.
