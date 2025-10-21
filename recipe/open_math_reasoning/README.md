# Open math reasoning
## Introduction
In this recipe, we perform SFT on the [open math reasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset using the new SFT trainer with backend agostic model engine. Note that our goal is not to replicate the [AIMO-2 Winning Solution](https://arxiv.org/abs/2504.16891) work, but to demonstrate a SFT demo from end to end.

Note that you may need to modify the path as needed in the following scripts.
## Dataset Preprocessing
### Download Dataset
```bash
hf download nvidia/OpenMathReasoning --repo-type dataset --include data/cot* --local-dir /path/to/dataset/nvidia/OpenMathReasoning
hf download math-ai/aime24 --repo-type dataset --local-dir /path/to/dataset/math-ai/aime24
hf download math-ai/aime25 --repo-type dataset --local-dir /path/to/dataset/math-ai/aime25
```

### Preprocess the dataset
```bash
python3 recipe/open_math_reasoning/prepare_nvidia-OpenMathReasoning_sft.py --local_dataset_path /path/to/nvidia/OpenMathReasoning --local_save_dir /path/to/open_math_reasoning
```

### Prepare the eval dataset
```bash
python3 recipe/open_math_reasoning/prepare_eval_dataset.py --local_dataset_path /path/to/dataset --local_save_dir /path/to/eval_dataset
```

## Train the model using SFT
export CKPT_HOME=/path/to/ckpt
export MODEL_ID=Qwen/Qwen3-8B-Base
export TRAIN_FILES=/path/to/open_math_reasoning/cot_dataset.parquet

### FSDP backend
export BACKEND=fsdp2
bash recipe/open_math_reasoning/run_sft_qwen3_8b.sh

### Megatron backend
export BACKEND=megatron
bash recipe/open_math_reasoning/run_sft_qwen3_8b.sh

## Eval the model
### Merge checkpoint into huggingface format
FSDP backend
```bash
python -m verl.model_merger merge --backend fsdp --local_dir /path/to/ckpt/global_step_19751 --target_dir /path/to/ckpt/global_step_19751/huggingface
```
Megatron backend
```bash
python -m verl.model_merger merge --backend megatron --local_dir /path/to/ckpt/global_step_19751 --target_dir /path/to/ckpt/global_step_19751/huggingface --use_cpu_initialization
```

### Generate the responses
```bash
export MODEL_PATH=/path/to/ckpt/global_step_19751/huggingface
bash recipe/open_math_reasoning/run_generation.sh
```

### Evaluate the responses
```bash
bash recipe/open_math_reasoning/run_eval.sh
```

You should see the results like:
```python
{'test_score/aime24': 0.584375, 'test_score/aime25': 0.43333333333333335}
```
