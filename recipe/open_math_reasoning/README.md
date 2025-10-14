# Open math reasoning
## Introduction
In this recipe, we perform SFT on the [open math reasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset using the new SFT trainer with backend agostic model engine.

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
### FSDP backend
BACKEND=fsdp2 bash recipe/open_math_reasoning/run_sft_qwen3_8b.sh

### Megatron backend
TODO

## Eval the model

### Generate the responses
```bash

```
