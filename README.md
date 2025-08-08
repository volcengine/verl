<h1 style="text-align: center;">AALC: Large Language Model Efficient Reasoning via Adaptive Accuracy-Length Control</h1>

The repo is built based on the verl GitHub repo.

## Getting Started
To set up an environment, please follow the following commands:
```bash
conda create -n lr python==3.10
pip install -r requirements.txt
pip install -e . --no-deps
```

To train a model, please confirm all parameters in the file `train_grpo_math_LR.sh` and then run:
```bash
bash train_grpo_math_LP.sh
```

To test a checkpoint, the procedure is similar to the training part, but the file is `test_grpo_math_LR.sh`.
