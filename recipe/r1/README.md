# DeepSeek R1 Reproduction

This recipe is under development, if you are interested, checkout the TODO list and join this project! https://github.com/volcengine/verl/issues/708 

## Reproducing Evaluation

Eval Results of DS-R1-Distill-Qwen2.5-1.5B (k=8)

Dataset | Test Results | Reported
-- | -- | --
GPQA Diamond | 35.3 | 33.8
LiveCodeBench | 16.9 | 16.9
AIME 2024 | 30.4 | 28.9
CNMO 2024 (en) | 45.1 | -
CNMO 2024 (zh) | 41.0 | -

---

Eval Results (DS-R1)

Dataset | Test Results (k=1) | Test Results (k=4) | Reported
-- | -- | -- | --
GPQA Diamond | 67.7 | 69.6 | 71.5
LiveCodeBench | 64.7 | 63.1 | 65.9
AIME 2024 | 86.7 | 79.2 | 79.8
CNMO 2024 | 75.0 | 78.5 | 78.8
