.. _algo-baseline-page:

Algorithm Baselines
===================

Datasets 
------------------

Assuming GSM8k/math dataset is preprocess via ``python3 examples/data_preprocess/*.py``

Refer to the table below to reproduce RL training from different pre-trained models.

NVIDIA GPUs
--------------------------------

.. _Huggingface: https://huggingface.co/google/gemma-2-2b-it#benchmark-results
.. _SFT Command and Logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-sft-0.411.log
.. _SFT+PPO Command and Logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-ppo-bsz512_4-prompt1024-resp-512-0.640.log
.. _wandb: https://api.wandb.ai/links/verl-team/h7ux8602
.. _Qwen Blog: https://qwenlm.github.io/blog/qwen2.5-llm/
.. _PPO Command and Logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log
.. _Megatron PPO Command and Logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/deepseek-llm-7b-chat-megatron-bsz256_4-prompt512-resp512-0.695.log
.. _Qwen7b GRPO Script: https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/examples/grpo_trainer/run_qwen2-7b_seq_balance.sh
.. _Megatron wandb: https://wandb.ai/verl-team/verl_megatron_gsm8k_examples/runs/10fetyr3
.. _Qwen7b ReMax Script: https://github.com/eric-haibin-lin/verl/blob/main/examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh
.. _Qwen7b ReMax Wandb: https://wandb.ai/liziniu1997/verl_remax_example_gsm8k/runs/vxl10pln
.. _Qwen0.5b PRIME Script: https://github.com/volcengine/verl/blob/main/recipe/prime/run_prime_qwen.sh
.. _Qwen0.5b PRIME Wandb: https://api.wandb.ai/links/zefan-wang-thu-tsinghua-university/rxd1btvb
.. _Megatron Qwen2 7b GRPO Script with Math and GSM8k: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b_math_megatron.log
.. _Qwen7b GRPO FSDP2 Script and Logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log
.. _Qwen 7B SPPO Script: https://github.com/volcengine/verl/tree/main/recipe/sppo/README.md
.. _Qwen 7B SPIN Script: https://github.com/volcengine/verl/tree/main/recipe/spin/README.md
.. _Mixtral-8x22B-Instruct RLOO Wandb: https://api.wandb.ai/links/ppo_dev/sbuiuf2d

.. To contributors: PLEASE make sure the table margins are aligned vertically, and check if the rendering is successful in preview.

+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Model                            | Method                 | Test score |  Details                                                                                      |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it             | pretrained checkpoint  | 23.9       |   `Huggingface`_                                                                              |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it             | SFT                    | 52.06      |   `SFT Command and Logs`_                                                                     |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it             | SFT + PPO              | 64.02      |   `SFT+PPO Command and Logs`_, `wandb`_                                                       |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct       | pretrained checkpoint  | 36.4       |   `Qwen Blog`_                                                                                |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct       | PPO                    | 56.7       |   `PPO Command and Logs`_                                                                     |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct       | PRIME                  | 58.7       |   `Qwen0.5b PRIME Script`_, `Qwen0.5b PRIME Wandb`_                                           |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| deepseek-ai/deepseek-llm-7b-chat | PPO (Megatron)         | 69.5 [1]_  |   `Megatron PPO Command and Logs`_, `Megatron wandb`_                                         |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2-7B-Instruct           | GRPO                   | 89         |   `Qwen7b GRPO Script`_                                                                       |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2-7B-Instruct           | GRPO (FSDP2)           | 89.8       |   `Qwen7b GRPO FSDP2 Script and Logs`_                                                        |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2-7B-Instruct           | GRPO (Megatron)        | 89.6       |   `Megatron Qwen2 7b GRPO Script with Math and GSM8k`_                                        |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-7B-Instruct         | ReMax                  | 97         |   `Qwen7b ReMax Script`_, `Qwen7b ReMax Wandb`_                                               |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-7B-Instruct         | SPPO                   | 65.6 (MATH)|   `Qwen 7B SPPO Script`_                                                                      |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Mixtral-8x22B-Instruct-v0.1      | Original instruct model| 83.7       |   `Qwen Blog`_                                                                                |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Mixtral-8x22B-Instruct-v0.1      | RLOO (Megatron)        | 92.3       |   `Mixtral-8x22B-Instruct RLOO Wandb`_                                                        |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-7B-Instruct         | SPIN                   | 92         |   `Qwen 7B SPIN Script`_                                                                      |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+

AMD GPUs (MI300)
--------------------------------

.. _ppo_run_deepseek7b_llm.sh:  https://github.com/yushengsu-thu/verl_training_log/blob/main/gsm8k/ppo_run_deepseek7b_llm.log
.. _grpo_run_deepseek7b_llm.sh: https://github.com/yushengsu-thu/verl_training_log/blob/main/gsm8k/grpo_run_deepseek7b_llm.log

+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Model                            | Method                 | Test score |  Details                                                                                      |
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| deepseek-ai/deepseek-llm-7b-chat | PPO                    | 70.5 [1]_  |   `ppo_run_deepseek7b_llm.sh`_                                                                |                   
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| deepseek-ai/deepseek-llm-7b-chat | GRPO                   | 71.4 [1]_  |   `grpo_run_deepseek7b_llm.sh`_                                                               |                   
+----------------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+

.. [1] During the evaluation, we have only extracted answers following the format "####". A more flexible answer exaction, longer response length and better prompt engineering may lead to higher score.
