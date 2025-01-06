.. _algo-baseline-page:

Algorithm Baselines
===================

GSM8k 
------------------

Assuming GSM8k dataset is preprocess via ``python3 examples/data_preprocess/gsm8k.py``

Refer to the table below to reproduce PPO training from different pre-trained models.

.. _Huggingface: https://huggingface.co/google/gemma-2-2b-it#benchmark-results
.. _SFT Command and logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-sft-0.411.log
.. _SFT+PPO Command and logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-ppo-bsz512_4-prompt1024-resp-512-0.640.log
.. _wandb: https://api.wandb.ai/links/verl-team/h7ux8602
.. _Qwen Blog: https://qwenlm.github.io/blog/qwen2.5-llm/
.. _PPO Command and logs: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log

+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Model                      | Method                 | Test score |  Details                                                                                      |
+============================+========================+============+=====================+=========================================================================+
| google/gemma-2-2b-it       | pretrained checkpoint  | 23.9       |   `Huggingface`_                                                                              |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it       | SFT                    | 52.06      |   `SFT Command and logs`_                                                                     |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it       | SFT + PPO              | 64.02      |   `SFT+PPO Command and logs`_, `wandb`_                                                       |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct | pretrained checkpoint  | 36.4       |   `Qwen Blog`_                                                                                |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct | PPO                    | 56.7       |   `PPO Command and logs`_                                                                     |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+