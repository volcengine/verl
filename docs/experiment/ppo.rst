.. _algo-baseline-page:

Algorithm Baselines
===================

GSM8k 
------------------

Assuming GSM8k dataset is preprocess via ``python3 examples/data_preprocess/gsm8k.py``

Refer to the table below to reproduce PPO training from different pre-trained models.

.. _hf_gemma: https://huggingface.co/google/gemma-2-2b-it#benchmark-results
.. _verl_data_gemma_sft: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-sft-0.411.log
.. _verl_data_gemma_sft_ppo: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/gemma-2-2b-it-ppo-bsz512_4-prompt1024-resp-512-0.640.log
.. _verl_data_gemma_sft_ppo_wandb: https://api.wandb.ai/links/verl-team/h7ux8602
.. _qwen_blog: https://qwenlm.github.io/blog/qwen2.5-llm/
.. _verl_data_qwen25_05_ppo: https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log

+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Model                      | Method                 | Test score |  Details                                                                                      |
+============================+========================+============+=====================+=========================================================================+
| google/gemma-2-2b-it       | pretrained checkpoint  | 23.9       |   `Huggingface <gemma_hf>`_                                                                   |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it       | SFT                    | 52.06      |   `Command and logs <verl_data_gemma_sft>`_                                                   |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| google/gemma-2-2b-it       | SFT + PPO              | 64.02      |   `Command and logs <verl_data_gemma_sft_ppo>`_, `wandb <verl_data_gemma_sft_ppo_wandb>`_     |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct | pretrained checkpoint  | 36.4       |   `Qwen blog <qwen_blog>`_                                                                    |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+
| Qwen/Qwen2.5-0.5B-Instruct | PPO                    | 56.7       |   `Command and logs <verl_data_qwen25_05_ppo>`_                                               |
+----------------------------+------------------------+------------+-----------------------------------------------------------------------------------------------+