Performance Tuning Guide
==============================

Author: `Guangming Sheng <https://github.com/PeterSH6>`_

In this section, we will discuss how to tune the performance of all the stages in verl, including:

1. Rollout generation throughput.

2. Enable ``use_remove_padding=True`` for sequence packing (i.e., data packing and remove padding).

3. Batch size tuning for forward and backward computation

4. Enable ``use_dynamic_bsz=True`` for higher throughput.

5. Utilize Ulysses Sequence Parallel for Long Context Training

6. LigerKernel for SFT performance optimization

Rollout Generation Tuning
--------------------------

verl currently supports two rollout backends: vLLM and TGI (with SGLang support coming soon). 

Below are key factors for tuning vLLM-based rollout. Before tuning, we recommend setting ``actor_rollout_ref.rollout.disable_log_stats=False`` so that rollout statistics are logged.

- Increase ``gpu_memory_utilization``.

  - For vLLM v0.5.4 and v0.6.3, the vLLM pre-allocates GPU KVCache by using gpu_memory_utilization of the **remaining** memory. 
  - For vLLM v0.7.0 and later, the vLLM instance will only use gpu_memory_utilization of the **total** memory.
  - For SGLang, it's the fraction of the free GPU memory used for **static** memory like model weights and KV cache. However, the remaining (1-gpu_memory_utilization) will also be used during inference.

  However, if model parameters and optimizer states are not offloaded, using too high a fraction can lead to OOM. 
  A value between 0.5 and 0.7 often strikes a good balance between high throughput and avoiding OOM.

  Note: since the definition of ``gpu_memory_utilization`` varies across inference engines, a value that works well for one engine may cause OOM for another.

- Adjust ``max_num_seqs`` or ``max_num_batched_tokens``.
  If the GPU cache utilization is relatively low in the log, increase ``max_num_seqs`` or ``max_num_batched_tokens`` 
  can enlarge the effective batch size in the decoding stage, allowing more concurrent requests per batch. 
  We recommend setting ``max_num_batched_tokens > 2048`` for higher throughput.

- Use a smaller ``tensor_parallel_size``. 
  When GPU resources allow, a smaller tensor parallel size spawns more vLLM replicas. 
  Data parallelism (DP) can yield higher throughput than tensor parallelism (TP), but also increases KVCache consumption. 
  Carefully balance the trade-off between more replicas and higher memory usage.
  Our experient in Sec. 8.4 of `HybridFlow paper <https://arxiv.org/pdf/2409.19256v2>`_ evaluate this trade-off.

More tuning details such as dealing with Preemption and Chunked-prefill
can be found in `vLLM official tuning guide <https://docs.vllm.ai/en/latest/performance/optimization.html>`_ 

The performance of vllm can be further increased if upgrading from v0.6.3 to v0.7. See https://github.com/volcengine/verl/blob/main/docs/README_vllm0.7.md for details on how to upgrade.

Enable remove padding (sequence packing)
-----------------------------------------

Currently, for llama, mistral, gemma1 and qwen based models, users can enable `use_remove_padding=True` to utilize the 
sequence packing implementation provided by transformers library.

For other models, transformers library may also support it but we haven't tested it yet.
Users can add the desired model config to the  `test_transformer.py <https://github.com/volcengine/verl/blob/main/tests/models/test_transformer.py#L24>`_ file.
And test its functionaility by running the following command:

.. code-block:: bash

  pytest -s tests/models/test_transformer.py

If the test passes, you can add your desired model into the model `registry.py <https://github.com/volcengine/verl/blob/main/verl/models/registry.py#L24>`_ file.
Then, you can enjoy the performance boost of sequence packing
and welcome to PR your tested model to verl!


Batch Size Tuning
-----------------

To achieve higher throughput in experience preparation (i.e., model fwd) and model update (i.e., actor/critic fwd/bwd), 
users may need to tune the ``*micro_batch_size_per_gpu`` for different computation.

In verl, the core principle for setting batch sizes is:

- **Algorithmic metrics** (train batch size, PPO mini-batch size) are *global* (from a single-controller perspective), 
  normalized in each worker. See the `normalization code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L120-L122>`_.

- **Performance-related parameters** (micro batch size, max token length for dynamic batch size) are *local* parameters that define the per-GPU data allocations. 
  See the `normalization code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L127>`_.

.. note:: In your training script, please use ``*micro_batch_size_per_gpu`` instead of ``*micro_batch_size``. 
  So that you don't need to consider the normalization of the ``micro_batch_size`` and ``micro_batch_size`` will be deprecated.

Batch Size Tuning tips
""""""""""""""""""""""

Therefore, users may need to tune the ``*micro_batch_size_per_gpu`` to accelerate training. Here're some tips:

1. **Enable gradient checkpointing**: 
   Set ``actor_rollout_ref.model.enable_gradient_checkpointing=True`` and ``critic.model.enable_gradient_checkpointing=True``. 
   This often allows for larger micro-batch sizes and will be beneficial for large mini-batch training.

2. Increase the ``*micro_batch_size_per_gpu`` as much as possible till equals to normalized ``mini_batch_size``.

3. **Use larger forward-only parameters**: 
   Forward only parameter, such as ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``, 
   ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``, ``critic.forward_micro_batch_size_per_gpu`` could be larger (e.g., 2x) than training related micro batch sizes,
   such as ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``, ``critic.ppo_micro_batch_size_per_gpu``.

4. **Allow larger micro-batch sizes for Critic and Reward models**:
   micro batch size of Critic and Reward model could be larger than Actor model. This is because the actor model has much larger vocab size in the final layer.

5. **Enable activation offloading**:
   Set ``actor_rollout_ref.model.enable_activation_offload=True`` and ``critic.model.enable_activation_offload=True``.
   This often works together with gradient checkpointing to get larger micro-batch sizes and it's only available in FSDP backend now.

Tuning for Dynamic Batch Size
-----------------------------

Dynamic batch size is a technique that allows the model to process similar number of tokens in a single forward pass (with different actual batch sizes).
This can significantly improve the training efficiency and reduce the memory usage.

To utilize this technique, users can set ``use_dynamic_bsz=True`` in actor, ref, critic and reward models.
With ``use_dynamic_bsz=True``, users don't need to tune ``*micro_batch_size_per_gpu``. 
Instead, users should tune the following parameters:

- ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``, ``critic.ppo_max_token_len_per_gpu``: 
  The maximum number of tokens to be processed in fwd and bwd of ``update_policy`` and ``update_critic``.

- ``actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`` and ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``: 
  The maximum number of tokens to be processed in a the fwd computation of ``compute_log_prob`` and ``comptue_ref_log_prob``.

- ``critic.forward_micro_batch_size_per_gpu``, ``reward_model.forward_micro_batch_size_per_gpu``: 
  The maximum number of tokens to be processed in a the fwd computation of ``compute_values``, ``compute_rm_score``.

Dynamic Batch Size Tuning tips
""""""""""""""""""""""""""""""

Here're some tips to tune the above parameters:

1. **Increase** ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``  
   Make it at least 2 x (max_prompt_length + max_response_length). We set it to 3x in `run_qwen2-7b_rm_seq_balance.sh <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh#L25>`_.
   Try to increase it to get higher throughput.

2. **Forward-only parameters can be larger**: 
   Similar to the non-dynamic-batch scenario, forward-only token limits can exceed those used in forward/backward operations.
 
3. **Use larger limits for Critic and Reward models**:
   Critic and Reward parameters can be set at least 2× the Actor’s limits. For instance, we set them to 4× here:  
   `run_qwen2-7b_rm_seq_balance.sh <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh#L40>`_
   
.. :math:`\text{critic.ppo_max_token_len_per_gpu}  = 2 \times  \text{actor.ppo_max_token_len_per_gpu})`.

Ulysses Sequence Parallel for Long Context Training
----------------------------------------------------

To utilize this technique, users can set ``ulysses_sequence_parallel_size>1`` in actor, ref, critic and reward models.

We support different model utilize different ulysses_sequence_parallel_size sizes.

To train log sequence (>32k), users may need to decrease the ``*micro_batch_size_per_gpu`` and ``*max_token_len_per_gpu`` to avoid OOM.

LigerKernel for SFT
----------------------

LigerKernel is a high-performance kernel for Supervised Fine-Tuning (SFT) that can improve training efficiency. To enable LigerKernel in your SFT training:

1. Install liger-kernel via ``pip3 install liger-kernel``. In your SFT configuration file (e.g., ``verl/trainer/config/sft_trainer.yaml``), set the ``use_liger`` parameter:

   .. code-block:: yaml

      model:
        use_liger: True  # Enable LigerKernel for SFT

2. The default value is ``False``. Enable it only when you want to use LigerKernel's optimizations.

3. LigerKernel is particularly useful for improving training performance in SFT scenarios.

