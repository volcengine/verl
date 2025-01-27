Performance Tuning Guide
=========================

In this ssection, we will discuss how to tune the performance of all the stages in veRL, including:

1. Rollout generation throughput.

2. Batch size tuning for fwd and bwd computation

3. Enable use_dynamic_bsz for even higher throughput.

4. Utilize Ulysses Sequence Parallel for Long Context Training

Rollout Generation Tuning
--------------------------

Currently, we support two types rollout backend: vLLM and TGI. We will support SGLang soon.
We will discuss some key factors to tune the vLLM rollout:

Before tuning, we recommend setting the ``actor_rollout_ref.rollout.disable_log_stats=False`` to get the statistics of the rollout generation.

- Increase ``gpu_memory_utilization``. The vLLM pre-allocates GPU KVCache by using gpu_memory_utilization% of the remaining memory. 
  However, if you don't offload other model parameters and optimizer, we cannot make this value too large as it would lead to OOM. 
  Setting it to 0.5 - 0.7 would be a good choice to ensure no preemption, achieve high throughput and avoid OOM.

- If the GPU cache utilization is relatively low in the log, try to increase ``max_num_seqs`` or ``max_num_batched_tokens`` to incrase the bsz in the decode stage. 
  This can help increase the number of concurrent requests in a batch, thereby raising the GPU cache utilization.
  We recommend setting ``max_num_batched_tokens > 2048`` for higher throughput.

- If the GPU memory is enough, try to use smaller ``tensor_parallel_size`` to get more vLLM replicas. 
  As DP could result in larger throughput than TP but will lead to larger KVCache consumption. 
  There're some trade-off between more replicas and higher memory usage. 
  Our experient in Sec. 8.4 of `HybridFlow paper <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py>`_ evaluate this trade-off.

More tuning details such as dealing with Preemption and Chunked-prefill, 
you can refer the `vLLM official tuning guide <https://docs.vllm.ai/en/latest/performance/optimization.html>`_ 


Batch Size Tuning
-----------------

To achieve higher throughput in experience preparation (i.e., model fwd) and model update (i.e., actor/critic fwd/bwd), 
users may need to tune the ``*micro_batch_size_per_gpu`` for different computation.

In veRL, the Core logic of setting batch size:

- All algorithmic metrics (train batch size, ppo mini batch size): are global (from the perspective of single-controller), 
  which will be normalized in each Worker. `See normalization code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L120-L122>`_.
- All performance-related parameters (micro batch size, max token length in dynamic batch size) are local parameters, which represent the data sizes per GPU.
  `See normalization code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L127>`_

.. note:: In your training script, please use ``*micro_batch_size_per_gpu`` instead of ``*micro_batch_size``. 
  So that you don't need to consider the normalization of the ``micro_batch_size`` and ``micro_batch_size`` will be deprecated.

Therefore, users may need to tune the ``*micro_batch_size_per_gpu`` to accelerate training. Here're some tips:

1. Turn on ``enable_gradient_checkpointing``: ``actor_rollout_ref.model.enable_gradient_checkpointing=True`` and ``critic.model.enable_gradient_checkpointing=True``.
   These parameters enable us to setting larger ``micro_batch_size_per_gpu``, which will be beneficial for large mini-batch training.

2. Increase the ``*micro_batch_size_per_gpu`` as much as possible till equals to normalized ``mini_batch_size``.

3. Forward only parameter, such as ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``, 
   ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``, ``critic.forward_micro_batch_size_per_gpu`` could be larger (e.g., 2x) than training related micro batch sizes,
   such as ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``, ``critic.ppo_micro_batch_size_per_gpu``.

4. The micro batch size of Critic and Reward model could be larger than Actor model. This is because the actor model has much larger vocab size in the final layer.


Tuning for Dynamic Batch Size
-----------------------------

Dynamic batch size is a technique that allows the model to process similar number of tokens in a single forward pass (with different actual batch sizes).
This can significantly improve the training efficiency and reduce the memory usage.

To utilize this technique, users can set ``use_dynamic_bsz=True`` in actor, ref, critic and reward models.
With ``use_dynamic_bsz=True``, users don't need to tune ``*micro_batch_size_per_gpu``. Instead, they should tune the following parameters:

- ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu``, ``critic.ppo_max_token_len_per_gpu``: 
  The maximum number of tokens to be processed in fwd and bwd of ``update_policy`` and ``update_critic``.

- ``actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`` and ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``: 
  The maximum number of tokens to be processed in a the fwd computation of ``compute_log_prob`` and ``comptue_ref_log_prob``.

- ``critic.forward_micro_batch_size_per_gpu``, ``reward_model.forward_micro_batch_size_per_gpu``: 
  The maximum number of tokens to be processed in a the fwd computation of ``compute_values``, ``compute_rm_score``.

Here're some tips to tune the above parameters:

1. The ``actor_rollout_ref.actor.ppo_max_token_len_per_gpu`` should be at least :math:`2 \times  (\text{max_prompt_length} + \text{max_response_length})`. 
   We set it to 3x in `run_qwen2-7b_rm_seq_balance.sh <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh#L25>`_.
   Try to increase it to get higher throughput.
   
2. Similarly in non-dynamic-batch-size scenarios, the fwd only parameter could be larger than fwd+bwd params.

3. Critic and Reward model related parameter can be at least 2x larger than Actor's. 
   We set it to 4x in `run_qwen2-7b_rm_seq_balance.sh <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh#L40>`_.
   
.. :math:`\text{critic.ppo_max_token_len_per_gpu}  = 2 \times  \text{actor.ppo_max_token_len_per_gpu})`.

Ulysses Sequence Parallel for Long Context Training
----------------------------------------------------

To utilize this technique, users can set ``ulysses_sequence_parallel_size>1`` in actor, ref, critic and reward models.
We support different model utilize different ulysses_sequence_parallel_size sizes.

To train log sequence (>32k), users may need to decrease the ``*micro_batch_size_per_gpu`` and ``*max_token_len_per_gpu`` to avoid OOM.