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
  Our experient in Sec.`HybridFlow paper <https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py>`_ evaluate this trade-off.

More tuning details such as dealing with Preemption and Chunked-prefill, you can refer the vLLM official tuning guide: https://docs.vllm.ai/en/latest/performance/optimization.html


Batch Size Tuning
-----------------

To achieve higher throughput in experience preparation (i.e., model fwd) and model update (i.e., actor/critic fwd/bwd), 
users may need to tune the ``*micro_batch_size_per_gpu`` for different computation.

In veRL, the Core logic of setting batch size:

- All algorithmic metrics (train batch size, ppo mini batch size): are global (from the perspective of single-controller), 
  which will be normalized in each Worker `Code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L120-L122>`_ .
- All performance-related parameters (micro batch size, max token length in dynamic batch size) are local parameters, which represent the data sizes per GPU.
  `Code <https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py#L127>`_

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


