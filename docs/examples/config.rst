.. _config-explain-page:

Config Explanation
===================

ppo_trainer.yaml for FSDP Backend
---------------------------------

Data
~~~~

.. code:: yaml

   data:
     tokenizer: null
     train_files: ~/data/rlhf/gsm8k/train.parquet
     val_files: ~/data/rlhf/gsm8k/test.parquet
     prompt_key: prompt
     max_prompt_length: 512
     max_response_length: 512
     train_batch_size: 1024
     return_raw_input_ids: False  # This should be set to true when the tokenizer between policy and rm differs
     return_raw_chat: False
     shuffle: True
     filter_overlong_prompts: False # for large-scale dataset, filtering overlong prompts could be timeconsuming. You should disable this and set `truncation='left'
     truncation: error
     image_key: images

- ``data.train_files``: Training set parquet. Can be a list or a single
  file. The program will read all files into memory, so it can't be too
  large (< 100GB). The path can be either local path or HDFS path. For
  HDFS path, we provide utils to download it to DRAM and convert the
  HDFS path to local path.
- ``data.val_files``: Validation parquet. Can be a list or a single
  file.
- ``data.prompt_key``: The field in the dataset where the prompt is
  located. Default is 'prompt'.
- ``data.max_prompt_length``: Maximum prompt length. All prompts will be
  left-padded to this length. An error will be reported if the length is
  too long
- ``data.max_response_length``: Maximum response length. Rollout in RL
  algorithms (e.g. PPO) generates up to this length
- ``data.train_batch_size``: Batch size sampled for one training
  iteration of different RL algorithms.
- ``data.return_raw_input_ids``: Whether to return the original
  input_ids without adding chat template. This is mainly used to
  accommodate situations where the reward model's chat template differs
  from the policy. It needs to be decoded first, then apply the RM's
  chat template. If using a model-based RM, and the policy and RM
  chat_templates are different, this flag needs to be set
- ``data.return_raw_chat``:
- ``data.shuffle``: Whether to shuffle the data in the dataloader.
- ``data.filter_overlong_prompts``: Default don't filter. You can filter for small-scale dataset. 
  For large-scale dataset, filtering overlong prompts could be timeconsuming. 
  You should disable this and set ``truncation='left``
- ``data.truncation``: Truncate the input_ids or prompt length if they
  exceed max_prompt_length. Default is 'error', not allow exceed the
  max_prompt_length. The users should increase the max_prompt_length if
  throwing the error. You can also set ``left`` and ``right``.
- ``data.image_key``: The field in the multi-modal dataset where the image is
  located. Default is 'images'.

Actor/Rollout/Reference Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   actor_rollout_ref:
    hybrid_engine: True
    model:
      path: ~/models/deepseek-llm-7b-chat
      external_lib: null
      override_config: { }
      enable_gradient_checkpointing: False
      use_remove_padding: False
    actor:
      strategy: fsdp  # This is for backward-compatibility
      ppo_mini_batch_size: 256
      ppo_micro_batch_size: null # will be deprecated, use ppo_micro_batch_size_per_gpu
      ppo_micro_batch_size_per_gpu: 8
      use_dynamic_bsz: False
      ppo_max_token_len_per_gpu: 16384 # n * ${data.max_prompt_length} + ${data.max_response_length}
      grad_clip: 1.0
      clip_ratio: 0.2
      entropy_coeff: 0.001
      use_kl_loss: False # True for GRPO
      use_torch_compile: True # False to disable torch compile
      kl_loss_coef: 0.001 # for grpo
      kl_loss_type: low_var_kl # for grpo
      ppo_epochs: 1
      shuffle: False
      ulysses_sequence_parallel_size: 1 # sp size
      optim:
        lr: 1e-6
        lr_warmup_steps: -1 # Prioritized. Negative values mean delegating to lr_warmup_steps_ratio.
        lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
        min_lr_ratio: null   # only useful for warmup with cosine
        warmup_style: constant  # select from constant/cosine
        total_training_steps: -1  # must be override by program
      fsdp_config:
        wrap_policy:
          # transformer_layer_cls_to_wrap: None
          min_num_params: 0
        param_offload: False
        optimizer_offload: False
        fsdp_size: -1
    ref:
      fsdp_config:
        param_offload: False
        wrap_policy:
          # transformer_layer_cls_to_wrap: None
          min_num_params: 0
      log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
      log_prob_micro_batch_size_per_gpu: 16
      log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
      log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
      ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
    rollout:
      name: vllm
      temperature: 1.0
      top_k: -1 # 0 for hf rollout, -1 for vllm rollout
      top_p: 1
      prompt_length: ${data.max_prompt_length}  # not use for opensource
      response_length: ${data.max_response_length}
      # for vllm rollout
      dtype: bfloat16 # should align with FSDP
      gpu_memory_utilization: 0.5
      ignore_eos: False
      enforce_eager: True
      free_cache_engine: True
      load_format: dummy_dtensor
      tensor_model_parallel_size: 2
      max_num_batched_tokens: 8192
      max_num_seqs: 1024
      log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
      log_prob_micro_batch_size_per_gpu: 16
      log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
      log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
      # for hf rollout
      do_sample: True
      # number of responses (i.e. num sample times)
      n: 1 # > 1 for grpo, rloo

**Common config for actor, rollout and reference model**

- ``actor_rollout_ref.hybrid_engine``: Whether it's a hybrid engine,
  currently only supports hybrid engine
- ``actor_rollout_ref.model.path``: Huggingface model path. This can be
  either local path or HDFS path. For HDFS path, we provide utils to
  download it to DRAM and convert the HDFS path to local path.
- ``actor_rollout_ref.model.external_libs``: Additional Python packages
  that need to be imported. Used to register models or tokenizers into
  the Huggingface system.
- ``actor_rollout_ref.model.override_config``: Used to override some of
  the model's original configurations, mainly dropout
- ``actor_rollout_ref.model.enable_gradient_checkpointing``: Whether to
  enable gradient checkpointing for the actor

**Actor model**

- ``actor_rollout_ref.actor.strategy``: fsdp or megatron. In this
  example, we use fsdp backend.

- ``actor_rollout_ref.actor.ppo_mini_batch_size``: One sample is split
  into multiple sub-batches with batch_size=ppo_mini_batch_size for PPO
  updates. The ppo_mini_batch_size is a global num across all workers/gpus

- ``actor_rollout_ref.actor.ppo_micro_batch_size``: [Will be deprecated, use ppo_micro_batch_size_per_gpu] 
  Similar to gradient accumulation, the micro_batch_size_per_gpu for one forward pass,
  trading speed for GPU memory. The value represent the global view.

- ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``: Similar to gradient
  accumulation, the micro_batch_size_per_gpu for one forward pass, trading speed
  for GPU memory. The value represent the local num per gpu.

- ``actor_rollout_ref.actor.grad_clip``: Gradient clipping for actor
  updates

- ``actor_rollout_ref.actor.clip_ratio``: PPO clip ratio

- ``actor_rollout_ref.actor.use_torch_compile``: Whether to use torch compile in actor

- ``actor_rollout_ref.actor.entropy_coeff``: The weight of entropy when
  calculating PPO loss

- ``actor_rollout_ref.actor.ppo_epochs``: Number of epochs for PPO
  updates on one set of sampled data

- ``actor_rollout_ref.actor.shuffle``: Whether to shuffle data when
  there are multiple epochs

- ``actor_rollout_ref.actor.optim``: Actor's optimizer parameters

- ``actor_rollout_ref.actor.fsdp_config``: FSDP config for actor
  training

  - ``wrap_policy``: FSDP wrap policy. By default, it uses Huggingface's
    wrap policy, i.e., wrapping by DecoderLayer

    - No need to set transformer_layer_cls_to_wrap, so we comment it.

  - ``*_offload``: Whether to enable parameter, gradient and optimizer
    offload

    - Trading speed for GPU memory.

**Reference Model**

- ``actor_rollout_ref.ref``: FSDP config same as actor. **For models
  larger than 7B, it's recommended to turn on offload for ref by
  default**

- ``actor_rollout_ref.ref.log_prob_micro_batch_size``: [Will be deprecate, use log_prob_micro_batch_size_per_gpu]
  The batch size for one forward pass in the computation of ``ref_log_prob``. The value represent the global num.

- ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``: The batch size
  for one forward pass in the computation of ``ref_log_prob``. The value represent the local num per gpu.

**Rollout Model**

- ``actor_rollout_ref.rollout.name``: hf/vllm. We use vLLM by default
  because it's much efficient and our hybrid engine is implemented with
  vLLM.

- Rollout (Auto-regressive) parameters. The key should be equal to the
  property name in vLLM's ``SamplingParams``.

  - ``temperature``, ``top_k``, ``top_p`` and others: Sampling
    parameters in ``SamplingParams``.

- ``dtype``: Rollout model parameters type. This should be align with
  the actor model parameter type in FSDP/Megatron backend.

- ``gpu_memory_utilization``: The proportion of the remaining GPU memory
  allocated for kv cache after other models have initialized when using
  vllm.

- ``tensor_model_parallel_size``: TP size for rollout. Only effective
  for vllm.

- ``actor_rollout_ref.ref.log_prob_micro_batch_size``: [Will be deprecate, use log_prob_micro_batch_size_per_gpu]
  The batch size for one forward pass in the computation of ``log_prob``. The value represent the global num.

- ``log_prob_micro_batch_size_per_gpu``: Micro batch size per gpu (The batch size for
  one forward pass) for recalculating ``log_prob``. The value represent the local num per gpu.

- ``do_sample``: Whether to sample. If set to False, the rollout model
  will perform greedy sampling. We disable ``do_sample`` during
  validation.

- ``actor_rollout_ref.rollout.ignore_eos``: Whether to ignore the EOS
  token and continue generating tokens after the EOS token is generated.

- ``actor_rollout_ref.rollout.free_cache_engine``: Offload the KVCache
  after rollout generation stage. Default is True. When set to True, we
  need to disable the usage of CUDAGraph (set ``enforce_eager`` to
  True.)

- ``actor_rollout_ref.rollout.enforce_eager``: Whether to use CUDAGraph
  in vLLM generation. Default set to True to disable CUDAGraph.

- ``actor_rollout_ref.rollout.load_format``: Which weight loader to use
  to load the actor model weights to the rollout model.

  - ``auto``: Use Megatron weight loader.
  - ``megatron``: Use Megatron weight loader. Deployed with Megatron
    backend. The input model ``state_dict()`` is already partitioned
    along TP dimension and already gathered along PP dimension. This
    weight loader requires that the Rollout model and Actor model's
    parameters shape and name should be identical.
  - ``dtensor``: Default solution when using Huggingface weight loader.
    Deployed with FSDP backend and the state_dict_type is
    ``StateDictType.SHARDED_STATE_DICT``. Recommend to use this weight
    loader
  - ``hf``: Use Huggingface weight loader. Deployed with FSDP backend
    and the state_dict_type is ``StateDictType.FULL_STATE_DICT``. This
    solution doesn't need to rewrite the weight loader for each model
    implemented in vLLM but it results in larger peak memory usage.
  - ``dummy_hf``, ``dummy_megatron``, ``dummy_dtensor``: Random
    initialization.

.. note:: **NOTED**: In this config field, users only need to select from ``dummy_megatron``, ``dummy_dtensor``, ``dummy_hf`` for rollout initialization and our hybrid engine will select the corresponding weight loader (i.e., ``megatron``, ``dtensor``, ``hf``) during actor/rollout weight synchronization.

Critic Model
~~~~~~~~~~~~

Most parameters for Critic are similar to Actor Model.

Reward Model
~~~~~~~~~~~~

.. code:: yaml

   reward_model:
     enable: False
     model:
       input_tokenizer: ${actor_rollout_ref.model.path}  # set this to null if the chat template is identical
       path: ~/models/Anomy-RM-v0.1
       external_lib: ${actor_rollout_ref.model.external_lib}
       fsdp_config:
         min_num_params: 0
         param_offload: False
     micro_batch_size_per_gpu: 16
     max_length: null
     reward_manager: naive

- ``reward_model.enable``: Whether to enable reward model. If False, we
  compute the reward only with the user-defined reward functions. In
  GSM8K and Math examples, we disable reward model. For RLHF alignment
  example using full_hh_rlhf, we utilize reward model to assess the
  responses. If False, the following parameters are not effective.
- ``reward_model.model``

  - ``input_tokenizer``: Input tokenizer. If the reward model's chat
    template is inconsistent with the policy, we need to first decode to
    plaintext, then apply the rm's chat_template. Then score with RM. If
    chat_templates are consistent, it can be set to null.
  - ``path``: RM's HDFS path or local path. Note that RM only supports
    AutoModelForSequenceClassification. Other model types need to define
    their own RewardModelWorker and pass it from the code.
- ``reward_model.reward_manager``:  Reward Manager. This defines the mechanism
  of computing rule-based reward and handling different reward sources. Default
  if ``naive``. If all verification functions are multiprocessing-safe, the reward
  manager can be set to ``prime`` for parallel verification.

Customized Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml
  
   custom_reward_function:
     path: null
     name: compute_score

- ``custom_reward_function.path``: The path to the file containing your customized reward function. If not specified, pre-implemented reward functions will be used.
- ``custom_reward_function.name`` (Optional) : The name of the reward function within the specified file. Default is 'compute_score'.

Algorithm
~~~~~~~~~

.. code:: yaml

   algorithm:
     gamma: 1.0
     lam: 1.0
     adv_estimator: gae
     kl_penalty: kl  # how to estimate kl divergence
     kl_ctrl:
       type: fixed
       kl_coef: 0.005

- ``gemma``: discount factor
- ``lam``: Trade-off between bias and variance in the GAE estimator
- ``adv_estimator``: Support ``gae``, ``grpo``, ``reinforce_plus_plus``, ``rloo``
- ``kl_penalty``: Support ``kl``, ``abs``, ``mse`` and ``full``. How to
  calculate the kl divergence between actor and reference policy. For
  specific options, refer to `core_algos.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py#L192>`_ .

Trainer
~~~~~~~

.. code:: yaml

   trainer:
     total_epochs: 30
     project_name: verl_examples
     experiment_name: gsm8k
     logger: ['console', 'wandb']
     nnodes: 1
     n_gpus_per_node: 8
     save_freq: -1
     test_freq: 2
     critic_warmup: 0
     default_hdfs_dir: ~/experiments/gsm8k/ppo/${trainer.experiment_name} # hdfs checkpoint path
     default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name} # local checkpoint path

- ``trainer.total_epochs``: Number of epochs in training.
- ``trainer.project_name``: For wandb, swanlab
- ``trainer.experiment_name``: For wandb, swanlab
- ``trainer.logger``: Support console and wandb, swanlab, mlflow, tensorboard
- ``trainer.nnodes``: Number of nodes used in the training.
- ``trainer.n_gpus_per_node``: Number of GPUs per node.
- ``trainer.save_freq``: The frequency (by iteration) to save checkpoint
  of the actor and critic model.
- ``trainer.test_freq``: The validation frequency (by iteration).
- ``trainer.critic_warmup``: The number of iteration to train the critic
  model before actual policy learning.
