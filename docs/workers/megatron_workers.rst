Megatron-LM Backend
=====================

We support Megatron Backend by implementing various workers for actor,
critic, reference, rollout and reward models. We also implement the
``3DHybridEngine`` using Megatron-LM and vLLM in `megatron_vllm.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/hybrid_engine/megatron_vllm.py>`_.

**Pros**

- Support 3D parallelism and sequence parallelism for best scalablility
  and throughput.
- 3D HybridEngine can significantly reduce peak memory usage and reduce
  weight synchronize overhead between actor and rollout.

**Cons**

- Users should implement their own models for Megatron-LM
- Users should implement the corresponding weight_loader to

  - synchronize the model weight between actor (in Megatron) and rollout
    (in vLLM).
  - load weights from checkpoints to corresponding model in Megatron-LM

Megatron Workers
----------------

MegatronWorker
^^^^^^^^^^^^^^

``MegatronWorker`` is the base class of different megatron worker
classes. In this class, ``get_megatron_global_info`` and
``get_megatron_rank_info`` function to retrive the 3D parallel world
size and rank of each ``Worker`` running on specific GPU. These information
will be used in transfer protocol for Megatron Backend.

The following ``Worker`` class for different models will be utilized to
construct the ``WorkerGroup`` .

We implement various of APIs for each ``Worker`` class decorated by the
``@register(dispatch_mode=)`` . These APIs can be called by the ray
driver process. The data can be correctly collect and dispatch following
the ``dispatch_mode`` on each function. The supported dispatch_model
(i.e., transfer protocols) can be found in `decorator.py <https://github.com/volcengine/verl/blob/main/verl/single_controller/base/decorator.py>`_.

ActorRolloutRefWorker
^^^^^^^^^^^^^^^^^^^^^

This class is implemented for Actor/Rollout HybridEngine or for the
reference model to initialize their model and perform computation.

Actor/Rollout HybridEngine
''''''''''''''''''''''''''

1. HybridEngine, Actor and Rollout initialization API.

.. code:: python

   @register(dispatch_mode=Dispatch.ONE_TO_ALL)
   def init_model(self):

``ONE_TO_ALL``: when calling the ``init_model`` function from the driver
process, each worker (on a GPU) will execute the following model
initialization process.

The initialization details of HybridEngine, Actor and Rollout are
highlighted below:

1. ``AllGatherPPModel`` holds memory buffer for both Actor and Rollout
   and support weight resharding between actor and rollout.
2. ``MegatronPPOActor`` implements the simple PPO computation logics
   when the model is built with Megatron, including compute log prob,
   model update.
3. ``vLLMRollout`` support generation with vLLM. We modify the vLLM
   Engine and make it executed under SPMD to fit into our
   ``WorkerGroup`` design.
4. ``MegatronVLLMShardingManager`` a context manager to perform actual
   resharding between actor and rollout.

See `source code <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/workers/megatron_workers.py#L63>`_ for more information.

.. code:: python

   # Initialize the 3D HybridEngine
   hybrid_engine = AllGatherPPModel(model_provider=megatron_actor_model_provider)
   # Fetch the model at current rank
   actor_module = hybrid_engine.this_rank_models
   ...

   # build actor model
   self.actor = MegatronPPOActor(config=self.config.actor,
                                 model_config=self.actor_model_config,
                                 megatron_config=megatron_config,
                                 actor_module=self.actor_module,
                                 actor_optimizer=self.actor_optimizer,
                                 actor_optimizer_config=self.actor_optim_config)

   # build rollout
   # rollout initialization
   rollout = vLLMRollout(actor_module=params,
                        config=self.config.rollout,
                        tokenizer=self.tokenizer,
                        model_hf_config=self.actor_model_config,
                        train_tp=mpu.get_tensor_model_parallel_world_size())
   # perform weight resharding between actor and rollout
   sharding_manager = MegatronVLLMShardingManager(module=self.hybrid_engine,
                                                  inference_engine=rollout.inference_engine,
                                                  model_config=self.actor_model_config,
                                                  layer_name_mapping=layer_name_mapping)
   ...

2. Generate sequence and recompute log prob

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
   def generate_sequences(self, prompts: DataProto):

- ``Dispatch.MEGATRON_PP_AS_DP_PROTO``: The PP dimension of the actor
  model will be regarded as DP dimension. Then the driver process will
  dispatch and collect the data according to this reorganization. This
  is because, in HybridEngine, the actor weight, which usually applied
  larger 3D parallel sizes, will be gathered along the PP dimension and
  TP dimension. Therefore, the corresponding data should be dispatched
  and collected through the 3D parallel group of the rollout model,
  rather than the actor model. However, the world_size and rank
  information can only be retrived from ``get_megatron_global_info`` and
  ``get_megatron_rank_info``, which records the 3D information for the
  actor model. Moreover, the data resharding inside TP dimension will be
  processed within the HybridEngine.

- In this function, the rollout model will perform auto-regressive
  generation and the actor model will recompute the old log prob for the
  generated response.

3. Update actor model

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def update_actor(self, data: DataProto):

- ``Dispatch.MEGATRON_COMPUTE_PROTO``: User passes the data partitioned
  by DP dimension. The data is dispatched to all tp/pp ranks within the
  same dp group, and ultimately only collects output data from tp=0 and
  the last pp.
- Update the actor model weight using PPO & entropy loss.

ReferenceModel
''''''''''''''

1. Reference model initialization

The reference model is initialized using the same function as the actor
model without initializing the HybridEngine and Optimizer. Then the
actor model is also wrapped by the ``MegatronPPOActor``.

2. Compute reference log prob

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_ref_log_prob(self, data: DataProto):

- In this function, the reference model will call the compute log prob
  function in ``MegatronPPOActor`` to compute the reference log prob.

CriticWorker and RewardWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Model initialization

Quite similar to reference model. The CriticWorker will perform
additional initialization for the Optimizer.

2. Compute Values for CriticWorker

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_values(self, data: DataProto):

3. Update Critic

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def update_critic(self, data: DataProto):

4. Compute Reward

.. code:: python

   @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
   def compute_rm_score(self, data: DataProto):

Context Parallel
----------------

This require the developer/contributor to implement the context parallel
both in Megatron-LM and models.
