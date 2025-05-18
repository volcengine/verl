Single Controller interface
============================

The Single Controller provides a unified interface for managing distributed workers
using Ray or other backends and executing functions across them.
It simplifies the process of dispatching tasks and collecting results, particularly 
when dealing with data parallelism or model parallelism. 


Core APIs
~~~~~~~~~~~~~~~~~

.. autoclass:: verl.single_controller.Worker
   :members: __init__, __new__, get_master_addr_port, get_cuda_visible_devices, world_size, rank

.. autoclass:: verl.single_controller.WorkerGroup
   :members: __init__,  world_size

.. autoclass:: verl.single_controller.ClassWithInitArgs
   :members: __init__, __call__

.. autoclass:: verl.single_controller.ResourcePool
   :members: __init__, world_size, local_world_size_list, local_rank_list

.. automodule:: verl.single_controller.ray
   :members: RayWorkerGroup, create_colocated_worker_cls

.. autoclass:: verl.single_controller.ray.megatron.NVMegatronRayWorkerGroup
   :members: __init__



Decorator APIs
~~~~~~~~~~~~~~~~~
.. autofunction:: verl.single_controller.base.decorator.register

.. autoclass:: verl.single_controller.base.decorator.Dispatch
   :members: RANK_ZERO, ONE_TO_ALL, ALL_TO_ALL, MEGATRON_COMPUTE, MEGATRON_PP_AS_DP, MEGATRON_PP_ONLY, MEGATRON_COMPUTE_PROTO, MEGATRON_PP_AS_DP_PROTO, DP_COMPUTE, DP_COMPUTE_PROTO, DP_COMPUTE_PROTO_WITH_FUNC, DP_COMPUTE_METRIC, DIRECT_ROLLOUT_METHOD
   :member-order: bysource

.. autoclass:: verl.single_controller.base.decorator.Execute
   :members: ALL, RANK_ZERO
   :member-order: bysource