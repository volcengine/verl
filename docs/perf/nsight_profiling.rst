Nsight Profiling in verl
==============================

This guide explains how to use NVIDIA Nsight Systems for profiling verl training runs.

Configuration
--------------

Profiling in verl can be configured through several parameters in the trainer configuration file (ppo_trainer.yaml or other files like dapo_trainer.yaml):

0. Prerequisites

Nsight Systems version is important, please reference ``docker/Dockerfile.vllm.sglang.megatron`` for the version we used.

1. Global Profiling Control
--------------------------

``trainer.profile_steps``
    List of step numbers at which profiling should be performed. For example: [1, 2, 5] will profile steps 1, 2, and 5. And ``null`` means no profiling.

verl has one single controller process and multiple worker processes. Both controller and worker processes can be profiled. Since the controller process can be executed in any nodes in the cluster, there is message printed in the logging to indicate the controller process node hostname and process id.

2. Profiling rank control
----------------

By default, verl profiles the whole training process in a single `` worker_process_<PID>.<step>.nsys-rep`` file for each process rank. If you want to profile only part of the process ranks, you can set the ``profile_ranks`` parameter for each component. ``null`` and ``[]`` means no ranks, and ``[0,1]`` means profile ranks 0 and 1. If you want to profile all ranks, you can set the ``profile_ranks_all`` parameter for each component.

``actor_rollout_ref.hybrid_engine``
``actor_rollout_ref.actor.profile_ranks``
``actor_rollout_ref.ref.profile_ranks``
``actor_rollout_ref.rollout.profile_ranks``
``actor_rollout_ref.critic.profile_ranks``
``actor_rollout_ref.rm.profile_ranks``

3. Component-specific Profiling
------------------------------

Due to verl combines different components into a hybrid engine, one ``*.nsys-rep`` file may be too large to analyze. You can set the ``profile_discrete`` parameter for each component to profile each component separately. By default, all components ``profile_discrete`` are False. But if one component is set to True, all components should be set to True, which should be guaranteed by the user. In discrete mode, each component will be profiled in a separate ``worker_process_<PID>.<discrete_step>.nsys-rep`` file. Be noted that different algorithm has different discrete steps in a single training step.

``actor_rollout_ref.actor.profile_discrete``
``actor_rollout_ref.ref.profile_discrete``
``actor_rollout_ref.rollout.profile_discrete``
``actor_rollout_ref.critic.profile_discrete``
``actor_rollout_ref.rm.profile_discrete``

Each component (actor, rollout, ref, critic, reward model) can be independently configured for profiling:

4. where to find the profiling data

By default the ``*.nsys-rep`` files are saved in the directory ``/tmp/ray/session_latest/logs/nsight/`` at each node. According to the Ray manual, this default directory is not changeable. ``however, Ray preserves the the --output option of the default config`` <https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html>.

Some users may think it is not convenient, but it is understandable that Ray my start hundreds of processes and it would be a big network file system pressure if we save the files in one central place.

Usage Example
------------

To enable profiling for specific components and steps, modify your ppo_trainer.yaml like this:

.. code-block:: yaml

    trainer:
        profile_steps: null # disable profile

.. code-block:: yaml

    trainer:
        profile_steps: [1, 2, 5]  # Profile steps 1, 2, and 5

    actor_rollout_ref:
        actor:
            profile_discrete: False
            profile_ranks: [0, 1]  # Only profile ranks 0 and 1 for actor
        rollout:
            profile_discrete: False
            profile_ranks: null    # Profile all ranks for rollout
        ref:
            profile_discrete: False  # Disable profiling for ref policy
    critic:
        profile_discrete: False
        profile_ranks: [0]      # Only profile rank 0 for critic

.. code-block:: yaml

    trainer:
        profile_steps: [1, 2, 5]  # Profile steps 1, 2, and 5

    actor_rollout_ref:
        actor:
            profile_discrete: True
            profile_ranks: [0, 1]  # Only profile ranks 0 and 1 for actor
        rollout:
            profile_discrete: True
            profile_ranks: null    # Profile all ranks for rollout
        ref:
            profile_discrete: False  # Disable profiling for ref policy
    critic:
        profile_discrete: True
        profile_ranks: [0]      # Only profile rank 0 for critic

Profiling Output
--------------

When profiling is enabled, verl will generate Nsight Systems profiles for the specified components and steps. The profiles will include:

- CUDA kernel execution
- Memory operations
- CPU-GPU synchronization
- NVTX markers for key operations

The profiling data can be analyzed using NVIDIA Nsight Systems GUI or command-line tools.

Notes
-----

1. Profiling adds overhead to training, so it's recommended to only enable it for specific steps and components you want to analyze.

2. For large models or high-throughput training, consider profiling only specific ranks to reduce overhead.

3. The profile_discrete flag allows you to enable/disable profiling without changing the profile_ranks configuration.

4. When using multiple GPUs, profile_ranks helps focus profiling on specific GPUs to reduce overhead and simplify analysis.

