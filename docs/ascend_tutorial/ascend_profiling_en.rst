Data collection based on FSDP (Fully Sharded Data Parallel) backend on Ascend devices(NPU)
==========================================================================================

Last updated: 07/14/2025.

This is a tutorial for data collection using the GRPO or DAPO algorithm
based on FSDP on Ascend devices.

Configuration
-------------

Reuse the configuration items in
verl/trainer/config/ppo_trainer.yaml to control the collection mode
and steps, you can also manage the collection behaviors such as
collection level via verl/trainer/config/npu_profile/npu_profile.yaml.

Global collection control
~~~~~~~~~~~~~~~~~~~~~~~~~

Use parameters in ppo_trainer.yaml to control the collection mode
and steps.

-  trainer.profile_steps: This parameter can be set as a list that has
   collection steps, such as [2, 4], which means it will collect steps 2
   and 4. If set to null, no collection occurs.
-  actor_rollout_ref.profiler: Control the ranks and mode of profiling

   -  all_ranks: Collects data from all ranks when set to true.
   -  ranks: This parameter specifies which ranks to collect (e.g., [0,
      1]) when all_ranks is False.
   -  discrete: Controls the collection mode. If False, end-to-end data
      is collected; if True, data is collected in discrete phases during
      training.

Use parameters in npu_profile.yaml to control collection behavior:

-  save_path: Storage path for collected data.
-  level: Collection level—options are level_none, level0, level1, and
   level2

   -  level_none: Disables all level-based data collection (turns off
      profiler_level).
   -  level0: Collect high-level application data, underlying NPU data,
      and operator execution details on NPU.
   -  level1: Extends level0 by adding CANN-layer AscendCL data and AI
      Core performance metrics on NPU.
   -  level2: Extends level1 by adding CANN-layer Runtime data and AI
      CPU metrics.

-  record_shapes: Whether to record tensor shapes.
-  with_memory: Whether to enable memory analysis.
-  with_npu: Whether to collect device-side performance data.
-  with_cpu: Whether to collect host-side performance data.
-  with_module: Whether to record framework-layer Python call stack
   information.
-  with_stack: Whether to record operator call stack information.
-  analysis: Enables automatic data parsing.

Examples
--------

Disabling collection
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

       trainer:
           profile_steps: null # disable profile

End-to-End collection
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

       trainer:
           profile_steps: [1, 2, 5]
       actor_rollout_ref:
            profiler:
                discrete: False
                all_ranks: True


Discrete Mode Collection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

       trainer:
           profile_steps: [1, 2, 5]
       actor_rollout_ref:
            profiler:
                discrete: True
                all_ranks: False
                ranks: [0, 1]


Visualization
-------------

Collected data is stored in the user-defined save_path and can be
visualized by using the `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ tool.

If the analysis parameter is set to False, offline parsing is required after data collection:

.. code:: python

    import torch_npu
    # Set profiler_path to the parent directory of the "localhost.localdomain_<PID>_<timestamp>_ascend_pt" folder
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)