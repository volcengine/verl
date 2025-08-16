Data collection based on FSDP backend on Ascend devices(en)
==========================================================================================

Last updated: 08/14/2025.

This is a tutorial for data collection using the GRPO or DAPO algorithm
based on FSDP on Ascend devices.

Configuration
-------------

Leverage two levels of configuration to control data collection:

1. **Global profiler control**: Use parameters in ``ppo_trainer.yaml`` to control the collection mode and steps.
2. **Role profile control**: Use parameters in each role's ``profile`` field to control the collection mode for each role.

Global collection control
~~~~~~~~~~~~~~~~~~~~~~~~~

Use parameters in ppo_trainer.yaml to control the collection mode
and steps.

-  global_profiler: Control the ranks and mode of profiling

   -  tool: The profiling tool to use, options are nsys, npu, torch,
      torch_memory.
   -  steps: This parameter can be set as a list that has
      collection steps, such as [2, 4], which means it will collect steps 2
      and 4. If set to null, no collection occurs.
   -  save_path: The path to save the collected data. Default is
      "outputs/profile".


Role collection control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In each role's ``profiler`` field, you can control the collection mode for that role.

-  enable: Whether to enable profiling for this role.
-  all_ranks: Whether to collect data from all ranks.
-  ranks: A list of ranks to collect data from. If empty, no data is collected.
-  tool_config: Configuration for the profiling tool used by this role.

Use parameters in each role's ``profiler.tool_config.npu`` to control npu profiler behavior:

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

-  contents: A list of options to control the collection content, such as
   npu, cpu, memory, shapes, module, stack.
   
   -  npu: Whether to collect device-side performance data.
   -  cpu: Whether to collect host-side performance data.
   -  memory: Whether to enable memory analysis.
   -  shapes: Whether to record tensor shapes.
   -  module: Whether to record framework-layer Python call stack
      information.
   -  stack: Whether to record operator call stack information.

-  analysis: Enables automatic data parsing.
-  discrete: Whether to enable discrete mode.


Examples
--------

Disabling collection
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: null # disable profile

End-to-End collection
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
      actor_rollout_ref:
         actor:
            profiler:
               enable: True
               all_ranks: True
               tool_config:
                  npu:
                     discrete: False
        # rollout & ref follow actor settings


Discrete Mode Collection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
      actor_rollout_ref:
         actor:
            profiler:
               enable: True
               all_ranks: True
               tool_config:
                  npu:
                     discrete: True
        # rollout & ref follow actor settings


Visualization
-------------

Collected data is stored in the user-defined save_path and can be
visualized by using the `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ tool.

If the analysis parameter is set to False, offline parsing is required after data collection:

.. code:: python

    import torch_npu
    # Set profiler_path to the parent directory of the "localhost.localdomain_<PID>_<timestamp>_ascend_pt" folder
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)