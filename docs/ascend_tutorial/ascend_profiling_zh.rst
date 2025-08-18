Data collection based on FSDP backend on Ascend devices(zh)
====================================

在昇腾设备上基于FSDP后端进行数据采集

Last updated: 08/14/2025.

这是一份在昇腾设备上基于FSDP后端使用GRPO或DAPO算法进行数据采集的教程。

配置
----

使用两级profile设置来控制数据采集

- 全局采集控制：使用verl/trainer/config/ppo_trainer.yaml中的配置项控制采集的模式和步数，
- 角色profile控制：通过每个角色中的配置项控制等参数。

全局采集控制
~~~~~~~~~~~~

通过 ppo_trainer.yaml 中的参数控制采集步数和模式：

-  global_profiler: 控制采集的rank和模式

   -  tool: 使用的采集工具，选项有 nsys、npu、torch、torch_memory。
   -  steps: 此参数可以设置为包含采集步数的列表，例如 [2, 4]，表示将采集第2步和第4步。如果设置为 null，则不进行采集。
   -  save_path: 保存采集数据的路径。默认值为 "outputs/profile"。

角色profiler控制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在每个角色的 ``profiler`` 字段中，您可以控制该角色的采集模式。

-  enable: 是否为此角色启用性能分析。
-  all_ranks: 是否从所有rank收集数据。
-  ranks: 要收集数据的rank列表。如果为空，则不收集数据。
-  tool_config: 此角色使用的性能分析工具的配置。

通过每个角色的 ``profiler.tool_config.npu`` 中的参数控制具体采集行为：

-  level: 采集级别—选项有 level_none、level0、level1 和 level2

   -  level_none: 禁用所有基于级别的数据采集（关闭 profiler_level）。
   -  level0: 采集高级应用数据、底层NPU数据和NPU上的算子执行详情。
   -  level1: 在level0基础上增加CANN层AscendCL数据和NPU上的AI Core性能指标。
   -  level2: 在level1基础上增加CANN层Runtime数据和AI CPU指标。

-  contents: 控制采集内容的选项列表，例如
   npu、cpu、memory、shapes、module、stack。
   
   -  npu: 是否采集设备端性能数据。
   -  cpu: 是否采集主机端性能数据。
   -  memory: 是否启用内存分析。
   -  shapes: 是否记录张量形状。
   -  module: 是否记录框架层Python调用栈信息。
   -  stack: 是否记录算子调用栈信息。

-  analysis: 启用自动数据解析。
-  discrete: 使用离散模式。

示例
----

禁用采集
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: null # disable profile

端到端采集
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


离散模式采集
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


可视化
------

采集后的数据存放在用户设置的save_path下，可通过 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 工具进行可视化。

如果analysis参数设置为False，采集之后需要进行离线解析：

.. code:: python

    import torch_npu
    # profiler_path请设置为"localhost.localdomain_<PID>_<timestamp>_ascend_pt"目录的上一级目录
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)