在昇腾设备上基于FSDP后端进行数据采集
====================================

Last updated: 07/08/2025.

这是一份在昇腾设备上基于FSDP后端使用GRPO或DAPO算法进行数据采集的教程。

配置
----

复用verl/trainer/config/ppo_trainer.yaml中的配置项控制采集的模式和步数，
通过verl/trainer/config/npu_profile/npu_profile.yaml中的配置项控制例如采集等级等参数。

全局采集控制
~~~~~~~~~~~~

通过 ppo_trainer.yaml 中的参数控制采集步数和模式：

-  trainer.profile_steps：
   该参数可以设置为一个包含采集步数的列表，例如[2，
   4]， 意味着将会采集第二步和第四步。如果该参数为null，则代表不进行采集
-  actor_rollout_ref：
   可以对actor、rollout和ref三种角色分别设置其profiler选项。由于actor和rollout在同一个worker中，在当前采集逻辑下，
   actor与rollout应设置相同的参数，若参数不同，则rollout的参数生效。端到端模式下，ref建议设置成与actor/rollout相同的参数，
   以便于生成完整的关键函数耗时信息，具体采集的rank由actor/rollout参数进行控制；
   离散模式下，ref的参数和actor/rollout的参数分别控制两种worker采集的rank

   -  all_ranks：设为True代表对所有rank进行采集
   -  ranks：当all_ranks不为True时，
      通过ranks参数控制需要采集的rank，该参数设置为一个包含采集rank的列表， 例如[0，
      1]
   -  discrete：
      控制采集的模式。当该参数设置为False，代表采集端到端的数据；当该参数设置为True，代表采用离散模式分训练阶段采集数据。
      actor、rollout和ref三种角色应设置相同的discrete参数

通过 npu_profile.yaml 中的参数控制具体采集行为：

-  save_path：采集数据的存放路径
-  level：采集等级，可选项为level_none、level0、level1和level2

   -  level_none：不采集所有Level层级控制的数据，即关闭profiler_level
   -  level0：采集上层应用数据、底层NPU数据以及NPU上执行的算子信息
   -  level1：在level0的基础上多采集CANN层AscendCL数据和NPU上执行的AI
      Core性能指标信息
   -  level2：在level1的基础上多采集CANN层Runtime数据以及AI CPU

-  record_shapes：是否记录张量形状
-  with_memory：是否启用内存分析
-  with_npu：是否采集device侧性能数据
-  with_cpu：是否采集host侧性能数据
-  with_module：是否记录框架层python调用栈信息
-  with_stack：是否记录算子调用栈信息
-  analysis：是否自动解析数据

示例
----

禁用采集
~~~~~~~~

.. code:: yaml

       trainer:
           profile_steps: null # disable profile

端到端采集
~~~~~~~~~~

.. code:: yaml

       trainer:
           profile_steps: [1, 2, 5]
       actor_rollout_ref:
           actor:
               profiler:
                   discrete: False
                   all_ranks: True
           rollout:
               profiler:
                   discrete: False
                   all_ranks: True
           ref:
               profiler:
                   discrete: False
                   all_ranks: True


离散模式采集
~~~~~~~~~~~~

.. code:: yaml

       trainer:
           profile_steps: [1, 2, 5]
       actor_rollout_ref:
           actor:
               profiler:
                   discrete: True
                   all_ranks: False
                   ranks: [0, 1]
           rollout:
               profiler:
                   discrete: True
                   all_ranks: False
                   ranks: [0, 1]
           ref:
               profiler:
                   discrete: True
                   all_ranks: False
                   ranks: [2, 3]

可视化
------

采集后的数据存放在用户设置的save_path下，可通过 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 工具进行可视化。

如果analysis参数设置为False，采集之后需要进行离线解析：

.. code:: python

    import torch_npu
    # profiler_path请设置为"localhost.localdomain_<PID>_<timestamp>_ascend_pt"目录的上一级目录
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)