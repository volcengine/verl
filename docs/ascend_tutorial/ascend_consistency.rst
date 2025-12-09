Enable deterministic computation on Ascend devices
===================================

Last updated: 12/09/2025.

在昇腾设备上开启verl端到端确定性的方式。


数据输入一致
-----------------------------------

在脚本的配置中设置数据shuffle参数

.. code:: bash

   data.shuffle=False
   data.validation_shuffle=False


使能端到端的确定性seed
----------------
对于fsdp训练后端和megatron后端开启方式分别为在fsdp_worker.py/megatron_worker.py文件开头增加seed函数。

.. code:: bash

   import random
   import numpy as np
   import torch
   import torch_npu
   import os

   def seed_all(seed=1234):
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      os.environ['HCCL_DETERMINISTIC'] = str(True)
      os.environ['LCCL_DETERMINISTIC'] = str(1)
      os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1)
      os.environ['ATB_LLM_LCOC_ENABLE'] = "0"
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.use_deterministic_algorithms(True)
      torch_npu.npu.manual_seed_all(seed)
      torch_npu.npu.manual_seed(seed)

   seed_all()
