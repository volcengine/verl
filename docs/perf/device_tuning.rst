Resource Needed for verl RL
==============================

Since RL requires more resources compared to regular training, 
determining how much resources are needed to successfully run it before training 
is a relatively difficult task. To provide more people with reference points for 
resource selection when dealing with different models and tasks, this section is 
mainly dedicated to introducing the environmental requirements based on experiments 
we have conducted.

However, due to limited manpower and equipment resources, we also hope for more 
contributions from the open-source community. When submitting a PR, it is necessary 
to provide a script to be added to the example/tuning scripts.

We need two types of scripts: one is the configuration that can run with the **minimum 
resources(min)**, and the other is the configuration that runs with **recommended resources(recommended)**. For the former, 
it can be understood as a script that can run after applying all memory optimization techniques 
(e.g., offload, gradient checkpointing). For the latter, it can be understood as a script that 
can run while avoiding operations that incur additional time overhead as much as possible (targetting best throughput).

When defining script names, please follow this format: 
``[model]_[task]_[gpunums]_[device]_[train]_[infer].sh``. This will effectively improve 
the script's recognizability. You can place the script under the ``examples/tuning/`` directory.

If you happen to have a configuration that has already been tested, we welcome you to submit 
a PR and include a screenshot from Wandb or other verifiable evidence.

----------------------------------------

7B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-7B
      - GRPO
      - 2*H800
      - fsdp
      - vllm0.8.2
      - `qwen2-7b_grpo_2_h800_fsdp_vllm <../../examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
      

14B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-14B
      - GRPO
      - 4*H800
      - fsdp
      - vllm0.8.2
      - `qwen2-14b_grpo_4_h800_fsdp_vllm <../../examples/tuning/14b/qwen2-14b_grpo_4_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_


32B
~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ====== ====== ======
   tag    model  task   resource train  infer  link
   ====== ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ====== ======== ====== ====== ======

70B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Tag
      - Model
      - Task
      - Resource
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H20
      - fsdp
      - vllm0.8.2
      - `qwen2-70b_grpo_32_h20_fsdp_vllm <../../examples/tuning/70b/qwen2-70b_grpo_32_h20_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H800
      - fsdp
      - vllm0.8.3
      - `qwen2-70b_grpo_32_h800_fsdp_vllm <../../examples/tuning/70b/qwen2-70b_grpo_32_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_

405B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ====== ====== ======
   tag    model  task   resource train  infer  link
   ====== ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ====== ======== ====== ====== ======


671B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ====== ====== ======
   tag    model  task   resource train  infer  link
   ====== ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ====== ======== ====== ====== ======
