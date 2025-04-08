Device needed In verl RLHF
==========================

Since RL requires more resources compared to regular training, determining how many resources are needed to successfully run it before training is a relatively difficult task. To provide more people with reference points for resource selection when dealing with different models and tasks, this section is mainly dedicated to introducing the environmental requirements based on experiments we have conducted.

However, due to limited manpower and equipment resources, we also hope for more assistance from the open-source community. When submitting a PR, it is necessary to provide a script to be added to the example/tuning scripts.

When defining script names, please follow this format: ``[model]_[task]_[gpunums]_[device]_[train]_[infer].sh``. This will effectively improve the script's recognizability.
----------------------------------------

13B
~~~

.. table::
   :widths: auto

   ====== ====== ======== ====== ====== ======
   model  task   resource train  infer  link
   ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ======== ====== ====== ======


32B
~~~

.. table::
   :widths: auto

   ====== ====== ======== ====== ====== ======
   model  task   resource train  infer  link
   ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ======== ====== ====== ======


70B
~~~

.. table::
   :widths: auto

   ============= ====== ======== ====== ========= =================================================
   model         task   resource train  infer     link
   ============= ====== ======== ====== ========= =================================================
   Qwen2-70B     GRPO   32*H20   fsdp   vllm0.8.2 `tuning script <../../examples/tuning/70b/qwen2-70b_grpo_32_h20_fsdp_vllm.sh>`_
   ============= ====== ======== ====== ========= =================================================


405B
~~~~

.. table::
   :widths: auto

   ====== ====== ======== ====== ====== ======
   model  task   resource train  infer  link
   ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ======== ====== ====== ======


671B
~~~~

.. table::
   :widths: auto

   ====== ====== ======== ====== ====== ======
   model  task   resource train  infer  link
   ====== ====== ======== ====== ====== ======
   \      \      \        \      \      \
   ====== ====== ======== ====== ====== ======