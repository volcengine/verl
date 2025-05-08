Using Checkpoints to Support Fault Tolerance Training
=====================================================

There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

.. note:: 

    Notice that the ``checkpoint.contents`` field has no effect to FSDP checkpoint except ``hf_model``, 
    the other 3 fields are binded together to save and load. We recommend to include ``model``, ``optimizer`` and ``extra`` all.

Checkpoint Saving Directory Structure
-------------------------------------

Commonly, we use the ``default_local_dir`` declared in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yml``
to work as preffix when saving checkpoints, which is ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``.

So the inner checkpoint structure of **FSDP** is like:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── actor_huggingface
    │   ├── critic
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   └── critic_huggingface
    └── latest_checkpointed_iteration.txt

All model shards, optimizers and extra states are stored togather, in a sharded and distributed way.

While **Megatron** current checkpoint structure is:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface     # default save tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
    │   │   ├── model           # save sharded model, naming the same as Megatron
    │   │   │   ├── mp_rank_xx_yyy          # xx is tp_rank in 2 digits, yyy is pp_rank in 3 digits
    │   │   │   │   └── model_states.pt
    │   │   │   └── mp_rank_xx_xxx
    │   │   ├── optim
    │   │   │   └── distrib_optim_pp{a}_tp{b}_cp{c}_dp{d}.pt
    │   │   └── rng_states
    │   └── critic
    │   │   ├── huggingface
    │   │   ├── model
    │   │   ├── optim
    │   │   └── rng_states
    └── latest_checkpointed_iteration.txt

Convert FSDP and Megatron Checkpoints to HuggingFace Format Model
-----------------------------------------------------------------

We provide a tool to convert the FSDP and Megatron checkpoints to HuggingFace format model.
The tool is located in ``scripts/model_merger.py``.

The arguments are as follows:

.. code:: bash

    usage: model_merger.py [-h] [--backend {fsdp,megatron}]
                           [--tie-word-embedding whether the model share embedding weights]
                           [--is-value-model whether the model is critic model]
                           [--hf_model_path $original_model_path, like {Qwen/Qwen2-7B}]
                           [--local_dir $local_directory saved fsdp or megatron models]
                           [--target_dir $target_dir to save converted models, default is tmp]
                           [--hf_upload_path $huggingface_repo to upload]

So example use of Megatron model merger is:

.. code:: bash

    python scripts/model_merger.py \
        --backend megatron \
        --tie-word-embedding \
        --hf_model_path Qwen/Qwen2.5-0.5B \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor

Megatron Merger details
-----------------------

Current implement of decoder layers uses ``nn.ModuleList`` to store the layers, 
and thus the model layers on every PP rank and VPP rank starts their index from 0.

There are 3 ways to correct this behavior:

1. Modify the decoder layer's state_dict, add ``offset`` to each layer's index, thus rewrite ``nn.ModuleList`` implementation.
2. Modify the layer index when saving checkpoint and recover them when loading checkpoint.
3. The Checkpoint merger do this work, calculate the actual ``offset`` from ``state_dict`` only, a little complex.

Current implementation use solution 2.


HuggingFace to Megatron DistCheckpoint details
----------------------------------------------

If your model is quite huge, we recommend you to use Megatron dist-checkpoint to load the model.
Megatron dist-checkpoint supports loading with different kinds of model parallelism,
and it is much faster than the original checkpoint loading.

To convert original HuggingFace model to Megatron dist-checkpoint,
you can use the ``scripts/converter_hf_to_mcore.py`` script. Large MoE models are temporarily supported with CPU initialization,
which is a little slower. While we are working on a better solution to support large models.

Example command to convert the model is as follows:

.. code:: bash

    python scripts/converter_hf_to_mcore.py \
        --hf_model_path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --output_path /mnt/disk/Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --use_cpu_initialization    # Only work for MoE models


Original Checkpoint Utils
-------------------------

Original Checkpoint Utils refer to original checkpoint implementation in ``verl/models/[model]/megatron/checkpoint_utils``.

We only need ``[model]_loader.py`` in original checkpoint utils now, since we get rid of storing ``hf_model`` every time (which is not recommended for large model training, try only saving sharded models if you can).

.. note:: 

    Note that ``[model]_loader`` only support environments where **storage clusters are able to connect with every calculation nodes**. 
    Because it utilizes **sharded load way to minimize the loading checkpoint overhead**. 
    Every rank loads its own data from ``state_dict`` which can be accessed by all of them.
    While there is also no need to broadcast among DP ranks, since the saved state_dict is only produced by DP rank 0.

    For users who can **only place the huggingface model on one device**, we keep the original costly implementation in ``[model]_loader_deprecated``. In this implementation, rank 0 broadcast all weights to each tp and pp rank, and then dp rank 0 broadcast to all dp ranks. There may be at risks of OOM.

    To use deprecated loader, change the import package of ``load_state_dict_to_megatron_llama``.
