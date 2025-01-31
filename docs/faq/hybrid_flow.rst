HybridFlow Programming Guide
====================================

In this section, we will introduce the basic concepts of HybridFlow, including the motivation and the API.

Motivation
--------------------------
In classic RL system, we use dataflow to abstract RL systems.

DataFlow
""""""""""""""""""""""
Dataflow is 


API
--------------------------

RayResourcePool
""""""""""""""""""""""
A partition of GPUs from the global ray cluster. Note that any two resource pools are mutually exclusive (They can't share GPUs).

Examples

.. code:: python

    from verl.single_controller import RayResourcePool

    # create a resource pool with 4 GPUs

    # create a resource pool with 2 * 8 GPUs



RayClassWithInitArgs
""""""""""""""""""""""


