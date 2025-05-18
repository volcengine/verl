Trainers
=========================

Trainers drive the training loop. Introducing new trainer classes in case of new training paradiam is encouraged.

.. autosummary::
   :nosignatures:

   verl.trainer.ppo.ray_trainer.RayPPOTrainer


Core APIs
~~~~~~~~~~~~~~~~~

.. autoclass::  verl.trainer.ppo.ray_trainer.RayPPOTrainer

.. automodule:: verl.utils.tokenizer
   :members: hf_tokenizer

.. automodule:: verl.single_controller
   :members: Worker, WorkerGroup, ClassWithInitArgs, ResourcePool
