.. _atropos-api-page:

Atropos Integration API
======================

AtroposRLTrainer
----------------

.. autoclass:: recipe.atropos.main_atropos.AtroposRLTrainer
   :members:
   :undoc-members:
   :show-inheritance:

AtroposDataLoader
-----------------

.. autoclass:: recipe.atropos.data_loader.AtroposDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

AtroposShardingManager
----------------------

.. autoclass:: recipe.atropos.main_atropos.AtroposShardingManager
   :members:
   :undoc-members:
   :show-inheritance:

AtroposAPI
----------

.. autoclass:: recipe.atropos.atropos_api.AtroposAPI
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
--------------------

AtroposConfig
~~~~~~~~~~~~~

.. autoclass:: recipe.atropos.config.AtroposConfig
   :members:
   :undoc-members:
   :show-inheritance:

DataConfig
~~~~~~~~~~

.. autoclass:: recipe.atropos.config.DataConfig
   :members:
   :undoc-members:
   :show-inheritance:

TrainingConfig
~~~~~~~~~~~~~~

.. autoclass:: recipe.atropos.config.TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: recipe.atropos.utils.compute_advantages_from_atropos
.. autofunction:: recipe.atropos.utils.compute_advantages_from_gpro
.. autofunction:: recipe.atropos.utils.compute_advantage_weighted_loss
.. autofunction:: recipe.atropos.utils.normalize_advantages
.. autofunction:: recipe.atropos.utils.clip_advantages 