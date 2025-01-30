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
```python
from verl.single_controller import 
```

RayClassWithInitArgs
""""""""""""""""""""""


