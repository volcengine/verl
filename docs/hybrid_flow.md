# HybridFlow Programming Guide
Author: Chi Zhang

In this section, we will introduce the basic concepts of HybridFlow, the motivation and how to program with verl APIs.

## Motivation
In classic RL, we use dataflow to represent RL systems.

### DataFlow

Dataflow is an abstraction of computations. Neural Netowork training is a typical dataflow. It can be represented by computational graph. 

Credit: CS231n 2024 lecture 4

This figure represents the computation graph of a polynomial function followed by a sigmoid function. In the data flow of neural network computation, each node represents an operator, and each edge represents the direction of forward/backward propagation. The computation graph determines the architecture of the neural network.

Reinforcement learning (RL) training can also be represented as a dataflow. However, the dataflow of RL has fundamental differences compared with dataflow of neural network training as follows:
| Workload    | Node | Edge      |
| -------- | ------- | ------- |
| Neural Network Training  |  Operator (+/-/matmul/softmax)   |   Tensor movement      |
| Reinforcement Learning | High-level operators (rollout/model forward)     |  Data Movement       |

In tabular RL cases, each operator is a simple scalar math operation (e.g., bellman update). In deep reinforcement learning, each operator is a high-level neural network computation such as model inference/update. This makes RL a two-level dataflow problem
- Control flow: defines how the high-level operators are executed (e.g., In PPO, we first perform autoreg. Then, we perform )
- Computation flow: defines the dataflow of neural network computation (e.g., forward/backward/optimizer)

The model size used in DRL is typically small. Thus, the high-level neural network computation can be done in a single process. This enables embedding the Computation flow inside the Control flow as a single process. However, 


### Single-Controller vs. Multi-Controller
The \cite{pathway}


```python
# single controller way of tensor parallelism using Ray

def column_mlp(x, w):
    # running on GPU workers
    pass


def row_mlp(x, w):
    pass

# 

```

```python
# multi-controller way of tensor parallelism using torch collective API
```

### Design Choices


## Programming Guide with verl APIs

### RayResourcePool

### RayClsWithInitArgs

### RayWorkerGroup


Note that the code under https://github.com/volcengine/verl/tree/main/verl/single_controller is difficult to understand. However, it defines 



### Codebase walk through
High level 



RayResourcePool

A partition of GPUs from the global ray cluster. Note that any two resource pools are mutually exclusive (They can't share GPUs).

Examples

.. code:: python

    from verl.single_controller import RayResourcePool

    # create a resource pool with 4 GPUs

    # create a resource pool with 2 * 8 GPUs



RayClassWithInitArgs
""""""""""""""""""""""


