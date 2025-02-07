# HybridFlow Programming Guide
Author: [Chi Zhang](https://github.com/vermouth1992)

In this section, we will introduce the basic concepts of HybridFlow, the motivation and how to program with verl APIs.

## Motivation
We use dataflow to represent RL systems. \cite{rlflow}

### DataFlow

Dataflow is an abstraction of computations. Neural Netowork training is a typical dataflow. It can be represented by computational graph. 

Credit: CS231n 2024 lecture 4

This figure represents the computation graph of a polynomial function followed by a sigmoid function. In the data flow of neural network computation, each node represents an operator, and each edge represents the direction of forward/backward propagation. The computation graph determines the architecture of the neural network.

#### RL as a dataflow problem

Reinforcement learning (RL) training can also be represented as a dataflow. However, the dataflow of RL has fundamental differences compared with dataflow of neural network training as follows:
| Workload    | Node | Edge      |
| -------- | ------- | ------- |
| Neural Network Training  |  Operator (+/-/matmul/softmax)   |   Tensor movement      |
| Reinforcement Learning | High-level operators (rollout/model forward)     |  Data Movement       |

In tabular RL cases, each operator is a simple scalar math operation (e.g., bellman update). In deep reinforcement learning, each operator is a high-level neural network computation such as model inference/update. This makes RL a two-level dataflow problem
- Control flow: defines how the high-level operators are executed (e.g., In PPO, we first perform rollout. Then, we perform advantage computation. Finally, we perform training.)
- Computation flow: defines the dataflow of neural network computation (e.g., model forward/backward/optimizer)

### Design Choices
The model size used in DRL is typically small. Thus, the high-level neural network computation can be done in a single process. This enables embedding the computation flow inside the control flow as a single process. 

However, in LLM era, the computation flow (e.g., training neural network) becomes a multi-process program. This naturally leads to two design choices: 
- Convert the control flow into a multi-process program as well. Then colocate with computation flow (unified multi-controller)
    - Advantages:
        - Achieve the **optimal performance** under fixed computation flow and control flow as the communication overhead in both training and data transfer is minimized.
    - Disadvantages:
        - The same computation flow is hard to reuse from software perspective as it is coupled with specific controller code. For example, the training loop of PPO is generic. However, 

### Overall Execution Diagram



## Codebase walk through (PPO)

### Entry function
Code: https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py

In this file, we define a remote function `main_task` that serves as the controller process as shown in Figure~\ref{}. We also define a `RewardManager`, where users can customize their reward function based on the data source in the dataset. Note that `RewardManager` should return the final token-level reward that is optimized by RL algorithms. Note that users can combine model-based rewards and rule-based rewards.
The `main_task` constructs a RayPPOTrainer instance and launch the fit. Note that `main_task` **runs as a single process**. 

We highly recommend that the `main_task` is NOT schduled on the head of the ray cluster as it usually contains very few resources.

### Ray trainer
Code: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py

The RayPPOTrainer manages 
- Worker and WorkerGroup construction
- Runs the main loop of PPO algorithm
Note that, the fit function of RayPPOTrainer **runs as a single process**.

### Worker and WorkerGroup construction
Each workerGroup manages a list of workers that runs remotely. Note that the worker group runs in the process of its construtor.
Each worker inside the WorkerGroup runs on a GPU. The worker group serves as a proxy for the controller process to interact with a list of workers, in order to perform certain computations. **In order to do so, we have to bind the methods of the worker into the method of the WorkerGroup and define the data dispatch and data collection**. This is done via simple decoration that will be introduced in the Worker definition section.

For example, in PPO, we define 3 worker groups
- ActorRolloutRef: manages actor, rollout and reference policy. ActorRolloutRefWorker can be instantiated as a single actor, a single rollout, a single reference policy, a combined actor/rollout or a combined actor/rollout/ref. This design is aimed for the maximum code reuse in various scenarios.
The reason for colocating actor and rollout is for fast weight transfer using nccl. The reason for coloating actor and reference is to implement an efficient lora PPO as the reference policy is simply the base model of PPO in lora.
- Critic: manages the critic model
- Reward: manages the reward model

The worker group will be constructed on the resource pool it designates. The resource pool is a set of GPUs in the ray cluster.

### Worker definition
We take ActorRolloutRefWorker (in https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py) for an exmaple. 
The APIs it should expose to the controller process are
- init_model: build the underlying model
- generate_sequences: given prompts, generate responses
- compute_log_prob: compute the log-probability of a generated sequence using actor
- compute_ref_log_prob: compute the log-probability of a generated sequence using reference policy
- save_checkpoint: save the checkpoint

Note that these methods are defined in the worker that can only be invoked via remote calls. For example, if the controller process wants to initialize the model, it has to call
```python
for worker in actor_rollout_ref_wg:
    worker.init_model.remote()
```
If the controller process wants to generate sequences, it has to call
```python
data = xxx
# split the data into dp chunks
data_dp_lst = data.split(dp_size)
output_dp_lst = []
for i, worker in enumerate(actor_rollout_ref_wg):
    output_future = worker.generate_sequences.remote(data_dp_lst[i])
    output_dp_lst.append(output_future)
output = torch.cat(ray.get(output_dp_lst), dim=0)
```
We observe that controll process calling worker group methods in general can be divided into 3 parts:
- Split the data into data parallel sizes
- Dispatch the corresponding data into each worker
- Collect and concatenate the data when the computation finishes

In verl, we design a syntax sugar to encapsulate the 3 processes into a single call from the controller process.
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(data):
    ...

# on the driver
output = actor_rollout_ref_wg.generate_sequences(data)
```
We decorate the method of the worker with a ``register`` that explicitly defines how the input data should be splitted and dispatch to each worker, and how the output data should be collected and concatenated by the controller. For example, ``Dispatch.DP_COMPUTE_PROTO`` splits the input data into dp chunks, dispatch each data to each worker, collect the output and concatenate the results. Note that this function requires the input and output to be a DataProto defined here (https://github.com/volcengine/verl/blob/main/verl/protocol.py).


### PPO main loop
With the aforementioned APIs, we can implement the main loop of PPO as if it is a single process program
```python
for prompt in dataloader:
    output = actor_rollout_ref_wg.generate_sequences(prompt)
    old_log_prob = actor_rollout_ref_wg.compute_log_prob(output)
    ref_log_prob = actor_rollout_ref_wg.compute_ref_log_prob(output)
    values = critic_wg.compute_values(output)
    rewards = reward_wg.compute_scores(output)
    # compute_advantages is running directly on the control process
    advantages = compute_advantages(values, rewards)
    output = output.union(old_log_prob)
    output = output.union(ref_log_prob)
    output = output.union(values)
    output = output.union(rewards)
    output = output.union(advantages)
    # update actor
    actor_rollout_ref_wg.update_actor(output)
    critic.update_critic(output)

```

### Several Takeaways
- This programming paradigm enables users to use different computation backend without modification of the control process.
- This programming paradigm enables flexible placement (by changing the mapping of WorkerGroup and ResourcePool) without modification of the control process.

