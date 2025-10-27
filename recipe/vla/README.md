# [WIP] Experimental VLA RL Support

This recipe introduces experimental support for training SimpleVLA-OFT, a VLA model.

A key challenge in VLA RL training, which differs from standard LLM RL training, is that the environment/simulation phase has a higher computational overhead than the generation phase. To achieve high efficiency, RL in this context requires an effective environment scheduling mechanism in addition to verl's existing efficient training and inference scheduling. The goal is to reduce the inefficiency caused by the environment and the model's generation process waiting on each other.

The core computational model of this PR is inspired by the pipeline parallelism design from RLinf. It aims to overlap the environment's execution time with the model's generation time, thereby maximizing environment utilization.

This PR also proposes a future direction: creating a unified `Env` class. This class would encapsulate functionalities like tool calling, MCP, etc., under a single interface. The environment would manage its state internally, allowing the agent to communicate simply by calling `step(action)` to submit an action and receive an observation.

Currently, this code is located independently within the `recipes` folder. Much of the design is tightly coupled with the SimpleVLA model and the Libero environment, serving as an initial version for demonstration and discussion.

**References:**
*   [https://github.com/PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
*   [https://github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)