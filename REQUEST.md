# INITIAL REQUEST: VERL + Atropos Integration

Hey! Nous Research here - we're the authors of an LLM RL environments repo called atropos which is designed to provide rollouts for multi-environment runs, and where each individual env can be single-turn, multi-turn, or multi-agent, R1-zero style, or have a custom chat template. Furthermore, environments can define token-level advantages and so are not necessarily tied to the same RL training algorithm.

Each environment is a self-contained microservice and we have an API that sequesters rollouts from all of the environments. A trainer can request a batch of data from the API. Here's our writeup of our API spec.

Here's an example integration into Axolotl from the Axolotl authors, and our own minimal example trainer.

===========================
EDIT:

- The integration must use GRPO (but support the optional token-level advantage overrides when provided.).
- Please expose the usual verl configurables as in this example.
- Atropos doesn't spin up inference servers - we need verl to spin these servers up and then provide their endpoints to Atropos.
- verl should also manage policy weight updates to the inference server.
- Both the verl team and Nous team must approve your PR.
- Most of the code is written under recipe
- Show an example script that trains a model that can improve on some metrics defined by the agent task
===========================

