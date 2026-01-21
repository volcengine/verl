# INITIAL REQUEST: VERL + Atropos Integration

Hey! Nous Research here - we're the authors of an LLM RL environments repo called atropos which is designed to provide rollouts for multi-environment runs, and where each individual env can be single-turn, multi-turn, or multi-agent, R1-zero style, or have a custom chat template. Furthermore, environments can define token-level advantages and so are not necessarily tied to the same RL training algorithm.

Each environment is a self-contained microservice and we have an API that sequesters rollouts from all of the environments. A trainer can request a batch of data from the API. Here's our writeup of our [API spec](https://github.com/NousResearch/atropos/tree/main/atroposlib/api).

Here's [an example integration into Axolotl](https://github.com/axolotl-ai-cloud/plugin-atropos) from the Axolotl authors, and our own [minimal example trainer](https://github.com/NousResearch/atropos/tree/main/example_trainer).

===========================
EDIT:

- The integration must use GRPO (but support the optional token-level advantage overrides when provided.).
- Please expose the usual verl configurables as in this [example](https://github.com/volcengine/verl/blob/main/examples%2Fgrpo_trainer%2Frun_qwen2_5-3b_gsm8k_grpo_lora.sh).
- Atropos doesn't spin up inference servers - we need verl to spin these servers up and then provide their endpoints to Atropos.
- verl should also manage policy weight updates to the inference server.
- Both the verl team and Nous team must approve your PR.
- Most of the code is written under recipe
- Show an example script that trains a model that can improve on some metrics defined by the agent task
- For a full integration into VeRL it should be able to take the groups from Atropos and perform GRPO training with them, as well as do full online RL - i.e. utilizing VeRL's vllm/sglang manager to spin up inference, update those inference weights, stay relatively on policy, as well as all the hyperparameters that GRPO comes alongside with, a reference model for KL Div, etc.
- Launch script that overrides configs for the environments or atropos to point them to VeRL's inference services (a script that launches atropos and verl together will allow this syncing)
- Check if we are using atropos api correctly 
- If Atropos' environments could sync to the verl inference engines in here, can you point to where that would be? How do the environments in atropos call for rollouts? 
===========================

