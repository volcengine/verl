# Agent Training with verl + Agent Lightning

**The example is tested with verl v0.5.0, vLLM v0.10.0, and Agent Lightning v0.1.1.**

This example might be useful to you if you are trying to use `verl` to:

- Diverge from the traditional RLHF setup;
- Customize an async server by inheriting from `AsyncServerBase`;
- Train on an arbitrary dataset which is not natively supported by `verl`.
- 