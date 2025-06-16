### Checklist Before Starting

- [ ] Searched for similar PR(s).
- [ ] Checked PR Title format
  - In format of: [modules] type: Title
  - modules are in `fsdp, megatron, sglang, vllm, rollout, trainer, ci, training_utils, recipe, hardware, deployment, ray, worker, single_controller, misc, perf, model, algo, env, tool, ckpt, doc, data`
  - type is in `feat, fix, refactor, chore`
  - can involve multiple modules, seperated by `,` or space, like `[megatron, fsdp, doc] feat: xxx`

### What does this PR do?

> Add one-line overview of what this PR aims to achieve or accomplish. Reference related github issues and PRs if that help review.

### Test

> For changes that can not be tested by CI (e.g., algorithm implementation, new model support), validate by experiment(s) and show results like training curve plots, evaluatuion results, etc.

### High-Level Design

> Demonstrate the high-level design if this PR is complex.

### Specific Changes

> List the specific changes.

### API

> Demonstrate how the API changes if any.

### Usage Example

> Provide usage example(s) for easier usage.

```python
# Add code snippet or script demonstrating how to use this 
```

### Checklist Before Submitting

- [ ] Read the [Contribute Guide](https://github.com/volcengine/verl?tab=readme-ov-file#contribution-guide).
- [ ] Apply [pre-commit checks](https://github.com/volcengine/verl?tab=readme-ov-file#code-linting-and-formatting).
- [ ] Add `[BREAKING]` to the PR title `description` if it breaks any API.
- [ ] Update the documentation about your changes in the [docs](https://github.com/volcengine/verl/tree/main/docs).
- [ ] New CI unit test(s) are added to cover the code path.
- [ ] Rely on existing unit tests on CI that covers the code path.
