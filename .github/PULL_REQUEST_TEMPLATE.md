### Checklist Before Starting

- [ ] Search for similar PR(s).

### What does this PR do?

> Add one-line overview of what this PR aims to achieve or accomplish. 

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

### Test

> For changes that can not be tested by CI (e.g., algorithm implementation, new model support), validate by experiment(s) and show results like training curve plots, evaluatuion results, etc.

### Additional Info.

- **Issue Number**: Fixes issue # or discussion # if any.
- **Training**: [Note which backend this PR will affect: FSDP, Megatron, both, or none]
- **Inference**: [Note which backend this PR will affect: vLLM, SGLang, both, or none]

### Checklist Before Submitting

- [ ] Read the [Contribute Guide](https://github.com/volcengine/verl?tab=readme-ov-file#contribution-guide).
- [ ] Apply [pre-commit checks](https://github.com/volcengine/verl?tab=readme-ov-file#code-linting-and-formatting).
- [ ] Add `[BREAKING]` to the PR title if it breaks any API.
- [ ] Update the documentation about your changes in the [docs](https://github.com/volcengine/verl/tree/main/docs).
- [ ] Add CI test(s) if neccessary.
