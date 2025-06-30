### Adding a New Workflow

When adding a new workflow for continuous integration (CI), you have two runner options: a fixed runner or a machine from the vemlp.

- **Fixed Runner**: To use a fixed runner, specify it in your workflow using the `runs-on` keyword, like `runs-on: [L20x8]`. 
- **Vemlp Runner**: Opting for a Vemlp machine allows you to launch tasks elastically. Each job can run in parallel, which can significantly speed up your CI process.

Here is a template to assist you. This template is designed for using Vemlp machines. Currently, for each job in the workflow, you need to create a `setup` and a `cleanup` job. This is a temporary workaround to ensure compatibility with GitHub's `rerun failed jobs` feature. We'll aim to reduce this redundancy in future versions. When using this template, the main parts you need to modify are the `IMAGE` environment variable and the specific `job steps`.

```yaml
name: Your Default Workflow

on:
  push:
    branches:
      - main
      - v0.* 
  pull_request:
    branches:
      - main
      - v0.* 
    paths:
      - "**/*.py"
      - ".github/workflows/template.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}

permissions:
  contents: read

env:
  IMAGE: "your vemlp image" # e.g. "verl-ci-cn-beijing.cr.volces.com/whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.0-te2.3"
  DYNAMIC_RUNNER_URL: "https://sd10g3clalm04ug7alq90.apigateway-cn-beijing.volceapi.com/runner" # public veFaas api

jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      runner-label: ${{ steps.create-runner.outputs.runner-label }}
      task-id: ${{ steps.create-runner.outputs.task-id }}
    steps:
      - uses: actions/checkout@v4
      - id: create-runner
        uses: plutoZZZZ/verl-github-runner-demo@v1 # This is a temporary version. In the future, a formal version will be released.
        with:
          mode: "create"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          image: "${{ env.DEFAULT_IMAGE }}"

  your_job:
    needs: setup
    runs-on: ["${{ needs.setup.outputs.runner-label || 'default-runner' }}"]
    steps:
      xxxx # your jobs

  cleanup:
    runs-on: ubuntu-latest
    needs: [setup, your_job]
    if: always()
    steps:
      - id: destroy-runner
        uses: plutoZZZZ/verl-github-runner-demo@v1
        with:
          mode: "destroy"
          faas-url: "${{ env.DYNAMIC_RUNNER_URL }}"
          task-id: "${{ needs.setup.outputs.task-id }}"