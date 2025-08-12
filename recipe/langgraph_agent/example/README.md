# MathExpression: LangGraph Agent Example

MathExpression is a tiny example to demonstrate multi-turn rollout with [LangGraph ReactAgent](https://langchain-ai.github.io/langgraph/agents/overview/).

### Define react agent with tool
Firstly, to force ReactAgent to evaluate math expression by tool, we define a special operand `@`:
```python
@tool(parse_docstring=True)
def calculate(a: int, b: int, operand: str) -> int:
    """
    Compute the results using operand with two integers

    Args:
        a: the first operand
        b: the second operand
        operand: '+' or '-' or '*' or '@'
    """
    assert operand in ["+", "-", "*", "@"], f"unknown operand {operand}"
    if operand == "@":
        return 3 * a - 2 * b
    return eval(f"{a} {operand} {b}")
```

Without calling `calculate`, ReactAgent is impossible to evaluate math expression correctly.

Then, we can equip ReactAgent with `calculate` tool:
```python
class MathExpressionReactAgentLoop(ReactAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer):
        cls.tools = [calculate]
        super().init_class(config, tokenizer)
```

We can define agent loop config in yaml file, which will be used by AgentLoopWorker to dynamic load custom AgentLoop class.
```yaml
- name: math_expression
  _target_: recipe.langgraph_agent.example.math_expression.MathExpressionReactAgentLoop
```

### Prepare dataset
Now, let's prepare two small datasets for training and evaluation:
```bash
python recipe/langgraph_agent/example/create_dataset.py
```

- Parameters: `--train_size` (default: 5000), `--test_size` (default: 500), `--output_dir` (default: `data/math_expression_tool`).
- Example with custom sizes/output:
```bash
python recipe/langgraph_agent/example/create_dataset.py \
  --train_size 10000 \
  --test_size 1000 \
  --output_dir data/math_expression_tool
```

Note that dataset should contain a column `agent_name` with `math_expression`, which is used by `AgentLoopWorker` to select the
agent loop class.
| prompt | reward_model | agent_name |
|--------------------------------------|------------------------------|-----------------|
| [{'role': 'user', 'content': '...'}] | {'ground_truth': '-10', ...} | math_expression |
| [{'role': 'user', 'content': '...'}] | {'ground_truth': '-10', ...} | math_expression |

Generated math expressions are like below, requiring model to call `calculate` multiple times to solve sub expressions.
```
(2 @ (8 @ 8 @ 5 @ 5 @ 3) @ 6 @ (1 @ 4 @ 4 @ 4) @ 2) @ 6
(4.6 @ (9.05 @ 4.0) @ 8.3 @ 1.21) @ 8.6
9 @ 4
((2 @ 2) @ (3 @ 3)) @ 4
```

### Training
Hook all these up and start training:
```bash
bash recipe/langgraph_agent/example/run_qwen2.5_3b.sh 2>&1 | tee train.log
```

To submit on a SLURM cluster (the script contains SBATCH headers):
```bash
sbatch recipe/langgraph_agent/example/run_qwen2.5_3b.sh
```

**Note on `GPUS_PER_NODE` and `NNODES`:**

- `GPUS_PER_NODE`: GPUs per node.  
  Detection order: `SLURM_GPUS_ON_NODE` (if set) → `GPUS_PER_NODE` → `2`.
- `NNODES`: number of nodes.  
  Detection order: `SLURM_JOB_NUM_NODES` (if set) → `NNODES` → `1`.
- Total GPUs = `GPUS_PER_NODE × NNODES` (must be ≥ 2).

Local override (no `SLURM_*` set):
```bash
GPUS_PER_NODE=4 NNODES=2 bash recipe/langgraph_agent/example/run_qwen2.5_3b.sh
```

After total 39 steps, model should achieve 100% accuray on test dataset:
- val-aux/lighteval/MATH/reward: 1.0
- val-aux/num_turns/mean: 9.0, average number of messages include assistant and tool turns.

We can see model can solve math expression by calling `calculate` tool:
```
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
We define a new math operator @, where you can only call an external tool to compute. Please put your final answer inside \boxed{} only in the last turn. Now answer the following questions:
Compute 5 @ (3 @ 6 @ 8)
assistant
Sure, let's break this down step by step. We need to compute the expression \(5 @ (3 @ 6 @ 8)\). First, we need to compute the inner expression \(3 @ 6 @ 8\).

Let's compute \(3 @ 6\) first using the external tool.
<tool_call>
{"name": "calculate", "arguments": {"a": 3, "b": 6, "operand": "@"}}
</tool_call>user
<tool_response>
-3
</tool_response>
assistant
We have computed that \(3 @ 6 = -3\). Now we need to compute \(-3 @ 8\).

Let's use the external tool to compute this.
<tool_call>
{"name": "calculate", "arguments": {"a": -3, "b": 8, "operand": "@"}}
</tool_call>user
<tool_response>
-25
</tool_response>
assistant
We have computed that \(-3 @ 8 = -25\). Now we can put this result into the original expression to get the final result of \(5 @ (3 @ 6 @ 8)\), which is \(5 @ (-25)\).

Let's compute \(5 @ (-25)\) using the external tool.
<tool_call>
{"name": "calculate", "arguments": {"a": 5, "b": -25, "operand": "@"}}
</tool_call>user
<tool_response>
65
</tool_response>
assistant
The final result of \(5 @ (3 @ 6 @ 8)\) is \(\boxed{65}\).
```
