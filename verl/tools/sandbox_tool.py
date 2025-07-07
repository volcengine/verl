from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema


from envs.base_sandbox import BaseSandbox, ToolDefinition
from verl.tools.schemas import OpenAIFunctionSchema

class SandboxToolAdapter(BaseTool):
    """
    Wrap one Sandbox tool behind the BaseTool interface expected by the
    verl evaluator/trainer stack.

    - `instance_id` ↔ `rollout_id`: we simply pass the value straight through.
    - Tool-level Rewards are left as zero;
    """

    def __init__(self, sandbox: BaseSandbox, tool_def: ToolDefinition):
        self._sandbox = sandbox
        self._tool_def = tool_def
        tool_schema = OpenAIFunctionToolSchema(
            type="object",
            function=OpenAIFunctionSchema(
                name=self._tool_def.name,
                description=self._tool_def.description or "",
                parameters=self._tool_def.input_schema or {},
            )
        )
        super().__init__(config={}, tool_schema=tool_schema)

    async def create(self, instance_id: Optional[str] = None, **_) -> str:
        """Mint an ID."""
        return instance_id or str(uuid4())

    async def execute(
        self, instance_id: str, parameters: Dict[str, Any], **_
    ) -> Tuple[str, float, Dict]:
        """
        Forward the call to the underlying sandbox tool, injecting `rollout_id`
        so everything downstream can treat it as part of the correct rollout.
        """
        response = self._sandbox.run_tool(
            instance_id,
            self._tool_def.name,
            **parameters
        )
        return str(response), 0.0, {}      # (tool_response, step_reward, metrics)

    async def calc_reward(self, *args, **kwargs) -> float:
        """No per-step reward in this generic adapter."""
        return 0.0

    async def release(self, instance_id: str, **_) -> None:
        """Nothing to clean up for now."""
        self._sandbox.cleanup_rollout(instance_id)


def sandbox_to_tool_list(sandbox: BaseSandbox) -> List[BaseTool]:
    """
    Convert all tools registered inside `sandbox` into `BaseTool` instances.
    """
    return [
        SandboxToolAdapter(sandbox, tool_def)
        for tool_def in sandbox.list_tools()
    ]
