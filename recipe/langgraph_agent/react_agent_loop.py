# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LangGraph React Agent Loop.

This implementation is exact same as `ToolAgentLoop`.

Ref: https://langchain-ai.github.io/langgraph/tutorials/workflows/
"""

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from recipe.langgraph_agent.chat_model import (
    ChatModel,
    MaxTokenExceededError,
    convert_to_agent_output,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

logger = logging.getLogger(__name__)


async def call_model(state: MessagesState, config: RunnableConfig):
    model = config["configurable"]["model"]
    sampling_params = config["configurable"]["sampling_params"]
    try:
        message = await model.ainvoke(state["messages"], sampling_params=sampling_params)
        return {"messages": [message]}
    except MaxTokenExceededError:
        # last message is ToolMessage
        return {"messages": []}


def should_continue(state: MessagesState, config: RunnableConfig) -> Literal["tools", END]:
    # Safely extract max_assistant_turns from config
    max_assistant_turns = None
    try:
        if config and "configurable" in config:
            max_assistant_turns = config["configurable"].get("max_assistant_turns")
    except Exception as e:
        logger.warning(f"Failed to extract max_assistant_turns from config: {e}")

    num_assistant_turns = 0
    for message in state["messages"]:
        if message.type == "ai":
            num_assistant_turns += 1

    last_message = state["messages"][-1]

    # LLM call failed, e.g: max response length exceeded
    if last_message.type == "tool":
        return END

    # max assistant turns exceeded
    # Use a reasonable default limit (25) if max_assistant_turns is not set
    # This prevents infinite loops
    effective_max_turns = max_assistant_turns if max_assistant_turns is not None else 25
    if num_assistant_turns >= effective_max_turns:
        return END

    # no tool calls
    if not getattr(last_message, "tool_calls", None):
        return END

    return "tools"


class ReactAgentLoop(AgentLoopBase):
    # Recursion limit calculation constants
    DEFAULT_MAX_ASSISTANT_TURNS = 25
    MIN_RECURSION_LIMIT = 50
    NODES_PER_TURN = 2  # Each AI turn involves agent + tools nodes
    RECURSION_LIMIT_SAFETY_FACTOR = 1.5  # 50% buffer for edge cases

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ReactAgentLoop initialization")

        # build graph
        cls.graph = cls.build_graph()

    @classmethod
    def build_graph(cls) -> StateGraph:
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(cls.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )

        workflow.add_edge("tools", "agent")
        graph = workflow.compile()
        return graph

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        model_path = self.config.actor_rollout_ref.model.path
        model_name = "/".join(model_path.split("/")[-2:])

        rollout = self.config.actor_rollout_ref.rollout
        model = ChatModel(
            model=model_name,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
            max_parallel_calls=rollout.multi_turn.max_parallel_calls,
            tool_parser=rollout.multi_turn.format,
        )

        model = model.bind_tools(self.tools, tool_choice="any")

        # Calculate recursion_limit dynamically based on max_assistant_turns
        max_assistant_turns = (
            rollout.multi_turn.max_assistant_turns
            if rollout.multi_turn.max_assistant_turns
            else self.DEFAULT_MAX_ASSISTANT_TURNS
        )

        # Formula: nodes_per_turn * max_turns * safety_buffer, with minimum threshold
        recursion_limit = max(
            self.MIN_RECURSION_LIMIT,
            int(max_assistant_turns * self.NODES_PER_TURN * self.RECURSION_LIMIT_SAFETY_FACTOR),
        )
        logger.info(f"Configured recursion_limit={recursion_limit} (max_assistant_turns={max_assistant_turns})")

        config = {
            "configurable": {
                "model": model,
                "sampling_params": sampling_params,
                "max_user_turns": rollout.multi_turn.max_user_turns,
                "max_assistant_turns": rollout.multi_turn.max_assistant_turns,
            },
            "recursion_limit": recursion_limit,
        }

        # TODO: how to handle multiple trajectories in an graph invocation?
        # Each graph node may has its own LLM calls and state, e.g:
        # https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
        try:
            state = await self.graph.ainvoke(input={"messages": messages}, config=config)
        except Exception as e:
            logger.error(f"Agent loop execution failed: {type(e).__name__}: {e}")
            logger.error("Attempting to recover by extracting last valid trajectory.")

            # Strategy: Find the last valid AI message with complete metadata
            # This preserves any partial progress made before the error
            last_valid_ai_message = None
            for msg in reversed(messages):
                if (
                    msg.type == "ai"
                    and hasattr(msg, "response_metadata")
                    and "prompt_ids" in msg.response_metadata
                    and "response_mask" in msg.response_metadata
                ):
                    last_valid_ai_message = msg
                    break

            if last_valid_ai_message:
                logger.info("Recovered valid trajectory from existing messages.")
                state = {"messages": messages}
            else:
                logger.warning("No valid trajectory found. Creating minimal fallback.")
                fallback_message = AIMessage(
                    content="[Agent execution failed - no valid trajectory]",
                    response_metadata={
                        "request_id": "fallback",
                        "prompt_ids": [],
                        "response_mask": [],
                    },
                )
                state = {"messages": messages + [fallback_message]}

        output = convert_to_agent_output(state["messages"], rollout.response_length)
        return output
