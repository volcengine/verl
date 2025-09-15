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

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from recipe.langgraph_agent.chat_model import (
    ChatModel,
    MaxTokenExceededError,
    convert_to_agent_output,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput


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
    max_assistant_turns = config["configurable"]["max_assistant_turns"]
    num_assistant_turns = 0
    for message in state["messages"]:
        if message.type == "ai":
            num_assistant_turns += 1

    last_message = state["messages"][-1]

    # LLM call failed, e.g: max response length exceeded
    if last_message.type == "tool":
        return END

    # max assistant turns exceeded
    if max_assistant_turns and num_assistant_turns >= max_assistant_turns:
        return END

    # no tool calls
    if not last_message.tool_calls:
        return END

    return "tools"


class ReactAgentLoop(AgentLoopBase):
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

        config = {
            "configurable": {
                "model": model,
                "sampling_params": sampling_params,
                "max_user_turns": rollout.multi_turn.max_user_turns,
                "max_assistant_turns": rollout.multi_turn.max_assistant_turns,
            }
        }

        # TODO: how to handle multiple trajectories in an graph invocation?
        # Each graph node may has its own LLM calls and state, e.g:
        # https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
        state = await self.graph.ainvoke(input={"messages": messages}, config=config)

        output = convert_to_agent_output(state["messages"], rollout.response_length)
        return output
