# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 JetBrains s.r.o. and/or its affiliates
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
LangGraph implementation with ReAct agent and reflection step.

This module creates a graph where:
1. A ReAct agent performs initial reasoning and action
2. A deterministic validation message is appended asking the agent to validate its work
3. The agent can go through multiple rounds of self-reflection
"""

import json
from typing import List, TypedDict

import requests
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent


class GraphState(TypedDict):
    """State for the reflection graph."""

    messages: List[BaseMessage]
    current_round: int
    max_rounds: int
    needs_validation: bool


def create_sandboxfusion_tool(sandboxfusion_api: str) -> BaseTool:
    """Create sample tools for the ReAct agent with optional SandboxFusion API."""

    @tool
    def code_execution_tool(code: str) -> str:
        """
        Execute Python code in a sandboxed environment.
        Returns the stdout and stderr of the code execution.
        Please call print() for any variables you want to see.
        If you don't call print(), the output will not be shown.
        """
        try:
            # Make request to SandboxFusion API
            response = requests.post(f"{sandboxfusion_api}/run_code", json={"code": code, "language": "python", "files": {}})

            result = response.json()

            # Format the output in a pretty way
            output_parts = []
            output_parts.append("=" * 50)
            output_parts.append("CODE EXECUTION RESULT")
            output_parts.append("=" * 50)

            # Add status
            status = result.get("status", "Unknown")
            output_parts.append(f"Status: {status}")

            if result.get("message"):
                output_parts.append(f"Message: {result['message']}")

            # Add run result details
            run_result = result.get("run_result", {})
            if run_result:
                output_parts.append(f"Execution Status: {run_result.get('status', 'Unknown')}")
                output_parts.append(f"Execution Time: {run_result.get('execution_time', 'N/A')}s")
                output_parts.append(f"Return Code: {run_result.get('return_code', 'N/A')}")

                # Add stdout if present
                stdout = run_result.get("stdout", "")
                stdout = stdout.rstrip() or "<No stdout. Use print() to see the output>"
                output_parts.append("-" * 30)
                output_parts.append("STDOUT:")
                output_parts.append("-" * 30)
                output_parts.append(stdout.rstrip())

                # Add stderr if present
                stderr = run_result.get("stderr", "")
                stderr = stderr.rstrip() or "<No stderr>"
                output_parts.append("-" * 30)
                output_parts.append("STDERR:")
                output_parts.append("-" * 30)
                output_parts.append(stderr.rstrip())

            output_parts.append("=" * 50)

            return "\n".join(output_parts)

        except requests.exceptions.RequestException as e:
            return f"Error connecting to SandboxFusion API: {e}"
        except json.JSONDecodeError as e:
            return f"Error parsing SandboxFusion response: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

    return code_execution_tool


def assign_initial_state(state: GraphState, max_rounds: int) -> GraphState:
    """Passthrough node to assign initial state variables."""
    return {**state, "current_round": 0, "max_rounds": max_rounds, "needs_validation": False}


def react_agent_node(state: GraphState, react_agent) -> GraphState:
    """Node that runs the ReAct agent."""

    # Run the agent with current messages
    response = react_agent.invoke({"messages": state["messages"]})

    return {**state, "messages": response["messages"], "needs_validation": True}


def validation_node(state: GraphState) -> GraphState:
    """Node that appends a deterministic validation message."""

    validation_message = HumanMessage(
        content="""
Please carefully review your previous response and reasoning. Check the following:

1. Is your reasoning logically sound and well-connected?
2. Are all your calculations correct?
3. Did you use the tools appropriately?
4. Is your final answer accurate and complete?
5. Are there any errors or missing steps?

If you find any issues, please provide a corrected response. If everything looks correct, please confirm your answer is final.
"""
    )

    updated_messages = state["messages"] + [validation_message]

    return {**state, "messages": updated_messages, "current_round": state["current_round"] + 1, "needs_validation": False}


def create_reflection_agent(model: BaseLanguageModel, sandboxfusion_api: str, max_rounds: int = 2) -> StateGraph:
    """
    Create a reflection agent that uses deterministic validation messages.

    Args:
        llm: The language model to use for the ReAct agent
        sandboxfusion_api: SandboxFusion API endpoint or identifier
        max_rounds: Maximum number of reflection rounds

    Returns:
        Compiled StateGraph for the reflection agent
    """

    # Create tools with SandboxFusion API
    tools = [create_sandboxfusion_tool(sandboxfusion_api)]

    # Create the ReAct agent
    react_agent = create_react_agent(model, tools)

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("assign_state", lambda state: assign_initial_state(state, max_rounds))
    workflow.add_node("agent", lambda state: react_agent_node(state, react_agent))
    workflow.add_node("validation", validation_node)

    # Set entry point to assign_state
    workflow.set_entry_point("assign_state")

    # Add edge from assign_state to agent
    workflow.add_edge("assign_state", "agent")

    def should_continue_from_agent(state: GraphState) -> str:
        """Determine what to do after agent generates a response."""

        # If we need validation and haven't done max validations yet
        if state["needs_validation"] and state["current_round"] < state["max_rounds"] - 1:
            return "validate"

        # Otherwise end
        return "end"

    def should_continue_from_validation(state: GraphState) -> str:
        """After validation, always go back to agent for final response."""
        return "agent"

    # Add conditional edges
    workflow.add_conditional_edges("agent", should_continue_from_agent, {"validate": "validation", "end": END})

    workflow.add_conditional_edges("validation", should_continue_from_validation, {"agent": "agent"})

    # Compile the graph with fallback in case of recursion error
    app = workflow.compile()

    return app


def run_reflection_agent(question: str, llm: BaseLanguageModel, max_rounds: int = 3, sandboxfusion_api: str = None) -> dict:
    """
    Run the reflection agent with a given question.

    Args:
        question: The question to ask the agent
        llm: The language model to use
        max_rounds: Maximum number of reflection rounds
        sandboxfusion_api: SandboxFusion API endpoint or identifier

    Returns:
        Dictionary containing the final state and results
    """

    # Create the graph with SandboxFusion API
    app = create_reflection_agent(llm, sandboxfusion_api, max_rounds)

    # Initial state with only messages
    initial_state = {"messages": [HumanMessage(content=question)]}

    # Run the graph with recursion limit
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    # Extract final answer from the last AI message
    final_answer = ""
    for message in reversed(final_state["messages"]):
        if isinstance(message, AIMessage):
            final_answer = message.content
            break

    return {**final_state, "final_answer": final_answer}


# Example usage
if __name__ == "__main__":
    # Example question
    question = r"""
    Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

Find the largest possible real part of [(75+117i)z+\frac{96+144i}{z}]where $z$ is a complex number with $|z|=4$.

Remember to put your answer on its own line after "Answer:"."""

    print(f"Question: {question}")
    print("-" * 50)

    # Create a custom LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create mock SandboxFusion API
    sandbox_api = "http://localhost:8080"

    # Run the reflection agent with SandboxFusion API
    result = run_reflection_agent(question, llm=llm, max_rounds=2, sandboxfusion_api=sandbox_api)

    print(f"Final Answer: {result['final_answer']}")
    print(f"Rounds Completed: {result['current_round']}")

    # Print the conversation history
    print("\nConversation History:")
    for i, message in enumerate(result["messages"]):
        message_type = type(message).__name__
        content_preview = message.content
        tool_calls = hasattr(message, "tool_calls") and message.tool_calls
        if tool_calls:
            print(f"{i + 1}. {message_type} with Tool Calls: {tool_calls}")
        else:
            print(f"{i + 1}. {message_type}: {content_preview}")
