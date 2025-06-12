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

from typing import List, TypedDict

import numpy as np
import pytest
import ray
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from omegaconf import DictConfig, OmegaConf

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto


class SimpleGraphState(TypedDict):
    """State for the simple two-step graph."""
    messages: List[BaseMessage]


def llm_node(state: SimpleGraphState, model: BaseLanguageModel) -> SimpleGraphState:
    """Node that calls the LLM with current messages."""
    response = model.invoke(state["messages"])
    updated_messages = state["messages"] + [response]
    
    return {
        **state,
        "messages": updated_messages
    }


def revision_node(state: SimpleGraphState) -> SimpleGraphState:
    """Node that adds a revision prompt."""
    revision_message = HumanMessage(content="Please revise your previous response to make it more concise.")
    updated_messages = state["messages"] + [revision_message]
    
    return {
        **state,
        "messages": updated_messages
    }


def create_simple_graph(model: BaseLanguageModel) -> StateGraph:
    """Create a simple graph: LLM -> revision prompt -> LLM."""
    
    # Create the graph
    workflow = StateGraph(SimpleGraphState)
    
    # Add nodes
    workflow.add_node("llm_first", lambda state: llm_node(state, model))
    workflow.add_node("revision", revision_node)
    workflow.add_node("llm_second", lambda state: llm_node(state, model))
    
    # Set entry point
    workflow.set_entry_point("llm_first")
    
    # Add edges
    workflow.add_edge("llm_first", "revision")
    workflow.add_edge("revision", "llm_second")
    workflow.add_edge("llm_second", END)
    
    # Compile the graph
    app = workflow.compile()
    return app


@pytest.fixture
def init_config() -> DictConfig:
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.max_model_len = 12000
    config.actor_rollout_ref.rollout.chat_scheduler = "examples.grpo_trainer.langgraph_chat_scheduler.LangGraphChatCompletionScheduler"
    config.actor_rollout_ref.rollout.n = 4
    config.trainer.n_gpus_per_node = 2
    
    # Configure LangGraph settings
    config.actor_rollout_ref.rollout.langgraph = {
        "graph": {
            "_target_": "tests.workers.rollout.test_langgraph_chat_scheduler.create_simple_graph",
            "_partial_": True
        },
        "chat_template_kwargs": {
            "chat_template": open("tests/workers/rollout/resource/chat_templates/qwen2.5.jinja").read()
        },
        "graph_config": {}
    }
    
    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_langgraph_async_rollout_simple_graph(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    async_rollout_manager = init_async_rollout_manager(init_config)

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Explain what machine learning is in simple terms.",
            }
        ],
        [
            {
                "role": "user", 
                "content": "What are the benefits of renewable energy?"
            }
        ],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
        },
    )
    result = async_rollout_manager.generate_sequences(prompts=batch)

    # =========================== 3. Check results ===========================
    # check result shape
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert len(result) == 2 * 4
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len
    
    assert "loss_mask" in result.batch, "loss_mask should be in batch"
    assert result.batch["loss_mask"].sum() > 0, "loss_mask should be non-zero"

    # Check that we have raw_responses in non_tensor_batch
    assert "raw_responses" in result.non_tensor_batch
    raw_responses = result.non_tensor_batch["raw_responses"]
    assert len(raw_responses) == 2 * 4
    
    # Check number of messages in responses
    # Each should have: assistant response -> human revision -> assistant response
    # So 3 messages per conversation
    for i, response_messages in enumerate(raw_responses):
        print(f"Response {i} has {len(response_messages)} messages")
        assert len(response_messages) == 3, f"Expected 3 messages, got {len(response_messages)}"
    
        # Check message types
        assert response_messages[0].get("role") == "assistant", "First response message should be assistant"
        assert response_messages[1].get("role") == "user", "Second response message should be user (revision)"
        assert response_messages[2].get("role") == "assistant", "Third response message should be assistant"
        
        # Check that revision message contains expected content
        revision_content = response_messages[1].get("content", "")
        assert "revise" in revision_content.lower(), f"Revision message should contain 'revise': {revision_content}"
        
    print("Test passed!")
    print(f"Generated {len(result)} sequences with expected shapes and message counts")
    ray.shutdown() 
