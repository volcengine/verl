# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# Adapted from tests/workers/rollout/test_sglang_async_rollout_sf_tools.py

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer
from utils_sglang import (
    get_rollout_config,
    prepare_inputs,
)

from verl.protocol import DataProto
from verl.tools.benchmax_tool import BenchmaxToolAdapter
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


def get_benchmax_messages():
    """
    Conversation scaffold for the WikipediaEnv environment.
    """

    # ── Turn 0 ───────────────────────────────────────────────────────────────
    user_prompt = {
        "role": "user",
        "content": "What is the capital city of France?",
    }

    expect_turn_0_msg = {
        "role": "assistant",
        "content": (
            "I'll search Wikipedia for the article on **France** to confirm its capital."
        ),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "arguments": {"q": "France", "limit": 5},
                },
            }
        ],
    }

    tool_return_0_msg = {
        "role": "tool",
        "content": """
[
  {
    "title": "France",
    "snippet": "France, officially the French Republic, is a country in Western Europe …"
  },
  {
    "title": "Paris",
    "snippet": "Paris is the capital and most populous city of France …"
  }
]
""",
    }

    # ── Turn 1 ───────────────────────────────────────────────────────────────
    expect_turn_1_msg = {
        "role": "assistant",
        "content": (
            "The top hit is the article “France”. I'll open that to read its summary."
        ),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_wikipedia_article",
                    "arguments": {"title": "France"},
                },
            }
        ],
    }

    tool_return_1_msg = {
        "role": "tool",
        "content": """
{
  "title": "France",
  "extract": "France, officially the French Republic, is a country in Western Europe. Its capital is Paris."
}
""",
    }

    # ── Turn 2 (final answer) ────────────────────────────────────────────────
    expect_turn_2_msg = {
        "role": "assistant",
        "content": (
            "The summary states that France’s capital is **Paris**.\n\n"
            "<answer>Paris</answer>"
        ),
    }

    # Assemble arrays in the format expected by the evaluator
    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    return user_prompts, expect_turn_array, tool_return_array



class TestRolloutWithBenchmaxTools:
    @pytest.fixture
    def qwen_tokenizer(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # we only need this for tokenizer
    @pytest.fixture
    def qwen_model_config(self):
        local_model_path = "Qwen/Qwen2.5-0.5B"
        config = AutoConfig.from_pretrained(local_model_path)
        return config

    @pytest.fixture
    def benchmax_tool_schemas(self):
        """Schemas for the WikipediaEnv tools."""
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Search Wikipedia articles by keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "Query string to search for.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max number of results (default 10).",
                            },
                        },
                        "required": ["q"],
                        "title": "search_wikipediaArguments",
                    },
                    "strict": False,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_wikipedia_article",
                    "description": "Fetch the summary paragraph for a Wikipedia article.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Exact title of the Wikipedia article.",
                            },
                        },
                        "required": ["title"],
                        "title": "get_wikipedia_articleArguments",
                    },
                    "strict": False,
                },
            },
        ]
        return tool_schemas

    @pytest.fixture
    def benchmax_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_benchmax_messages()
        prompts = [[message] for message in user_prompt]
        preencode_turn_array = [
            qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False)
            for turn in expect_turn_array
        ]
        preencode_tool_return_array = [
            qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True)
            for turn in tool_return_array
        ]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def benchmax_rollout_config(self):
        max_prompt_length = 4096
        max_response_length = 3000
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/benchmax_tool_config"
        rollout_config = get_rollout_config(
            max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path
        )
        return rollout_config

    @pytest.fixture
    def benchmax_data_proto(self, benchmax_data, qwen_tokenizer):
        preencode_prompts, _, _ = benchmax_data
        prompts = [
            qwen_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in preencode_prompts
        ]
        input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
        tools_kwargs = np.array(
            [{}],
            dtype=object,
        )
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        messages = np.asarray(preencode_prompts)
        index = np.array([0], dtype=object)
        prompts = DataProto(
            batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index}
        )
        return prompts

    @pytest.fixture
    def mock_rollout(self, benchmax_rollout_config, qwen_tokenizer, qwen_model_config):
        """Mock the rollout instance with sampling_params initialized."""
        with patch.object(SGLangRollout, "_init_distributed_env", return_value=None), patch.object(
            SGLangRollout, "_init_inference_engine", return_value=None
        ), patch.object(SGLangRollout, "_init_sampling_params", return_value=None):
            rollout = SGLangRollout(
                actor_module="",
                config=benchmax_rollout_config,
                processing_class=qwen_tokenizer,
                model_hf_config=qwen_model_config,
            )
            rollout.sampling_params = {
                "n": 1,
                "max_new_tokens": benchmax_rollout_config.response_length,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            }
            return rollout

    def test_tools_registration(self, mock_rollout):
        assert len(mock_rollout._tool_schemas) != 0
        assert "get_wikipedia_article" in mock_rollout._tool_map.keys()
        assert "search_wikipedia" in mock_rollout._tool_map.keys()

        assert isinstance(mock_rollout._tool_map["get_wikipedia_article"], BenchmaxToolAdapter)
        assert isinstance(mock_rollout._tool_map["search_wikipedia"], BenchmaxToolAdapter)
        # depend on the tokenizer
        assert mock_rollout._tool_call_parser_type == "qwen25"

    def test_rollout_req_creation(self, mock_rollout, benchmax_data_proto):
        req_list = mock_rollout._preprocess_prompt_to_async_rollout_requests(benchmax_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tool_schemas) == 2

    def test_over_size_case(self, mock_rollout, benchmax_data_proto, benchmax_data):
        mock_rollout.config.multi_turn.max_assistant_turns = 1
        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(benchmax_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, _ = benchmax_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        mock_rollout._handle_engine_call = MagicMock()
        future = asyncio.Future()
        future.set_result(
            {
                "text": expect_turn_array[0],
                "meta_info": {
                    "id": "d1188d81cba840359df5b352b344bc8e",
                    "finish_reason": {"type": "length", "length": 1024},
                    "prompt_tokens": 132,
                    "completion_tokens": 100,
                    "cached_tokens": 0,
                    "e2e_latency": 9.9304039478302,
                },
            }
        )
        mock_rollout._handle_engine_call.return_value = future
        mock_rollout._tp_rank = 0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[mock_rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert output_req.reward_scores.get("search_wikipedia") == 0.0
        assert output_req.reward_scores.get("get_wikipedia_article") == 0.0
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @patch.object(BenchmaxToolAdapter, "execute", new_callable=AsyncMock)
    def test_tool_call_basic_case(self, mock_execute, mock_rollout, benchmax_data_proto, benchmax_data):
        _, expect_turn_array, tool_return_array = benchmax_data
        # Mock benchmax env tool execution to return predefined responses
        mock_execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]

        mock_rollout.config.multi_turn.max_assistant_turns = 10
        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(benchmax_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        mock_rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
            i.set_result(
                {
                    "text": turn,
                    "meta_info": {
                        "id": "d1188d81cba840359df5b352b344bc8e",
                        "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"},
                        "prompt_tokens": len(turn),
                        "completion_tokens": 100,
                        "cached_tokens": 0,
                        "e2e_latency": 9.9304039478302,
                    },
                }
            )
            if idx < len(expect_turn_array) - 1:
                assert mock_rollout._function_call_parser.has_tool_call(turn)
                assert mock_rollout._function_call_parser.parse_non_stream(turn)

        mock_rollout._handle_engine_call.side_effect = futures
        mock_rollout._tp_rank = 0

        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(*[mock_rollout._async_rollout_a_request(req, True, False) for req in req_list])
        )

        # Verify conversation completed successfully with proper tool usage
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert mock_execute.await_count == 2
        assert len(output_req.messages) == 6
        # Verify tool response messages contain expected content
        benchmax_counter = 0
        for msg in output_req.messages:
            if msg.role == "tool":
                assert msg.content == tool_return_array[benchmax_counter]
                benchmax_counter += 1
        assert benchmax_counter == 2

    @patch.object(BenchmaxToolAdapter, "execute", new_callable=AsyncMock)
    def test_tool_call_batch_case(self, mock_execute, mock_rollout, benchmax_data_proto, benchmax_data):
        _, expect_turn_array, tool_return_array = benchmax_data
        # Mock tool execution for large batch (100 requests * 2 calls each)
        mock_execute.side_effect = [
            (tool_return_array[0], 0.0, {"status": "success"}),
            (tool_return_array[1], 0.0, {"status": "success"}),
        ] * 100

        mock_rollout.config.multi_turn.max_assistant_turns = 10
        base_req = mock_rollout._preprocess_prompt_to_async_rollout_requests(benchmax_data_proto, n=1)[0]

        req_nums = 100
        req_list = []
        req_turns_map = {}
        req_turns_counter = {}

        for i in range(req_nums):
            tmp_req = deepcopy(base_req)
            tmp_req.batch_data_id = i
            tmp_req.request_id = i
            req_list.append(MagicMock(wraps=tmp_req, spec=AsyncRolloutRequest))

            futures = [asyncio.Future() for _ in expect_turn_array]
            for idx, (fut, turn) in enumerate(zip(futures, expect_turn_array)):
                fut.set_result(
                    {
                        "text": turn,
                        "meta_info": {
                            "id": "dummy",
                            "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"},
                            "prompt_tokens": len(turn),
                            "completion_tokens": 100,
                        },
                    }
                )
            req_turns_map[i] = futures
            req_turns_counter[i] = 0

        async def hacked_handle_engine_call(self, _req: AsyncRolloutRequest, *_args, **_kwargs):
            fut = req_turns_map[_req.batch_data_id][req_turns_counter[_req.batch_data_id]]
            req_turns_counter[_req.batch_data_id] += 1
            return await fut

        with patch.object(SGLangRollout, "_handle_engine_call", new=hacked_handle_engine_call):
            mock_rollout._tp_rank = 0
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(*[mock_rollout._async_rollout_a_request(r, True, False) for r in req_list])
            )

        # Verify all requests completed successfully
        assert len(output_req_list) == req_nums
        for out_req in output_req_list:
            assert out_req.state == AsyncRolloutRequestStateEnum.COMPLETED
            assert len(out_req.messages) == 6
            assert sum(1 for m in out_req.messages if m.role == "tool") == 2

        assert mock_execute.await_count == 2 * req_nums
