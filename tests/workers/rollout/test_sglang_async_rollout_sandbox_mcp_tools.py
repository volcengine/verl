# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
from verl.tools.mcp_sandbox_tool import MCPSandboxTool
from verl.tools.utils.mcp_clients.McpClientManager import MCPClientManager
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


def get_sandbox_messages():
    user_prompt = {
        "role": "user",
        "content": """
            Solve the following problem step by step. You now have the ability to selectively 
            write executable Python code to enhance your reasoning process. \n\n**user question:**\nThere 
            are 152 students at Dala High School. Assume the following: \n- 100 students take a Math class \n- 94 
            students take a Science class \n- 57 students take an English class \n- 73 students take a Math class 
            and a Science class \n- 24 students take a Math class and an English class \n- 27 students take a Science 
            class and an English class \n- 22 students take a Math class and a Science class and an English class\n \nHow 
            many students take neither a Math class nor a Science class nor an Eglish class?\n\nRemember to place the final 
            answer in the last part using the format: \n<answer>\n\boxed{'The final answer goes here.'}\n</answer>
        """,
    }
    expect_turn_0_msg = {
        "role": "assistant",
        "content": """
            Okay, so I need to find out how many students at Dala High School are not taking any of the three classes: Math, 
            Science, or English. The total number of students is 152. Let me see... I remember this is a problem about sets 
            and maybe using the principle of inclusion-exclusion. Let me recall how that works.\n\nFirst, the inclusion-exclusion 
            principle for three sets says that the total number of students taking at least one of the classes is equal to the 
            sum of the numbers in each individual class, minus the sum of the numbers in each pair of classes, plus the number in 
            all three classes. Then, subtract that total from the overall number of students to get those not taking any of the 
            three. \n\nLet me write that down step by step. Let M be the set of students taking Math, S for Science, and E for English. 
            Then:\n\nTotal in at least one class = |M ∪ S ∪ E| = |M| + |S| + |E| - |M ∩ S| - |M ∩ E| - |S ∩ E| + |M ∩ S ∩ E|\n\nGiven the 
            numbers:\n\n|M| = 100\n\n|S| = 94\n\n|E| = 57\n\n|M ∩ S| = 73\n\n|M ∩ E| = 24\n\n|S ∩ E| = 27\n\n|M ∩ S ∩ E| = 22\n\nSo plugging 
            these into the formula:\n\nTotal = 100 + 94 + 57 - 73 - 24 - 27 + 22\n\nLet me compute that step by step using code to ensure 
            accuracy.\n
        """,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "arguments": {
                        "codeStr": """M = 100\\nS = 94\\nE = 57\\nM_S = 73\\nM_E = 24\\nS_E = 27\\nM_S_E = 
                        22\\n\\ntotal_in_any = M + S + E - M_S - M_E - S_E + M_S_E\\nstudents_neither = 152 - total_in_any\\nprint(students_neither)""",
                        "language": "python",
                    },
                },
            }
        ],
    }

    tool_return_0_msg = {
        "role": "tool",
        "content": """3""",
    }

    expect_turn_1_msg = {
        "role": "assistant",
        "content": """
            The code returns 3, indicating that 3 students take none of the classes. To verify, I'll break down each
            category to ensure the total matches:\n\n- Only Math and Science: 73 - 22 = 51\n- Only Math and English:
            24 - 22 = 2\n- Only Science and English: 27 - 22 = 5\n- Only Math: 100 - 51 - 2 - 22 = 25\n- Only Science:
            94 - 51 - 5 - 22 = 16\n- Only English: 57 - 2 - 5 - 22 = 28\n\nSumming all categories:\n
        """,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "arguments": {
                        "code": """only_M_S = 73 - 22\\nonly_M_E = 24 - 22\\nonly_S_E = 27 - 22\\n\\nonly_M = 100 - only_M_S -
                        only_M_E - 22\\nonly_S = 94 - only_M_S - only_S_E - 22\\nonly_E = 57 - only_M_E - only_S_E - 22\\n\\ntotal_verify
                        = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22\\nprint(total_verify)""",
                        "language": "python",
                    },
                },
            }
        ],
    }

    tool_return_1_msg = {
        "role": "tool",
        "content": """149""",
    }
    expect_turn_2_msg = {
        "role": "assistant",
        "content": """
            The verification total is 149, so students not taking any classes are 152 - 149 = 3, confirming the initial
            result.\n\n<answer>\n\\boxed{3}\n</answer>
        """,
    }

    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg, expect_turn_2_msg]
    tool_return_array = [tool_return_0_msg, tool_return_1_msg]

    return user_prompts, expect_turn_array, tool_return_array


class TestRolloutWithMCPSandboxTools:
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
    def sandbox_tool_schemas(self):
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": "run your code str in sandbox server with your provided language,...",
                    "parameters": {
                        "properties": {
                            "codeStr": {
                                "title": "codeStr",
                                "type": "string",
                            },
                            "language": {
                                "title": "language",
                                "type": "string",
                            },
                        },
                        "required": ["codeStr", "language"],
                        "title": "run_codeArguments",
                        "type": "object",
                    },
                    "strict": False,
                },
            }
        ]
        return tool_schemas

    @pytest.fixture
    def sandbox_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_sandbox_messages()
        prompts = [[message] for message in user_prompt]
        preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
        preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def sandbox_rollout_config(self):
        max_prompt_length = 4096
        max_response_length = 3000
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/sandbox_mcp_tool_config"
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
        return rollout_config

    @pytest.fixture
    def sandbox_data_proto(self, sandbox_data, qwen_tokenizer):
        preencode_prompts, _, _ = sandbox_data
        prompts = [qwen_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
        input_ids, attention_mask, position_ids = prepare_inputs(qwen_tokenizer, prompts, 1000)
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        messages = np.asarray(preencode_prompts)

        tools_kwargs = np.array(
            [
                {
                    "run_code": {
                        "create_kwargs": {"ground_truth": "test-solution-str"},
                    },
                }
            ],
            dtype=object,
        )
        index = np.array([0], dtype=object)
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})
        return prompts

    @pytest.fixture
    def mock_rollout(self, sandbox_rollout_config, qwen_tokenizer, qwen_model_config):
        """Mock the rollout instance with sampling_params initialized."""
        tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": "run your code str in sandbox server with your provided language,...",
                    "parameters": {
                        "properties": {
                            "codeStr": {
                                "title": "codeStr",
                                "type": "string",
                            },
                            "language": {
                                "title": "language",
                                "type": "string",
                            },
                        },
                        "required": ["codeStr", "language"],
                        "title": "run_codeArguments",
                        "type": "object",
                    },
                    "strict": False,
                },
            }
        ]
        with patch.object(MCPClientManager, "fetch_tool_schemas", return_value=tool_schema), patch.object(MCPClientManager, "initialize", return_value=None), patch.object(SGLangRollout, "_init_distributed_env", return_value=None), patch.object(
            SGLangRollout, "_init_inference_engine", return_value=None
        ), patch.object(SGLangRollout, "_init_sampling_params", return_value=None):
            rollout = SGLangRollout(actor_module="", config=sandbox_rollout_config, processing_class=qwen_tokenizer, model_hf_config=qwen_model_config)
            rollout.sampling_params = {
                "n": 1,
                "max_new_tokens": sandbox_rollout_config.response_length,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            }
            return rollout

    def test_tools_registration(self, mock_rollout):
        assert len(mock_rollout._tool_schemas) != 0
        assert "run_code" in mock_rollout._tool_map.keys()
        from verl.tools.mcp_sandbox_tool import MCPSandboxTool

        assert isinstance(mock_rollout._tool_map["run_code"], MCPSandboxTool)
        # depend on the tokenizer
        assert mock_rollout._tool_call_parser_type == "qwen25"

    def test_rollout_req_creation(self, mock_rollout, sandbox_data_proto):
        req_list = mock_rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tool_schemas) == 1

    def test_over_size_case(self, mock_rollout, sandbox_data_proto, sandbox_data):
        mock_rollout.config.multi_turn.max_assistant_turns = 1
        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, _ = sandbox_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        mock_rollout._handle_engine_call = MagicMock()
        future = asyncio.Future()
        future.set_result({"text": expect_turn_array[0], "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "length", "length": 1024}, "prompt_tokens": 132, "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
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
        assert output_req.reward_scores.get("run_code") == []
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @patch.object(MCPSandboxTool, "execute", new_callable=AsyncMock)
    def test_tool_call_basic_case(self, mock_execute, mock_rollout, sandbox_data_proto, sandbox_data):
        _, expect_turn_array, tool_return_array = sandbox_data
        # Mock sandbox tool execution to return predefined responses
        mock_execute.side_effect = [(msg, 0.0, {"status": "success"}) for msg in tool_return_array]

        mock_rollout.config.multi_turn.max_assistant_turns = 10
        req = mock_rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        mock_rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
            i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
            if idx < len(expect_turn_array) - 1:
                assert mock_rollout._function_call_parser.has_tool_call(turn)
                assert mock_rollout._function_call_parser.parse_non_stream(turn)

        mock_rollout._handle_engine_call.side_effect = futures
        mock_rollout._tp_rank = 0

        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(asyncio.gather(*[mock_rollout._async_rollout_a_request(req, True, False) for req in req_list]))

        # Verify conversation completed successfully with proper tool usage
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert "run_code" in output_req.metrics
        assert output_req.metrics["run_code"][0]["status"] == "success"
        assert mock_execute.await_count == 2
        assert len(output_req.messages) == 6
        # Verify tool response messages contain expected content
        sandbox_counter = 0
        for msg in output_req.messages:
            if msg.role == "tool":
                assert msg.content == tool_return_array[sandbox_counter]
                sandbox_counter += 1
        assert sandbox_counter == 2

    @patch.object(MCPSandboxTool, "execute", new_callable=AsyncMock)
    def test_tool_call_batch_case(self, mock_execute, mock_rollout, sandbox_data_proto, sandbox_data):
        _, expect_turn_array, tool_return_array = sandbox_data
        # Mock tool execution for large batch (100 requests * 2 calls each)
        mock_execute.side_effect = [
            (tool_return_array[0], 0.0, {"status": "success"}),
            (tool_return_array[1], 0.0, {"status": "success"}),
        ] * 100

        mock_rollout.config.multi_turn.max_assistant_turns = 10
        base_req = mock_rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]

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
            output_req_list = loop.run_until_complete(asyncio.gather(*[mock_rollout._async_rollout_a_request(r, True, False) for r in req_list]))

        # Verify all requests completed successfully
        assert len(output_req_list) == req_nums
        for out_req in output_req_list:
            assert out_req.state == AsyncRolloutRequestStateEnum.COMPLETED
            assert "run_code" in out_req.metrics
            for metric in out_req.metrics["run_code"]:
                assert metric["status"] == "success"
            assert len(out_req.messages) == 6
            assert sum(1 for m in out_req.messages if m.role == "tool") == 2

        assert mock_execute.await_count == 2 * req_nums
