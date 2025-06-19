# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import time
from copy import deepcopy
from functools import wraps
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ray
from tensordict import TensorDict
from torch.testing._internal.common_distributed import MultiProcessTestCase
from transformers import AutoConfig, AutoTokenizer
from utils_sglang import (
    get_rollout_config,
    prepare_inputs,
)

from verl.protocol import DataProto
from verl.tools.sandbox_fusion_tools import TokenBucketWorker
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, Message
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

sandbox_url = ""


def get_sandbox_fusion_messages():
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
                    "name": "code_interpreter",
                    "arguments": {
                        "code": """M = 100\\nS = 94\\nE = 57\\nM_S = 73\\nM_E = 24\\nS_E = 27\\nM_S_E = 
                        22\\n\\ntotal_in_any = M + S + E - M_S - M_E - S_E + M_S_E\\nstudents_neither = 152 - total_in_any\\nprint(students_neither)""",
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
                    "name": "code_interpreter",
                    "arguments": {
                        "code": """only_M_S = 73 - 22\\nonly_M_E = 24 - 22\\nonly_S_E = 27 - 22\\n\\nonly_M = 100 - only_M_S -
                        only_M_E - 22\\nonly_S = 94 - only_M_S - only_S_E - 22\\nonly_E = 57 - only_M_E - only_S_E - 22\\n\\ntotal_verify
                        = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22\\nprint(total_verify)""",
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


def skip_if_valid_sandbox(url):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if url == "" or url is None:
                pytest.skip("No valid sandbox url provided")

        return wrapper

    return decorator


class TestRolloutWithTools:
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
    def sandbox_fusion_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_sandbox_fusion_messages()
        prompts = [[message] for message in user_prompt]
        preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
        preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def sandbox_fusion_rollout_config(self):
        max_prompt_length = 1024
        max_response_length = 1024
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/sandbox_fusion_tool_config"
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
        return rollout_config

    @pytest.fixture
    def sandbox_data_proto(self, sandbox_fusion_data, qwen_tokenizer):
        preencode_prompts, _, _ = sandbox_fusion_data
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
                    "code_interpreter": {
                        "create_kwargs": {"ground_truth": "test-solution-str"},
                    },
                }
            ],
            dtype=object,
        )
        index = np.array([0], dtype=object)
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})
        return prompts

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tools_registration(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config):
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        assert len(rollout._tool_schemas) == 1
        assert "code_interpreter" in rollout._tool_map.keys()
        from verl.tools.sandbox_fusion_tools import SandboxFusionTool

        assert isinstance(rollout._tool_map["code_interpreter"], SandboxFusionTool)
        assert rollout._tool_call_parser_type == "qwen25"

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_rollout_req_creation(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto):
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tool_schemas) == 1
        print(type(req_list[0].tool_schemas[0]))
        assert req_list[0].tool_schemas[0] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="code_interpreter",
                description="A tool for executing code.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "code": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The code to execute.",
                            enum=None,
                        )
                    },
                    required=["code"],
                ),
                strict=False,
            ),
        )

    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_over_size_case(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data):
        sandbox_fusion_rollout_config.multi_turn.max_turns = 1
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, tool_return_array = sandbox_fusion_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call = MagicMock()
        future = asyncio.Future()
        future.set_result({"text": expect_turn_array[0], "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "length", "length": 1024}, "prompt_tokens": 132, "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
        rollout._handle_engine_call.return_value = future
        rollout._tp_rank = 0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert output_req.reward_scores == {"code_interpreter": []}
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @skip_if_valid_sandbox(sandbox_url)
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_basic_case(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data):
        sandbox_fusion_rollout_config.multi_turn.max_turns = 10
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        self._tool_map["code_interpreter"].sandbox_fusion_url = sandbox_url
        req = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]
        _, expect_turn_array, tool_return_array = sandbox_fusion_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
            i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
            if idx < len(expect_turn_array) - 1:
                assert rollout._function_call_parser.has_tool_call(turn)
                assert rollout._function_call_parser.parse_non_stream(turn)

        rollout._handle_engine_call.side_effect = futures
        rollout._tp_rank = 0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        # here we verify whether the code sandbox is executed correctly
        assert output_req.metrics == {"code_interpreter": ["3", "149"]}
        assert rollout._handle_engine_call.call_count == 3
        assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
        code_counter = 0
        for msg in output_req.messages:
            if msg.role == "tool":
                code_counter += 1
                assert msg.content == tool_return_array[code_counter]
        assert code_counter == 2

    @skip_if_valid_sandbox(sandbox_url)
    @patch.object(SGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(SGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(SGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_batch_case(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data):
        sandbox_fusion_rollout_config.multi_turn.max_turns = 10
        rollout = SGLangRollout(actor_module="", config=sandbox_fusion_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        self._tool_map["code_interpreter"].sandbox_fusion_url = sandbox_url
        req = rollout._preprocess_prompt_to_async_rollout_requests(sandbox_data_proto, n=1)[0]
        req_nums = 100
        req_list = []
        req_turns_counter = {}
        # this map should a Map[id:List[Futures]]
        req_turns_map = {}
        _, expect_turn_array, tool_return_array = sandbox_fusion_data
        for i in range(req_nums):
            _temp_req = deepcopy(req)
            _temp_req.batch_data_id = i
            _temp_req.request_id = i
            req_list.append(MagicMock(wraps=_temp_req, spec=AsyncRolloutRequest))
            futures = [asyncio.Future() for i in expect_turn_array]
            for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
                i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 9.9304039478302}})
                if idx < len(expect_turn_array) - 1:
                    assert rollout._function_call_parser.has_tool_call(turn)
                    assert rollout._function_call_parser.parse_non_stream(turn)
            req_turns_map[_temp_req.batch_data_id] = futures
            req_turns_counter[_temp_req.batch_data_id] = 0

        async def hacked_handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, **kwargs):
            result = req_turns_map[_req.batch_data_id][req_turns_counter[_req.batch_data_id]]
            req_turns_counter[_req.batch_data_id] += 1
            re = await result
            return re

        with patch.object(SGLangRollout, "_handle_engine_call", new=hacked_handle_engine_call):
            rollout._tp_rank = 0
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
                )
            )
            assert len(output_req_list) == req_nums
            # FIGUER out how to count this
            # assert rollout._handle_engine_call.call_count == 3 * req_nums
            for output_req in output_req_list:
                assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
                # here we verify whether the code sandbox is executed correctly
                assert output_req.metrics == {"code_interpreter": ["3", "149"]}
                assert len(output_req.messages) == 6  # user + 3*assistant + 2*tool_call
                code_counter = 0
                for msg in output_req.messages:
                    if msg.role == "tool":
                        code_counter += 1
                assert code_counter == 2


class RayMultiProcessTestCase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        ray.init(ignore_reinit_error=True)
        print("init_single cluster")
        self._spawn_processes()

    def tearDown(self):
        print("tearDown_single cluster")
        ray.shutdown()


@ray.remote
class TestActor:
    def __init__(self, rank, world_size):
        self._world_size = world_size
        self._rank = rank
        self.rank_list = []
        self.time_list = []

    def record_rank(self, rank):
        self.rank_list.append(rank)

    def get_rank(self):
        return self._rank

    def ping(self):
        return True

    def record_execution_time(self, time):
        self.time_list.append(time)

    def get_time(self, timeout):
        import time

        now = time.time()
        while time.time() - now < timeout:
            # for start and end time
            if len(self.time_list) == self._world_size * 2:
                self.time_list.sort()
                return self.time_list[-1] - self.time_list[0]
            else:
                time.sleep(1)
                continue
        return False

    def verify_rank(self):
        import time

        now = time.time()
        while time.time() - now < 10:
            if len(self.rank_list) == self._world_size:
                print(self.rank_list)
                self.rank_list.sort()
                for i in range(self._world_size):
                    if self.rank_list[i] != i:
                        return False
                return True
            else:
                time.sleep(1)
                continue
        return False


class TestRayGlobalActorCase(RayMultiProcessTestCase):
    @property
    def world_size(self) -> int:
        # for DP = 8
        return 2

    def test_basic_multi_process_init(self):
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        handle = TestActor.remote(self.rank, self.world_size)
        re = ray.get(handle.get_rank.remote())
        assert re == self.rank, f"rank not match: {re} != {self.rank}"

    # def test_global_actor(self):
    #     ray.init("auto",namespace="test",ignore_reinit_error=True)
    #     handle = TestActor.options(get_if_exists=True,name="test-actor").remote(self.rank,self.world_size)
    #     handle.record_rank.remote(self.rank)
    #     # since test actor's concurrency is 1, we need to wait for all processes to finish
    #     time.sleep(5)
    #     assert ray.get(handle.ping.remote()) == True # make sure actor handle is valid
    #     if self.rank == 0:
    #         assert ray.get(handle.verify_rank.remote()) == True
    #     else:
    #         # get_actor use weak_ref, so we need to make sure the actor is not garbage collected
    #         time.sleep(10)


class TestSingleNodeRateLimiterCase(RayMultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_rate_limiter(self):
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        from verl.tools.sandbox_fusion_tools import PoolMode, init_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=3)
        exec_worker = init_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=3, mode=PoolMode.ThreadMode)
        center = TestActor.options(get_if_exists=True, name="test-actor").remote(self.rank, self.world_size)
        ray.get(exec_worker.ping.remote())

        def fn(i):
            import time

            time.sleep(3)
            return i

        start = time.time()
        tasks = [exec_worker.execute.remote(fn, i) for i in range(6)]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        end = time.time()
        duration = end - start
        center.record_execution_time.remote(start)
        center.record_execution_time.remote(end)
        print(f"Total time: {duration:.2f} seconds for rank: {self.rank}")

        assert results == list(range(6))
        # we have 6 task with rate limit of 3, therefore we need at least 2 round: 3*2=6 seconds
        assert duration > 6
        assert duration < 10

    def test_rotten_execution(self):
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        from verl.tools.sandbox_fusion_tools import PoolMode, init_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=6)
        exec_worker = init_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=6, mode=PoolMode.ThreadMode)
        ray.get(exec_worker.ping.remote())

        def fn(i):
            if i == 10:
                raise Exception("test")
            else:
                return i

        tasks = [exec_worker.execute.remote(fn, i) for i in range(20)]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        expect_result = [None] + list(range(10)) + list(range(11, 20))
        sorted_data = sorted(results, key=lambda x: (x is not None, x))
        assert sorted_data == expect_result, f"results: {results}, expect_result: {expect_result}"
        rate_limiter = TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote()
        rate = ray.get(rate_limiter.get_current_count.remote())
        assert rate == 0, f"rate: {rate}"


class TestMultiNodeRateLimiterCase(RayMultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def test_rate_limiter(self):
        ray.init("auto", namespace="test", ignore_reinit_error=True)
        from verl.tools.sandbox_fusion_tools import PoolMode, init_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=6)
        exec_worker = init_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=6, mode=PoolMode.ThreadMode)
        center = TestActor.options(get_if_exists=True, name="test-actor").remote(self.rank, self.world_size)
        ray.get(exec_worker.ping.remote())

        def fn(i):
            import time

            time.sleep(2)
            return i

        start = time.time()
        tasks = [exec_worker.execute.remote(fn, i) for i in range(6)]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        end = time.time()
        duration = end - start
        center.record_execution_time.remote(start)
        center.record_execution_time.remote(end)
        print(f"Total time: {duration:.2f} seconds for rank: {self.rank}")
        assert results == list(range(6))
        time.sleep(5)
        if self.rank == 0:
            total_cost = ray.get(center.get_time.remote(10))
            print(f"for total cost: {total_cost}")
            # # we have 6 task each node * 2node = 12 task, each task take 2 second.
            # with rate limit of 6,
            # therefore we need at least 2 round: 12/6*2=4 seconds
            assert total_cost > 4, total_cost
        else:
            time.sleep(10)
