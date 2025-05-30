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
from verl.workers.rollout.sglang_rollout.async_sglang_rollout import AsyncSGLangRollout

DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + "Please check today's weather."

search_url = ""


def get_search_messages():
    user_prompt = {
        "role": "user",
        "content": user_content,
    }

    expect_turn_0_msg = {"role": "assistant", "content": "Let me search the web.", "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": "today's weather"}}}]}
    tool_return_0_msg = {"role": "tool", "content": "Today's weather in Beijing is sunny."}
    expect_turn_1_msg = {"role": "assistant", "content": "The weather in Beijing is sunny."}
    user_prompts = [user_prompt]
    expect_turn_array = [expect_turn_0_msg, expect_turn_1_msg]
    tool_return_array = [tool_return_0_msg]

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
    def search_data(self, qwen_tokenizer):
        user_prompt, expect_turn_array, tool_return_array = get_search_messages()
        prompts = [[message] for message in user_prompt]
        preencode_turn_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False) for turn in expect_turn_array]
        preencode_tool_return_array = [qwen_tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True) for turn in tool_return_array]
        return prompts, preencode_turn_array, preencode_tool_return_array

    @pytest.fixture
    def search_rollout_config(self):
        max_prompt_length = 4096
        max_response_length = 3000
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/search_tool_config"
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_path)
        return rollout_config

    @pytest.fixture
    def search_data_proto(self, search_data, qwen_tokenizer):
        preencode_prompts, _, _ = search_data
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
                    "search": {
                        "create_kwargs": {"ground_truth": "sunny", "question": "Please check today's weather.", "data_source": "searchR1_nq"},
                    },
                }
            ],
            dtype=object,
        )
        index = np.array([0], dtype=object)
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages, "tools_kwargs": tools_kwargs, "index": index})
        return prompts

    @patch.object(AsyncSGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_sampling_params", return_value=None)
    def test_tools_registration(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config):
        rollout = AsyncSGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        assert len(rollout._tool_schemas) == 1
        assert "search" in rollout._tool_map.keys()
        from verl.tools.search_tool import SearchTool

        assert isinstance(rollout._tool_map["search"], SearchTool)
        # depend on the tokenizer
        assert rollout._tool_call_parser_type == "qwen25"

    @patch.object(AsyncSGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_sampling_params", return_value=None)
    def test_rollout_req_creation(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto):
        rollout = AsyncSGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert len(req_list[0].tools) == 1
        print(type(req_list[0].tools[0]))
        assert req_list[0].tools[0] == OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search",
                description="Searches the web for relevant information based on the given query.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query_list": OpenAIFunctionPropertySchema(
                            type="array",
                            description="A list of fully-formed semantic queries. The tool will return search results for each query.",
                            items={"type": "string"},
                        )
                    },
                    required=["query_list"],
                ),
                strict=False,
            ),
        )

    @patch.object(AsyncSGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_sampling_params", return_value=None)
    def test_over_size_case(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data):
        search_rollout_config.multi_turn.max_turns = 1
        rollout = AsyncSGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]

        _, expect_turn_array, tool_return_array = search_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call = MagicMock()
        future = asyncio.Future()
        future.set_result({"text": expect_turn_array[0], "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "length", "length": 3000}, "prompt_tokens": 132, "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
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
        assert output_req.reward_scores == {"search": []}, f"output_req.reward_scores: {output_req.reward_scores}"
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )

    @skip_if_valid_sandbox(search_url)
    @patch.object(AsyncSGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_basic_case(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data):
        search_rollout_config.multi_turn.max_turns = 10
        rollout = AsyncSGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        self._tool_map["search"].retrieval_service_url = search_url
        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]
        _, expect_turn_array, tool_return_array = search_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call = MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
            i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
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
        # here we verify the search tool is executed correctly
        assert "search" in output_req.metrics
        assert isinstance(output_req.metrics["search"], list)
        assert output_req.metrics["search"][0]["status"] == "success"
        assert rollout._handle_engine_call.call_count == 2
        assert len(output_req.messages) == 4  # user + 2*assistant + 1*tool_call
        search_counter = 0
        for msg in output_req.messages:
            if msg.role == "tool":
                search_counter += 1
                assert msg.content == tool_return_array[search_counter]
        assert search_counter == 1

    @skip_if_valid_sandbox(search_url)
    @patch.object(AsyncSGLangRollout, "_init_distributed_env", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_inference_engine", return_value=None)
    @patch.object(AsyncSGLangRollout, "_init_sampling_params", return_value=None)
    def test_tool_call_batch_case(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data):
        search_rollout_config.multi_turn.max_turns = 10
        rollout = AsyncSGLangRollout(actor_module="", config=search_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        self._tool_map["search"].retrieval_service_url = search_url
        req = rollout._preprocess_prompt_to_async_rollout_requests(search_data_proto, n=1)[0]
        req_nums = 100
        req_list = []
        req_turns_counter = {}
        # this map should a Map[id:List[Futures]]
        req_turns_map = {}
        _, expect_turn_array, tool_return_array = search_data
        for i in range(req_nums):
            _temp_req = deepcopy(req)
            _temp_req.batch_data_id = i
            _temp_req.request_id = i
            req_list.append(MagicMock(wraps=_temp_req, spec=AsyncRolloutRequest))
            futures = [asyncio.Future() for i in expect_turn_array]
            for idx, (i, turn) in enumerate(zip(futures, expect_turn_array)):
                i.set_result({"text": turn, "meta_info": {"id": "d1188d81cba840359df5b352b344bc8e", "finish_reason": {"type": "tool_calls" if idx < len(expect_turn_array) - 1 else "stop"}, "prompt_tokens": len(turn), "completion_tokens": 100, "cached_tokens": 0, "e2e_latency": 2.23543}})
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

        with patch.object(AsyncSGLangRollout, "_handle_engine_call", new=hacked_handle_engine_call):
            rollout._tp_rank = 0
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
                )
            )
            assert len(output_req_list) == req_nums
            # FIGUER out how to count this
            # assert rollout._handle_engine_call.call_count == 2 * req_nums
            for output_req in output_req_list:
                assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
                # here we verify the search tool is executed correctly
                assert "search" in output_req.metrics
                assert isinstance(output_req.metrics["search"], list)
                assert output_req.metrics["search"][0]["status"] == "success"
                assert rollout._handle_engine_call.call_count == 2
                assert len(output_req.messages) == 4  # user + 2*assistant + 1*tool_call
                search_counter = 0
                for msg in output_req.messages:
                    if msg.role == "tool":
                        search_counter += 1
                        assert msg.content == tool_return_array[search_counter]
                assert search_counter == 1


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
        from verl.tools.search_tool import PoolMode, init_search_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=3)
        exec_worker = init_search_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=3, mode=PoolMode.ThreadMode)
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
        from verl.tools.search_tool import PoolMode, init_search_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=6)
        exec_worker = init_search_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=6, mode=PoolMode.ThreadMode)
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
        from verl.tools.search_tool import PoolMode, init_search_execution_pool

        # exec_worker = ExecutionWorker.options(max_concurrency=10).remote(enable_global_rate_limit=True, rate_limit=6)
        exec_worker = init_search_execution_pool(num_workers=10, enable_global_rate_limit=True, rate_limit=6, mode=PoolMode.ThreadMode)
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
