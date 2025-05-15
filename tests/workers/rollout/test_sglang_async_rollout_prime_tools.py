import asyncio
from copy import deepcopy
from os import path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)
from datasets import Dataset
import numpy as np
from tensordict import TensorDict
import verl

import pytest
from transformers import AutoConfig, AutoTokenizer

from verl.protocol import DataProto
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema, OpenAIFunctionToolSchema
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, FinishReasonTypeEnum, Message
from verl.workers.rollout.sglang_rollout.async_sglang_rollout import AsyncSGLangRollout
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
    def multiple_code_data(self):
        """
        This dialogue is formatted in ShareGPT style and demonstrates multi-step reasoning
        using Python code to solve a set theory problem.
        """
        return Dataset.from_list([
            {
                "conversations": [
                    {
                        "from": "system",
                        "value": (
                            "Solve the following problem step by step. You now have the ability to selectively write "
                            "executable Python code to enhance your reasoning process.\n\n"
                            "**user question:**\n"
                            "There are 152 students at Dala High School. Assume the following: \n"
                            "- 100 students take a Math class\n"
                            "- 94 students take a Science class\n"
                            "- 57 students take an English class\n"
                            "- 73 students take a Math class and a Science class\n"
                            "- 24 students take a Math class and an English class\n"
                            "- 27 students take a Science class and an English class\n"
                            "- 22 students take a Math class and a Science class and an English class\n\n"
                            "How many students take neither a Math class nor a Science class nor an English class?\n\n"
                            "Remember to place the final answer in the last part using the format:\n"
                            "<answer>\n\\boxed{'The final answer goes here.'}\n</answer>"
                        )
                    },
                    {
                        "from": "human",
                        "value": "Please solve the above problem."
                    },
                    {
                        "from": "gpt",
                        "value": (
                            "Okay, so I need to find out how many students at Dala High School are not taking any of the "
                            "three classes: Math, Science, or English. The total number of students is 152. Let me see... "
                            "I remember this is a problem about sets and maybe using the principle of inclusion-exclusion. "
                            "Let me recall how that works.\n\n"
                            "First, the inclusion-exclusion principle for three sets says that the total number of students "
                            "taking at least one of the classes is equal to the sum of the numbers in each individual class, "
                            "minus the sum of the numbers in each pair of classes, plus the number in all three classes. "
                            "Then, subtract that total from the overall number of students to get those not taking any of the three.\n\n"
                            "Let me write that down step by step. Let M be the set of students taking Math, S for Science, and E for English. Then:\n\n"
                            "Total in at least one class = |M ∪ S ∪ E| = |M| + |S| + |E| - |M ∩ S| - |M ∩ E| - |S ∩ E| + |M ∩ S ∩ E|\n\n"
                            "Given the numbers:\n"
                            "|M| = 100\n"
                            "|S| = 94\n"
                            "|E| = 57\n"
                            "|M ∩ S| = 73\n"
                            "|M ∩ E| = 24\n"
                            "|S ∩ E| = 27\n"
                            "|M ∩ S ∩ E| = 22\n\n"
                            "So plugging these into the formula:\n\n"
                            "Total = 100 + 94 + 57 - 73 - 24 - 27 + 22\n\n"
                            "Let me compute that step by step using code to ensure accuracy."
                        )
                    },
                    {
                        "from": "gpt",
                        "value": (
                            "```python\n"
                            "M = 100\nS = 94\nE = 57\nM_S = 73\nM_E = 24\nS_E = 27\nM_S_E = 22\n\n"
                            "total_in_any = M + S + E - M_S - M_E - S_E + M_S_E\n"
                            "students_neither = 152 - total_in_any\n"
                            "print(students_neither)\n"
                            "```"
                        )
                    },
                    {
                        "from": "tool",
                        "value": "3"
                    },
                    {
                        "from": "gpt",
                        "value": (
                            "The code returns 3, indicating that 3 students take none of the classes. To verify, I'll break down "
                            "each category to ensure the total matches:\n\n"
                            "- Only Math and Science: 73 - 22 = 51\n"
                            "- Only Math and English: 24 - 22 = 2\n"
                            "- Only Science and English: 27 - 22 = 5\n"
                            "- Only Math: 100 - 51 - 2 - 22 = 25\n"
                            "- Only Science: 94 - 51 - 5 - 22 = 16\n"
                            "- Only English: 57 - 2 - 5 - 22 = 28\n\n"
                            "Summing all categories:"
                        )
                    },
                    {
                        "from": "gpt",
                        "value": (
                            "```python\n"
                            "only_M_S = 73 - 22\nonly_M_E = 24 - 22\nonly_S_E = 27 - 22\n\n"
                            "only_M = 100 - only_M_S - only_M_E - 22\n"
                            "only_S = 94 - only_M_S - only_S_E - 22\n"
                            "only_E = 57 - only_M_E - only_S_E - 22\n\n"
                            "total_verify = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22\n"
                            "print(total_verify)\n"
                            "```"
                        )
                    },
                    {
                        "from": "tool",
                        "value": "149"
                    },
                    {
                        "from": "gpt",
                        "value": (
                            "The verification total is 149, so students not taking any classes are 152 - 149 = 3, confirming the initial result.\n\n"
                            "<answer>\n\\boxed{3}\n</answer>"
                        )
                    }
                ]
            }
        ])

    @pytest.fixture
    def prime_data(self):

        prompt = '''
            Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. \n\n**user question:**\nThere are 152 students at Dala High School. Assume the following: \n- 100 students take a Math class \n- 94 students take a Science class \n- 57 students take an English class \n- 73 students take a Math class and a Science class \n- 24 students take a Math class and an English class \n- 27 students take a Science class and an English class \n- 22 students take a Math class and a Science class and an English class\n \nHow many students take neither a Math class nor a Science class nor an Eglish class?\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\boxed{'The final answer goes here.'}\n</answer>
        '''

        expect_turn_0 = '''
        Okay, so I need to find out how many students at Dala High School are not taking any of the three classes: Math, Science, or English. The total number of students is 152. Let me see... I remember this is a problem about sets and maybe using the principle of inclusion-exclusion. Let me recall how that works.\n\nFirst, the inclusion-exclusion principle for three sets says that the total number of students taking at least one of the classes is equal to the sum of the numbers in each individual class, minus the sum of the numbers in each pair of classes, plus the number in all three classes. Then, subtract that total from the overall number of students to get those not taking any of the three. \n\nLet me write that down step by step. Let M be the set of students taking Math, S for Science, and E for English. Then:\n\nTotal in at least one class = |M ∪ S ∪ E| = |M| + |S| + |E| - |M ∩ S| - |M ∩ E| - |S ∩ E| + |M ∩ S ∩ E|\n\nGiven the numbers:\n\n|M| = 100\n\n|S| = 94\n\n|E| = 57\n\n|M ∩ S| = 73\n\n|M ∩ E| = 24\n\n|S ∩ E| = 27\n\n|M ∩ S ∩ E| = 22\n\nSo plugging these into the formula:\n\nTotal = 100 + 94 + 57 - 73 - 24 - 27 + 22\n\nLet me compute that step by step using code to ensure accuracy.\n\n<tool_call>[{"arguments": {"code": "M = 100\\nS = 94\\nE = 57\\nM_S = 73\\nM_E = 24\\nS_E = 27\\nM_S_E = 22\\n\\ntotal_in_any = M + S + E - M_S - M_E - S_E + M_S_E\\nstudents_neither = 152 - total_in_any\\nprint(students_neither)", "language": "python"}, "name": "calc_code_result"}]</tool_call>\n
        '''

        tool_return_0 = '''
        \n<interpreter>\n3\n</interpreter>\n
        '''

        expect_turn_1 = '''
        \nThe code returns 3, indicating that 3 students take none of the classes. To verify, I'll break down each category to ensure the total matches:\n\n- Only Math and Science: 73 - 22 = 51\n- Only Math and English: 24 - 22 = 2\n- Only Science and English: 27 - 22 = 5\n- Only Math: 100 - 51 - 2 - 22 = 25\n- Only Science: 94 - 51 - 5 - 22 = 16\n- Only English: 57 - 2 - 5 - 22 = 28\n\nSumming all categories:\n\n<tool_call>[{"arguments": {"code": "only_M_S = 73 - 22\\nonly_M_E = 24 - 22\\nonly_S_E = 27 - 22\\n\\nonly_M = 100 - only_M_S - only_M_E - 22\\nonly_S = 94 - only_M_S - only_S_E - 22\\nonly_E = 57 - only_M_E - only_S_E - 22\\n\\ntotal_verify = only_M + only_S + only_E + only_M_S + only_M_E + only_S_E + 22\\nprint(total_verify)", "language": "python"}, "name": "calc_code_result"}]</tool_call>\n
        '''

        tool_return_1 = '''
        \n<interpreter>\n149\n</interpreter>\n
        '''
        expect_turn_2 = '''
        \nThe verification total is 149, so students not taking any classes are 152 - 149 = 3, confirming the initial result.\n\n<answer>\n\\boxed{3}\n</answer>
        '''
        
        preencode_prompts = [
            [{"role": "user", "content": prompt, "tool_calls": None}]
        ]
        expect_turn_array = [
            expect_turn_0,expect_turn_1,expect_turn_2
        ]
        tool_return_array = [
            tool_return_0,tool_return_1
        ]

        return preencode_prompts,expect_turn_array,tool_return_array
    
    @pytest.fixture
    def prime_rollout_config(self):
        max_prompt_length = 1024
        max_prompt_length = 1024
        max_response_length = 1024
        dtype = "bfloat16"
        tensor_parallel_size = 1
        tool_path = "./resource/tool_configs/prime_tool_config"
        rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size,tool_path)
        return rollout_config

    @pytest.fixture
    def prime_data_proto(self,prime_data,qwen_tokenizer):
        preencode_prompts,_,_ = prime_data
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
        tools_kwargs = np.array([{
                    "calc_code_result": {
                        "create_kwargs": {"ground_truth": "test-solution-str"},
                    },
                }],dtype=object)
        index = np.array([0],dtype=object)
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": messages,"tools_kwargs": tools_kwargs,"index":index})
        return prompts


    @patch.object(AsyncSGLangRollout, '_init_distributed_env', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_inference_engine', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_sampling_params', return_value=None)
    def test_tools_registration(self, mock_env,mock_engine,mock_sampling,prime_rollout_config,qwen_tokenizer,qwen_model_config):
        print(prime_rollout_config)
        rollout = AsyncSGLangRollout(actor_module="", config=prime_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        assert len(rollout._tool_schemas) == 1
        assert "calc_code_result" in rollout._tool_map.keys()
        from verl.tools.prime_tools import PrimeTool
        assert isinstance(rollout._tool_map["calc_code_result"],PrimeTool)
        assert rollout._tool_call_parser_type == "qwen25"
    

    @patch.object(AsyncSGLangRollout, '_init_distributed_env', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_inference_engine', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_sampling_params', return_value=None)
    def test_rollout_req_creation(self,mock_env,mock_engine,mock_sampling,prime_rollout_config,qwen_tokenizer,qwen_model_config,prime_data_proto):
        rollout = AsyncSGLangRollout(actor_module="", config=prime_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req_list = rollout._preprocess_prompt_to_async_rollout_requests(prime_data_proto,n=1)
        assert len(req_list) == 1
        assert req_list[0].state == AsyncRolloutRequestStateEnum.PENDING
        assert req_list[0].tools == [
            OpenAIFunctionToolSchema(
                type="function",
                function=OpenAIFunctionSchema(
                    name="calc_code_result",
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
                )
            )
        ]


    @patch.object(AsyncSGLangRollout, '_init_distributed_env', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_inference_engine', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_sampling_params', return_value=None)
    def test_over_size_case(self,mock_env,mock_engine,mock_sampling,prime_rollout_config,qwen_tokenizer,qwen_model_config,prime_data_proto,prime_data):
        prime_rollout_config.multi_turn.max_turns = 1
        rollout = AsyncSGLangRollout(actor_module="", config=prime_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req = rollout._preprocess_prompt_to_async_rollout_requests(prime_data_proto,n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]
        _,expect_turn_array,tool_return_array = prime_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call=MagicMock()
        future = asyncio.Future()
        future.set_result({'text': expect_turn_array[0], 
        'meta_info': {'id': 'd1188d81cba840359df5b352b344bc8e', 'finish_reason': {'type': 'length', 'length': 1024},'prompt_tokens': 132, 'completion_tokens': 100, 'cached_tokens': 0, 'e2e_latency': 9.9304039478302}})
        rollout._handle_engine_call.return_value = future
        rollout._tp_rank=0
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[rollout._async_rollout_a_request(req, True, False) for req in req_list],
            )
        )
        assert len(output_req_list) == 1
        output_req = output_req_list[0]
        print(output_req)
        assert output_req.state == AsyncRolloutRequestStateEnum.COMPLETED
        assert output_req.reward_scores == {'calc_code_result': []}
        # we should only have two message, one for prompt, second for response.
        assert len(output_req.messages) == 2
        assert output_req.messages[1] == Message(
            role="assistant",
            content=expect_turn_array[0],
            tool_calls=None,
        )


    @patch.object(AsyncSGLangRollout, '_init_distributed_env', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_inference_engine', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_sampling_params', return_value=None)
    def test_tool_call_basic_case(self,mock_env,mock_engine,mock_sampling,prime_rollout_config,qwen_tokenizer,qwen_model_config,prime_data_proto,prime_data):
        prime_rollout_config.multi_turn.max_turns = 10
        rollout = AsyncSGLangRollout(actor_module="", config=prime_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req = rollout._preprocess_prompt_to_async_rollout_requests(prime_data_proto,n=1)[0]
        req = MagicMock(wraps=req, spec=AsyncRolloutRequest)
        req.finalize = MagicMock()
        req_list = [req]
        _,expect_turn_array,tool_return_array = prime_data
        # here we mock a meta info with 'length'. indicate the response is truncate
        rollout._handle_engine_call=MagicMock()
        futures = [asyncio.Future() for i in expect_turn_array]
        for idx,(i,turn) in enumerate(zip(futures,expect_turn_array)):
            i.set_result({'text': turn, 
        'meta_info': {'id': 'd1188d81cba840359df5b352b344bc8e','finish_reason': {'type': 'tool_calls' if idx<len(expect_turn_array)-1 else 'stop'},'prompt_tokens': len(turn), 'completion_tokens': 100, 'cached_tokens': 0, 'e2e_latency': 9.9304039478302}})
            if idx<len(expect_turn_array)-1:
                assert rollout._function_call_parser.has_tool_call(turn)
                assert rollout._function_call_parser.parse_non_stream(turn)
        
        rollout._handle_engine_call.side_effect = futures
        rollout._tp_rank=0
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
        assert output_req.reward_scores == {'calc_code_result': ['3', '149']} 
        assert rollout._handle_engine_call.call_count == 3
        assert len(output_req.messages) == 6 # user + 3*assistant + 2*tool_call
        code_counter =0
        for msg in output_req.messages:
            if msg.role == "tool":
                code_counter+=1
        assert code_counter == 2



    
    @patch.object(AsyncSGLangRollout, '_init_distributed_env', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_inference_engine', return_value=None)
    @patch.object(AsyncSGLangRollout, '_init_sampling_params', return_value=None)
    def test_tool_call_batch_case(self,mock_env,mock_engine,mock_sampling,prime_rollout_config,qwen_tokenizer,qwen_model_config,prime_data_proto,prime_data):
        prime_rollout_config.multi_turn.max_turns = 10
        rollout = AsyncSGLangRollout(actor_module="", config=prime_rollout_config, tokenizer=qwen_tokenizer, model_hf_config=qwen_model_config)
        req = rollout._preprocess_prompt_to_async_rollout_requests(prime_data_proto,n=1)[0]
        req_nums = 100
        req_list = []
        req_turns_counter = {}
        # this map should a Map[id:List[Futures]]
        req_turns_map ={}
        _,expect_turn_array,tool_return_array = prime_data
        for i in range(req_nums):
            _temp_req = deepcopy(req)
            _temp_req.batch_data_id = i
            _temp_req.request_id = i
            req_list.append(MagicMock(wraps=_temp_req, spec=AsyncRolloutRequest))
            futures = [asyncio.Future() for i in expect_turn_array]
            for idx,(i,turn) in enumerate(zip(futures,expect_turn_array)):
                i.set_result({'text': turn, 
            'meta_info': {'id': 'd1188d81cba840359df5b352b344bc8e','finish_reason': {'type': 'tool_calls' if idx<len(expect_turn_array)-1 else 'stop'},'prompt_tokens': len(turn), 'completion_tokens': 100, 'cached_tokens': 0, 'e2e_latency': 9.9304039478302}})
                if idx<len(expect_turn_array)-1:
                    assert rollout._function_call_parser.has_tool_call(turn)
                    assert rollout._function_call_parser.parse_non_stream(turn)
            req_turns_map[_temp_req.batch_data_id] = futures
            req_turns_counter[_temp_req.batch_data_id] = 0

        async def hacked_handle_engine_call(self, _req: AsyncRolloutRequest,do_sample: bool,is_validate:bool,**kwargs):
            result =  req_turns_map[_req.batch_data_id][req_turns_counter[_req.batch_data_id]]
            req_turns_counter[_req.batch_data_id]+=1
            re = await result
            return re

        with patch.object(AsyncSGLangRollout, '_handle_engine_call', new=hacked_handle_engine_call):
            rollout._tp_rank=0
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
                assert output_req.reward_scores == {'calc_code_result': ['3', '149']} 
                assert len(output_req.messages) == 6 # user + 3*assistant + 2*tool_call
                code_counter =0
                for msg in output_req.messages:
                    if msg.role == "tool":
                        code_counter+=1
                assert code_counter == 2