# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

import os

from recipe.agent_lightning_like.notify import notify_llm_server_address

from .utils import agent_loop_mgr, agent_server_addr, init_config, llm_router_and_addr, tokenizer  # noqa: F401


def test_agent_loop(init_config, tokenizer, agent_loop_mgr, llm_router_and_addr):  # noqa: F811
    assert agent_loop_mgr
    llm_router, router_addr = llm_router_and_addr
    assert llm_router is not None
    assert router_addr
    notify_llm_server_address(router_addr)
    assert os.path.isfile(os.environ["LLM_SERVER_NOTIFY_FILE"])

    from recipe.agent_lightning_like.example.dataset import CustomDataset
    from verl.protocol import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn

    dataset = CustomDataset(
        data_files=os.path.expanduser("~/data/gsm8k/train.parquet"), config=init_config.data, tokenizer=tokenizer
    )
    batch_size = 4
    samples = [dataset[i] for i in range(batch_size)]
    batch = DataProto.from_single_dict(collate_fn(samples))

    output = agent_loop_mgr.generate_sequences(batch)

    assert len(output) == batch_size
    assert "prompts" in output.batch and output.batch["prompts"].shape[-1] > 0
    assert "responses" in output.batch and output.batch["responses"].shape[-1] > 0
    assert "input_ids" in output.batch
    assert (
        output.batch["input_ids"].shape[-1] == output.batch["prompts"].shape[-1] + output.batch["responses"].shape[-1]
    )
    assert "attention_mask" in output.batch
    assert output.batch["attention_mask"].shape[-1] == output.batch["input_ids"].shape[-1]
    assert "position_ids" in output.batch
    assert output.batch["position_ids"].shape[-1] == output.batch["input_ids"].shape[-1]
    assert "response_mask" in output.batch
    assert output.batch["response_mask"].shape[-1] == output.batch["responses"].shape[-1]
    assert "rm_scores" in output.batch
    assert "__num_turns__" in output.non_tensor_batch and len(output.non_tensor_batch["__num_turns__"]) == batch_size

    for i in range(batch_size):
        if not (batch_size > 10 and i % (batch_size // 10) == 0):
            continue
        prompt = tokenizer.decode(output.batch["prompts"][i], skip_special_tokens=True)
        response = tokenizer.decode(output.batch["responses"][i], skip_special_tokens=True)
        unmasked_response_ids = output.batch["responses"][i][output.batch["response_mask"][i] == 1]
        unmasked_response = tokenizer.decode(unmasked_response_ids, skip_special_tokens=True)
        num_turns = output.non_tensor_batch["__num_turns__"][i]
        score = output.batch["rm_scores"][i].sum().item()
        print("=" * 40)
        print(f"Sample {i}:")
        print(f"Num Turns: {num_turns}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Unmasked Response: {unmasked_response}")
        print(f"Reward Score: {score}")
        print(f"Ground Truth: {samples[i]['reward_model']['ground_truth']}")

    print("=" * 40)
    print("Test passed!")
