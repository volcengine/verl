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
import os
import pickle
import tempfile

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    tokenizer = hf_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert "input_ids" in data_proto.batch

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    print(f"type: type{output}")
    print(f"\n\noutput: {output}")


def test_image_rl_data():
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    tokenizer = hf_tokenizer("Qwen/Qwen2-VL-2B-Instruct")
    processor = hf_processor("Qwen/Qwen2-VL-2B-Instruct")
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )
    dataset = RLHFDataset(
        data_files=os.path.expanduser("~/data/geo3k/train.parquet"),
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    assert "multi_modal_data" in data_proto.non_tensor_batch, data_proto
    assert "multi_modal_inputs" in data_proto.non_tensor_batch, data_proto

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    print(f"type: type{output}")
    print(f"\n\noutput: {output}")


def test_rl_dataset_pickle_unpickle():
    """Test that RLHFDataset can be pickled and unpickled correctly."""
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    tokenizer = hf_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )

    # Test case 1: serialize_dataset=False (default behavior)
    dataset_no_serialize = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)
    original_length = len(dataset_no_serialize)
    original_item = dataset_no_serialize[0]

    # Pickle and unpickle
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        pickle.dump(dataset_no_serialize, f)
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            unpickled_dataset = pickle.load(f)

        # Verify the unpickled dataset works correctly
        assert len(unpickled_dataset) == original_length, "Dataset length should be preserved after unpickling"
        assert hasattr(unpickled_dataset, "dataframe"), "Dataframe should be reconstructed after unpickling"

        # Verify we can access items and they have the same structure
        unpickled_item = unpickled_dataset[0]
        assert set(unpickled_item.keys()) == set(original_item.keys()), "Item keys should be the same"
        assert unpickled_item["input_ids"].shape == original_item["input_ids"].shape, "Input IDs shape should match"
        assert unpickled_item["attention_mask"].shape == original_item["attention_mask"].shape, (
            "Attention mask shape should match"
        )

        # Verify the dataset can be used in a DataLoader
        dataloader = DataLoader(dataset=unpickled_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        assert "input_ids" in batch, "Batch should contain input_ids"
        assert "attention_mask" in batch, "Batch should contain attention_mask"

    finally:
        # Clean up temporary file
        os.unlink(temp_path)

    # Test case 2: serialize_dataset=True
    dataset_serialize = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)
    dataset_serialize.serialize_dataset = True

    # Pickle and unpickle
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        pickle.dump(dataset_serialize, f)
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            unpickled_dataset_serialize = pickle.load(f)

        # Verify the unpickled dataset works correctly
        assert len(unpickled_dataset_serialize) == original_length, (
            "Dataset length should be preserved after unpickling"
        )
        assert hasattr(unpickled_dataset_serialize, "dataframe"), (
            "Dataframe should be preserved when serialize_dataset=True"
        )

        # Verify we can access items
        unpickled_item = unpickled_dataset_serialize[0]
        assert set(unpickled_item.keys()) == set(original_item.keys()), "Item keys should be the same"

    finally:
        # Clean up temporary file
        os.unlink(temp_path)

    print("All pickle/unpickle tests passed!")
