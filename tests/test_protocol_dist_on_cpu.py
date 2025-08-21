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

from typing import List
import ray
import random
import time
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.protocol import DataProto, DataProtoFuture
from verl.protocol_dist import DistDataProto, DataWorker


# A handle to wait for asynchronous operations to complete
class WaitHandle:
    def __init__(self, object_refs: List[ray.ObjectRef]):
        self.object_refs = object_refs

    def wait(self):
        ray.get(self.object_refs)

@pytest.fixture(scope="module")
def ray_cluster():
    """启动Ray并初始化DataWorkers"""
    ray.init(num_cpus=4)
    workers = [DataWorker.remote() for _ in range(4)]
    dataworker_handles = workers
    yield dataworker_handles
    ray.shutdown()

def test_tensor_dict_constructor(ray_cluster):
    obs = torch.randn(100, 10)
    act = torch.randn(100, 10, 3)
    data = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, dataworker_handles=ray_cluster)

    assert len(data.row_ids) == 100
    assert len(data) == 100

    with pytest.raises(AssertionError):
        data = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=2, dataworker_handles=ray_cluster)

    with pytest.raises(AssertionError):
        data = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=3, dataworker_handles=ray_cluster)

def test_construct_consistency(ray_cluster):
    obs = torch.randn(100, 10)
    act = torch.randn(100, 10, 3)
    dp = DataProto.from_dict(tensors={"obs": obs, "act": act})
    ddp = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, dataworker_handles=ray_cluster)
    dp_data = dp.batch["act"]
    ddp_data = ddp.select().batch["act"]
    torch.testing.assert_close(dp_data, ddp_data)
    ddp.destroy()
    data_dict = {
        "features": torch.randn(100, 128),
        "labels": np.random.randint(0, 10, 100)
    }
    dp = DataProto.from_single_dict(data_dict)
    ddp = DistDataProto.from_single_dict(data_dict, dataworker_handles=ray_cluster)
    dp_data = dp.batch["features"]
    ddp_data = ddp.select().batch["features"]
    torch.testing.assert_close(dp_data, ddp_data)

def test_update_consistency(ray_cluster):
    obs = torch.randn(100, 10)
    act = torch.randn(100, 10, 3)
    dp = DataProto.from_dict(tensors={"obs": obs, "act": act})
    ddp = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, dataworker_handles=ray_cluster)
    dp.batch["obs"] += 3.3
    ddp_data = ddp.select(batch_keys=["obs"])
    ddp_data.batch["obs"] += 3.3
    ddp.update(ddp_data)
    dp_data = dp.batch["obs"]
    ddp_data = ddp.select().batch["obs"]
    torch.testing.assert_close(dp_data, ddp_data)
    dp_data = dp.batch["act"]
    ddp_data = ddp.select().batch["act"]
    torch.testing.assert_close(dp_data, ddp_data)



# def test_destory(ray_cluster):
#     obs = torch.zeros(1024, 256000)
#     act = torch.zeros(1024, 256000, 16)
#     time.sleep(1)
#     object_store_mb_1 = ray.available_resources().get('object_store_memory', 0) / (1024**2)
#     ddp = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, dataworker_handles=ray_cluster)
#     time.sleep(1)
#     object_store_mb_2 = ray.available_resources().get('object_store_memory', 0) / (1024**2)
#     ddp.destroy()
#     time.sleep(1)
#     object_store_mb_3 = ray.available_resources().get('object_store_memory', 0) / (1024**2)
#     print(object_store_mb_1, object_store_mb_2, object_store_mb_3)
#     assert abs(object_store_mb_1 - object_store_mb_3) < 2
#     assert (object_store_mb_1 - object_store_mb_2) > 10

def test_get_meta_data(ray_cluster):
    data_dict = {
        "features": torch.randn(100, 128),
        "labels": np.random.randint(0, 10, 100)
    }
    ddp = DistDataProto.from_single_dict(data_dict, dataworker_handles=ray_cluster)
    target_id = ddp.row_ids[0]
    assert ddp.get_metadata()[target_id]["shape"]["features"] == torch.Size([1, 128])

def test_async_select(ray_cluster):
    data_dict = {
        "features": torch.randn(100, 128),
        "labels": np.random.randint(0, 10, 100)
    }
    dp = DataProto.from_single_dict(data_dict)
    ddp = DistDataProto.from_single_dict(data_dict, dataworker_handles=ray_cluster)
    dp_data = dp.batch["features"]
    ddp_data_future = ddp.select(async_mode=True)
    assert isinstance(ddp_data_future, DataProtoFuture)
    ddp_result = ddp_data_future.get()
    assert isinstance(ddp_result, DataProto)
    ddp_data = ddp_result.batch["features"]
    torch.testing.assert_close(dp_data, ddp_data)

def test_make_iterator_consistency(ray_cluster):
    obs = torch.randn(100, 10)
    labels = [random.choice(["abc", "cde"]) for _ in range(100)]
    dp = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})
    ddp = DistDataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, dataworker_handles=ray_cluster)

    def test_iterator(mini_batch_size=10, shuffle=False):
        if shuffle:
            dataloader_kwargs = {"shuffle": True}
        else:
            dataloader_kwargs = None
        data_iter_dp_sequential = dp.make_iterator(mini_batch_size=mini_batch_size, epochs=2, seed=1, dataloader_kwargs=dataloader_kwargs)
        data_list_dp_sequential = []
        for data in data_iter_dp_sequential:
            data_list_dp_sequential.append(data)

        data_iter_ddp_sequential_1 = ddp.make_iterator(mini_batch_size=mini_batch_size, epochs=2, seed=1, dataloader_kwargs=dataloader_kwargs)
        data_list_ddp_sequential_1 = []
        for data in data_iter_ddp_sequential_1:
            data_list_ddp_sequential_1.append(data)

        data_iter_ddp_sequential_2 = ddp.make_iterator(mini_batch_size=mini_batch_size, epochs=2, seed=1, dataloader_kwargs=dataloader_kwargs)
        data_list_ddp_sequential_2 = []
        for data in data_iter_ddp_sequential_2:
            data_list_ddp_sequential_2.append(data)

        for data1, data2, data3 in zip(data_list_ddp_sequential_1, data_list_ddp_sequential_2, data_list_dp_sequential, strict=True):
            assert isinstance(data1, DataProto)
            assert isinstance(data2, DataProto)
            result = torch.all(torch.eq(data1.batch["obs"], data2.batch["obs"]))
            if not result.item():
                print(data1.batch["obs"])
                print(data2.batch["obs"])
                raise AssertionError()
            
            result = torch.all(torch.eq(data1.batch["obs"], data3.batch["obs"]))
            if not result.item():
                print(data1.batch["obs"])
                print(data3.batch["obs"])
                raise AssertionError()
            non_tensor_result = np.all(np.equal(data1.non_tensor_batch["labels"], data2.non_tensor_batch["labels"]))
            if not non_tensor_result.item():
                print(data1.non_tensor_batch["labels"])
                print(data2.non_tensor_batch["labels"])

            non_tensor_result = np.all(np.equal(data1.non_tensor_batch["labels"], data3.non_tensor_batch["labels"]))
            if not non_tensor_result.item():
                print(data1.non_tensor_batch["labels"])
                print(data3.non_tensor_batch["labels"])

    test_iterator(mini_batch_size=10, shuffle=False)
    test_iterator(mini_batch_size=10, shuffle=True)

def test_make_iterator_var_batchsize(ray_cluster):
    obs = torch.randn(100, 10)
    labels = [random.choice(["abc", "cde"]) for _ in range(100)]
    dp = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})
    ddp = DistDataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, dataworker_handles=ray_cluster)

    data_iter_dp_sequential = dp.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    data_list_dp_sequential = []
    for data in data_iter_dp_sequential:
        data_list_dp_sequential.append(data)
    
    indices = ddp.row_ids * 2
    max_size = 16
    min_size = 5
    num_batches = 20
    avg_size = 200 // num_batches
    batches = []
    
    start_idx = 0
    for i in range(num_batches):
        if i == num_batches - 1:
            batch_size = len(indices) - start_idx
        else:
            offset = np.random.randint(-1, 2) 
            batch_size = max(min_size, min(max_size, avg_size + offset))
            
            remaining = len(indices) - start_idx
            remaining_batches = num_batches - i
            min_remaining = (remaining_batches - 1) * min_size
            
            if remaining - batch_size < min_remaining:
                batch_size = max(min_size, remaining - min_remaining)
        
        end_idx = min(start_idx + batch_size, len(indices))
        batch = indices[start_idx:end_idx]
        batches.append(batch)
        
        start_idx = end_idx
        
        if start_idx >= len(indices):
            break
    
    # print(len(batches))
    # print([len(batch) for batch in batches])
    # print(sum([len(batch) for batch in batches]))
    data_iter_ddp_sequential_var_batchsize = ddp.make_iterator(read_order=batches)

    data_list_ddp_sequential_var_batchsizes = []
    for batch_idx, batch in zip(batches, data_iter_ddp_sequential_var_batchsize):
        assert len(batch_idx) == len(batch)
        data_list_ddp_sequential_var_batchsizes.append(batch)

    data1 = DataProto.concat(data_list_dp_sequential)
    data2 = DataProto.concat(data_list_ddp_sequential_var_batchsizes)
    assert torch.allclose(data1.batch["obs"], data2.batch["obs"])
    assert np.all(np.equal(data1.non_tensor_batch["labels"], data2.non_tensor_batch["labels"]))

def test_chunk_concat_consistency(ray_cluster):
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ["a", "b", "c", "d", "e", "f"]
    dp = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})
    ddp = DistDataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"}, dataworker_handles=ray_cluster)
    with pytest.raises(AssertionError):
        dp.chunk(5)
        ddp.chunk(5)

    dp_split = dp.chunk(2)
    ddp_split = ddp.chunk(2)
    assert len(ddp_split) == 2
    assert torch.all(torch.eq(ddp_split[0].select().batch["obs"], torch.tensor([1, 2, 3])))
    assert np.all(ddp_split[0].select().non_tensor_batch["labels"] == np.array(["a", "b", "c"]))
    assert ddp_split[0].select().meta_info == {"name": "abdce"}

    assert torch.all(torch.eq(ddp_split[1].select().batch["obs"], torch.tensor([4, 5, 6])))
    assert np.all(ddp_split[1].select().non_tensor_batch["labels"] == np.array(["d", "e", "f"]))
    assert ddp_split[1].select().meta_info == {"name": "abdce"}

    ddp_concat_data = DistDataProto.concat(ddp_split)
    assert torch.all(torch.eq(ddp_concat_data.select().batch["obs"], dp.batch["obs"]))
    assert np.all(ddp_concat_data.select().non_tensor_batch["labels"] == dp.non_tensor_batch["labels"])
    assert ddp_concat_data.meta_info == dp.meta_info
    assert ddp_concat_data.select().meta_info == dp.meta_info

def test_reorder_consistency(ray_cluster):
    data_dict = {
        "features": torch.tensor([1, 2, 3, 4, 5, 6]),
        "labels": np.array(["a", "b", "c", "d", "e", "f"])
    }
    dp = DataProto.from_single_dict(data_dict)
    ddp = DistDataProto.from_single_dict(data_dict, dataworker_handles=ray_cluster)
    dp.reorder(torch.tensor([3, 4, 2, 0, 1, 5]))
    ddp.reorder(torch.tensor([3, 4, 2, 0, 1, 5]))
    dp_data = dp.batch["features"]
    ddp_data = ddp.select().batch["features"]
    torch.testing.assert_close(dp_data, ddp_data)
    assert torch.all(torch.eq(ddp_data, torch.tensor([4, 5, 3, 1, 2, 6])))
    dp_labels = dp.non_tensor_batch["labels"]
    ddp_labels = ddp.select().non_tensor_batch["labels"]
    assert np.all(ddp_labels == dp_labels)
    assert np.all(ddp_labels == np.array(["d", "e", "c", "a", "b", "f"]))

def test_repeat(ray_cluster):
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DistDataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"}, dataworker_handles=ray_cluster)
    print(ray.get(ray_cluster[0].get_data_len.remote()))
    # Test interleave=True
    repeated_data_interleave = data.repeat(repeat_times=2, interleave=True).select()
    print(ray.get(ray_cluster[0].get_data_len.remote()))
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    expected_labels_interleave = ["a", "a", "b", "b", "c", "c"]

    assert torch.all(torch.eq(repeated_data_interleave.batch["obs"], expected_obs_interleave))
    assert (repeated_data_interleave.non_tensor_batch["labels"] == expected_labels_interleave).all()
    assert repeated_data_interleave.meta_info == {"info": "test_info"}

    # Test interleave=False
    repeated_data_no_interleave = data.repeat(repeat_times=2, interleave=False).select()
    print(ray.get(ray_cluster[0].get_data_len.remote()))  # NOTE(caiyunke.astra): memory not released when doing multiple repeats. Manually call destroy() in actual use.
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
    expected_labels_no_interleave = ["a", "b", "c", "a", "b", "c"]

    assert torch.all(torch.eq(repeated_data_no_interleave.batch["obs"], expected_obs_no_interleave))
    assert (repeated_data_no_interleave.non_tensor_batch["labels"] == expected_labels_no_interleave).all()
    assert repeated_data_no_interleave.meta_info == {"info": "test_info"}

def test_dataproto_pad_unpad(ray_cluster):
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DistDataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"}, dataworker_handles=ray_cluster)

    from verl.protocol_dist import pad_distdataproto_to_divisor, unpad_distdataproto

    padded_ddp, pad_size = pad_distdataproto_to_divisor(data, size_divisor=2)
    assert pad_size == 1

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ["a", "b", "c", "a"]

    padded_data = padded_ddp.select()
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    assert padded_data.meta_info == {"info": "test_info"}

    unpadd_ddp = unpad_distdataproto(padded_ddp, pad_size=pad_size)
    unpadd_data = unpadd_ddp.select()
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    assert unpadd_data.meta_info == {"info": "test_info"}

    padded_ddp, pad_size = pad_distdataproto_to_divisor(data, size_divisor=3)
    assert pad_size == 0

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    expected_labels = ["a", "b", "c"]

    padded_data = padded_ddp.select()
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    assert padded_data.meta_info == {"info": "test_info"}

    unpadd_ddp = unpad_distdataproto(padded_ddp, pad_size=pad_size)
    unpadd_data = unpadd_ddp.select()
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    assert unpadd_data.meta_info == {"info": "test_info"}

    padded_ddp, pad_size = pad_distdataproto_to_divisor(data, size_divisor=7)
    assert pad_size == 4

    padded_data = padded_ddp.select()
    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ["a", "b", "c", "a", "b", "c", "a"]
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    assert padded_data.meta_info == {"info": "test_info"}


    unpadd_ddp = unpad_distdataproto(padded_ddp, pad_size=pad_size)
    unpadd_data = unpadd_ddp.select()
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    assert unpadd_data.meta_info == {"info": "test_info"}


if __name__ == "__main__":
    ray.init(num_cpus=4)
    workers = [DataWorker.remote() for _ in range(4)]
    # obs = torch.randn(100, 10)
    # act = torch.randn(100, 10, 3)
    # dp = DataProto.from_dict(tensors={"obs": obs, "act": act})
    # ddp = DistDataProto.from_dict(tensors={"obs": obs, "act": act}, dataworker_handles=workers)
    # print(ddp.row_to_worker)
    # print(data.row_ids[0])
    # print(ray.get(workers[0].get.remote(data.row_ids[0])))
    # dp_data = dp.batch["act"]
    # ddp_data = ddp.select().batch["act"]
    # print(ddp_data == dp_data)
    test_get_meta_data(workers)
    test_reorder_consistency(workers)
    test_async_select(workers)
    test_make_iterator_consistency(workers)
    test_make_iterator_var_batchsize(workers)
    test_chunk_concat_consistency(workers)
    test_repeat(workers)
    # test_destory(workers)
    test_dataproto_pad_unpad(workers)
    test_update_consistency(workers)
    ray.shutdown()