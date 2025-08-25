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


import ray
import torch

from verl import DataProto
from verl.protocol_dist import DistDataProto, DataWorker
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register


@ray.remote
class TestActor(Worker):
    def __init__(self):
        super().__init__()

        import torch.distributed

        torch.distributed.init_process_group(backend="nccl")
        self.infer_device_mesh = torch.distributed.device_mesh.init_device_mesh(
            device_type="cuda", mesh_shape=[2, 4], mesh_dim_names=["dp", "tp"]
        )
        self.train_device_mesh = torch.distributed.device_mesh.init_device_mesh(
            device_type="cuda", mesh_shape=[2, 2, 2], mesh_dim_names=["pp", "dp", "tp"]
        )

        self._register_dispatch_collect_info(
            "infer",
            dp_rank=self.infer_device_mesh["dp"].get_local_rank(),
            is_collect=self.infer_device_mesh["tp"].get_local_rank() == 0,
        )
        self._register_dispatch_collect_info(
            "train",
            dp_rank=self.train_device_mesh["dp"].get_local_rank(),
            is_collect=self.train_device_mesh["tp"].get_local_rank() == 0
            and self.train_device_mesh["pp"].get_local_rank() == 1,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_data_proto(self, data: DataProto | DistDataProto):
        print(type(data))
        tp_rank = self.infer_device_mesh["tp"].get_local_rank()
        dp_rank = self.infer_device_mesh["dp"].get_local_rank()
        if type(data) is DataProto:
            data.batch["a"] += (tp_rank + 1) * dp_rank
        else:
            # print("collect_info", self._query_collect_info("infer"))
            data_dp = data.select(batch_keys=["a"])
            data_dp.batch["a"] += (tp_rank + 1) * dp_rank
            # if tp_rank == 0:  # NOTE(caiyunke.astra): Must restrict update operation as we do inplace data update, which cannot controlled by collect_fn
            if self._query_collect_info("infer"): 
                data.update(data_dp)
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def train_data_proto(self, data: DataProto| DistDataProto):
        tp_rank = self.train_device_mesh["tp"].get_local_rank()
        dp_rank = self.train_device_mesh["dp"].get_local_rank()
        pp_rank = self.train_device_mesh["pp"].get_local_rank()
        if type(data) is DataProto:
            data.batch["a"] += (tp_rank + 1) * (dp_rank + 2) * (pp_rank + 3)
        else:
            print(self._query_collect_info("infer"))
            data_dp = data.select(batch_keys=["a"])
            data_dp.batch["a"] += (tp_rank + 1) * (dp_rank + 2) * (pp_rank + 3)
            if self._query_collect_info("train"):  # NOTE(caiyunke.astra): Must restrict update operation as we do inplace data update, which cannot controlled by collect_mask in collect_fn
                data.update(data_dp)
        # tp rank 0, pp rank 1, dp rank 0, output data added: 8 + 3 = 11
        # tp rank 0, pp rank 1, dp rank 1, output data added: 12 + 4 = 16
        return data


def test_dist_global_info_wg():
    # create a worker group with size 8
    # register a infer dist info with tp=4, dp=2
    # register a train dist info with tp=2, dp=2, pp=2
    # test the correctness of data dispatch and computation
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    ray.init()
    ray_cls = RayClassWithInitArgs(TestActor)
    resource_pool = RayResourcePool(process_on_nodes=[8])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)

    infer_input_data_proto = DataProto.from_single_dict(data={"a": torch.tensor([1, 2])})
    infer_output_data_proto = wg.generate_data_proto(infer_input_data_proto)

    assert wg._dispatch_info["infer"] == [0, 0, 0, 0, 1, 1, 1, 1]

    assert torch.all(torch.eq(infer_output_data_proto.batch["a"], torch.tensor([1, 3])))

    train_input_data_proto = DataProto.from_single_dict(data={"a": torch.tensor([3, 4])})
    train_output_data_proto = wg.train_data_proto(train_input_data_proto)

    assert wg._dispatch_info["train"] == [0, 0, 1, 1, 0, 0, 1, 1]

    assert torch.all(torch.eq(train_output_data_proto.batch["a"], torch.tensor([11, 16])))

    ray.shutdown()


def test_dist_global_info_wg_distdataproto():
    # create a worker group with size 8
    # register a infer dist info with tp=4, dp=2
    # register a train dist info with tp=2, dp=2, pp=2
    # test the correctness of data dispatch and computation
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    ray.init()
    # breakpoint()
    ray_cls = RayClassWithInitArgs(TestActor)
    resource_pool = RayResourcePool(process_on_nodes=[8])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)


    alive_nodes = []
    for node in ray.nodes():
        if node["alive"]:
            available_cpu = node["Resources"].get("CPU", 0.0)
            alive_nodes.append((node["NodeID"], available_cpu))
    print(len(alive_nodes))
    bundles = []
    for _, available_cpu in alive_nodes:
        bundles.append({"CPU": min(available_cpu, 8)})
    pg = ray.util.placement_group(bundles=bundles, strategy="SPREAD")

    dataworkers = []
    for bundle_idx, (node_id, available_cpu) in enumerate(alive_nodes):
        for _ in range(4):
            worker = DataWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx
                )
            ).remote()
            dataworkers.append(worker)

    infer_input_dist_data_proto = DistDataProto.from_single_dict(data={"a": torch.tensor([1, 2])}, dataworker_handles=dataworkers)
    infer_output_dist_data_proto = wg.generate_data_proto(infer_input_dist_data_proto)

    print(infer_output_dist_data_proto)
    assert wg._dispatch_info["infer"] == [0, 0, 0, 0, 1, 1, 1, 1]
    print(infer_output_dist_data_proto.select().batch["a"])
    assert torch.all(torch.eq(infer_output_dist_data_proto.select().batch["a"], torch.tensor([1, 3])))

    train_input_data_proto = DistDataProto.from_single_dict(data={"a": torch.tensor([3, 4])}, dataworker_handles=dataworkers)
    train_output_data_proto = wg.train_data_proto(train_input_data_proto)

    assert wg._dispatch_info["train"] == [0, 0, 1, 1, 0, 0, 1, 1]

    assert torch.all(torch.eq(train_output_data_proto.select().batch["a"], torch.tensor([11, 16])))

    ray.shutdown()

if __name__ == "__main__":

    test_dist_global_info_wg()
    test_dist_global_info_wg_distdataproto()
