"""
Test the global distributed info
"""

import torch
from verl.single_controller.base.worker import DistRankInfo, DistGlobalInfo


def test_dist_global_info():
    dist_info = DistGlobalInfo(tp_size=2, cp_size=2, ep_size=2, dp_size=2, pp_size=2)

    result = []

    for i in range(dist_info.get_world_size()):
        result.append(dist_info.get_rank_info(global_rank=i, order='tp-pp-dp-ep-cp'))

    expected_result = [
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=0, ep_rank=1, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=0, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=0, pp_rank=1),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=0),
        DistRankInfo(tp_rank=0, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=1),
        DistRankInfo(tp_rank=1, cp_rank=1, ep_rank=1, dp_rank=1, pp_rank=1),
    ]

    for info, expected_info in zip(result, expected_result):
        assert info == expected_info


from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch, make_nd_compute_dataproto_dispatch_fn, make_nd_compute_dispatch_fn
from verl import DataProto
import ray

@ray.remote
class TestActor(Worker):
    def __init__(self) -> None:
        super().__init__()

        import torch.distributed
        torch.distributed.init_process_group(backend='nccl')
        self.infer_device_mesh = torch.distributed.device_mesh.init_device_mesh(device_type='cuda', 
                                                                                mesh_shape=[2, 4], mesh_dim_names=['dp', 'tp'])
        self.train_device_mesh = torch.distributed.device_mesh.init_device_mesh(device_type='cuda', 
                                                                                mesh_shape=[2, 2, 2], mesh_dim_names=['pp', 'dp', 'tp'])


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='infer'))
    def generate_data_proto(self, data: DataProto):
        tp_rank = self.infer_device_mesh['tp'].get_local_rank()
        dp_rank = self.infer_device_mesh['dp'].get_local_rank()
        data.batch['a'] += (tp_rank + 1) * dp_rank
        return data
    
    @register(dispatch_mode=make_nd_compute_dispatch_fn(mesh_name='infer'))
    def generate(self, data: int):
        tp_rank = self.infer_device_mesh['tp'].get_local_rank()
        dp_rank = self.infer_device_mesh['dp'].get_local_rank()
        print(f'tp_rank: {tp_rank}, dp_rank: {dp_rank}, init data: {data}')
        data += (tp_rank + 1) * dp_rank
        print(f'tp_rank: {tp_rank}, dp_rank: {dp_rank}, output data: {data}')
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='train'))
    def train_data_proto(self, data: DataProto):
        tp_rank = self.train_device_mesh['tp'].get_local_rank()
        dp_rank = self.train_device_mesh['dp'].get_local_rank()
        pp_rank = self.train_device_mesh['pp'].get_local_rank()
        data.batch['a'] += (tp_rank + 1) * (dp_rank + 2) * (pp_rank + 3)
        # tp rank 0, pp rank 1, dp rank 0, output data added: 8 + 3 = 11
        # tp rank 0, pp rank 1, dp rank 1, output data added: 12 + 4 = 16
        return data
        

# def test_dist_global_info_wg():
# create a worker group with size 8
# register a infer dist info with tp=4, dp=2
# register a train dist info with tp=2, dp=2, pp=2
# test the correctness of data dispatch and computation
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


ray.init()

ray_cls = RayClassWithInitArgs(TestActor)
resource_pool = RayResourcePool(process_on_nodes=[8])
wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)

wg.register_dist_info(name='infer', dist_global_info=DistGlobalInfo(tp_size=4, dp_size=2))
wg.register_dist_info(name='train', dist_global_info=DistGlobalInfo(tp_size=2, dp_size=2, pp_size=2))

infer_input_data = [1, 2]
infer_output_data = wg.generate(infer_input_data)

assert infer_output_data == [1, 3]

infer_input_data_proto = DataProto.from_single_dict(data={'a': torch.tensor([1, 2])})
infer_output_data_proto = wg.generate_data_proto(infer_input_data_proto)

assert torch.all(torch.eq(infer_output_data_proto.batch['a'], torch.tensor([1, 3])))

train_input_data_proto = DataProto.from_single_dict(data={'a': torch.tensor([3, 4])})
train_output_data_proto = wg.train_data_proto(train_input_data_proto)

assert torch.all(torch.eq(train_output_data_proto.batch['a'], torch.tensor([11, 16])))
