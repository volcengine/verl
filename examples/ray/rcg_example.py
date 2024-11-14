import os
import ray
import torch
import warnings
from single_controller.ray.base import (
    RayResourcePool,
    RayClassWithInitArgs,
    RayWorkerGroup,
    merge_resource_pool,
)
from single_controller.base import Worker
from single_controller.ray.decorator import register, Dispatch, Execute, dispatch_dp_compute_data_proto, collect_dp_compute_data_proto

from ray.dag import InputNode, MultiOutputNode
from verl.protocol import DataProto

warnings.filterwarnings("ignore")

@ray.remote(num_gpus=1)
class GPUWorker(Worker):

    def __init__(self) -> None:
        super().__init__()

    def mult_x_by_rank(self, data: DataProto):
        data = data.to('cuda')
        res = data.batch['x'] * self.rank
        print(f"rank {self.rank}, shape: {res.shape}")
        data.pop(batch_keys=['x'])
        output = data.union(DataProto.from_dict(tensors={'x': res}))
        return output
    

    def mult_y_by_rank(self, data: DataProto):
        data = data.to('cuda')
        res = data.batch['y'] * self.rank
        print(f"rank {self.rank}, shape: {res.shape}")
        data.pop(batch_keys=['y'])
        output = data.union(DataProto.from_dict(tensors={'y': res}))
        return output

if __name__ == "__main__":
    resource_pool = RayResourcePool(process_on_nodes=[4], max_colocate_count=1, use_gpu=True)

    class_with_args = RayClassWithInitArgs(cls=GPUWorker)
    worker_group = RayWorkerGroup(resource_pool, class_with_args)
    
    tensors = {
        'x': torch.ones((256, 512), dtype=torch.int64), 
        'y': torch.ones((256, 1024), dtype=torch.int64) * 2,
    }   

    data_proto = DataProto.from_dict(tensors=tensors)

    
    # Standard Ray execution
    splitted_data_proto, _ = dispatch_dp_compute_data_proto(worker_group, data_proto)
    splitted_ray_output = worker_group.execute_all_sync("mult_x_by_rank", *splitted_data_proto)
    ray_output = collect_dp_compute_data_proto(worker_group, splitted_ray_output)
    
    splitted_data_proto, _ = dispatch_dp_compute_data_proto(worker_group, ray_output)
    splitted_ray_output = worker_group.execute_all_sync("mult_y_by_rank", *splitted_data_proto)
    ray_output = collect_dp_compute_data_proto(worker_group, splitted_ray_output)
    print("Ray x:", ray_output.batch['x'].shape, ray_output.batch['x'])
    print("Ray y:", ray_output.batch['y'].shape, ray_output.batch['y'])
    
    # RCG execution
    with InputNode() as splitted_input:
        splitted_output = worker_group.bind_all("mult_x_by_rank", splitted_input)
        splitted_output = worker_group.bind_all("mult_y_by_rank", splitted_output)
        dag = MultiOutputNode(splitted_output)
    dag = dag.experimental_compile()
    
    # split
    splitted_data_proto, _ = dispatch_dp_compute_data_proto(worker_group, data_proto)
    # execute
    splitted_output = ray.get(dag.execute(*splitted_data_proto[0]))
    # concatenate
    rcg_output = collect_dp_compute_data_proto(worker_group, splitted_output)
    print("RCG x:", rcg_output.batch['x'].shape, rcg_output.batch['x'])
    print("RCG y:", rcg_output.batch['y'].shape, rcg_output.batch['y'])

