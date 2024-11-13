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

    def mult_by_rank(self, data: DataProto):
        data = data.to('cuda')
        res = data.batch['x'] * self.rank
        print(f"rank {self.rank}, shape: {res.shape}")
        output = DataProto.from_dict(tensors={'values': res})
        return output

if __name__ == "__main__":
    resource_pool = RayResourcePool(process_on_nodes=[4], max_colocate_count=1, use_gpu=True)

    class_with_args = RayClassWithInitArgs(cls=GPUWorker)
    worker_group = RayWorkerGroup(resource_pool, class_with_args)
    
    tensors = {
        'x': torch.ones((256, 512), dtype=torch.int64),
    }   

    data_proto = DataProto.from_dict(tensors=tensors)
    
    # Standard Ray execution
    splitted_data_proto, _ = dispatch_dp_compute_data_proto(worker_group, data_proto)
    ray_output = worker_group.execute_all_sync("mult_by_rank", *splitted_data_proto)
    ray_output = collect_dp_compute_data_proto(worker_group, ray_output)
    
    # RCG execution
    with InputNode() as splitted_input:
        splitted_output = worker_group.bind_all("mult_by_rank", splitted_input)
        dag = MultiOutputNode(splitted_output)
    dag = dag.experimental_compile()
    
    # split
    splitted_data_proto, _ = dispatch_dp_compute_data_proto(worker_group, data_proto)
    # execute
    splitted_output = ray.get(dag.execute(*splitted_data_proto[0]))
    # concatenate
    rcg_output = collect_dp_compute_data_proto(worker_group, splitted_output)

    assert torch.allclose(ray_output.batch['values'], rcg_output.batch['values'])
    print("RCG:", rcg_output.batch['values'].shape, rcg_output.batch['values'], rcg_output.batch['values'].device)
