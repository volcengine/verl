import os

os.environ['NCCL_DEBUG'] = 'WARN'

from verl.protocol import all_gather_data_proto, DataProto
from verl.utils.distributed import initialize_global_process_group
import torch
import torch.distributed
import numpy as np


def test_all_gather_data_proto():
    device_mesh = torch.distributed.device_mesh.init_device_mesh('cuda', mesh_shape=[2, 2], mesh_dim_names=['dp', 'tp'])

    global_rank = torch.distributed.get_rank()

    obs = torch.tensor([[1 * global_rank, 2 * global_rank + 1], [3 * global_rank, 4 * global_rank + 1]])

    labels = ['a', 'b'] if global_rank % 2 == 0 else ['b', 'a']
    labels = np.array(labels, dtype=object)
    data = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels}, meta_info={'info': 'test_info'})

    all_gather_data_proto(data=data, process_group=device_mesh.get_group('dp'))

    if global_rank == 0:
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device='cuda')
        expected_labels = ['a', 'b', 'a', 'b']
    elif global_rank == 1:
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device='cuda')
        expected_labels = ['b', 'a', 'b', 'a']
    elif global_rank == 2:
        expected_obs = torch.tensor([[0, 1], [0, 1], [2, 5], [6, 9]], device='cuda')
        expected_labels = ['a', 'b', 'a', 'b']
    elif global_rank == 3:
        expected_obs = torch.tensor([[1, 3], [3, 5], [3, 7], [9, 13]], device='cuda')
        expected_labels = ['b', 'a', 'b', 'a']

    torch.testing.assert_close(data.batch['obs'], expected_obs, atol=0, rtol=0)
    assert (data.non_tensor_batch['labels'] == expected_labels).all()
    assert data.meta_info == {'info': 'test_info'}


if __name__ == '__main__':
    local_rank, rank, world_size = initialize_global_process_group()
    test_all_gather_data_proto()
