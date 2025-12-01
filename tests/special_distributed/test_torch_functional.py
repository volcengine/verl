import torch
import os

from verl.utils.torch_functional import allgather_dict_into_dict


if __name__ == '__main__':
    torch.distributed.init_process_group(backend='gloo')

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    metrics_dict = {
        'loss': [0 + rank, 1 + rank, 2 + rank],
        'grad_norm': rank
    }

    result = allgather_dict_into_dict(data=metrics_dict, group=None)

    assert result['loss'] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert result['grad_norm'] == [0, 1, 2, 3]

    print(result)
