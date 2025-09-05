import random
import torch
from teacher import TeacherClient

def main():
    teacher_client = TeacherClient("127.0.0.1", 15555)
    tokens = [[random.randint(1, 99999) for _ in range(100)] for _ in range(2)]
    tokens[0][40] = 128858
    _, teacher_topk_logps, teacher_topk_indices = teacher_client.submit(tokens).result()
    assert all(logps.shape == (100, 256) for logps in teacher_topk_logps)
    assert all(logps.dtype == torch.float32 for logps in teacher_topk_logps)
    assert all(indices.shape == (100, 256) for indices in teacher_topk_indices)
    assert all(indices.dtype == torch.int32 for indices in teacher_topk_indices)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()