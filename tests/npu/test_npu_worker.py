import pytest
import ray
import torch


@pytest.fixture
def ray_cluster_with_npus():
    # Assume in the NPU environment.
    ray.init()
    yield
    ray.shutdown()


@ray.remote(resources={"NPU": 1})
class Worker:
    def __init__(self):
        pass
    
    def test_torch_npu_avalable(self):
        available = False
        try:
            import torch_npu
            available = torch.npu.is_available()
        except Exception as e:
            pass
        return available
        

def test_torch_npu(ray_cluster_with_npus):
    worker = Worker.remote()
    assert ray.get(worker.remote())
