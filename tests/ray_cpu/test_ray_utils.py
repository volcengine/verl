import pytest
import ray

from verl.utils.ray_utils import parallel_put


# Initialize Ray for testing if not already done globally
@pytest.fixture()
def init_ray():
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


def test_parallel_put_basic(init_ray):
    data = [1, "hello", {"a": 2}, [3, 4]]
    refs = parallel_put(data)
    assert len(refs) == len(data)
    retrieved_data = [ray.get(ref) for ref in refs]
    assert retrieved_data == data


def test_parallel_put_empty(init_ray):
    data = []
    refs = parallel_put(data)
    assert len(refs) == 0


def test_parallel_put_workers(init_ray):
    data = list(range(20))
    # Test with specific number of workers
    refs = parallel_put(data, max_workers=4)
    assert len(refs) == len(data)
    retrieved_data = [ray.get(ref) for ref in refs]
    assert retrieved_data == data
    # Test with default workers (should cap)
    refs_default = parallel_put(data)
    assert len(refs_default) == len(data)
    retrieved_data_default = [ray.get(ref) for ref in refs_default]
    assert retrieved_data_default == data
