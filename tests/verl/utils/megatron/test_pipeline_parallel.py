from verl.utils.megatron.pipeline_parallel import make_batch_generator


def test_make_batch_generator_no_vpp():
    batches = [1, 2, 3]
    vpp_size = 1
    generator = make_batch_generator(batches, vpp_size)
    assert list(generator) == batches


def test_make_batch_generator_with_vpp():
    batches = [{"data": 1}, {"data": 2}]
    vpp_size = 2
    generators = make_batch_generator(batches, vpp_size)
    assert isinstance(generators, list)
    assert len(generators) == vpp_size

    # Check each generator yields the original batches
    for gen in generators:
        assert list(gen) == batches


def test_make_batch_generator_empty():
    batches = []
    vpp_size = 1
    generator = make_batch_generator(batches, vpp_size)
    assert list(generator) == []

    vpp_size = 3
    generators = make_batch_generator(batches, vpp_size)
    assert len(generators) == vpp_size
    for gen in generators:
        assert list(gen) == []
