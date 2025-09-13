import time

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def _initialize_kv_caches(self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
    start = time.time()

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # Profiles the peak memory usage of the model to determine how much
    # memory can be allocated for kv cache.
    available_gpu_memory = self.model_executor.determine_available_memory()

    assert len(kv_cache_specs) == len(available_gpu_memory)
    # Get the kv cache tensor size
    self.kv_cache_configs = [
        get_kv_cache_config(vllm_config, kv_cache_spec_one_worker, available_gpu_memory_one_worker)
        for kv_cache_spec_one_worker, available_gpu_memory_one_worker in zip(
            kv_cache_specs, available_gpu_memory
        )
    ]

    # Since we use a shared centralized controller, we need the
    # `kv_cache_config` to be consistent across all workers to make sure
    # all the memory operators can be applied to all workers.
    unify_kv_cache_configs(self.kv_cache_configs)

    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    assert all([cfg.num_blocks == self.kv_cache_configs[0].num_blocks for cfg in self.kv_cache_configs])
    num_gpu_blocks = self.kv_cache_configs[0].num_blocks
    num_cpu_blocks = 0
    scheduler_kv_cache_config = self.kv_cache_configs[0]

    # Initialize kv cache and warmup the execution
    self.model_executor.initialize_from_config(self.kv_cache_configs)

    elapsed = time.time() - start
    logger.info(("init engine (profile, create kv cache, warmup model) took %.2f seconds"), elapsed)
    return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config


EngineCore._initialize_kv_caches = _initialize_kv_caches
