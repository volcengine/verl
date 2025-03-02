import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version

from ..sharding_manager.base import BaseShardingManager
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import init_custom_process_group
from torch.distributed import ProcessGroup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPSGLShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: Engine,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # addr = os.environ["MASTER_ADDR"]
        # dp_rank = device_mesh.get_local_rank(0)
        # self.rank = device_mesh.get_rank()
        # if self.rank == 0:
        #     self.update_weight_pg: ProcessGroup = init_custom_process_group(
        #         backend="nccl",
        #         init_method=f"tcp://{addr}:65500",
        #         world_size=device_mesh.size(0),
        #         rank=dp_rank,
        #     )
        # self.inference_engine.init_weights_update_group(
        #     master_address=addr,
        #     master_port=65500,
        #     rank_offset=dp_rank,
        #     world_size=device_mesh.size(0),
        #     group_name="weight_update_group",
        #     backend="nccl",
        # )

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        st = self.module.state_dict()
        dtype = next(iter(st.values())).dtype
        print(f"state_dict dtype: {dtype}")
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        self.inference_engine.update_weights_from_tensor([(k, v) for k, v in st.items()])
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del st
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before sglang offload in sharding manager', logger=logger)
        log_gpu_memory_usage('After sglang offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=vllm_ps.get_tensor_model_parallel_world_size(),
                                                group=vllm_ps.get_tensor_model_parallel_group(),
                                                dim=0)
        else:
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=vllm_ps.get_tensor_model_parallel_world_size(),
                                                group=vllm_ps.get_tensor_model_parallel_group().device_group,
                                                dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        local_world_size = vllm_ps.get_tensor_model_parallel_world_size()
        src_rank = (torch.distributed.get_rank() // local_world_size) * local_world_size
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            broadcast_dict_tensor(data.batch, src=src_rank, group=vllm_ps.get_tensor_model_parallel_group())
        else:
            broadcast_dict_tensor(data.batch,
                                  src=src_rank,
                                  group=vllm_ps.get_tensor_model_parallel_group().device_group)
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
