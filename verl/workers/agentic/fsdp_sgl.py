import logging
import os

import torch
from sglang.srt.entrypoints.engine import Engine
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors, all_gather_dict_non_tensors,
                                         broadcast_dict_non_tensor)
from ..sharding_manager.base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


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

        # dp_rank = device_mesh.get_local_rank(0)
        # tp_rank = device_mesh.get_local_rank(1)
        # if tp_rank == 0:
        #     # addr = os.environ["MASTER_ADDR"]
        #     addr = "127.0.0.1"
        #     port = 40000 + random.randint(0, 1000)
        #     print(f"nodedup sharding manager starting weight pg dp_rank: {dp_rank}, addr: {addr}, port: {port}")
        #     def t():
        #         self.inference_engine.init_weights_update_group(
        #             master_address=addr,
        #             master_port=port,
        #             rank_offset=dp_rank * device_mesh.size(1),
        #             world_size=device_mesh.size(0),
        #             group_name=f"weight_update_group_{dp_rank}",
        #             backend="nccl",
        #         )
        #     threading.Thread(target=t).start()
        #     self.update_weight_pg: ProcessGroup = init_custom_process_group(
        #         backend="nccl",
        #         init_method=f"tcp://{addr}:{port}",
        #         world_size=device_mesh.size(0),
        #         rank=dp_rank,
        #     )
        #     print(f"nodedup sharding manager started weight pg dp_rank: {dp_rank}, addr: {addr}, port: {port}")

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
        k, v = next(iter(st.items()))
        print(f"state_dict dtype of {k}: {v.dtype}")
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # print(f'Weight keys: {st.keys()}')
        tensor_list = [(k, v.full_tensor() if isinstance(v, DTensor) else v) for k, v in st.items()]
        param_count = sum([v.numel() for k, v in tensor_list])
        print(f"param count: {param_count}")
        print(f"nodedup sharding manager {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.device_mesh.get_local_rank(1)=} {self.device_mesh.get_local_rank(0)=}")
        if self.device_mesh.get_local_rank(1) == 0:
            print("resuming memory occupation")
            self.inference_engine.resume_memory_occupation()
            print("resumed memory occupation")
        torch.cuda.synchronize()
        if self.device_mesh.get_local_rank(1) == 0:
            self.inference_engine.update_weights_from_tensor(tensor_list)
            # for k, t in tensor_list:
            #     torch.distributed.broadcast(t, src=self.device_mesh.get_rank(), group=self.update_weight_pg, async_op=True)
            #     self.inference_engine.update_weights_from_distributed(k, t.dtype, t.shape)
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del st
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before sglang offload in sharding manager', logger=logger)
        if self.device_mesh.get_local_rank(1) == 0:
            self.inference_engine.release_memory_occupation()
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
        data.batch = allgather_dict_tensors(
            data.batch.contiguous(),
            size=self.device_mesh.size(1),
            group=self.device_mesh.get_group(1),
            dim=0,
        )
        data.non_tensor_batch = all_gather_dict_non_tensors(
            data.non_tensor_batch,
            size=self.device_mesh.size(1),
            group=self.device_mesh.get_group(1),
        )

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        tp_size = self.device_mesh.size(1)
        tp_rank = self.device_mesh.get_local_rank(1)
        src_rank = self.device_mesh.get_local_rank(0) * tp_size
        # obs metrics are dynamically acquired, so we should build a same shape tensor dynamically, communicate shapes and dtypes first
        if tp_rank == 0:
            description: dict = {k: (v.shape, v.dtype) for k, v in data.batch.items()}
            description['batch_size'] = data.batch.batch_size
            lst = [description]
        else:
            lst = [None]
        torch.distributed.broadcast_object_list(lst, src=src_rank, group=self.device_mesh.get_group(1))
        description = lst[0]
        print(f"{self.device_mesh.get_rank()=} {tp_size=} {src_rank=} {tp_rank=}, description: {description=}")
        if tp_rank != 0:
            batch_size = description.pop('batch_size')
            batch = TensorDict(
                {k: torch.empty(shape, dtype=dtype, device='cuda') for k, (shape, dtype) in description.items()}, batch_size=batch_size)
            data = DataProto(batch=batch)
        broadcast_dict_tensor(
            data.batch,
            src=src_rank,
            group=self.device_mesh.get_group(1),
        )
        broadcast_dict_non_tensor(
            data.non_tensor_batch,
            src=src_rank,
            group=self.device_mesh.get_group(1),
        )
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[tp_rank]
        return data
