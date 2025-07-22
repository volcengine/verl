import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.training.initialize import _set_random_seed
from megatron.core.distributed import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from typing import List
from tests.unit_tests.test_utilities import Utils
from loguru import logger
from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBuffer,
)

from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model

# from verl.utils.device import get_device_id, get_device_name


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
    override_ddp_config=None,
):
    """Build the model."""
    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert model_type != ModelType.encoder_and_decoder, (
            "Interleaved schedule not supported for model with both encoder and decoder"
        )
        model = []
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            this_model.model_type = model_type
            model.append(this_model)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, (
                    "Split rank needs to be specified for model with both encoder and decoder"
                )
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        for model_module in model:
            model_module.to(torch.cuda.current_device())

    # Fp16 conversion.
    config: TransformerConfig = get_model_config(model[0])
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        ddp_config_dict = {
            "use_distributed_optimizer": use_distributed_optimizer,
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
        }
        if override_ddp_config is not None:
            ddp_config_dict.update(override_ddp_config)
        ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=ddp_config,
            )
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model


def get_named_tensor_buckets(
    expert_buffers: List[_ParamAndGradBuffer],
    bucket_bytes: int,
    num_experts_per_rank: int,
    ep_rank: int,
):
    # ) -> List[Tuple[str, torch.Tensor, Tuple[int, int, int]]]:
    """Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    def get_flattened_params(expert_buffer, param_infos):
        start_index = param_infos[0][2]
        end_index = param_infos[-1][3]
        return expert_buffer.param_data[start_index:end_index]

    def update_param_name(param_name: str):
        name_prefix, local_expert_id = param_name.split(".weight")
        local_expert_id = int(local_expert_id)
        global_expert_id = num_experts_per_rank * ep_rank + local_expert_id
        return f"{name_prefix}.weight{global_expert_id}"

    for expert_buffer in expert_buffers:
        param_infos = []
        current_size = 0
        for param in expert_buffer.params[::-1]:
            name = update_param_name(expert_buffer.param_to_name[param])
            tensor_size = param.element_size() * param.numel()
            if current_size + tensor_size > bucket_bytes:
                if param_infos:
                    yield param_infos, get_flattened_params(expert_buffer, param_infos)
                param_infos = [
                    (
                        name,
                        param.shape,
                        expert_buffer.param_index_map[param][0],
                        expert_buffer.param_index_map[param][1],
                    )
                ]
                current_size = tensor_size

            else:
                param_infos.append(
                    (
                        name,
                        param.shape,
                        expert_buffer.param_index_map[param][0],
                        expert_buffer.param_index_map[param][1],
                    )
                )
                current_size += tensor_size

        if param_infos:
            yield param_infos, get_flattened_params(expert_buffer, param_infos)


def get_param(
    param_data: torch.Tensor,
    shape: torch.Size,
    start_index: int,
    flattened_experts_start_index: int,
):
    end_index = start_index + shape.numel() - flattened_experts_start_index
    assert end_index <= param_data.numel(), "Requested tensor is out of buffer range"
    buffer_tensor = param_data[start_index:end_index]
    buffer_tensor = buffer_tensor.view(shape)
    return buffer_tensor


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def bucketed_all_gather_param_generator(
    expert_parallel_buffers,
    num_moe_experts,
    ep_size,
    ep_rank,
    pp_rank,
):
    bucket_size = (1 << 30) * 10
    for experts_param_names, flattened_experts in get_named_tensor_buckets(
        expert_parallel_buffers,
        bucket_bytes=bucket_size,
        num_experts_per_rank=num_moe_experts // ep_size,
        ep_rank=ep_rank,
    ):
        all_experts_param_infos = [None for _ in range(ep_size)]
        all_flattened_experts = [torch.empty_like(flattened_experts) for _ in range(ep_size)]
        torch.distributed.all_gather_object(
            all_experts_param_infos,
            experts_param_names,
            group=mpu.get_expert_model_parallel_group(),
        )
        torch.distributed.all_gather(
            all_flattened_experts,
            flattened_experts,
            group=mpu.get_expert_model_parallel_group(),
        )
        for index, flattened_expert in enumerate(all_flattened_experts):
            expert_infos = all_experts_param_infos[index]
            flattened_experts_start_index = expert_infos[0][2]
            for param_name, shape, start_index, _ in expert_infos:
                expert_weight = get_param(
                    flattened_expert,
                    shape,
                    start_index,
                    flattened_experts_start_index,
                )
                logger.info(f"{param_name=} {expert_weight=}")
    pass


def ddp_all_gather_param_generator(
    module,
):
    vpp_size = len(module)
    pass


class TestMoELayerInit:
    def __init__(
        self,
        expert_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
    ) -> None:
        self.expert_model_parallel_size = expert_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        Utils.initialize_model_parallel(
            expert_model_parallel_size=expert_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        )

    def _get(
        self,
        param_data: torch.Tensor,
        shape: torch.Size,
        start_index: int,
        numel: int,
    ) -> torch.Tensor:
        end_index = start_index + shape.numel()
        assert end_index <= numel, "Requested tensor is out of buffer range"
        buffer_tensor = param_data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def test_te_moe_layer(
        self,
    ):
        self.num_moe_experts = 8
        self.grouped_gemm = True
        self.num_layers = 4
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=self.num_layers,
            hidden_size=4096,
            num_attention_heads=64,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=False,
            moe_shared_expert_intermediate_size=1024,
            # moe_token_dispatcher_type=moe_token_dispatcher_type,
            expert_model_parallel_size=self.expert_model_parallel_size,
            moe_router_topk=4,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=self.grouped_gemm,
            moe_ffn_hidden_size=12288,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.num_moe_experts,
            moe_grouped_gemm=self.grouped_gemm,
        )

        def megatron_actor_model_provider(pre_process, post_process):
            gpt_model = GPTModel(
                config=self.transformer_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                max_sequence_length=4,
                pre_process=pre_process,
                post_process=post_process,
            ).cuda()
            return gpt_model

        rank = torch.distributed.get_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()
        ep_size = mpu.get_expert_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        actor_module = get_model(
            model_provider_func=megatron_actor_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
        )
        vpp_size = len(actor_module)
        # model = unwrap_model(actor_module)[0]
        # for name, _ in model.named_parameters():
        #     logger.info(f"{pp_rank=} {name=}")
        for i in range(vpp_size):
            logger.info(f"{pp_rank=} {actor_module[i]=}")
        # ddp_config = DistributedDataParallelConfig(
        #     overlap_grad_reduce=True,
        #     # bucket_size=10000,
        #     use_distributed_optimizer=True,
        # )
        # ddp_model = DistributedDataParallel(
        #     self.transformer_config,
        #     ddp_config=ddp_config,
        #     module=self.gpt_model,
        # )
        # logger.info(
        #     f"{ ep_size= }{ep_rank=} {mpu.get_data_parallel_world_size()=} {mpu.get_data_parallel_rank()=}"
        # )

    def destory(self):
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    test_moe = TestMoELayerInit(
        expert_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    test_moe.test_te_moe_layer()
    test_moe.destory()
