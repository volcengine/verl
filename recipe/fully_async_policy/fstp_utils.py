
import torch
import torch.distributed as dist
from typing import Tuple, Dict, Optional
from torch.distributed.tensor import Shard, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec

def fsdp2_sharded_save_to_cpu(
    model: torch.nn.Module  # 你的FSDP2包装模型（如FSDPQwen2ForCausalLM）
) -> Tuple[Dict[str, Tuple[torch.Tensor, DTensorSpec]], DTensorSpec]:
    """
    分片保存：每个进程仅保存自身GPU上的DTensor本地分片到CPU内存
    Args:
        model: FSDP2包装后的模型，参数为DTensor类型
    Returns:
        cpu_sharded_state: 本进程的CPU分片字典，key=参数名，value=(CPU分片张量, 原DTensorSpec)
        global_spec: 首个参数的DTensorSpec（用于加载时验证全局规则）
    """
    cpu_sharded_state = {}
    global_spec = None  # 记录全局分片规则（所有参数的spec应一致）

    for param_name, param in model.named_parameters():
        # 仅处理DTensor类型的分片参数（FSDP2的核心参数）
        if not isinstance(param, DTensor):
            # 非分片参数（如BatchNorm的running_mean）也按本地数据保存
            cpu_tensor = param.detach().cpu()
            cpu_sharded_state[param_name] = (cpu_tensor, None)
            continue

        # 记录全局分片规则（取首个DTensor的spec，确保所有参数遵循同一规则）
        if global_spec is None:
            global_spec = param._spec
            assert hasattr(global_spec, "device_mesh"), "DTensorSpec必须包含device_mesh属性"
            assert hasattr(global_spec, "placements"), "DTensorSpec必须包含placements属性"

        # 1. 提取当前GPU的本地分片数据（_local_tensor）
        local_gpu_tensor = param._local_tensor  # 你的DTensor类定义的本地分片属性
        # 2. 转移到CPU内存并脱离计算图
        local_cpu_tensor = local_gpu_tensor.detach().cpu()
        # 3. 保存CPU分片 + 原DTensorSpec（确保分片规则不变）
        cpu_sharded_state[param_name] = (local_cpu_tensor, param._spec)

    assert global_spec is not None, "模型中未找到DTensor类型参数，可能未启用FSDP2分片"
    return cpu_sharded_state, global_spec


def fsdp2_sharded_load_from_cpu(
    model: torch.nn.Module,
    cpu_sharded_state: Dict[str, Tuple[torch.Tensor, Optional[DTensorSpec]]],
    target_spec: DTensorSpec
) -> None:
    """
    分片加载：每个进程仅加载自身负责的CPU分片到GPU，保持分片规则不变
    Args:
        model: 待恢复的FSDP2模型（需与保存时结构一致）
        cpu_sharded_state: 本进程从CPU内存读取的分片数据（来自fsdp2_sharded_save_to_cpu）
        target_spec: 保存时的全局DTensorSpec（用于验证分片规则一致性）
    """
    # 验证device_mesh是否一致（核心：确保加载的分片对应原GPU）
    current_device_mesh = None
    for param in model.parameters():
        if isinstance(param, DTensor):
            current_device_mesh = param._spec.device_mesh
            break
    assert current_device_mesh is not None, "待加载模型未初始化DTensor参数"
    assert current_device_mesh == target_spec.device_mesh, \
        f"加载时device_mesh与原分片不一致！原：{target_spec.device_mesh}，当前：{current_device_mesh}"

    for param_name, param in model.named_parameters():
        # 跳过未在保存状态中的参数（如新增参数）
        if param_name not in cpu_sharded_state:
            continue

        # 提取CPU分片数据和原Spec
        local_cpu_tensor, saved_spec = cpu_sharded_state[param_name]

        # 分情况处理：DTensor分片参数 vs 普通参数
        if isinstance(param, DTensor):
            # 1. 验证分片规则一致性（placements必须与原Spec一致）
            assert saved_spec is not None, f"参数{param_name}的保存状态中缺少DTensorSpec"
            assert saved_spec.placements == target_spec.placements, \
                f"参数{param_name}的分片策略与全局规则不一致！"

            # 2. 将CPU分片数据转移到当前GPU（param._local_tensor的设备）
            target_device = param._local_tensor.device
            local_gpu_tensor = local_cpu_tensor.to(target_device)

            # 3. 恢复到DTensor的本地分片（直接赋值给_local_tensor，保持spec不变）
            param._local_tensor.copy_(local_gpu_tensor)

        else:
            # 普通参数：直接加载到原设备
            target_device = param.device
            param.data.copy_(local_cpu_tensor.to(target_device))

    # 进程同步：确保所有进程加载完成后再继续
    dist.barrier()
