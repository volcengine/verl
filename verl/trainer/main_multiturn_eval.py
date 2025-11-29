"""
Multi-turn tool call evaluation module for batch processing of tool-use conversations.
Independent from PPO training, supports batch reading, tool calling, and reward recording.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, RewardModelWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@hydra.main(config_path="config", config_name="multiturn_eval", version_base=None)
def main(config: DictConfig):
    """Main entry point for multi-turn evaluation with Hydra configuration."""
    run_multiturn_evaluation(config)


def initialize_ray_cluster(config: DictConfig):
    """Initialize Ray with the same runtime env defaults used during training."""
    if ray.is_initialized():
        logger.info("Ray is already initialized, skipping initialization")
        return

    ray_kwargs = config.get("ray_kwargs", {}) or {}
    ray_init_kwargs = ray_kwargs.get("ray_init", {})
    runtime_env_cfg = ray_init_kwargs.get("runtime_env", {})
    default_runtime_env = get_ppo_ray_runtime_env()
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_cfg)
    ray_init_with_env = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    ray_init_dict = OmegaConf.to_container(ray_init_with_env, resolve=True)
    print(f"ray init dict: {ray_init_dict}")

    # Check if we're connecting to an existing cluster and remove resource specifications
    # to avoid conflicts when connecting to existing Ray clusters
    try:
        ray.init(**ray_init_dict)
    except:
        _remove_num_resources(ray_init_dict)
        logger.info(
            "Connecting to existing Ray cluster detected, removing num_cpus/num_gpus from init parameters"
        )
        print(f"ray init dict: {ray_init_dict}")
        ray.init(**ray_init_dict)

    logger.info("Initialized Ray cluster with config: %s", ray_init_dict)


def _remove_num_resources(ray_init_dict: Dict[str, Any]) -> None:
    """Remove num_cpus and num_gpus from the init dict when connecting to another Ray cluster."""
    for key in ("num_cpus", "num_gpus"):
        ray_init_dict.pop(key, None)


def load_tokenizer_and_processor(config: DictConfig):
    """Load tokenizer/processor based on actor rollout model configuration."""
    model_cfg = config.actor_rollout_ref.model
    local_model_path = copy_to_local(model_cfg.path, use_shm=model_cfg.get("use_shm", False))
    tokenizer_path = model_cfg.get("tokenizer_path") or local_model_path
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    processor = None
    if model_cfg.get("use_processor", False):
        processor = hf_processor(tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True)

    chat_template = model_cfg.get("custom_chat_template") or model_cfg.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
        if processor:
            processor.chat_template = chat_template

    return tokenizer, processor


def _resolve_eval_files(data_cfg: DictConfig):
    eval_files = data_cfg.get("eval_files")
    if eval_files is None:
        eval_files = data_cfg.get("val_files")
    if eval_files is None:
        eval_files = data_cfg.get("train_files")
    if eval_files is None:
        raise ValueError("Please provide data.eval_files (or val/train files) for evaluation.")
    return eval_files


def create_eval_dataloader(config: DictConfig, tokenizer, processor):
    """Create RL dataset/dataloader that matches the PPO training pipeline."""
    data_cfg = config.data
    eval_files = _resolve_eval_files(data_cfg)
    dataset = create_rl_dataset(eval_files, data_cfg, tokenizer, processor, is_train=False)
    sampler = create_rl_sampler(data_cfg, dataset)

    batch_size = (
        data_cfg.get("eval_batch_size")
        or data_cfg.get("val_batch_size")
        or data_cfg.get("train_batch_size")
    )
    if batch_size is None:
        raise ValueError("Set data.eval_batch_size (or val/train batch size) for evaluation.")

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=data_cfg.get("dataloader_num_workers", 0),
        drop_last=False,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return dataset, dataloader


def ensure_batch_uids(batch: DataProto):
    """Attach unique ids to each sample if they are missing."""
    if "uid" not in batch.non_tensor_batch:
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
            dtype=object,
        )


def _maybe_patch_device_mesh() -> str:
    """
    Torch 2.8.0 DeviceMesh does not expose `_dim_group_names`, but
    FSDP2 sharded load path in torch.distributed assumes it exists.
    Add a lightweight compatibility property so checkpoint loading
    does not crash on AttributeError.
    """
    try:
        from torch.distributed.device_mesh import DeviceMesh
    except Exception as e:  # pragma: no cover - defensive
        return f"skip:torch_import_failed:{e}"

    if hasattr(DeviceMesh, "_dim_group_names"):
        return "skip:already_patched"

    def _dim_group_names(self):
        infos = getattr(self, "_dim_group_infos", None)
        if infos is not None:
            names = []
            for idx, info in enumerate(infos):
                name = getattr(info, "name", None)
                if name is None:
                    mesh_dims = getattr(self, "mesh_dim_names", None)
                    name = mesh_dims[idx] if mesh_dims and idx < len(mesh_dims) else None
                names.append(name)
            return names

        mesh_dims = getattr(self, "mesh_dim_names", None)
        return list(mesh_dims) if mesh_dims is not None else []

    DeviceMesh._dim_group_names = property(_dim_group_names)
    return "patched"


def _patch_device_mesh_on_workers(actor_rollout_wg: RayWorkerGroup):
    """Apply the DeviceMesh compatibility patch inside all rollout workers."""
    try:
        import ray
    except Exception:  # pragma: no cover - defensive
        return

    def _register_device_mesh_process_groups(worker_container):
        """Register simple named process groups (fsdp/ddp/sp etc.) to default WORLD group."""
        try:
            import torch.distributed as dist
            from torch.distributed.distributed_c10d import _get_group_size_by_name, _register_process_group
        except Exception as e:  # pragma: no cover - defensive
            return [f"skip:dist_import_failed:{e}"]

        if not dist.is_initialized():
            return ["skip:dist_not_initialized"]

        def _register_for_worker(worker):
            results = []
            meshes = []
            if hasattr(worker, "device_mesh"):
                meshes.append(getattr(worker, "device_mesh"))
            ulysses_mesh = getattr(worker, "ulysses_device_mesh", None)
            if ulysses_mesh is not None:
                meshes.append(ulysses_mesh)

            for mesh in meshes:
                if mesh is None:
                    continue
                names = getattr(mesh, "mesh_dim_names", None) or getattr(mesh, "_dim_group_names", None)
                if not names:
                    names = ["fsdp"]
                for name in names:
                    if not name:
                        continue
                    try:
                        _get_group_size_by_name(name)
                        results.append(f"skip:exists:{name}")
                        continue
                    except Exception:
                        pass
                    try:
                        _register_process_group(name, dist.group.WORLD)
                        results.append(f"registered:{name}")
                    except Exception as e:  # pragma: no cover - defensive
                        results.append(f"error:register:{name}:{e}")
            return results

        if hasattr(worker_container, "worker_dict"):
            all_results = []
            for sub_worker in worker_container.worker_dict.values():
                all_results.extend(_register_for_worker(sub_worker))
            return all_results
        return _register_for_worker(worker_container)

    def _patch_and_register(worker_container):
        patch_status = _maybe_patch_device_mesh()
        register_result = _register_device_mesh_process_groups(worker_container)
        return {"patch": patch_status, "register": register_result}

    patch_results = ray.get(
        [
            worker.__ray_call__.remote(lambda self: _patch_and_register(self))  # type: ignore[misc]
            for worker in actor_rollout_wg.workers
        ]
    )
    logger.warning("DeviceMesh patch/register results: %s", patch_results)


def prepare_generation_batch(batch: DataProto, async_mode: bool) -> DataProto:
    """Subset the batch to the tensors required by rollout workers."""
    reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"}
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_keys_to_pop = [k for k in batch.non_tensor_batch.keys() if k not in reward_model_keys]
    gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_keys_to_pop)
    if async_mode:
        # AgentLoop needs access to metadata (tool specs, etc.) on each request.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    return gen_batch


def load_checkpoint_if_needed(config: DictConfig, actor_rollout_wg: RayWorkerGroup):
    """Load checkpoint if checkpoint_dir is specified in config.
    
    This function loads model weights from a checkpoint directory, similar to how
    ray_trainer.py loads checkpoints during training resume.
    
    Note on checkpoint_dir vs model.path:
    - model.path is ALWAYS required for loading tokenizer and processor
    - checkpoint_dir is OPTIONAL: if provided, model weights will be loaded from checkpoint
      (overriding the initial weights loaded from model.path during init_model)
    - If checkpoint_dir is None, model weights will come from model.path (original model evaluation)
    
    Args:
        config: Configuration object containing checkpoint_dir
        actor_rollout_wg: Actor rollout worker group to load checkpoint into
        
    Raises:
        ValueError: If checkpoint_dir is invalid or checkpoint not found
    """
    checkpoint_dir = config.get("checkpoint_dir")
    if checkpoint_dir is None:
        logger.info("No checkpoint_dir specified, using model weights from model.path (original model)")
        return
    
    # Resolve checkpoint directory path
    if not os.path.isabs(checkpoint_dir):
        working_dir = os.getcwd()
        checkpoint_dir = os.path.join(working_dir, checkpoint_dir)
    
    # Check if checkpoint_dir points to a specific global_step_* folder or parent directory
    if "global_step_" in checkpoint_dir:
        # Direct path to a specific checkpoint
        global_step_folder = checkpoint_dir
        if not os.path.isabs(global_step_folder):
            working_dir = os.getcwd()
            global_step_folder = os.path.join(working_dir, global_step_folder)
    else:
        # Parent directory, find latest checkpoint
        global_step_folder = find_latest_ckpt_path(checkpoint_dir)
        if global_step_folder is None:
            raise ValueError(
                f"No checkpoint found in {checkpoint_dir}. "
                f"Please ensure the directory contains global_step_* subdirectories "
                f"or specify the full path to a specific checkpoint."
            )
    
    logger.info(f"Loading checkpoint from: {global_step_folder}")
    
    # Extract global step number for logging
    if "global_step_" in global_step_folder:
        try:
            global_step = int(global_step_folder.split("global_step_")[-1])
            logger.info(f"Checkpoint global step: {global_step}")
        except ValueError:
            logger.warning(f"Could not parse global step from path: {global_step_folder}")
    
    # Construct actor checkpoint path (same structure as ray_trainer.py)
    actor_path = os.path.join(global_step_folder, "actor")
    if not os.path.exists(actor_path):
        raise ValueError(
            f"Actor checkpoint not found at {actor_path}. "
            f"Expected checkpoint structure: {global_step_folder}/actor/"
        )
    
    # Load checkpoint (del_local_after_load=False for evaluation)
    logger.info(f"Loading actor checkpoint from: {actor_path}")
    try:
        actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=False)
    except RuntimeError as e:
        # Torch 2.8 FSDP may fail if named process groups were not registered.
        if "process group" in str(e) and "fsdp" in str(e).lower():
            logger.warning("Checkpoint load hit process group error, retrying after registering device mesh groups.")
            _patch_device_mesh_on_workers(actor_rollout_wg)
            actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=False)
        else:
            raise
    logger.info("Checkpoint loaded successfully")


def initialize_worker_groups(
    config: DictConfig,
) -> tuple[AgentLoopManager | RayWorkerGroup, RayWorkerGroup, Optional[RayWorkerGroup]]:
    """Create rollout (and optional reward) workers using the same logic as PPO training."""

    rollout_cfg = config.actor_rollout_ref.rollout
    actor_strategy = config.actor_rollout_ref.actor.strategy
    if actor_strategy in {"fsdp", "fsdp2"}:
        actor_worker_cls = AsyncActorRolloutRefWorker if rollout_cfg.mode == "async" else ActorRolloutRefWorker
    elif actor_strategy == "megatron":
        from verl.workers.megatron_workers import ActorRolloutRefWorker as MegaActorWorker
        from verl.workers.megatron_workers import AsyncActorRolloutRefWorker as MegaAsyncActorWorker

        actor_worker_cls = MegaAsyncActorWorker if rollout_cfg.mode == "async" else MegaActorWorker
    else:
        raise NotImplementedError(f"Unsupported actor strategy: {actor_strategy}")

    role_worker_mapping = {Role.ActorRollout: ray.remote(actor_worker_cls)}

    reward_wg_cls = None
    if config.reward_model.enable:
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl in ["auto", "enable"]:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                reward_worker_impl = RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker as MegatronRewardWorker

                reward_worker_impl = MegatronRewardWorker
            else:
                raise NotImplementedError(f"Unsupported reward strategy: {config.reward_model.strategy}")
        elif use_legacy_worker_impl == "disable":
            from verl.workers.roles import RewardModelWorker as NewRewardWorker

            reward_worker_impl = NewRewardWorker
            logger.info("Using new reward worker implementation")
        else:
            raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        reward_wg_cls = ray.remote(reward_worker_impl)
        role_worker_mapping[Role.RewardModel] = reward_wg_cls

    trainer_cfg = config.get("trainer", {})
    n_gpus_per_node = trainer_cfg.get("n_gpus_per_node", 1)
    nnodes = trainer_cfg.get("nnodes", 1)
    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [n_gpus_per_node] * nnodes}
    mapping = {Role.ActorRollout: global_pool_id}

    if config.reward_model.enable:
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0 or config.reward_model.nnodes <= 0:
                raise ValueError("reward_model resource pool sizes must be positive.")
            reward_pool_id = "reward_pool"
            resource_pool_spec[reward_pool_id] = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            mapping[Role.RewardModel] = reward_pool_id
        else:
            mapping[Role.RewardModel] = global_pool_id

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    actor_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    resource_pool_to_cls[actor_pool]["actor_rollout"] = actor_cls

    if config.reward_model.enable:
        rm_pool = resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(role_worker_mapping[Role.RewardModel], config=config.reward_model)
        resource_pool_to_cls[rm_pool]["rm"] = rm_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    reward_wg = None
    if config.reward_model.enable:
        reward_wg = all_wg["rm"]
        reward_wg.init_model()

    if rollout_cfg.mode == "async":
        agent_loop_handle = AgentLoopManager(config=config, worker_group=actor_rollout_wg, rm_wg=reward_wg)
    else:
        agent_loop_handle = actor_rollout_wg

    return agent_loop_handle, actor_rollout_wg, reward_wg


def _determine_size_divisor(actor_rollout_wg: RayWorkerGroup, config: DictConfig, async_mode: bool) -> int:
    if not async_mode:
        return actor_rollout_wg.world_size
    agent_cfg = config.actor_rollout_ref.rollout.agent
    return max(1, agent_cfg.get("num_workers", 1))


def run_generation_step(
    agent_handle: AgentLoopManager | RayWorkerGroup,
    actor_rollout_wg: RayWorkerGroup,
    batch: DataProto,
    tokenizer,
    config: DictConfig,
) -> tuple[DataProto, float]:
    async_mode = isinstance(agent_handle, AgentLoopManager)
    gen_batch = prepare_generation_batch(batch, async_mode)
    gen_batch.meta_info["eos_token_id"] = tokenizer.eos_token_id
    gen_batch.meta_info["pad_token_id"] = tokenizer.pad_token_id
    gen_batch.meta_info["recompute_log_prob"] = False
    gen_batch.meta_info["validate"] = True
    val_kwargs = config.actor_rollout_ref.rollout.get("val_kwargs", None)
    if val_kwargs and val_kwargs.get("do_sample") is not None:
        gen_batch.meta_info["do_sample"] = val_kwargs.do_sample

    size_divisor = _determine_size_divisor(actor_rollout_wg, config, async_mode)
    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
    start = time.time()
    if async_mode:
        output_padded = agent_handle.generate_sequences(prompts=gen_batch_padded)
    else:
        output_padded = agent_handle.generate_sequences(data=gen_batch_padded)
    generation_time = time.time() - start
    timing = output_padded.meta_info.pop("timing", None)
    if timing:
        logger.debug("Generation timing stats: %s", timing)
    output_batch = unpad_dataproto(output_padded, pad_size=pad_size)
    return output_batch, generation_time


def compute_reward_outputs(batch: DataProto, reward_wg: Optional[RayWorkerGroup], reward_fn):
    """Compute reward outputs for evaluation batch.
    
    Args:
        batch: DataProto containing prompts and responses
        reward_wg: Optional reward model worker group
        reward_fn: Reward manager function
        
    Returns:
        Tuple of (scores list, reward_result dict)
    """
    if reward_wg is not None:
        reward_tensor = reward_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
    if reward_fn is None:
        batch_size = batch.batch["attention_mask"].shape[0]
        scores = [0.0] * batch_size
        return scores, {"reward_tensor": None, "reward_extra_info": {}}
    
    try:
        reward_result = reward_fn(batch, return_dict=True)
        reward_tensor = reward_result.get("reward_tensor")
        if reward_tensor is None:
            batch_size = batch.batch["attention_mask"].shape[0]
            scores = [0.0] * batch_size
        else:
            scores = reward_tensor.sum(-1).detach().cpu().tolist()
    except Exception as e:
        logger.warning(f"Error in reward_fn: {e}, using zero rewards")
        batch_size = batch.batch["attention_mask"].shape[0]
        scores = [0.0] * batch_size
        try:
            # Fallback to non-dict return
            reward_tensor = reward_fn(batch)
            if reward_tensor is not None:
                scores = reward_tensor.sum(-1).detach().cpu().tolist()
            reward_result = {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        except Exception as e2:
            logger.error(f"Error in reward_fn fallback: {e2}")
            reward_result = {"reward_tensor": None, "reward_extra_info": {}}
    
    return scores, reward_result

def collect_sample_records(
    combined_batch: DataProto,
    generation_time: float,
    reward_scores: List[float],
    reward_result: Dict[str, Any],
    rollout_metrics: Optional[List[Dict[str, Any]]],
    tokenizer,
):
    prompts = tokenizer.batch_decode(combined_batch.batch["prompts"], skip_special_tokens=True)
    responses = tokenizer.batch_decode(combined_batch.batch["responses"], skip_special_tokens=True)
    sample_ids = combined_batch.non_tensor_batch["uid"].tolist()
    num_turns = combined_batch.non_tensor_batch.get("__num_turns__", np.ones(len(responses), dtype=np.int32))
    reward_extra = reward_result.get("reward_extra_info", {})

    # Extract tool_rewards and interaction_rewards from non_tensor_batch
    # For agent loop: extra_fields contains "tool_rewards" (list) and "turn_scores" (list)
    # For sglang rollout: reward_scores contains tool_reward_scores (dict) and user_turn_rewards (list)
    tool_rewards_list = combined_batch.non_tensor_batch.get("tool_rewards", None)
    turn_scores_list = combined_batch.non_tensor_batch.get("turn_scores", None)
    reward_scores_dict = combined_batch.non_tensor_batch.get("reward_scores", None)

    records = []
    for idx, sample_id in enumerate(sample_ids):
        record = {
            "sample_id": sample_id,
            "prompt": prompts[idx],
            "response": responses[idx],
            "turn_score": reward_scores[idx],
            "num_turns": int(num_turns[idx]) if len(num_turns) else 0,
            "generation_time": generation_time,
        }
        
        # Extract tool_rewards (from agent loop extra_fields)
        if tool_rewards_list is not None and idx < len(tool_rewards_list):
            tool_rewards = tool_rewards_list[idx]
            if isinstance(tool_rewards, (list, np.ndarray)):
                record["tool_rewards"] = list(tool_rewards) if isinstance(tool_rewards, np.ndarray) else tool_rewards
            else:
                record["tool_rewards"] = tool_rewards
        
        # Extract interaction_rewards/turn_scores (from agent loop extra_fields)
        if turn_scores_list is not None and idx < len(turn_scores_list):
            turn_scores = turn_scores_list[idx]
            if isinstance(turn_scores, (list, np.ndarray)):
                record["interaction_rewards"] = list(turn_scores) if isinstance(turn_scores, np.ndarray) else turn_scores
            else:
                record["interaction_rewards"] = turn_scores
        
        # Extract tool_rewards and user_turn_rewards (from sglang rollout reward_scores)
        if reward_scores_dict is not None and idx < len(reward_scores_dict):
            reward_scores_item = reward_scores_dict[idx]
            if isinstance(reward_scores_item, dict):
                # Extract tool_reward_scores (dict with tool names as keys)
                tool_reward_scores = {k: v for k, v in reward_scores_item.items() if k != "user_turn_rewards"}
                if tool_reward_scores:
                    record["tool_rewards"] = tool_reward_scores
                # Extract user_turn_rewards (list)
                if "user_turn_rewards" in reward_scores_item:
                    user_turn_rewards = reward_scores_item["user_turn_rewards"]
                    if isinstance(user_turn_rewards, (list, np.ndarray)):
                        record["interaction_rewards"] = list(user_turn_rewards) if isinstance(user_turn_rewards, np.ndarray) else user_turn_rewards
                    else:
                        record["interaction_rewards"] = user_turn_rewards
        
        if rollout_metrics and idx < len(rollout_metrics):
            record["agent_metrics"] = rollout_metrics[idx]
        for key, values in reward_extra.items():
            if len(values) > idx:
                record[f"reward_extra.{key}"] = values[idx]
        records.append(record)
    return records


def aggregate_summary(turn_scores: List[float], generation_times: List[float], reward_times: List[float]) -> Dict[str, Any]:
    total_samples = len(turn_scores)
    return {
        "total_samples": total_samples,
        "mean_turn_score": float(np.mean(turn_scores)) if turn_scores else 0.0,
        "std_turn_score": float(np.std(turn_scores)) if turn_scores else 0.0,
        "total_generation_time": float(sum(generation_times)),
        "total_reward_time": float(sum(reward_times)),
        "avg_generation_time_per_sample": float(np.mean(generation_times)) if generation_times else 0.0,
        "avg_reward_time_per_sample": float(np.mean(reward_times)) if reward_times else 0.0,
    }


def append_results_to_file(records: List[Dict[str, Any]], scores_path: Path, is_first_batch: bool):
    """Append records to output file incrementally to avoid memory overflow.
    
    Args:
        records: List of record dictionaries to append
        scores_path: Path to the output file
        is_first_batch: Whether this is the first batch (determines write mode)
    """
    if not records:
        return
    
    if scores_path.suffix == ".parquet":
        # For parquet, use pandas append mode
        df = pd.DataFrame(records)
        if is_first_batch:
            df.to_parquet(scores_path, index=False, engine='pyarrow')
        else:
            # Append to existing parquet file
            existing_df = pd.read_parquet(scores_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(scores_path, index=False, engine='pyarrow')
    elif scores_path.suffix == ".jsonl":
        # JSONL format: append line by line (recommended for large datasets)
        mode = 'w' if is_first_batch else 'a'
        with open(scores_path, mode, encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    else:
        # JSON format: append to array (less efficient for large files)
        if is_first_batch:
            with open(scores_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        else:
            # Read existing, append, and write back
            with open(scores_path, 'r', encoding='utf-8') as f:
                existing_records = json.load(f)
            existing_records.extend(records)
            with open(scores_path, 'w', encoding='utf-8') as f:
                json.dump(existing_records, f, indent=2, ensure_ascii=False)


def save_summary_and_trace(metrics: Dict[str, Any], trace_records: List[Dict[str, Any]], config: DictConfig):
    """Save evaluation summary and trace (sample records) at the end of evaluation.
    
    Args:
        metrics: Summary metrics dictionary
        trace_records: Sample records for trace (first 100)
        config: Configuration object
    """
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output.get("trace_path") and trace_records:
        trace_path = output_dir / config.output.trace_path
        trace_payload = {
            "config": OmegaConf.to_container(config, resolve=True),
            "records": trace_records,
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved evaluation trace to %s", trace_path)

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation summary to %s", summary_path)


def run_multiturn_evaluation(config: DictConfig):
    """Main entry that mirrors PPO rollout but skips optimization."""
    logger.info("Starting multi-turn evaluation...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    initialize_ray_cluster(config)
    tokenizer, processor = load_tokenizer_and_processor(config)
    _, dataloader = create_eval_dataloader(config, tokenizer, processor)
    agent_handle, actor_rollout_wg, reward_wg = initialize_worker_groups(config)

    # Patch DeviceMesh on workers to avoid torch 2.8.0 AttributeError when loading FSDP2 sharded checkpoints.
    _patch_device_mesh_on_workers(actor_rollout_wg)
    _maybe_patch_device_mesh()

    # Load checkpoint if checkpoint_dir is specified
    # Note: checkpoint_dir and model.path are NOT mutually exclusive:
    # - model.path is still needed for tokenizer/processor loading
    # - checkpoint_dir is used to load model weights (overrides initial weights from model.path)
    load_checkpoint_if_needed(config, actor_rollout_wg)

    reward_kwargs = config.reward_model.get("reward_kwargs", {})
    reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)

    eval_cfg = config.get("evaluation", {})
    max_batches = eval_cfg.get("max_batches")
    max_samples = eval_cfg.get("max_samples")

    # Prepare output paths
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = None
    if config.output.get("scores_path"):
        scores_path = output_dir / config.output.scores_path
        # Remove existing file if present
        if scores_path.exists():
            scores_path.unlink()
            logger.info("Removed existing scores file: %s", scores_path)

    # Keep only metrics in memory, not full records
    turn_scores: List[float] = []
    generation_times: List[float] = []
    reward_times: List[float] = []
    trace_records: List[Dict[str, Any]] = []  # Keep first 100 for trace
    
    consumed_samples = 0
    progress = tqdm(total=len(dataloader), desc="Batches", disable=len(dataloader) == 0)

    try:
        for batch_idx, batch_dict in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = DataProto.from_single_dict(batch_dict)
            ensure_batch_uids(batch)
            
            generation_output, generation_time = run_generation_step(
                agent_handle=agent_handle,
                actor_rollout_wg=actor_rollout_wg,
                batch=batch,
                tokenizer=tokenizer,
                config=config,
            )
            combined_batch = batch.union(generation_output)
            combined_batch.meta_info["validate"] = True

            scores, reward_result = compute_reward_outputs(combined_batch, reward_wg, reward_fn)
            per_sample_gen_time = (generation_time / len(scores)) if scores else 0.0
            reward_time = reward_result.get("reward_time", 0.0)
            per_sample_reward_time = (reward_time / len(scores)) if scores else 0.0

            sample_records = collect_sample_records(
                combined_batch=combined_batch,
                generation_time=per_sample_gen_time,
                reward_scores=scores,
                reward_result=reward_result,
                rollout_metrics=generation_output.meta_info.get("metrics"),
                tokenizer=tokenizer,
            )
            
            # Write results incrementally to avoid memory overflow
            if scores_path:
                is_first_batch = (batch_idx == 0)
                append_results_to_file(sample_records, scores_path, is_first_batch)
                if is_first_batch:
                    logger.info("Started writing results to %s", scores_path)
            
            # Keep first 100 records for trace
            if len(trace_records) < 100:
                trace_records.extend(sample_records[:100 - len(trace_records)])
            
            # Only keep metrics in memory
            turn_scores.extend(scores)
            generation_times.extend([per_sample_gen_time] * len(scores))
            reward_times.extend([per_sample_reward_time] * len(scores))

            consumed_samples += len(scores)
            if max_samples is not None and consumed_samples >= max_samples:
                break
            progress.update(1)
    finally:
        progress.close()

    if scores_path:
        logger.info("Finished writing all results to %s", scores_path)

    summary_metrics = aggregate_summary(turn_scores, generation_times, reward_times)
    save_summary_and_trace(summary_metrics, trace_records, config)
    logger.info("Multi-turn evaluation completed!")
    logger.info("Summary metrics: %s", json.dumps(summary_metrics, indent=2))

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
