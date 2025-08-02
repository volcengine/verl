
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Set

from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool
from verl.single_controller.ray.base import (
    RayWorkerGroup, 
    create_colocated_worker_cls,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, ResourcePoolManager, Role

from verl.utils.tracking import ValidationGenerationsLogger
WorkerType = Type[Worker]

@dataclass
class RoleLayoutConfigItem:
    role: Role
    role_remote_worker_cls: Any
    resource_pool_name: str
    group_name: str
    

class RoleLayoutConfig:
    def __init__(self, roles: List[RoleLayoutConfigItem], config) -> None:
        self.config = config
        self.role_map = {role_item.role: role_item for role_item in roles}
        self.resource_pools: Set[str] = set([role.resource_pool_name for role in roles])
        self.groups: Set[str] = set([role.group_name for role in roles])
        print(f"{self.resource_pools=} \n {self.groups=}")
        print("\n".join([f"{item.role}  |  {item.group_name}  |  {item.resource_pool_name}" for item in roles]))
        self._validate()

    def _validate(self):
        """
        Validates the integrity and consistency of the layout configuration.
        
        It primarily ensures that all roles within a single process group are
        mapped to the exact same resource pool.
        """
        # TODO: Validate that all required roles from the main config are present in the layout.
        
        for group in self.groups:
            # Get all resource pool names associated with this group
            resource_pool = [v.resource_pool_name for v in self.role_map.values() if v.group_name == group]
            assert all(pool == resource_pool[0] for pool in resource_pool)
        
        print("RoleLayoutConfig validation passed successfully.")
    
    def __getitem__(self, role: Role) -> RoleLayoutConfigItem:
        return self.role_map[role]


class ColocateResourcePoolManager(ResourcePoolManager):

    def create_resource_pool(self, pool_map: Dict[str, Set]):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            assert resource_pool_name in pool_map.keys()
            max_colocate_count = len(pool_map[resource_pool_name])
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=max_colocate_count, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()


class SharedPoolRayPPOTrainer(RayPPOTrainer):

    # support each role have individual ray_worker_group_cls,
    # TODO: support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_layout_config: RoleLayoutConfig, # for shared pool setting
        resource_pool_manager: ColocateResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initializes the distributed PPO trainer with a flexible resource layout.
        Note that this trainer itself runs on the driver process on a single node.

        Args:
            config: The main configuration object for the training session.
            tokenizer: The tokenizer for processing text data.
            role_layout_config (RoleLayoutConfig): A configuration object that defines the
                entire distributed topology. It maps each role (e.g., Actor, Critic) to a
                specific process group and a resource pool. This is the core component
                that enables flexible, shared resource pools.
            resource_pool_manager (ColocateResourcePoolManager): A manager responsible for
                creating and providing access to the underlying Ray resource pools based
                on the specifications in `role_layout_config`.
            ray_worker_group_cls (type[RayWorkerGroup], optional): The class used to instantiate
                a worker group (process). Defaults to RayWorkerGroup.
            processor: An optional data processor, typically used for multimodal inputs.
            reward_fn: The function used to compute rewards during training.
            val_reward_fn: The function used to compute rewards during validation.
            train_dataset (Optional[Dataset], optional): The training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): The validation dataset. Defaults to None.
            collate_fn: The function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): The sampler for the training dataset. Defaults to None.
            device_name (str, optional): The target device for computation (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_layout_config.role_map.keys(), f"{role_layout_config.role_map=}"

        self.role_layout_config = role_layout_config
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_layout_config.role_map.keys()
        self.use_rm = Role.RewardModel in role_layout_config.role_map.keys()
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)


    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        pool_2_group = {pool: set() for pool in self.role_layout_config.resource_pools}
        for item in self.role_layout_config.role_map.values():
            pool_2_group[item.resource_pool_name].add(item.group_name)
        self.resource_pool_manager.create_resource_pool(pool_map=pool_2_group)

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        role_2_name: Dict[Role, str] = {
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
        }
        role_2_config: Dict[Role, Any] = {
            Role.ActorRollout: self.config.actor_rollout_ref,
            Role.Critic: self.config.critic,
            Role.RefPolicy: self.config.actor_rollout_ref,
            Role.RewardModel: self.config.reward_model,
        }
        no_role_kwargs = [Role.Critic, Role.RewardModel]

        for work_group_id in self.role_layout_config.groups:
            roles: List[Role] = [item.role for item in self.role_layout_config.role_map.values() 
                     if item.group_name == work_group_id]

            group_class_dict: Dict[str, RayClassWithInitArgs] = {
                role_2_name[role]: (
                    RayClassWithInitArgs(
                        cls=self.role_layout_config[role].role_remote_worker_cls,
                        config=role_2_config[role],
                    )
                ) if role in no_role_kwargs else (
                    RayClassWithInitArgs(
                        cls=self.role_layout_config[role].role_remote_worker_cls,
                        config=role_2_config[role],
                        role=role_2_name[role],
                    )
                )
                for role in roles
            }

            worker_dict_cls = create_colocated_worker_cls(class_dict=group_class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=self.resource_pool_manager.get_resource_pool(roles[0]),
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=group_class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )
