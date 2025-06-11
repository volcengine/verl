import collections
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional


# use mapping-protol to make dataclass behaving like a dict for backward compatibility
class MappingProtocol(collections.abc.Mapping):
    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


@dataclass
class CustomCls(MappingProtocol):
    # The path to the file containing your customized dataset class. If not specified, pre-implemented dataset will be used.
    path: Optional[str] = None
    # The name of the dataset class within the specified file.
    name: Optional[str] = None


@dataclass
class DataConfig(MappingProtocol):
    # Tokenizer class or path. If null, it will be inferred from the model.
    tokenizer: Optional[str] = None
    # Whether to use shared memory for data loading.
    use_shm: bool = False
    # Training set parquet. Can be a list or a single file.
    # The program will read all files into memory, so it can't be too large (< 100GB).
    # The path can be either a local path or an HDFS path.
    # For HDFS path, we provide utils to download it to DRAM and convert it to a local path.
    train_files: Optional[Any] = None
    # Validation parquet. Can be a list or a single file.
    val_files: Optional[Any] = None
    # The field in the dataset where the prompt is located. Default is 'prompt'.
    prompt_key: Optional[str] = None
    # The field used to select the reward function (if using different ones per example).
    reward_fn_key: Optional[str] = None
    # Maximum prompt length. All prompts will be left-padded to this length.
    # An error will be reported if the length is too long.
    max_prompt_length: int = 512
    # Maximum response length. Rollout in RL algorithms (e.g. PPO) generates up to this length.
    max_response_length: int = 512
    # Batch size sampled for one training iteration of different RL algorithms.
    train_batch_size: int = 1024
    # Batch size used during validation. Can be null.
    val_batch_size: Optional[int] = None
    # Whether to return the original input_ids without adding chat template.
    # This is used when the reward model's chat template differs from the policy.
    # If using a model-based RM with different templates, this should be True.
    return_raw_input_ids: bool = False
    # Whether to return the original chat (prompt) without applying chat template.
    return_raw_chat: bool = False
    # Whether to return the full prompt with chat template.
    return_full_prompt: bool = False
    # Whether to shuffle the data in the dataloader.
    shuffle: bool = True
    # Whether to shuffle the validation set.
    validation_shuffle: bool = False
    # Whether to filter overlong prompts.
    filter_overlong_prompts: bool = False
    # Number of workers for filtering overlong prompts.
    # For large-scale datasets, filtering can be time-consuming.
    # Use multiprocessing to speed up. Default is 1.
    filter_overlong_prompts_workers: int = 1
    # Truncate the input_ids or prompt if they exceed max_prompt_length.
    # Options: 'error', 'left', or 'right'. Default is 'error'.
    truncation: str = "error"
    # The field in the multi-modal dataset where the image is located. Default is 'images'.
    image_key: str = "images"
    # The field in the multi-modal dataset where the video is located.
    video_key: str = "videos"
    # If the remote tokenizer has a Python file, this flag determines whether to allow using it.
    trust_remote_code: bool = False
    # Optional: specify a custom dataset class path and name if overriding default loading behavior.
    custom_cls: Optional[CustomCls] = None
    # data seed
    seed: int = 0


@dataclass
class KlCtrlConfig(MappingProtocol):
    # KL control type: "fixed" or "adaptive"
    type: str = "fixed"
    # Initial coefficient for KL penalty
    kl_coef: float = "0.001"
    # Horizon value for adaptive controller (if enabled)
    horizon: int = 10000
    # Target KL divergence (used for adaptive controller)
    target_kl: float = 0.1


@dataclass
class PfPPOConfig(MappingProtocol):
    # Method for reweighting samples: "pow", "max_min", or "max_random"
    reweight_method: pow
    # Power used for weight scaling in "pow" method
    weight_pow: 2.0


@dataclass
class AlgorithmConfig(MappingProtocol):
    # Discount factor for future rewards
    gamma: float = 1.0
    # Trade-off between bias and variance in the GAE estimator
    lam: float = 1.0
    # Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
    adv_estimator: str = "gae"
    # Whether to normalize advantages by std (specific to GRPO)
    norm_adv_by_std_in_grpo: bool = True
    # Whether to enable in-reward KL penalty
    use_kl_in_reward: bool = False
    # How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full"
    kl_penalty: str = "kl"
    # KL control configuration
    kl_ctrl: Optional[KlCtrlConfig] = None
    # Whether to enable preference feedback PPO
    use_pf_ppo: bool = False
    # Preference feedback PPO settings
    pf_ppo: Optional[KlCtrlConfig] = None

    def __post_init__(self):
        pass


@dataclass
class CustomRewardFunc(MappingProtocol):
    # The path to the file containing your customized reward function.
    # If not specified, pre-implemented reward functions will be used.
    path: Optional[str] = None
    # The name of the reward function within the specified file. Default is 'compute_score'
    name: str = "compute_score"


@dataclass
class RayInit(MappingProtocol):
    # Number of CPUs for Ray. Use a fixed number instead of null when using SLURM.
    num_cpus: Optional[int] = None
    # Path to save Ray timeline JSON for performance profiling
    timeline_json_file: Optional[str] = None


@dataclass
class TrainerConfig(MappingProtocol):
    # Whether to balance batch sizes across distributed workers
    balance_batch: bool = False
    # Number of epochs in training
    total_epochs: int = 30
    # Total training steps (can be set explicitly or derived from epochs)
    total_training_steps: Optional[int] = None
    # Project name for experiment tracking (e.g., wandb)
    project_name: str = "verl_examples"
    # Experiment name for run identification in tracking tools
    experiment_name: str = "gsm8k"
    # Logging backends to use: "console", "wandb", etc.
    logger: List[str] = field(default_factory=lambda: ["console", "wandb"])
    # Number of generations to log during validation
    log_val_generations: int = 0
    # Directory for logging rollout data; no dump if null
    rollout_data_dir: Optional[str] = None
    # Directory for logging validation data; no dump if null
    validation_data_dir: Optional[str] = None
    # Number of nodes used in the training
    nnodes: int = 1
    # Number of GPUs per node
    n_gpus_per_node: int = 8
    # Save frequency (by iteration) for model checkpoints
    save_freq: int = -1
    # Resume mode: "auto", "disable", or "resume_path"
    # "auto": resume from last checkpoint if available
    # "disable": start from scratch
    # "resume_path": resume from a user-defined path
    resume_mode: str = "auto"
    # Path to resume training from (only used when resume_mode is "resume_path")
    resume_from_path: Optional[str] = None

    # Whether to run validation before training begins
    val_before_train: bool = True
    # Validation frequency (in training iterations)
    test_freq: int = -1
    # Number of iterations to warm up the critic before updating policy
    critic_warmup: int = 0
    # Default path to distributed filesystem for saving checkpoints
    default_hdfs_dir: Optional[str] = None
    # Whether to delete local checkpoints after loading
    del_local_ckpt_after_load: bool = False
    # Default local directory for saving checkpoints
    default_local_dir: Optional[str] = None
    # Maximum number of actor checkpoints to keep
    max_actor_ckpt_to_keep: Optional[int] = None
    # Maximum number of critic checkpoints to keep
    max_critic_ckpt_to_keep: Optional[int] = None
    # Timeout (in seconds) for Ray worker to wait for registration
    ray_wait_register_center_timeout: int = 300
    # Device to run training on (e.g., "cuda", "cpu")
    device: str = "cuda"

    def __post_init__(self):
        if self.default_local_dir is None:
            self.default_local_dir = f"checkpoints/{self.project_name}/{self.experiment_name}"
        assert self.resume_mode in ["auto", "disable", "resume_path"]


# protocol
@dataclass
class PPOConfig(MappingProtocol):
    data: Optional[DataConfig] = None
    trainer: Optional[TrainerConfig] = None
    algorithm: Optional[AlgorithmConfig] = None
    custom_reward_function: Optional[CustomRewardFunc] = None
    ray_init: Optional[RayInit] = None
    actor_rollout_ref: Optional[Any] = None
    critic: Optional[Any] = None
    reward_model: Optional[Any] = None
