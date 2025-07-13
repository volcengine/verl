import hydra


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    print(config)
    from verl.utils.config import omega_conf_to_dataclass

    profiler_config = omega_conf_to_dataclass(config.critic.profiler)
    print(profiler_config)


if __name__ == "__main__":
    main()
