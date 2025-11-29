# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


def apply_partialrollout_patch():
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from recipe.partial_rollout.dapo_ray_trainer import dapo_ray_trainer_fit
    from recipe.partial_rollout.ray_trainer import ray_trainer__init__, ray_trainer_init_workers, ray_trainer_fit
    RayPPOTrainer.__init__ = ray_trainer__init__
    RayPPOTrainer.init_workers = ray_trainer_init_workers
    RayPPOTrainer.fit = ray_trainer_fit
    RayDAPOTrainer.fit = dapo_ray_trainer_fit
