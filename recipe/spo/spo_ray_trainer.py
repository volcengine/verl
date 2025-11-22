# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Modifications Copyright 2025 SPO authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SPO Trainer extending PPO Trainer with Self-Play Optimization.
This trainer inherits from the base PPO trainer and adds SPO-specific logic.
"""

import json
import os
import random
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as BaseRayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.torch_functional import masked_mean

# Re-export for backward compatibility
__all__ = ["RayPPOTrainer", "ResourcePoolManager", "Role", "apply_kl_penalty", "compute_advantage", "compute_response_mask"]


class RayPPOTrainer(BaseRayPPOTrainer):
    """SPO-specific PPO trainer that extends the base trainer with Self-Play Optimization logic.

    This trainer inherits most functionality from the base RayPPOTrainer and adds:
    - SPO-specific weighted sampling based on Thompson sampling
    - SPO advantage calculation using Bayesian framework
    - Alpha/beta updates with KL-based rho smoothing
    """

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        reward_extra_infos_dict.pop("acc", None)
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Override: Get generation batch with SPO-specific keys.

        SPO modification: Includes "raw_prompt" in reward_model_keys.
        """
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch


    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from recipe.spo.agent_loop import SPOAgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = SPOAgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _get_spo_rho(
        self,
        prompt2protodata: dict[str, DataProto],
        prompt2log_probs: dict[str, torch.Tensor],
        prompt2D: dict[str, torch.Tensor],
        micro_prompts: list[str],
        spo_log_prob_batch_backup: DataProto,
    ) -> torch.Tensor:
        """Calculate rho for alpha and beta updating."""
        rho_metrics = {}

        if self.config.trainer.spo.rho.type == "constant":
            # Repeat a constant to len(micro_prompts) as a torch.Tensor
            rho = torch.full((len(micro_prompts),), self.config.trainer.spo.rho.value, dtype=torch.float32)
            D = torch.full((len(micro_prompts),), 0.0, dtype=torch.float32)
            rho_metrics["spo/rho"] = rho.mean().item()
            rho_metrics["spo/D"] = D.mean().item()
            return rho, prompt2protodata, prompt2log_probs, prompt2D, rho_metrics
        elif self.config.trainer.spo.rho.type == "kl":
            # Extract past dataprotos of micro_prompts
            past_dataprotos = []
            first_sampled_number = 0
            for pid_, p_ in enumerate(micro_prompts):
                if p_ in prompt2protodata.keys():
                    proto = prompt2protodata[p_]
                else:
                    first_sampled_number += 1
                    proto = spo_log_prob_batch_backup.select_idxs([pid_])
                # Remove per-sample meta_info fields to avoid conflicts during concat
                proto.meta_info.pop("global_token_num", None)
                past_dataprotos.append(proto)
            past_dataprotos = DataProto.concat(past_dataprotos)
            response_mask = compute_response_mask(past_dataprotos)
            first_sampled_ratio = first_sampled_number / len(micro_prompts)
            rho_metrics["spo/first_sampled_ratio"] = first_sampled_ratio

            cur_log_probs = self.actor_rollout_wg.compute_log_prob(past_dataprotos)
            cur_log_probs = cur_log_probs.batch["old_log_probs"]
            old_log_probs = []
            for pid_, p_ in enumerate(micro_prompts):
                if p_ in prompt2log_probs.keys():
                    old_log_probs.append(prompt2log_probs[p_])
                else:
                    old_log_probs.append(cur_log_probs[pid_].unsqueeze(0))
            old_log_probs = torch.cat(old_log_probs, dim=0)  # (M, seq_len)

            kl = (old_log_probs - cur_log_probs).abs()
            D = masked_mean(kl, response_mask, axis=-1)  # (M,)
            rho_metrics["spo/D"] = D.mean().item()
            D_half = torch.as_tensor(0.06, dtype=D.dtype, device=D.device)
            rho = torch.pow(2.0, -D / D_half)
            rho_metrics["spo/rho"] = rho.mean().item()
            rho_clipped = torch.clamp(rho, min=self.config.trainer.spo.rho.clip_lower, max=0.96)
            rho_metrics["spo/rho_clipped"] = rho_clipped.mean().item()
            rho_metrics["spo/rho_clip_ratio"] = (rho_clipped != rho).type(torch.float).mean().item()

            # Update prompt2protodata and prompt2log_probs
            new_log_probs = self.actor_rollout_wg.compute_log_prob(spo_log_prob_batch_backup)
            for pid_, p_ in enumerate(micro_prompts):
                prompt2protodata[p_] = spo_log_prob_batch_backup.select_idxs([pid_])
                prompt2log_probs[p_] = new_log_probs.batch["old_log_probs"][pid_].unsqueeze(0)
                prompt2D[p_] = D[pid_].item()

            return rho_clipped, prompt2protodata, prompt2log_probs, prompt2D, rho_metrics
        else:
            raise ValueError(f"Unknown rho type: {self.config.trainer.spo.rho.type}")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        if self.config.trainer.spo.enable:
            prompt2scores = json.load(open(self.config.trainer.spo.offline_values))
            Neff = self.config.trainer.spo.offline_N
            prompt2scores = {k: [int(_ > 0) for _ in v] for k, v in prompt2scores.items()}
            print(f"[DEBUG] Select {Neff} samples for each prompt to calculate offline values.")
            full_prompts = list(prompt2scores.keys())
            if Neff == 0:
                prompt2alpha = {k: 0.5 for k in full_prompts}
                prompt2beta = {k: 0.5 for k in full_prompts}
            else:
                for k, v in prompt2scores.items():
                    if len(v) > Neff:
                        prompt2scores[k] = random.sample(v, Neff)
                N_init = 1 / (1 - self.config.trainer.spo.rho.clip_lower)
                print(f"[DEBUG] N_init: {N_init}")
                prompt2alpha = {k: N_init * (sum(prompt2scores[k]) + 0.5) / (Neff + 1) for k in full_prompts}
                prompt2beta = {k: N_init * (Neff - sum(prompt2scores[k]) + 0.5) / (Neff + 1) for k in full_prompts}
            prompt2protodata = {}
            prompt2log_probs = {}
            prompt2D = {}
            prompt2sampled_number = defaultdict(int)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.config.trainer.spo.enable:
                    EXPLORATION_EPSILON = 0.05

                    prompt2phat = {
                        k: float(prompt2alpha[k]) / float(prompt2alpha[k] + prompt2beta[k]) for k in full_prompts
                    }
                    prompt2weight = {
                        k: ((prompt2phat[k] * (1.0 - prompt2phat[k])) ** 0.5) + EXPLORATION_EPSILON
                        for k in full_prompts
                    }

                    items = []
                    weights = []
                    for i, p in enumerate(batch_dict["raw_prompt"]):
                        p_str = p[0]["content"].strip()
                        w = float(prompt2weight.get(p_str, 0.0))
                        items.append(i)
                        weights.append(w)

                    M = len(items)
                    if M > 0:
                        weights_np = np.asarray(weights, dtype=np.float64)
                        wsum = float(weights_np.sum())

                        if wsum > 0.0:
                            probs = weights_np / wsum
                        else:
                            probs = np.full(M, 1.0 / M, dtype=np.float64)

                        probs = probs / probs.sum()

                        target_bs = int(self.config.data.train_batch_size)
                        replace = target_bs > M

                        selected_pos = np.random.choice(M, size=target_bs, replace=replace, p=probs)
                        keep_idx = [items[j] for j in selected_pos.tolist()]
                        keep_weights = weights_np[selected_pos]
                        keep_probs = probs[selected_pos]

                        print(f"[DEBUG] Weighted sampling: M={M}, target_bs={target_bs}, replace={replace}")
                        print(
                            f"[DEBUG] Sampled probs: min={keep_probs.min():.6f}, max={keep_probs.max():.6f}, mean={keep_probs.mean():.6f}"
                        )
                        print(
                            f"[DEBUG] Sampled weights: min={keep_weights.min():.6f}, max={keep_weights.max():.6f}, mean={keep_weights.mean():.6f}"
                        )

                        if keep_idx:
                            sampled_batch_dict = {}
                            for k, v in batch_dict.items():
                                try:
                                    sampled_batch_dict[k] = v[keep_idx]
                                    continue
                                except Exception:
                                    pass

                                if isinstance(v, (list, tuple)):
                                    sampled_batch_dict[k] = type(v)(v[i] for i in keep_idx)
                                else:
                                    sampled_batch_dict[k] = v

                            batch_dict = sampled_batch_dict
                            print(f"[DEBUG] Final size of keep_idx: {len(keep_idx)}")

                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    micro_prompts = batch.non_tensor_batch.get("raw_prompt", None)
                    micro_prompts = [_[0]["content"].strip() for _ in micro_prompts]
                    if self.config.trainer.spo.enable:
                        alpha = [prompt2alpha[_] for _ in micro_prompts]
                        beta = [prompt2beta[_] for _ in micro_prompts]
                        sum_reward_tensor = reward_tensor.sum(dim=-1)

                        spo_metrics = {}
                        r = sum_reward_tensor
                        alpha = torch.tensor(alpha, dtype=torch.float).to(r)
                        beta = torch.tensor(beta, dtype=torch.float).to(r)
                        spo_metrics["spo/reward"] = r.mean().detach().item()
                        spo_metrics["spo/alpha"] = alpha.mean().detach().item()
                        spo_metrics["spo/beta"] = beta.mean().detach().item()
                        Neff = alpha + beta
                        spo_metrics["spo/Neff"] = Neff.mean().detach().item()
                        p_hats = alpha / Neff
                        spo_metrics["spo/p_hats"] = p_hats.mean().detach().item()

                        # Recalculate advantages
                        advantages = r - p_hats
                        spo_metrics["spo/adv_before_norm"] = advantages.mean().detach().item()

                        response_mask = compute_response_mask(batch)
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        quantiles = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=advantages.device)
                        q_vals = torch.quantile(advantages, quantiles)
                        spo_metrics["spo/adv_after_norm/p10"] = q_vals[0].item()
                        spo_metrics["spo/adv_after_norm/p30"] = q_vals[1].item()
                        spo_metrics["spo/adv_after_norm/p50"] = q_vals[2].item()
                        spo_metrics["spo/adv_after_norm/p70"] = q_vals[3].item()
                        spo_metrics["spo/adv_after_norm/p90"] = q_vals[4].item()
                        advantages = advantages.unsqueeze(-1) * response_mask
                        batch.batch["advantages"] = advantages
                        batch.batch["returns"] = advantages

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if self.config.trainer.spo.enable:
                            spo_log_prob_batch_backup = batch.select(deepcopy=True)

                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if "advantages" not in batch.batch:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.spo.enable:
                        rho, prompt2protodata, prompt2log_probs, prompt2D, rho_metrics = self._get_spo_rho(
                            prompt2protodata, prompt2log_probs, prompt2D, micro_prompts, spo_log_prob_batch_backup
                        )
                        spo_metrics.update(rho_metrics)

                        # if you want exact Beta intervals, maintain alpha/beta as well:
                        alpha = rho * alpha + r
                        beta = rho * beta + (1 - r)

                        cur_sampled_numbers = []
                        for i in range(len(alpha)):
                            prompt2alpha[micro_prompts[i]] = alpha[i].item()
                            prompt2beta[micro_prompts[i]] = beta[i].item()
                            prompt2sampled_number[micro_prompts[i]] += 1
                            cur_sampled_numbers.append(prompt2sampled_number[micro_prompts[i]])

                        cur_sampled_numbers = np.array(cur_sampled_numbers, dtype=np.int32)
                        spo_metrics["spo/cur_sampled_number/min"] = cur_sampled_numbers.min()
                        spo_metrics["spo/cur_sampled_number/max"] = cur_sampled_numbers.max()
                        spo_metrics["spo/cur_sampled_number/mean"] = cur_sampled_numbers.mean()

                        metrics.update(spo_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
