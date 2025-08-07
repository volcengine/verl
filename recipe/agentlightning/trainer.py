# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import random
from contextlib import contextmanager
from copy import deepcopy
from pprint import pprint

import torch
from codetiming import Timer
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .daemon import AgentModeDaemon


@contextmanager
def _timer(name: str, timing_raw: dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class AgentLightningTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentLightningServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:
    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        test_data = next(iter(self.val_dataloader))
        test_batch = DataProto.from_single_dict(test_data)

        self.async_rollout_manager.wake_up()
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )
        self.agent_mode_daemon.run_until_all_finished()
        test_metrics = self.agent_mode_daemon.get_test_metrics()
        self.agent_mode_daemon.clear_data_and_server()
        self.async_rollout_manager.sleep()
        return test_metrics

    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                self.async_rollout_manager.wake_up()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )
                self.agent_mode_daemon.run_until_all_finished()
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=gen_batch.batch["fake_ids"].device,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.async_rollout_manager.sleep()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # after advantages are assinged, we round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)
            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_mini_batch"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with _timer("dump_rollout_generations", timing_raw):
                    print(batch.batch.keys())
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    self._dump_generations(
                        inputs=inputs,
                        outputs=outputs,
                        scores=scores,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        dump_path=rollout_data_dir,
                    )

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        self.agent_mode_daemon = AgentModeDaemon(
            self.config.agentlightning.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": self.config.actor_rollout_ref.model.path,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.agent_mode_daemon.start()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # train step
                metrics = self._train_step(batch_dict)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint("Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1
