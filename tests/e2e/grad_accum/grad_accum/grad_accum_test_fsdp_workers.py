import logging
import os
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import torch
from omegaconf import open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo import core_algos
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu, offload_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.utils.seqlen_balancing import get_uniform_data_chunks
from verl.utils.torch_functional import compute_response_mask
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.critic.dp_critic import DataParallelPPOCritic
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

ALL_LOSS_AGG_MODES: list[str] = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"]
GRAD_ACCUM_RTOL: float = 0.01

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class GradAccumulationTestActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = GradAccumulationTestDPActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = GradAccumulationTestDPActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp, optimizer=self.actor.actor_optimizer, lr_scheduler=self.actor_lr_scheduler, processing_class=self.processor if self.processor is not None else self.tokenizer, checkpoint_contents=self.config.actor.checkpoint.contents
            )


class GradAccumulationTestCriticWorker(CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module, optimizer=self.critic_optimizer, lr_scheduler=self.critic_lr_scheduler, processing_class=self.processor if self.processor is not None else self.tokenizer, checkpoint_contents=self.config.checkpoint.contents)


class GradAccumulationTestDPActor(DataParallelPPOActor):
    def compute_batch_loss(self, data: DataProto, loss_agg_mode: str = "token-mean", mini_batch_loss_token_num: Optional[int] = None, disable_grad_accum: bool = False) -> tuple[torch.Tensor, int]:
        accum_loss = None
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        if disable_grad_accum:
            micro_data_chunks = [data]
        else:
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_data_chunks, _ = get_uniform_data_chunks(data=data, max_token_len=max_token_len)
            else:
                num_micro_batches = len(data) // self.config.ppo_micro_batch_size_per_gpu
                micro_data_chunks = data.chunk(num_micro_batches)

            assert len(micro_data_chunks) > 1, f"len(micro_data_chunks) must be greater than 1 to test grad accumulation, but got {len(micro_data_chunks)=}"

        micro_weights = []
        raw_micro_losses = []
        for micro_data_chunk in micro_data_chunks:
            micro_batch = {**micro_data_chunk.batch, **micro_data_chunk.non_tensor_batch}

            response_mask = compute_response_mask(response_ids=micro_batch["responses"], attention_mask=micro_batch["attention_mask"])
            old_log_prob = micro_batch["old_log_probs"]
            advantages = micro_batch["advantages"]

            clip_ratio = self.config.clip_ratio
            clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
            clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
            clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
            entropy_coeff = self.config.entropy_coeff

            # all return: (bsz, response_length)
            entropy, log_prob = self._forward_micro_batch(micro_batch=micro_batch, temperature=temperature)

            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages, response_mask=response_mask, cliprange=clip_ratio, cliprange_low=clip_ratio_low, cliprange_high=clip_ratio_high, clip_ratio_c=clip_ratio_c, loss_agg_mode=loss_agg_mode
            )
            loss = pg_loss

            # compute entropy loss from entropy
            entropy_loss = core_algos.compute_entropy_loss(entropy=entropy, response_mask=response_mask, loss_agg_mode=loss_agg_mode)
            loss += -entropy_loss * entropy_coeff

            if self.config.use_kl_loss:
                ref_log_prob = micro_batch["ref_log_prob"]
                # compute kl loss
                kld = core_algos.kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                kl_loss = core_algos.compute_kl_loss(kld=kld, response_mask=response_mask, loss_agg_mode=loss_agg_mode)

                loss += kl_loss * self.config.kl_loss_coef

            # Rescale the final model loss together instead of separately in core_algos
            if loss_agg_mode == "token-mean":
                num_valid_toks = response_mask.sum()
                micro_weight = num_valid_toks / mini_batch_loss_token_num
            else:  # seq-mean
                micro_weight = len(micro_data_chunk) / self.config.ppo_mini_batch_size

            micro_loss = loss * micro_weight
            if accum_loss is None:
                accum_loss = micro_loss
            else:
                accum_loss += micro_loss

            micro_weights.append(micro_weight)
            raw_micro_losses.append(loss.detach().cpu().item())

        print(f"{raw_micro_losses=}")
        print(f"{sum(micro_weights)=}")
        print(f"{micro_weights=}")

        return accum_loss, len(micro_data_chunks)

    def update_policy(self, data: DataProto):
        """
        Tests gradient accumulation by comparing loss computed with mini-batches vs single batch
        """
        # make sure we are in training mode
        self.actor_module.train()
        metrics = defaultdict(list)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        non_tensor_select_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch.keys() else []

        selected_data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        num_mini_batches = len(selected_data) // self.config.ppo_mini_batch_size
        assert num_mini_batches > 1, f"num_mini_batches must be greater than 1 to test grad accumulation, but got {num_mini_batches=}"

        mini_dataloader = selected_data.chunk(num_mini_batches)

        test_infos: list[dict[str, Any]] = []

        for mini_idx, mini_data_chunk in enumerate(mini_dataloader):
            for loss_agg_mode in ALL_LOSS_AGG_MODES:
                mini_loss_w_grad_accum, num_micro_batches = self.compute_batch_loss(data=mini_data_chunk, loss_agg_mode=loss_agg_mode, mini_batch_loss_token_num=data.meta_info["mini_batch_loss_token_nums"][mini_idx], disable_grad_accum=False)
                mini_loss, _ = self.compute_batch_loss(data=mini_data_chunk, loss_agg_mode=loss_agg_mode, mini_batch_loss_token_num=data.meta_info["mini_batch_loss_token_nums"][mini_idx], disable_grad_accum=True)
                if loss_agg_mode == self.config.loss_agg_mode:
                    mini_loss.backward()
                mini_loss_w_grad_accum = mini_loss_w_grad_accum.detach().cpu()
                mini_loss = mini_loss.detach().cpu()

                test_infos.append(
                    {
                        "mini_idx": mini_idx,
                        "loss_agg_mode": loss_agg_mode,
                        "num_micro_batches": num_micro_batches,
                        "mini_loss": mini_loss.item(),
                        "mini_loss_w_grad_accum": mini_loss_w_grad_accum.item(),
                        "rtol": GRAD_ACCUM_RTOL,
                        "isclose": torch.isclose(mini_loss_w_grad_accum, mini_loss, rtol=GRAD_ACCUM_RTOL),
                    }
                )
            self._optimizer_step()
        self.actor_optimizer.zero_grad()

        test_info_df = pd.DataFrame(test_infos)
        print(test_info_df)

        return metrics


class GradAccumulationTestDPCritic(DataParallelPPOCritic):
    def compute_batch_loss(self, data: DataProto, loss_agg_mode: str = "token-mean", mini_batch_loss_token_num: Optional[int] = None, disable_grad_accum: bool = False) -> tuple[torch.Tensor, int]:
        accum_loss = None

        if disable_grad_accum:
            micro_data_chunks = [data]
        else:
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_data_chunks, _ = get_uniform_data_chunks(data=data, max_token_len=max_token_len)
            else:
                num_micro_batches = len(data) // self.config.ppo_micro_batch_size_per_gpu
                micro_data_chunks = data.chunk(num_micro_batches)

            assert len(micro_data_chunks) > 1, f"len(micro_data_chunks) must be greater than 1 to test grad accumulation, but got {len(micro_data_chunks)=}"

        micro_weights = []
        raw_micro_losses = []
        for micro_data_chunk in micro_data_chunks:
            micro_batch = {**micro_data_chunk.batch, **micro_data_chunk.non_tensor_batch}

            responses = micro_batch["responses"]
            attention_mask = micro_batch["attention_mask"]
            values = micro_batch["values"]
            returns = micro_batch["returns"]
            response_mask = compute_response_mask(response_ids=responses, attention_mask=attention_mask)

            vpreds = self._forward_micro_batch(micro_batch)

            # assert not torch.any(torch.isnan(vpreds)).item()

            vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds, values=values, returns=returns, response_mask=response_mask, cliprange_value=self.config.cliprange_value, loss_agg_mode=loss_agg_mode)

            loss = vf_loss
            # Rescale the final model loss together instead of separately in core_algos
            if loss_agg_mode == "token-mean":
                num_valid_toks = response_mask.sum()
                micro_weight = num_valid_toks / mini_batch_loss_token_num
            else:  # seq-mean
                micro_weight = len(micro_data_chunk) / self.config.ppo_mini_batch_size

            micro_loss = loss * micro_weight
            if accum_loss is None:
                accum_loss = micro_loss
            else:
                accum_loss += micro_loss

            micro_weights.append(micro_weight)
            raw_micro_losses.append(loss.detach().cpu().item())

        print(f"{raw_micro_losses=}")
        print(f"{sum(micro_weights)=}")
        print(f"{micro_weights=}")

        assert accum_loss is not None, "accum_loss must not be None"
        return accum_loss, len(micro_data_chunks)

    def update_critic(self, data: DataProto):
        """
        TODO: Merge common part with update_actor as update
        """
        # make sure we are in training mode
        self.critic_module.train()
        metrics = defaultdict(list)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        non_tensor_select_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch.keys() else []

        selected_data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        num_mini_batches = len(selected_data) // self.config.ppo_mini_batch_size

        mini_dataloader = data.chunk(num_mini_batches)  # TODO: `make_minibatch_iterator`` as in megatron

        test_infos: list[dict[str, Any]] = []

        for mini_idx, mini_data_chunk in enumerate(mini_dataloader):
            for loss_agg_mode in ALL_LOSS_AGG_MODES:
                mini_loss_w_grad_accum, num_micro_batches = self.compute_batch_loss(data=mini_data_chunk, loss_agg_mode=loss_agg_mode, mini_batch_loss_token_num=data.meta_info["mini_batch_loss_token_nums"][mini_idx], disable_grad_accum=False)
                mini_loss, _ = self.compute_batch_loss(data=mini_data_chunk, loss_agg_mode=loss_agg_mode, mini_batch_loss_token_num=data.meta_info["mini_batch_loss_token_nums"][mini_idx], disable_grad_accum=True)
                if loss_agg_mode == self.config.loss_agg_mode:
                    mini_loss.backward()
                mini_loss_w_grad_accum = mini_loss_w_grad_accum.detach().cpu()
                mini_loss = mini_loss.detach().cpu()
                test_infos.append(
                    {
                        "mini_idx": mini_idx,
                        "loss_agg_mode": loss_agg_mode,
                        "num_micro_batches": num_micro_batches,
                        "mini_loss": mini_loss.item(),
                        "mini_loss_w_grad_accum": mini_loss_w_grad_accum.item(),
                        "rtol": GRAD_ACCUM_RTOL,
                        "isclose": torch.isclose(mini_loss_w_grad_accum, mini_loss, rtol=GRAD_ACCUM_RTOL),
                    }
                )
            self._optimizer_step()
        self.critic_optimizer.zero_grad()

        test_info_df = pd.DataFrame(test_infos)
        print(test_info_df)

        return metrics
