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
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.utils import hf_tokenizer
from verl.utils.device import (
    get_device_id,
    get_device_name,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
)
from verl.utils.profiler import DistProfiler
from verl.workers.fsdp_workers import RewardModelWorker, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class EllipticalRewardModelWorker(RewardModelWorker):
    def __init__(self, config):
        super().__init__(config)
        self.lamb = config.elliptical.lamb
        self.normalization = config.elliptical.normalization
        self.sparse_dim = config.elliptical.sparse_dim
        self.sparse_matrix = None
        self.randomize_sparse_matrix = config.elliptical.randomize_sparse_matrix
        self.persist_covariance = config.elliptical.persist_covariance
        self.cov_inv_dict = {}
        self.mean_hidden_states_mu_dict = {}
        self.hidden_mean_counter_dict = {}

    @staticmethod
    def _construct_sparse_matrix(features: torch.Tensor, sparse_dim: int) -> torch.Tensor:
        from sklearn.random_projection import SparseRandomProjection

        sparse_proj = SparseRandomProjection(sparse_dim, density="auto")
        sparse_proj.fit(features)
        sparse_matrix = sparse_proj.components_
        sparse_matrix_coo = sparse_matrix.tocoo()

        # Convert the row and col lists to numpy arrays and then to a LongTensor (speed up)
        indices = torch.LongTensor(np.array([sparse_matrix_coo.row, sparse_matrix_coo.col]))
        values = torch.FloatTensor(sparse_matrix_coo.data)

        sparse_mat = torch.sparse_coo_tensor(indices, values, [sparse_dim, features.shape[1]]).t()

        return sparse_mat

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModel

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModel.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    def _forward_micro_batch(self, micro_batch, start_of_response: int):
        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                raise NotImplementedError("Remove padding is not implemented for elliptical reward model")
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )

                sequence_lengths = attention_mask[:, start_of_response:].sum(dim=1)
                mean_hidden_states = []
                for i, seq_len in enumerate(sequence_lengths):
                    mean_hidden_states.append(
                        output.last_hidden_state[i, start_of_response : start_of_response + seq_len].mean(dim=0)
                    )
                mean_hidden_states = torch.stack(mean_hidden_states)

            return mean_hidden_states

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_hidden_states(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data = rm_data.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                mean_hidden_states = self._forward_micro_batch(
                    micro_batch, start_of_response=data.batch["prompts"].shape[-1]
                )
                output.append(mean_hidden_states)
            mean_hidden_states = torch.cat(output, dim=0)  # (batch_size)

            # NOTE(Jens): this has not been thoroughly checked
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == mean_hidden_states.size(0), f"{len(indices)} vs. {mean_hidden_states.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                mean_hidden_states = mean_hidden_states[revert_indices]

            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"mean_hidden_states": mean_hidden_states})

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output

    def _compute_bonuses(self, hidden_states, cov_inv, prompt_index: int):
        if self.config.elliptical.reward_type == "leave_one_out":
            if self.persist_covariance:
                raise NotImplementedError("Leave-one-out with persistence is not implemented")
            else:
                bonuses = []
                for i, hidden_state in enumerate(hidden_states):
                    chosen_samp = hidden_state.unsqueeze(1)
                    middle_part = torch.inverse(1 - chosen_samp.t() @ cov_inv @ chosen_samp)
                    leave_one_out_cov_inv = cov_inv + cov_inv @ chosen_samp @ middle_part @ chosen_samp.t() @ cov_inv
                    bonus = (chosen_samp.t() @ leave_one_out_cov_inv @ chosen_samp).flatten().float()
                    bonuses.append(bonus)

                bonuses = torch.concat(bonuses)

        elif self.config.elliptical.reward_type == "leverage":
            if self.persist_covariance:
                hidden_mean = self.mean_hidden_states_mu_dict[prompt_index]
                hidden_mean_counter = self.hidden_mean_counter_dict[prompt_index]

                hidden_states = hidden_states - hidden_mean

                numerator = cov_inv @ hidden_mean.unsqueeze(1) @ hidden_mean.unsqueeze(0) @ cov_inv
                denominator = -1 / hidden_mean_counter + hidden_mean.t() @ cov_inv @ hidden_mean
                cov_inv_mean_adjusted = cov_inv - numerator / denominator
                batch_cov_inv = cov_inv_mean_adjusted.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
            else:
                batch_cov_inv = cov_inv.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)

            bonuses = (hidden_states.unsqueeze(1) @ batch_cov_inv @ hidden_states.unsqueeze(2)).flatten().float()

        return bonuses

    def _normalize_bonuses(self, bonuses):
        if self.normalization == "none":
            pass
        elif self.normalization == "rnd":
            std = torch.std(bonuses)
            if std > 0:
                bonuses = bonuses / std
        elif self.normalization == "z_score":
            mean = torch.mean(bonuses)
            std = torch.std(bonuses)
            if std > 0:
                bonuses = (bonuses - mean) / std
            else:
                bonuses = bonuses - mean
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        return bonuses

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        if self.sparse_matrix is None:
            d = data.batch["mean_hidden_states"].shape[-1]
            sparse_matrix = self._construct_sparse_matrix(torch.randn(1, d), self.sparse_dim)
            if not self.randomize_sparse_matrix:
                self.sparse_matrix = sparse_matrix
        else:
            sparse_matrix = self.sparse_matrix

        mean_hidden_states = data.batch["mean_hidden_states"].to(get_device_id()).float()

        # sparse project
        mean_hidden_states = mean_hidden_states @ sparse_matrix.to(get_device_id())

        # upgrade to float64
        mean_hidden_states = mean_hidden_states.to(torch.float64)

        seen_uids = set()
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32).to(get_device_id())
        raw_bonuses_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32).to(get_device_id())
        for i in range(len(data)):
            data_item = data[i]
            uid = data_item.non_tensor_batch["uid"]
            if uid in seen_uids:
                continue

            seen_uids.add(uid)
            mask = data.non_tensor_batch["uid"] == uid
            filtered_mean_hidden_states = mean_hidden_states[mask]

            prompt_index = data_item.non_tensor_batch["extra_info"]["index"]

            if self.persist_covariance:
                # first update the mean hidden states mu
                if prompt_index not in self.mean_hidden_states_mu_dict:
                    self.mean_hidden_states_mu_dict[prompt_index] = filtered_mean_hidden_states.mean(dim=0)
                    self.hidden_mean_counter_dict[prompt_index] = mask.sum()
                else:
                    total_count = self.hidden_mean_counter_dict[prompt_index] + mask.sum()
                    old_mu = self.mean_hidden_states_mu_dict[prompt_index]
                    new_mu = (
                        old_mu * self.hidden_mean_counter_dict[prompt_index]
                        + filtered_mean_hidden_states.mean(dim=0) * mask.sum()
                    ) / total_count
                    self.mean_hidden_states_mu_dict[prompt_index] = new_mu
                    self.hidden_mean_counter_dict[prompt_index] = total_count

                # NOTE: we don't center here since otherwise the covariance will accumulate stale means
                final_mean_hidden_states = filtered_mean_hidden_states

                if prompt_index not in self.cov_inv_dict:
                    d = final_mean_hidden_states.shape[-1]
                    self.cov_inv_dict[prompt_index] = (
                        torch.eye(d, dtype=torch.float64).to(get_device_id()) * self.lamb**-1
                    )
                cov_inv = self.cov_inv_dict[prompt_index]
            else:
                centered_mean_hidden_states = filtered_mean_hidden_states - filtered_mean_hidden_states.mean(dim=0)
                final_mean_hidden_states = centered_mean_hidden_states

                d = final_mean_hidden_states.shape[-1]
                cov_inv = torch.eye(d, dtype=torch.float64).to(get_device_id()) * self.lamb**-1

            # update inverse covariance matrix with rank-1 updates
            for hidden_state in final_mean_hidden_states:
                chosen_samp = hidden_state.unsqueeze(1)
                middle_part = torch.inverse(1 + chosen_samp.t() @ cov_inv @ chosen_samp)
                cov_inv = cov_inv - cov_inv @ chosen_samp @ middle_part @ chosen_samp.t() @ cov_inv

            if self.persist_covariance:
                self.cov_inv_dict[prompt_index] = cov_inv

            raw_bonuses = self._compute_bonuses(final_mean_hidden_states, cov_inv, prompt_index)
            normalized_bonuses = self._normalize_bonuses(raw_bonuses)

            prompt_ids = data.batch["prompts"][mask]
            prompt_length = prompt_ids.shape[-1]
            valid_response_lengths = data.batch["attention_mask"][mask, prompt_length:].sum(-1)

            raw_bonuses_tensor[mask, valid_response_lengths - 1] = raw_bonuses
            reward_tensor[mask, valid_response_lengths - 1] = normalized_bonuses

        output = DataProto.from_dict(
            tensors={"rm_scores": reward_tensor}, non_tensors={"raw_bonuses": raw_bonuses_tensor.cpu().numpy()}
        )
        return output.to("cpu")
