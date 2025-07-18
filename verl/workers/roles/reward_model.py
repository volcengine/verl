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

import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_nccl_backend,
)
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.workers.engine import EngineRegistry


class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())

        self.config = config
        self.engine = EngineRegistry.new(self.config.strategy, self.config)

        if self.config.model.input_tokenizer is None:
            self.use_different_tokenizer = False
        else:
            self.use_different_tokenizer = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.init_model(build_optim=False)

    def _reencode_by_rm_tokenizer(self, data: DataProto):
        ori_values = {
            "input_ids": data.batch["input_ids"],
            "attention_mask": data.batch["attention_mask"],
            "position_ids": data.batch["position_ids"],
        }

        src_max_length = data.batch["attention_mask"].shape[-1]
        src_tokenizer = self.engine.input_tokenizer
        target_tokenizer = self.engine.tokenizer

        rm_input_ids = []
        rm_attention_mask = []
        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        # NOTE(jhz): need to replace into origin values after compute reward!
        data.batch["input_ids"] = rm_input_ids
        data.batch["attention_mask"] = rm_attention_mask
        data.batch["position_ids"] = rm_position_ids
        return data, ori_values

    def _fn_post_scores(self, micro_batch, preds):
        batch_size = micro_batch["input_ids"].shape[0]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        scores = preds.squeeze(-1)  # (batch_size, seq_len)

        # extract the result of the last valid token
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        scores = scores[torch.arange(batch_size), eos_mask_idx]

        return scores, {"scores": scores.clone().detach()}

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        if self.use_different_tokenizer:
            data, ori_values = self._reencode_by_rm_tokenizer(data)

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output = self.engine.infer_batch(data, post_fn=self._fn_post_scores)

            if self.use_different_tokenizer:
                data.batch.update(ori_values)

            scores = output["scores"]
            token_level_scores = self._expand_to_token_level(data, scores)
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.engine.unshard_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.engine.module) == 1:
            self.engine.module._handle.reshard(True)

        output = output.to("cpu")
        return output
