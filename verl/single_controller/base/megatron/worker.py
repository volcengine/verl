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

from verl.single_controller.base.worker import DistGlobalInfo, DistRankInfo, Worker


class MegatronWorker(Worker):
    def __init__(self, cuda_visible_devices=None) -> None:
        super().__init__(cuda_visible_devices)

    def get_megatron_global_info(self):
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, cp_size=cp_size)
        return info

    def get_megatron_rank_info(self):
        from megatron.core import parallel_state as mpu

        tp_rank = mpu.get_tensor_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank)
        return info

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
    ):
        from transformers import AutoConfig

        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        def add_optimization_config_to_tf_config(tf_config):
            # add optimization config to tf_config, e.g. checkpointing
            if self.config.model.get("enable_gradient_checkpointing", False):
                gradient_checkpointing_cfg = dict(self.config.model.get("gradient_checkpointing_kwargs", dict()))
                tf_config.recompute_method = gradient_checkpointing_cfg.get("activations_checkpoint_method", "full")
                tf_config.recompute_granularity = gradient_checkpointing_cfg.get("activations_checkpoint_granularity", "full")
                tf_config.recompute_num_layers = gradient_checkpointing_cfg.get("activations_checkpoint_num_layers", -1)
            if megatron_config := self.config.get("megatron", {}):
                if extra := megatron_config.get("extra", {}):
                    for k, v in extra.items():
                        setattr(tf_config, k, v)

        add_optimization_config_to_tf_config(tf_config)

        print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config
