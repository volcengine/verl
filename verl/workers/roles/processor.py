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

from verl.utils.device import (
    is_cuda_available,
    is_npu_available,
)
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


class MicroBatchProcessor:
    def __init__(self):
        """
        Initialize the MicroBatchProcessor instance.

        This method is intended to set up the initial state of the MicroBatchProcessor.
        However, as it's a base class, the concrete implementation is left to the subclasses.
        Any subclass of MicroBatchProcessor should override this method to provide
        specific initialization logic.

        Raises:
            NotImplementedError: Always raised because this is an abstract initialization method.
        """
        raise NotImplementedError


    def preprocess(self, batch):
        """
        Preprocess a micro-batch of data before passing it to the model.

        This is an abstract method that should be implemented by subclasses.
        The implementation should transform the input batch data into a format
        suitable for the model.

        Args:
            batch: A micro-batch of data, typically a dictionary containing tensors
                   and metadata. The exact structure depends on the specific use case.

        Returns:
            dict: A dictionary containing the preprocessed input data ready for the model.

        Raises:
            NotImplementedError: This base method does not provide an implementation.
                                 Subclasses must override this method.
        """
        raise NotImplementedError


    def postprocess(self, outputs):
        """
        Postprocess the raw outputs from the model.

        This is an abstract method that should be implemented by subclasses.
        The implementation should transform the raw model outputs into a more
        usable format, such as extracting relevant information or performing
        additional calculations.

        Args:
            outputs: The raw outputs from the model. The exact structure depends
                     on the specific model and task.

        Returns:
            The post-processed outputs. The return type depends on the specific
            implementation in subclasses.

        Raises:
            NotImplementedError: This base method does not provide an implementation.
                                 Subclasses must override this method.
        """
        raise NotImplementedError


class CriticFSDPWithoutRmpadProcessor(MicroBatchProcessor):
    def __init__(self, config):
        """
        Initialize the CriticFSDPWithoutRmpadProcessor instance.

        Args:
            config: Configuration object containing necessary parameters for the processor.
        """
        self.config = config
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.engine_info = {}   # additional info required from the engine
        self.mb_ctx = {}        # context for the micro batch

    def mb_entry_guard(self):
        """
        Ensure that the necessary engine information is available before processing a micro-batch.
        This method acts as a guard at the entry point of micro-batch processing.
        It checks if the required key exists in the engine information dictionary and clears the 
        micro-batch context to prepare for new processing.

        Raises:
            AssertionError: If the 'use_value_head_model' key is not found in the engine_info dictionary.
        """
        # check required engine info
        assert "use_value_head_model" in self.engine_info.keys(), "use_value_head_model must be set in engine_info"
        self.mb_ctx = {} # clear the microbatch context


    def preprocess(self, batch):
        """
        Preprocess a micro-batch of data before passing it to the model.

        Args:
            batch (dict): A dictionary containing the input data for the micro-batch.
                          Expected keys include 'responses', 'input_ids', 'attention_mask',
                          'position_ids', and optionally 'multi_modal_inputs'.

        Returns:
            dict: A dictionary containing the preprocessed input data ready for the model.
        """
        self.mb_entry_guard()

        self.mb_ctx["response_length"] = batch["responses"].size(-1)
        inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                inputs[key] = torch.cat([inputs[key] for inputs in batch["multi_modal_inputs"]], dim=0)

        inputs["input_ids"] = batch["input_ids"]
        inputs["attention_mask"] = batch["attention_mask"]
        position_ids = batch["position_ids"]
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)
        inputs["position_ids"] = position_ids
        return inputs


    def postprocess(self, outputs):        
        """
        Postprocess the model outputs to extract the relevant value predictions.

        Args:
            outputs: The raw outputs from the model. The structure depends on whether the model 
                     uses a value head.

        Returns:
            torch.Tensor: The post-processed value predictions corresponding to the response sequence.
        """
        response_length = self.mb_ctx["response_length"]
        use_value_head_model = self.engine_info["use_value_head_model"]
        if use_value_head_model:
            # For trl.AutoModelForCausalLMWithValueHead
            values = outputs[2]
        else:
            values = outputs.logits
        values = values[:, -response_length - 1 : -1].squeeze(-1)
        return values



class CriticFSDPWithRmpadProcessor(MicroBatchProcessor):
    def __init__(self, config):
        """
        Initialize the CriticFSDPWithRmpadProcessor instance.

        Args:
            config: Configuration object containing necessary parameters for the processor.
        """
        self.config = config
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.engine_info = {}   # additional info required from the engine
        self.mb_ctx = {}        # context for the micro batch


    def mb_entry_guard(self):
        """
        Ensure that the necessary engine information is available before processing a micro-batch.
        This method acts as a guard at the entry point of micro-batch processing.
        It checks if the required key exists in the engine information dictionary and clears the 
        micro-batch context to prepare for new processing.

        Raises:
            AssertionError: If the 'use_value_head_model' key is not found in the engine_info dictionary.
        """
        # check required engine info
        assert "use_value_head_model" in self.engine_info.keys(), "use_value_head_model must be set in engine_info"
        self.mb_ctx = {} # clear the microbatch context


    def preprocess(self, batch):
        """
        Preprocess a micro-batch of data before passing it to the model. This method
        ensures the necessary engine information is available, extracts relevant information
        from the batch, unpads the input data, and pads and slices the inputs if sequence parallelism
        is enabled.

        Args:
            batch (dict): A dictionary containing the input data for the micro-batch.
                          Expected keys include 'responses', 'input_ids', 'attention_mask',
                          'position_ids', and optionally 'multi_modal_inputs'.

        Returns:
            dict: A dictionary containing the preprocessed input data ready for the model.
        """
        self.mb_entry_guard()
        self.mb_ctx["response_length"] = batch["responses"].size(-1)

        inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                inputs[key] = torch.cat([inputs[key] for inputs in batch["multi_modal_inputs"]], dim=0)

        input_ids = batch["input_ids"]
        bs, seqlen = input_ids.shape
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        if position_ids.dim() == 3:
            position_ids_rmpad = (
                index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                .transpose(0, 1)
                .unsqueeze(1)
            )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
        else:
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)

        # pad and slice the inputs if sp > 1
        if self.ulysses_sequence_parallel_size > 1:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
            )
            self.pad_size = pad_size

        inputs["input_ids"] = input_ids_rmpad
        inputs["attention_mask"] = None
        inputs["position_ids"] = position_ids_rmpad

        self.mb_ctx["indices"] = indices
        self.mb_ctx["seqlen"] = seqlen
        self.mb_ctx["bs"] = bs
        return inputs

    def postprocess(self, outputs):
        """
        Postprocess the model outputs to extract and format the value predictions.
        This method handles different types of model outputs based on whether the model
        uses a value head and also manages sequence parallelism if enabled.

        Args:
            outputs: The raw outputs from the model. The structure depends on whether the model 
                     uses a value head.

        Returns:
            torch.Tensor: The post-processed value predictions corresponding to the response sequence.
        """
        use_value_head_model = self.engine_info["use_value_head_model"]
        response_length = self.mb_ctx["response_length"]
        if use_value_head_model:
            # For trl.AutoModelForCausalLMWithValueHead
            values_rmpad = outputs[2].squeeze(0).unsqueeze(-1)
        else:
            values_rmpad = outputs.logits
            values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

        # gather output if sp > 1
        if self.ulysses_sequence_parallel_size > 1:
            values_rmpad = gather_outpus_and_unpad(
                values_rmpad, gather_dim=0, unpad_dim=0, padding_size=self.mb_ctx["pad_size"]
            )

        # pad it back
        values = pad_input(values_rmpad, 
                           indices=self.mb_ctx["indices"],
                           batch=self.mb_ctx["bs"],
                           seqlen=self.mb_ctx["seqlen"]).squeeze(-1)
        values = values[:, -response_length - 1 : -1]
        return values




def get_processor_cls(worker_name, config):
    """
    Retrieve the appropriate micro-batch processor class based on the worker name and configuration.

    Args:
        worker_name (str): The name of the worker, used to determine the type of processor needed.
        config (object): A configuration object containing various settings, such as the strategy
                         and model-specific parameters.

    Returns:
        class: The class of the micro-batch processor that matches the given worker name and configuration.

    Raises:
        NotImplementedError: If there is no matching processor class for the provided worker name and strategy.
    """
    if worker_name == "critic" and config.strategy == "fsdp":
        use_remove_padding = config.model.get("use_remove_padding", False)
        if use_remove_padding:
            return CriticFSDPWithRmpadProcessor
        else:
            return CriticFSDPWithoutRmpadProcessor
    else:
        raise NotImplementedError(f"Processor {worker_name} {config.strategy} not implemented")
