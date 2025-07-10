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
        raise NotImplementedError


    def preprocess(self, batch):
        raise NotImplementedError


    def postprocess(self, outputs):
        raise NotImplementedError


class CriticFSDPWithoutRmpadProcessor(MicroBatchProcessor):
    def __init__(self, config):
        self.config = config
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.engine_info = {}   # additional info required from the engine
        self.mb_ctx = {}        # context for the micro batch

    def mb_entry_guard(self):
        # check required engine info
        assert "use_value_head_model" in self.engine_info.keys(), "use_value_head_model must be set in engine_info"
        self.mb_ctx = {} # clear the microbatch context


    def preprocess(self, batch):
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
        self.config = config
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.engine_info = {}   # additional info required from the engine
        self.mb_ctx = {}        # context for the micro batch


    def mb_entry_guard(self):
        # check required engine info
        assert "use_value_head_model" in self.engine_info.keys(), "use_value_head_model must be set in engine_info"
        self.mb_ctx = {} # clear the microbatch context


    def preprocess(self, batch):
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
        values = pad_input(values_rmpad, indices=self.mb_ctx["indices"], batch=self.mb_ctx["bs"], seqlen=self.mb_ctx["seqlen"]).squeeze(-1)
        values = values[:, -response_length - 1 : -1]
        return values




def get_processor_cls(worker_name, config):
    if worker_name == "critic" and config.strategy == "fsdp":
        use_remove_padding = config.model.get("use_remove_padding", False)
        if use_remove_padding:
            return CriticFSDPWithRmpadProcessor
        else:
            return CriticFSDPWithoutRmpadProcessor
    else:
        raise NotImplementedError(f"Processor {worker_name} {config.strategy} not implemented")
