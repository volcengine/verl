import json
import torch
import numpy as np
from typing import Any, Optional
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

"""
{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}
    {%- if ns.is_last_user %}
      {{'<｜Assistant｜></think>'}}
    {%- endif %}
    {%- set ns.is_last_user = false -%}
    {%- set ns.is_first = false %}
    {%- set ns.is_tool = false -%}
    {%- for tool in message['tool_calls'] %}
      {%- set formatted_args = tool['function']['arguments'] if tool['function']['arguments'] is string else tool['function']['arguments']|tojson %}
      {%- if not ns.is_first %}
        {%- if message['content'] is none %}
          {{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + formatted_args + '<｜tool▁call▁end｜>'}}
        {%- else %}
          {{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + formatted_args + '<｜tool▁call▁end｜>'}}
        {%- endif %}
        {%- set ns.is_first = true -%}
      {%- else %}
        {{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + formatted_args + '<｜tool▁call▁end｜>'}}
      {%- endif %}
    {%- endfor %}
    {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
  {%- endif %}
  {%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none) %}
    {%- if ns.is_last_user %}
      {{'<｜Assistant｜>'}}
      {%- if message['prefix'] is defined and message['prefix'] and thinking %}
        {{'<think>'}}
      {%- else %}
        {{'</think>'}}
      {%- endif %}
    {%- endif %}
    {%- set ns.is_last_user = false -%}
    {%- if ns.is_tool %}
      {{message['content'] + '<｜end▁of▁sentence｜>'}}
      {%- set ns.is_tool = false -%}
    {%- else %}
      {%- set content = message['content'] -%}
      {%- if '</think>' in content %}
        {%- set content = content.split('</think>', 1)[1] -%}
      {%- endif %}
      {{content + '<｜end▁of▁sentence｜>'}}
    {%- endif %}
  {%- endif %}
"""
class MultiTurnSFTDatasetDeepseek(MultiTurnSFTDataset):
    def tokenize_assistant(self, index, message, full_message, tools, enable_thinking):
        """
        reimplement the jinja logic to suit multiturn data for dsv31
        """
        # check if assistant is followed by user
        #is_last_user = False
        #if index > 0 and full_message[index - 1]["role"] == "user":
        #    is_last_user = True

        # 判断当前assistant消息是否有tool_calls
        has_tool_calls = "tool_calls" in message and message["tool_calls"] is not None

        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
            apply_chat_template_kwargs["thinking"] = enable_thinking

        tokens = []
        # tool_calls分支
        if has_tool_calls:
            #if is_last_user:
            #    tokens += processor.encode("<｜Assistant｜></think>", add_special_tokens=False)
            is_first = False
            for tool in message["tool_calls"]:
                # 格式化参数
                formatted_args = tool["function"]["arguments"]
                if not isinstance(formatted_args, str):
                    for k, v in formatted_args.items():
                        if isinsance(v, np.ndarray):
                            formatted_args[k] = list(v)
                    formatted_args = json.dumps(formatted_args, ensure_ascii=False)

                if not is_first:
                    if message.get("content") is None:
                        tokens += processor.encode(
                            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                            + tool["function"]["name"]
                            + "<｜tool▁sep｜>"
                            + formatted_args
                            + "<｜tool▁call▁end｜>",
                            add_special_tokens=False,
                        )
                    else:
                        tokens += processor.encode(
                            message["content"]
                            + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                            + tool["function"]["name"]
                            + "<｜tool▁sep｜>"
                            + formatted_args
                            + "<｜tool▁call▁end｜>",
                            add_special_tokens=False,
                        )
                    is_first = True
                else:
                    tokens += processor.encode(
                        "<｜tool▁call▁begin｜>"
                        + tool["function"]["name"]
                        + "<｜tool▁sep｜>"
                        + formatted_args
                        + "<｜tool▁call▁end｜>",
                        add_special_tokens=False,
                    )
            tokens += processor.encode("<｜tool▁calls▁end｜><｜end▁of▁sentence｜>", add_special_tokens=False)
        else:
            #if is_last_user:
            #    tokens += processor.encode("<｜Assistant｜>", add_special_tokens=False)
            #    if message.get("prefix") and enable_thinking:
            #        tokens += processor.encode("<think>", add_special_tokens=False)
            #    else:
            #        tokens += processor.encode("</think>", add_special_tokens=False)
            content = message.get("content", "")
            # remove <think>, when previous turn is not tool
            #if not (index > 0 and full_message[index - 1]["role"] == "tool"):
            #    if "</think>" in content:
            #        content = content.split("</think>", 1)[1]
            tokens += processor.encode(content + "<｜end▁of▁sentence｜>", add_special_tokens=False)

        # 构造inputs
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}
        return inputs

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list, 
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process a single message and return its tokenized representation.

        Args:
            index: turn index in the conversation
            message: A single message dictionary
            images: List of images to be used
            videos: List of videos to be used
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        # the ns.is_last_user logic in chat template
        # if ns.is_last_user: add <｜Assistant｜>, add </think>
        # strip the assistant's <think> if previous turn is not a tool (which is always the case if tokenizing differnt turns individually)
        #if message["role"] == "assistant":
        #    # The first assistant turn after user, add <｜Assistant｜></think> at begining
        #    inputs = self.tokenize_assistant(index, message, full_message, tools, enable_thinking)
        if message["role"] == "system":
            inputs = processor.apply_chat_template(
                [message],
                tools=tools,
                # add generation prompt to True, for the '<｜Assistant｜>' token
                # Only USER will have this triggerd for dsv3
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
        elif message["role"] == "user":
            message_mod = message.copy()
            # Assert message has only one message
            #if apply_chat_template_kwargs["enable_thinking"]:
            #  message_mod["content"] += "      <｜Assistant｜><think>"
            #else:
            message_mod["content"] += "      <｜Assistant｜></think>"
            inputs = processor.apply_chat_template(
                [message_mod],
                # add generation prompt to True, for the '<｜Assistant｜>' token
                # Only USER will have this triggerd for dsv3
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
        else:
            inputs = processor.apply_chat_template(
                [message],
                # add generation prompt to True, for the '<｜Assistant｜>' token
                # Only USER will have this triggerd for dsv3
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # remove system prompt if exists
        if index != 0 and message["role"] not in ["system"]:
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        #if message["role"] == "assistant":
        #    loss_mask = torch.ones_like(attention_mask)
        #    # mask out generation prompt if assistant message
        #    loss_mask[: len(self.generation_prompt)] = 0
        if message["role"] == "assistant":
          loss_mask = torch.ones_like(attention_mask)
        else:
          loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs


if __name__ == "__main__":
  from transformers import AutoTokenizer

  data_paths = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yumingxuan/dataset/rl/ds_data/train_data.parquet"
  tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/deepseek-ai/DeepSeek-V3.1-bf16")

  dataset = MultiTurnSFTDatasetDeepseek(
      parquet_files=data_paths, tokenizer=tokenizer, config=None
  )
  
  for data in dataset:
    pass
    