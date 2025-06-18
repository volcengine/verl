import asyncio
import inspect
import logging
import os
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.messages.tool import invalid_tool_call, tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from omegaconf import DictConfig
from pydantic import Field
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.openai_api.protocol import Tool
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from verl.tools.schemas import OpenAIFunctionCallSchema

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

sampling_params_names = set(inspect.signature(SamplingParams.__init__).parameters)
sampling_params_names.discard("self")


class SGLangChatModel(BaseChatModel):
    """LangChain ChatModel wrapper for SGLang engine."""

    engine: Engine = Field(description="SGLang engine instance")
    sampling_params: Dict[str, Any] = Field(description="Detault sampling parameters for generation")
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional kwargs for chat template rendering")
    tokenizer: PreTrainedTokenizerBase = Field(description="Tokenizer for the model")
    tool_call_parser_type: str = Field(description="Type of tool call parser to use")
    rollout_config: DictConfig = Field(description="Rollout configuration")

    # Disable LangChain cache for SGLang models - not configurable by users
    cache: Literal[False] = Field(default=False, exclude=True, init=False, description="Cache setting (always disabled)")

    @property
    def _llm_type(self) -> str:
        return "sglang"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result_metadata = {"chat_template_kwargs": {**self.chat_template_kwargs, **(chat_template_kwargs or {})}}

        input_ids = self.tokenizer.apply_chat_template(convert_to_openai_messages(messages), tools=kwargs.get("tools"), add_generation_prompt=True, tokenize=True, **result_metadata["chat_template_kwargs"])
        max_new_tokens = min(self.rollout_config.response_length, self.rollout_config.max_model_len - len(input_ids) - 1)
        if max_new_tokens <= 0:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))], llm_output={"meta_info": {"finish_reason": {"type": "length"}}, **result_metadata})

        # Update sampling params with any additional kwargs
        result_metadata["sampling_params"] = sampling_params = {k: v for k, v in {**self.sampling_params, **(sampling_params or {})}.items() if k in sampling_params_names}
        if stop:
            sampling_params["stop"] = stop
        sampling_params["max_new_tokens"] = max_new_tokens
        output = await self.engine.async_generate(input_ids=input_ids, sampling_params=sampling_params)

        # Parse tool calls
        content, meta_info, tool_calls, invalid_tool_calls = output["text"], output["meta_info"], [], []
        message = AIMessage(content=content)
        if (tool_call_parser := kwargs.get("tool_call_parser")) and tool_call_parser.has_tool_call(content):
            try:
                normed_content, raw_tool_calls = tool_call_parser.parse_non_stream(content)
                for i, raw_tool_call in enumerate(raw_tool_calls):
                    tool_call_id = f"{meta_info['id']}-{i}"
                    tool_call_schema = OpenAIFunctionCallSchema(name=raw_tool_call.name, arguments=raw_tool_call.parameters)
                    if tool_call_schema.has_decode_error:
                        logger.warning(f"{raw_tool_call.name} has invalid tool call arguments: {repr(raw_tool_call.parameters)}")
                        invalid_tool_calls.append(invalid_tool_call(id=tool_call_id, name=raw_tool_call.name, args=raw_tool_call.parameters))
                    else:
                        tool_calls.append(tool_call(id=tool_call_id, name=tool_call_schema.name, args=tool_call_schema.arguments))
                message = AIMessage(content=normed_content, tool_calls=tool_calls, invalid_tool_calls=invalid_tool_calls)
            except (JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse tool calls: {e}", exc_info=True)

        return ChatResult(generations=[ChatGeneration(message=message)], llm_output={**meta_info, **result_metadata})

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        strict: bool = False,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> Runnable:
        tools = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        return super().bind(
            tools=tools,
            tool_call_parser=FunctionCallParser(tools=[Tool.model_validate(tool) for tool in tools], tool_call_parser=self.tool_call_parser_type),
        )
