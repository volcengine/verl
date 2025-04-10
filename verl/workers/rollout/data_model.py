from enum import Enum
from typing import List, Optional, Literal, Dict

from pydantic import BaseModel
from transformers import PreTrainedTokenizer
from verl.workers.tool.data_model import OpenAIFunctionToolSchema, OpenAIFunctionToolCall
from verl.utils.model import compute_position_id_with_mask


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""
    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""
    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tools: Optional[List[OpenAIFunctionToolSchema]] = None
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    position_ids: List[int]
    loss_mask: List[int]
    reward_scores: Dict[str, float]

    format_config: dict = {
        "chatml": {
            "assistant_prefix_msg": "<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>\n",
            "tool_prefix_msg": "<|im_start|>tool\n",
            "tool_suffix_msg": "<|im_end|>\n",
        }
    }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> str:
        return tokenizer.apply_chat_template( # type: ignore
            conversation=[msg.model_dump() for msg in self.messages], 
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None, 
            add_generation_prompt=True, 
            tokenize=False
        )

    def add_assistant_message(
        self, 
        tokenizer: PreTrainedTokenizer, 
        content: str, 
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        format: Literal["chatml"] = "chatml"
    ) -> None:
        """Currently, we only support chatml format."""
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        self.messages.append(msg)
        if tool_calls is not None:
            content_with_tool_calls: str = tokenizer.apply_chat_template(  # type: ignore
                conversation=[msg.model_dump()], 
                add_generation_prompt=False, 
                tokenize=False
            )
        else:
            content_with_tool_calls = content
        # TODO: support other formats
        if format in self.format_config:
            prefix_msg = self.format_config[format]["assistant_prefix_msg"]
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            suffix_msg = self.format_config[format]["assistant_suffix_msg"]
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            if tool_calls is not None:
                content = content_with_tool_calls.split(f"{prefix_msg}")[-1].split(f"{suffix_msg}")[0]
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
                append_token_ids = content_token_ids + suffix_token_ids
            elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
                append_token_ids = prefix_token_ids + content_token_ids + suffix_token_ids
            else:
                raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-len(prefix_token_ids):])}")
            self.input_ids += append_token_ids
            _attention_mask = [1] * len(append_token_ids)
            self.attention_mask += _attention_mask
            _delta_position_ids = compute_position_id_with_mask(_attention_mask).tolist()
            last_position_id = self.position_ids[-1]
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            self.loss_mask += [1] * len(append_token_ids)
            self.position_ids += _position_ids
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def add_tool_response_message(
        self, 
        tokenizer: PreTrainedTokenizer, 
        content: str, 
        format: Literal["chatml"] = "chatml"
    ) -> None:
        """Currently, we only support chatml format."""
        msg = Message(role="tool", content=content)
        self.messages.append(msg)
        # TODO: support other formats
        if format in self.format_config:
            prefix_msg = self.format_config[format]["tool_prefix_msg"]
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            suffix_msg = self.format_config[format]["tool_suffix_msg"]
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
                append_token_ids = content_token_ids + suffix_token_ids
            elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
                append_token_ids = prefix_token_ids + content_token_ids + suffix_token_ids
            else:
                raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-len(prefix_token_ids):])}")
            self.input_ids += append_token_ids
            _attention_mask = [1] * len(append_token_ids)
            self.attention_mask += _attention_mask
            _delta_position_ids = compute_position_id_with_mask(_attention_mask).tolist()
            last_position_id = self.position_ids[-1]
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            self.loss_mask += [0] * len(append_token_ids)
            self.position_ids += _position_ids
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def finalize(
        self, 
        tokenizer: PreTrainedTokenizer, 
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP, 
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        self.response_ids = self.input_ids[len(self.prompt_ids):]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            eos_token_id = tokenizer.eos_token_id
            self.input_ids.append(eos_token_id)
            self.attention_mask.append(1)
            self.position_ids.append(self.position_ids[-1] + 1)
            self.loss_mask.append(0)
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")

