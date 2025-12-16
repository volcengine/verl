# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
from typing import Any

from jinja2 import TemplateError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    try:
        return tokenizer.apply_chat_template([{}], tokenize=True, **apply_chat_template_kwargs)
    except TemplateError as e:
        logger.warning(f"Chat template does not support system prompt: {e}")
        return []


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


# =============================================================================
# Tokenization utility functions
# These functions are extracted from ToolAgentLoop for reusability in external
# projects like remote rollout servers.
# =============================================================================


def apply_chat_template_with_processor(
    processor,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Apply chat template via processor (tokenize=False).

    NOTE: To preserve ToolAgentLoop's original behavior, callers should decide
    whether this runs in an executor or on the event loop thread.
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    if tools is None:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
    return processor.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **apply_chat_template_kwargs,
    )


def apply_chat_template_with_tokenizer(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Apply chat template via tokenizer (tokenize=True).

    IMPORTANT: Some call sites intentionally do NOT pass any extra kwargs
    (e.g. incremental tool/user message tokenization). When no kwargs are needed,
    simply omit the apply_chat_template_kwargs parameter.
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    if tools is None:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **apply_chat_template_kwargs,
    )


def tokenize_with_processor(
    processor,
    *,
    raw_prompt: str,
    image_data: Any | None,
) -> list[int]:
    """Tokenize a pre-rendered prompt string via processor."""
    model_inputs = processor(text=[raw_prompt], images=image_data, return_tensors="pt")
    return model_inputs.pop("input_ids").squeeze(0).tolist()
