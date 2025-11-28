# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

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
        return tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    except TemplateError as e:
        logger.warning(f"Chat template does not support system prompt: {e}")
        return []
