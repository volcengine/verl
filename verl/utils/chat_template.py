# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from jinja2 import TemplateError

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, apply_chat_template_kwargs):
    try:
        # {% if loop.first and message['role'] != 'system' %} matches the system prompt
        return tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    except TemplateError as e:
        print(f"chat_template not support system prompt: {e}")
        return []
