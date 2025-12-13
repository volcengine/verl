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

import os
from typing import Any


def resolve_config_path(config_path: str) -> str:
    """Resolve agent loop configuration file path.

    In multi-node Ray training, relative paths may not resolve correctly
    because the working directory on remote nodes can differ from the driver node.
    This function resolves relative paths by checking multiple locations in order:
    1. If already absolute, return as-is
    2. Try current working directory
    3. Try relative to verl package installation (project root)

    Args:
        config_path: Configuration file path (relative or absolute)

    Returns:
        Absolute path to the configuration file

    Raises:
        FileNotFoundError: If the configuration file cannot be found
    """
    # Return absolute paths unchanged
    if os.path.isabs(config_path):
        return config_path

    # Try current working directory first
    cwd = os.path.abspath(os.getcwd())
    cwd_path = os.path.abspath(os.path.join(cwd, config_path))
    if (cwd_path == cwd or cwd_path.startswith(cwd + os.sep)) and os.path.exists(cwd_path):
        return cwd_path

    # Try relative to verl project root (where verl package is installed)
    try:
        import verl

        verl_package_dir = os.path.abspath(os.path.dirname(verl.__file__))

        # Strategy 1: For development/editable installs.
        project_root = os.path.dirname(verl_package_dir)
        dev_path = os.path.abspath(os.path.join(project_root, config_path))
        if (dev_path == project_root or dev_path.startswith(project_root + os.sep)) and os.path.exists(dev_path):
            return dev_path

        # Strategy 2: For standard package installations.
        install_path = os.path.abspath(os.path.join(verl_package_dir, config_path))
        if (install_path == verl_package_dir or install_path.startswith(verl_package_dir + os.sep)) and os.path.exists(
            install_path
        ):
            return install_path
    except (ImportError, AttributeError):
        pass  # verl not installed or __file__ not available

    # File not found - raise clear error
    raise FileNotFoundError(
        f"Agent loop configuration file not found: {config_path}. Tried current directory and verl project root."
    )


# tokenizer.apply_chat_template is not working properly for gpt-oss model.
# Because the chat template requires tool call messages to parse tool response messages
# so we need to format the tool response manually.
def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"


# =============================================================================
# Tokenization utility functions
# These functions are extracted from ToolAgentLoop for reusability in external
# projects like remote rollout servers.
# =============================================================================


def compute_system_prompt(
    tokenizer,
    apply_chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Compute system prompt tokens for prefix stripping in incremental tokenization.

    Args:
        tokenizer: HuggingFace tokenizer instance
        apply_chat_template_kwargs: Model-specific kwargs for apply_chat_template

    Returns:
        Token IDs representing the system/template prefix
    """
    return tokenizer.apply_chat_template(
        [{}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )


def apply_chat_template_with_processor(
    processor,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any],
) -> str:
    """Apply chat template via processor (tokenize=False).

    NOTE: To preserve ToolAgentLoop's original behavior, callers should decide
    whether this runs in an executor or on the event loop thread.
    """
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
    apply_chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Apply chat template via tokenizer (tokenize=True).

    IMPORTANT: Some call sites intentionally do NOT pass any extra kwargs
    (e.g. incremental tool/user message tokenization). Callers should pass an
    empty dict ({}), which is equivalent to not passing kwargs at all.
    """
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


def build_gpt_oss_tool_response_text(messages: list[dict[str, Any]], tool_call_names: list[str]) -> str:
    """Build gpt-oss tool response text (manual formatting + generation prompt)."""
    tool_response_texts: list[str] = []
    for i, tool_msg in enumerate(messages):
        actual_tool_name = tool_call_names[i]
        formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
        tool_response_texts.append(formatted)
    return add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
