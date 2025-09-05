import re

def validate_hybird_format(response: str, prompt: str) -> bool:
    """
    Validate the usage of <think>...</think> tags in the response based on the prompt instructions.

    Rules:
    - If prompt ends with "\\think" or has no label, expect exactly one <think> section with non-empty content.
    - If prompt ends with "\\no_think", expect exactly one <think> section with empty content (exactly "\n\n").
    
    Returns True if the response matches the expected format, False otherwise.
    """
    prompt_text = prompt.strip()

    expects_think = False
    expects_no_think = False

    if prompt_text.endswith(r"no_think"):
        expects_no_think = True
    elif prompt_text.endswith(r"think"):
        expects_think = True
    else:
        # Default: assume thinking is expected if no explicit label
        expects_think = True

    # Use regex to find all <think>...</think> blocks (non-greedy, DOTALL to match newlines)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_sections = pattern.findall(response)

    # Must have exactly one <think> block
    if len(think_sections) != 1:
        return 0

    content = think_sections[0]

    if expects_think:
        # Content must be non-empty when stripped (not just whitespace or newlines)
        return 1 if content.strip() != "" else 0

    elif expects_no_think:
        # Content must be exactly two newlines: "\n\n"
        return 1 if content == "\n\n" else 0

    return 0

