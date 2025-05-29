import re
from typing import Literal, TypedDict


VALID_ACTION_TYPES = [
    "click",
    "complete",
    "drag_and_release",
    "hover",
    "key_press",
    "scroll",
    "type",
    "wait",
]


class ActionInfo(TypedDict):
    action_type: Literal[
        "click",
        "complete",
        "drag_and_release",
        "hover",
        "key_press",
        "scroll",
        "type",
        "wait",
    ]
    coordinates: list[tuple[float, float]] | None
    args: dict[str, str] | None


def extract_action(text: str) -> str:
    """
    Extracts the text before the first instance of (...) in the string.

    Args:
        text (str): The input string.

    Returns:
        str: The action name before `(...)`. Returns an empty string if not found.
    """
    match = re.search(r"\b(\w+)\s*\(", text)
    return match.group(1) if match else ""


def extract_content_by_tags(text: str, tags: list[str]) -> dict[str, str | None]:
    """
    Extracts the first occurrence of content inside specified tags and returns a dictionary.

    Parameters:
        text (str): The input string containing various tags.
        tags (list[str]): A list of tag names to extract content from.

    Returns:
        dict[str, Optional[str]]: A dictionary where keys are tag names,
            and values are the first content string or None if the tag is not found.
    """
    extracted: dict[str, str | None] = {}

    for tag in tags:
        # Build a regex pattern dynamically for each tag
        pattern = rf"<{tag}>(.*?)</{tag}>"
        # Find the first match for the current tag
        match = re.search(pattern, text, re.DOTALL)
        # Assign None if no match, otherwise assign the matched string
        extracted[tag] = match.group(1) if match else None

    return extracted


def get_action_info(action: str) -> ActionInfo:
    action_type = extract_action(action)
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"Invalid action type: {action_type}")
    return eval(action)


def click(
    x: float, y: float, button: Literal["left", "right"] = "left", double: bool = False
) -> ActionInfo:
    return ActionInfo(
        action_type="click",
        coordinates=[(x, y)],
        args={"button": str(button), "double": str(double)},
    )


def complete(answer: str = "", infeasible_reason: str = "") -> ActionInfo:
    return ActionInfo(
        action_type="complete",
        coordinates=None,
        args={"answer": str(answer), "infeasible_reason": str(infeasible_reason)},
    )


def drag_and_release(x1: float, y1: float, x2: float, y2: float) -> ActionInfo:
    return ActionInfo(
        action_type="drag_and_release",
        coordinates=[(x1, y1), (x2, y2)],
        args=None,
    )


def hover(x: float, y: float) -> ActionInfo:
    return ActionInfo(
        action_type="hover",
        coordinates=[(x, y)],
        args=None,
    )


def key_press(keys: list[str]) -> ActionInfo:
    return ActionInfo(
        action_type="key_press",
        coordinates=None,
        args={"keys": str(keys)},
    )


def scroll(x: float, y: float, delta_x: float = 0, delta_y: float = 100) -> ActionInfo:
    return ActionInfo(
        action_type="scroll",
        coordinates=[(x, y)],
        args={
            "horizontal": str("left" if delta_x < 0 else "right"),
            "vertical": str("up" if delta_y < 0 else "down"),
        },
    )


def type(x: float, y: float, text: str) -> ActionInfo:
    return ActionInfo(
        action_type="type",
        coordinates=[(x, y)],
        args={"text": str(text)},
    )


def wait(ms: int = 1000) -> ActionInfo:
    return ActionInfo(
        action_type="wait",
        coordinates=None,
        args={"ms": str(ms)},
    )
