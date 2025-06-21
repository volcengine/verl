def get_task_cls(task_type):
    import sys

    assert sys.version_info >= (3, 10), "match expression only supported since Python 3.10"
    match task_type:
        case "openai/gsm8k":
            from .gsm8k_calculator import GSM8K

            return GSM8K
        case "sobokan":
            from .sobokan import Sobokan

            return Sobokan
        case _:
            from .regular_prompt import RegularPrompt

            return RegularPrompt
