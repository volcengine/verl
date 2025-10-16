from functools import partial

from latex2sympy2 import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex


def auto_extract(task, all_outputs, extra_info=None):
    task2extract_fn = {
        "math": partial(extract_answer, data_name=task),
        "gpqa": partial(extract_answer, data_name=task),
    }
    assert task in task2extract_fn, f"{task} not in {list(task2extract_fn.keys())}"
    extract_fn = task2extract_fn[task]

    # First perform LaTeX normalization to reduce parsing library macro substitution failures
    cleaned_outputs = [normalize_latex(x) for x in all_outputs]
    model_answers = [extract_fn(generated_text) for generated_text in cleaned_outputs]

    return [answer for answer in model_answers if answer is not None]