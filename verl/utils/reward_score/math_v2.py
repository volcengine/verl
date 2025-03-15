import re
import signal
from typing import Optional, Any
import torch

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]"                                                                             ,
    )


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem":
                "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution":
                "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot":
                "1",
        },
        {
            "problem":
                "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution":
                "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot":
                "1",
        },
        {
            "problem":
                "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution":
                "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot":
                "1",
        },
        {
            "problem":
                "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution":
                "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot":
                "1",
        },
    ]

def last_boxed_only_string_v2(string: str) -> Optional[str]:
    """
    find last \\boxed{...}
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    left = "\\boxed{"

    assert s[:len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"

    return s[len(left):-1]


class timeout:

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=10):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                    sympy.parsing.latex.errors.LaTeXParsingError,
                    sympy.SympifyError,
                    TypeError,
            ):
                # eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                # eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                # eval_logger.debug(
                # f"Had some trouble simplifying when comparing {x1} and {x2}"
                # )
                return False

    except TimeoutError:
        # eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        # eval_logger.error(e)
        raise
    except Exception as e:
        # eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    # "ft", #this is dangerous, infty, left will be damaged!
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


INVALID_ANS_GSM8k = "[invalid]"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
STRICT_BOX_PATTERN = r"\\boxed\{([^}]*)\}"


def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st


def is_correct_integer(
    og_pred,
    gt,
):
    numbers = re.findall(r'-?\d+', og_pred[-100:])
    numbers = numbers[-1] if len(numbers) > 0 else ""  # 很难通过枚举把最后一个搞成正确答案
    correctness = gt == numbers
    return correctness, og_pred[-100:]


def is_correct_minerva(og_pred, gt, gt_need_extract=False):
    og_pred = og_pred[-300:]  #math500最长answer为159
    match = re.findall(ANSWER_PATTERN, og_pred)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string_v2(gt)))
    else:
        gt = normalize_final_answer(gt)
    # return (pred == gt or is_equiv(pred, gt)), pred
    return (pred == gt), pred


def is_correct_strict_box(pred, gt, pause_tokens_index):
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100:]
    else:
        pred = pred[-100:]
    pred = last_boxed_only_string_v2(pred)
    pred = remove_boxed(pred) if pred is not None else None
    return 1 if (pred == gt) else -1, pred


def verify(pred, answer, resp_len, max_resp_len, add_int_verify=True, strict_box_verify=False, pause_tokens_index=None) -> bool:

    if strict_box_verify:
        corr_strict_box, pred_strict_box = is_correct_strict_box(pred, answer, pause_tokens_index)
        return corr_strict_box

    corr_minerva, pred_minerva = is_correct_minerva(pred,
                                                    answer)  # To remove if math is also converted to interger format
    if add_int_verify:
        corr_integer, pred_integer = is_correct_integer(pred, answer)
        pred = pred_minerva if corr_minerva else pred_integer
        corr = corr_minerva or corr_integer
    else:
        pred = pred_minerva
        corr = corr_minerva

    return corr


def compute_score(batch_info, solution_str, ground_truth, config, rm_name, pause_tokens_index, **argv) -> tuple[float, dict[str, Any]]:
    """
    default行为：对给1，其余给-1
    punish_no_answer:
    * v0: 0
    * v1: -0.1
    * v2: -0.2
    """
    prompt_length = batch_info['prompts'].shape[-1]
    max_resp_len = batch_info['responses'].shape[-1]
    resp_len = sum(batch_info['attention_mask'][prompt_length:].tolist())
    add_int_verify = config.reward_model.add_int_verify
    strict_box_verify = config.reward_model.strict_box_verify

    correct = verify(solution_str, ground_truth, resp_len, max_resp_len, add_int_verify, strict_box_verify,
                            pause_tokens_index)

    final_reward = 1 if correct else -1

    # extra_info = {"acc": correct, "format/answer": "answer:" in solution_str, "format/boxed": "boxed{" in solution_str, "format/option": False}

    return final_reward


if __name__ == "__main__":

    # 不含合法答案的长/短回复，启用不含answer惩罚
    # pred = r"""Another way is to look at the height of one corner of our shape considering the big square we are trying to find where this outer tangent line from the big circle to the adjacent small circles is touching the corners of those small circles at specific heights relative to the center of the big circle which may be related to the radius of the circle such as the bottom of the outer tangent line from the big excess outer circle to the adjacent small circles is some height relative to the center of the big outer circle which directly relates to the radius length of the small circles in relationship to the big outer shape in their outer area. We draw a line from the corner of one of the small circles to the outer edge of the big outer shape and from the center of the small circle perpendicular to the outer edge of the big outer shape so that it hits the outer edge of the big outer shape at a right angle which we call $ h $ which may be related to the radius of the small circle and the shape formed at that corner in relationship to the outer edges of the outer shapes. And the distance from the corner of one of the small circles to the outer edge of the big outer shape which includes the half of this figure that represents how high the outer tangent line from the big excess outer circle to the adjacent small circles is touching the corners of those small circles at specific heights relative to the center of the big circle which includes the radius part of the small circle which we call $ l $ . If we know the height of this line from the corner of the big outer main shape and the line perpendicular to the outer edge of the big outer shape which is what we call it as coming from the corner of the big outer main shape and is hitting the outer edge of the big outer shape with the half of this outer tangent line from the big excess outer circle to the adjacent small circles which we call it with the radius of the small circles which we call this value $ h $ which is what we are looking at in relationship to how that relates to the rest of the features of this outer shape. We might use some trigonometry involving the angle between this half part of the outer tangent line from the big excess outer circle to the adjacent small circles being some angle at one corner of one of the small circles in relationship to the other parts of the outer shape to figure out what the value of the radius of the small circle is. And if we have this value of the height of this line from the corner of the big outer main shape and the line perpendicular to the outer edge of the big outer shape which is what we have already called it as coming from the corner of the big outer main shape and is hitting the outer edge of the big outer shape with the half of this outer tangent line from the big excess outer circle to the adjacent small circles which we call it with the radius of the small circles which we call this value $ h $ which is what we have been using to think about in relationship to how that relates to the rest of the features of this outer shape and the angle at one corner of one of the small circles in relationship to the other parts of the outer shape which we call the "angle between relevant sides value" let's call it $ \theta_{rs} $ . Then using the height of this line from the corner of the big outer main shape and the line perpendicular to the outer edge of the big outer shape which is what we call it as coming from the corner of the big outer main shape and is hitting the outer edge of the big outer shape with the half of this outer tangent line"""
    answer = "22+12\sqrt{2}"
    pred = r"""Answer:22+12\sqrt{2}"""
    assert verify(pred, answer, 0, 16384, False, "v0", strict_box_verify=True) == -1

    pred = "So we have 235 as answer."
    answer = "235"
    assert verify(pred, answer, 0, 16384, False, "v0", strict_box_verify=True) == -1

    pred = "So we have \\boxed{235} as answer."
    answer = "235"
    assert verify(pred, answer, 0, 16384, False, "v0", strict_box_verify=True) == 1

    pred = "Answer: 11"
    answer = "11"
    assert verify(pred, answer, 0, 16384, False, "v0", strict_box_verify=True) == -1

    pred = "Answer: \\boxed{11}"
    answer = "-13"
    assert verify(pred, answer, 0, 16384, False, "v0", strict_box_verify=True) == -1

    pred = "Answer: \\boxed{11}"
    answer = "11"
    assert verify(pred, answer, 0, 16384, False, "v0", strict_box_verify=True) == 1
