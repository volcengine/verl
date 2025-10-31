import re
import regex
import multiprocessing
from math import isclose
from typing import Union

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

from timeout_decorator import timeout, TimeoutError


def parse_digits(num):
    num = regex.sub(',', '', str(num))
    try:
        return float(num)
    except:
        if num.endswith('%'):
            num = num[:-1]
            if num.endswith('\\'):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r'\{.*,.*\}', input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip('{}')
        pmatrix = r'\begin{pmatrix}' + m.replace(',', '\\') + r'\end{pmatrix}'
        pmatrix_list.append(pmatrix)

    return ', '.join(pmatrix_list)

def extract_inside_str(input_str):

    if type(input_str) == str and input_str.strip().startswith("\\(") and input_str.strip().endswith("\\)") :
        try:
            input_str = input_str.strip().split("\\(")[1].split("\\)")[-2]
        except:
            pass
    if type(input_str) == str and input_str.strip().startswith("\\[") and input_str.strip().endswith("\\]"):
        try:
            input_str = input_str.strip().split("\\[")[1].split("\\]")[-2]
        except:
            pass
    if type(input_str) == str and input_str.strip().startswith("(") and input_str.strip().endswith(")"):
        try:
            input_str = input_str.strip().split("(")[1].split(")")[-2]
        except:
            pass
    if type(input_str) == str and input_str.strip().startswith("[") and input_str.strip().endswith("]"):
        try:
            input_str = input_str.strip().split("[")[1].split("]")[-2]
        except:
            pass
    
    if type(input_str) == str and input_str.strip().startswith("\\text{"):
        try:
            input_str = input_str.strip().split("\\text{")[1].split("}")[-2]
        except:
            pass
    if type(input_str) == str and input_str.strip().endswith("```"):
        try:
            input_str = input_str[:-len("```")]
        except:
            pass

    return input_str

def math_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: bool = True,
                use_timeout: bool = False,
                ) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if str(prediction) == str(reference):
        return True
    prediction = extract_inside_str(prediction)
    reference = extract_inside_str(reference)
    try: # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False
    # print("try math_eval")

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not 'pmatrix' in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
        (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ['{', "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if regex.match(r'(\(|\[).+(\)|\])', prediction) is not None and regex.match(r'(\(|\[).+(\)|\])', reference) is not None:
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                return True
    if (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}")) and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}")) and \
        (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}")) and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")):
        pred_lines = [line.strip() for line in prediction[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\") if line.strip()]
        ref_lines = [line.strip() for line in reference[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\") if line.strip()]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count('=') == 1 and reference.count('=') == 1:
        pred = prediction.split('=')
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split('=')
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count('=') == 1 and len(prediction.split('=')[0].strip()) <= 2 and '=' not in reference:
        if math_equal(prediction.split('=')[1], reference, include_percentage, is_close):
            return True
    elif reference.count('=') == 1 and len(reference.split('=')[0].strip()) <= 2 and '=' not in prediction:
        if math_equal(prediction, reference.split('=')[1], include_percentage, is_close):
            return True

    if use_timeout:
        try:
            if timeout(use_timeout)(symbolic_equal)(prediction, reference):
                return True
        except TimeoutError:
            print({"type": "timeout", "prediction": prediction, "reference": reference})
            pass
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, abs_tol=1e-3)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s
    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # print("try simplify")
    # simplify equal
    try:
        if a.equals(b) or simplify(a-b) == 0:
            return True
    except:
        pass

    # print("try equation")
    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    return False


def call_with_timeout(func, args=(), kwargs=None, timeout_duration=5):
    if kwargs is None:
        kwargs = {}
    try:
        with multiprocessing.get_start_method("spawn").Pool(1) as p:
            result = p.apply_async(func, args, kwargs)
            return result.get(timeout=timeout_duration)
    except TimeoutError:
        print("Timeout reached")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def _test_math_equal():
    # print(
    #     math_equal(
    #         "\\begin{pmatrix} a & b \\\\ c & d\\end{pmatrix}",
    #         "\\begin{pmatrix} a & b \\\\ c & d\\end{pmatrix}",
    #         use_timeout=True
    #     )
    # )
    print(
        math_equal(
            "a = b",
            "b = a"
        )
    )
