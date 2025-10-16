import re


def normalize_frac(text: str) -> str:
    """
    Normalize LaTeX fraction notation, converting various loose forms to standard \frac{a}{b}
    to reduce downstream parsing library macro substitution failure warnings.
    """
    if not isinstance(text, str) or not text:
        return text

    s = text
    # 1) Unify \dfrac / \tfrac to \frac
    s = re.sub(r"\\[dt]frac\b", r"\\frac", s)

    # 2) \frac a b => \frac{a}{b}
    s = re.sub(r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)", r"\\frac{\1}{\2}", s)

    # 3) \frac{a} b => \frac{a}{b}
    s = re.sub(r"\\frac\s*{([^{}]+)}\s+([^\s{}]+)", r"\\frac{\1}{\2}", s)

    # 4) \frac a {b} => \frac{a}{b}
    s = re.sub(r"\\frac\s+([^\s{}]+)\s*{([^{}]+)}", r"\\frac{\1}{\2}", s)

    # 5) When only one parameter, pad denominator with 1: \frac{a} => \frac{a}{1}
    s = re.sub(r"\\frac\s*{([^{}]+)}\s*(?!{)", r"\\frac{\1}{1}", s)

    # 6) Extreme case: empty parameters, remove the fragment
    s = re.sub(r"\\frac\s*{\s*}\s*{\s*}", "", s)

    return s


def normalize_latex(text: str) -> str:
    """Aggregate common LaTeX normalization, currently only handles \frac related.
    More cleaning rules can be extended here in the future.
    """
    if not isinstance(text, str) or not text:
        return text
    s = text
    s = normalize_frac(s)
    return s


