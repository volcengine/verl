import re
import jieba


def contains_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def remove_latex_tags(text: str) -> str:
    # 去除 $...$ 和 $$...$$
    return re.sub(r'\${1,2}.*?\${1,2}', '', text, flags=re.DOTALL)

def remove_code_blocks(text: str) -> str:
    # 移除所有``` ```包裹的代码块
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)


def detect_language_mix(question: str, response: str) -> bool:
    # 先去除 LaTeX 标签
    question_clean = remove_latex_tags(question)
    clean_response = remove_code_blocks(response)

    if contains_chinese(question_clean):
        # question 包含中文，检测 response 中是否有大段非中文内容
        non_chinese_spans = re.findall(r'[^\u4e00-\u9fff]{2000,}', clean_response)
        return bool(non_chinese_spans)
    elif re.search("[a-zA-Z]", question_clean):
        return contains_chinese(clean_response)

    return False


def check_repetition(text: str, ngram: int = 60, threshold: int = 20) -> bool:
    if contains_chinese(text):
        words = list(jieba.cut(text))
    else:
        words = text.split()
    if len(words) < ngram:
        return False

    ngram_counts = {}
    for i in range(len(words) - ngram + 1):
        current_ngram = ' '.join(words[i:i+ngram])
        ngram_counts[current_ngram] = ngram_counts.get(current_ngram, 0) + 1
        if ngram_counts[current_ngram] >= threshold:
            return True

    for i in range(0, len(text) - 512, 512):
        chunk = text[i : i + 512]
        if chunk in text[:i] or chunk in text[i + 512:]:
            return True
    return False


def check_truncation(text: str) -> bool:
    text = text.strip().lower()
    if not text:
        return True
    incomplete_endings = [
        "等", "为", "是", "得出", "因为", "所以", "即", "可得", "如上", "如下", "如下所示", "见上式", 
        "因此", "最终", "最后", "可知", "得到"
    ]
    if any(text.endswith(ending) for ending in incomplete_endings):
        return True
    if re.search(r'[\=\+\-\*/\^_\$\\]$', text):
        return True
    if text.endswith(("...", "……", "---")):
        return True
    
    if text.strip()[-1] not in "。！？|.?!$}])" and text.strip()[-1] not in "m":
        return True

    return False

