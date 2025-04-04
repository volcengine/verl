from typing import List
from verl.utils.reward_score.hotpotqa import query_sglang_chat

unfaith_judge_prompt_template = '''You are a rigorous information analysis expert. Below are the historical search records and reasoning process. Your task is to evaluate whether the reasoning process accurately reflects the search results and if it produces content that contradicts the search results.

<|start of search histories|>
<<search_history>>
<|end of search histories|>

<|start of reasoning|>
<<reasoning>>
<|end of reasoning|>

If the reasoning is consistent to the search results, show "Yes". otherwise show "No". 
Please analyze the content and provide your answer in the following format: 
<start of my answer>
[Analysis] <your analysis> 
[Answer] Yes or No
<end of my answer>'''


def extract_real_content(text, label):
    return text.split(f"<{label}>\n")[-1].split(f"</{label}>")[0].strip()


def is_unfaithful_judge(history: List[str], reasoning: str):
    refine_history = [extract_real_content(h, 'observation') for h in history]
    prompt = unfaith_judge_prompt_template.replace(
        "<<search_history>>",
        "\n".join(refine_history)
    )
    prompt = prompt.replace(
        "<<reasoning>>",
        extract_real_content(reasoning, 'think')
    )
    for _ in range(5):
        response = query_sglang_chat(
            prompt, 
            # ['http://172.20.68.119:8000/v1'], 
            ['http://172.21.1.152:8000/v1'], 
            max_tokens=512
        )
        if response and len(response) == 0:
            continue
        else:
            break
    if len(response) == 0:
        return 0

    extract_answer = response.split("[Answer]")[-1].strip()
    # import json
    # print(json.dumps({"history": history, "reasoning": reasoning, "prompt": prompt, "response": response}, ensure_ascii=False))
    return 'no' in extract_answer.lower()


if __name__ == '__main__':
    history = [
        "<observation>\nThe sun rises in the east and sets in the west.\n</observation>",
        "<observation>\nThe earth is round.\n</observation>",
        "<observation>\nThe earth orbits the sun.\n</observation>"
    ]
    reasoning = "<think>\nThe sun orbits the earth.\n</think>"
    # reasoning = "<think>\nThe earth is flat.\n</think>"
    # reasoning = "<think>\nearth orbits the sun.\n</think>"
    print(is_unfaithful_judge(history, reasoning))
