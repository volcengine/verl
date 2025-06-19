from openai import OpenAI

from verl.utils.reward_score.math import last_boxed_only_string


GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    problem = extra_info["question"]
    prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
    messages = [
        {"role": "user", "content": prompt}
    ]
    reward_score = 0.0
    try:
        output = client.chat.completions.create(
            model="qwen2-5b",
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )
        response = output.choices[0].message.content
        result = last_boxed_only_string(response)
        reward_score = float(result == "True")
    except Exception as e:
        print("Error:", e)

    return reward_score
