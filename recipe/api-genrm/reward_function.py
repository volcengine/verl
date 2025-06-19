from openai import AsyncOpenAI
import asyncio
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"


client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=300,
)

async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    problem = extra_info["question"]
    prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
    messages = [
        {"role": "user", "content": prompt}
    ]
    reward_score = 0.0
    try:
        output = await client.chat.completions.create(
            model="genrm-demo",
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )
        response = output.choices[0].message.content
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
        reward_score = float(result == "True")
    except Exception as e:
        print("Error:", e)

    return reward_score

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tasks = [compute_score(data_source, solution_str, ground_truth, extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results
