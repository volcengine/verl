import re
EXTRACTION_TEMPLATE = """
Look at the following question and an answer, your job is to extract the final answer.
## Question: 
{question}
## Answer: 
{answer}
Put the answer in the format of the following example: 
<ANSWER>: <your answer>
Example:
<ANSWER>: yes
<ANSWER>: Atom
Make sure your answer format matches the format required by the question. All final answers are strings.
"""

EQUALITY_TEMPLATE = r"""
Look at the following question and two statements (two answers to the question) and judge whether the two answers are equivalent or with exactly the same meaning considering the context of the question.
Question
%(question)s

YOUR TASK:
Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


RETRY_COUNT = 3
import requests
import time
import random

def query_sglang_chat(
    prompt,
    urls,
    use_logits=False,
    skip_special_tokens=True,
    n=1,
    max_tokens=512,
    temperature=0.0,
    top_p=0.5
):
    messages = [{"role": "user", "content": prompt}]
    extra_body = {
        "skip_special_tokens": skip_special_tokens
    }

    for try_counter in range(RETRY_COUNT):
        try:
            request_data = {
                "model": "default",
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages,
                "n": n,
                "max_tokens": max_tokens,
                "seed": random.randint(0, 100000),
                **extra_body
            }

            api_base = random.choice(urls)
            url = api_base + "/chat/completions"
            response = requests.post(
                url,
                json=request_data,
                headers={"content-type": "application/json"},
                timeout=180
            )

            if response.status_code == 200:
                resp_json = response.json()
                choices = resp_json['choices']
                content_list = [choice['message']['content'].strip()
                                for choice in choices]

                return content_list[0]
            else:
                print(
                    f"Failed to fetch response, chat: {response.status_code}, {response.text},{api_base}")
        except Exception as e:
            sleep_time = 2 * try_counter + 1
            if sleep_time > 30:
                exit(1)
            print("url: ", url)
            print(f"Error: {str(e)}, sleeping for {sleep_time} seconds")
            # with open("/workspace/lurui/openrlhf-glm/logs/outputs/api_error.jsonl", "a") as f:
            # f.write(json.dumps({"url": urls, "error": str(
            # e), "request_data": request_data}) + "\n")
            time.sleep(sleep_time)

    return ""

def extract_solution(question, response, extractor_urls, eos_token):
    # response = response.strip().split('\n')
    # resp_text = [x for x in response if x.strip()]
    # resp_text = "\n".join(resp_text[-3:])
    response = response.replace("<|endoftext|>", "").strip()
    resp_text = response.strip().split("<|assistant|>")[-1]

    # for glm
    # if not resp_text.strip().endswith("<|user|>"):
    # for qwen
    # if not resp_text.strip().endswith("<|im_end|>"):

    # general eos_token
    if not resp_text.strip().endswith(eos_token):
        print(f"not end with {eos_token}")
        print(resp_text)
        return -1

    answer_template = EXTRACTION_TEMPLATE.format(
        question=question,
        answer=resp_text
    )
    answer = None
    for _ in range(6):
        extracted_answer = query_sglang_chat(
            prompt=answer_template, urls=extractor_urls, temperature=0.0, top_p=0.5)
        if extracted_answer is None:
            answer = None
            continue
        else:
            answer = extracted_answer.replace("<ANSWER>: ", "").strip()
            break
    return answer

def checker_check_equality(question: str, expr1: str, expr2: str, urls):
    prompt = EQUALITY_TEMPLATE % {"question":question,"expression1": expr1, "expression2": expr2}
    # print(prompt)
    response = []
    for _ in range(10):
        response = query_sglang_chat(prompt, urls)
        if response and len(response) == 0:
            continue
        else:
            break
    # print(response)
    if len(response) == 0:
        return 0
    return response.lower().strip() == "yes"

def compute_score(solution_str, ground_truth, format_score=0., score=1., question="", extractor_urls=[], checker_urls=[], tokenizer=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # for test
    # return random.random()
    # print(f"compute score {tokenizer=}")

    eos_token = tokenizer.eos_token

    # for llm-as-judge
    response = solution_str.replace("<|endoftext|>", "").strip()
    resp_print = response.strip().split("<|assistant|>")[-1]
    answer = extract_solution(question, response=solution_str, extractor_urls=extractor_urls, eos_token=eos_token)
    print(f"computing score of {answer=} {ground_truth=}")

    # not end with eos_token(length or max turns)

    if answer == -1:
        return float(-1)

    if answer is None:
        ans = float(0)
    else:
        if answer == ground_truth:
            ans = float(score)
        else:
            if checker_check_equality(question, answer, ground_truth, checker_urls):
                ans = float(score)
            else:
                ans = format_score

    import json
    print(json.dumps({"resp": resp_print, "answer": answer, "ground_truth": ground_truth, "judge": ans}, ensure_ascii=False))
    return ans