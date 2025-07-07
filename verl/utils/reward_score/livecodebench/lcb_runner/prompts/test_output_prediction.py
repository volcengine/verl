import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from verl.utils.reward_score.livecodebench.lcb_runner.lm_styles import LMStyle
from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks import TestOutputPredictionProblem


class PromptConstants:
    SYSTEM_MESSAGE_CHAT_GENERIC = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function.\
 The user has written a input for the testcase.\
 You will calculate the output of the testcase and\
 write the whole assertion statement in the markdown code block with the correct output."

    SYSTEM_MESSAGE_COMPLETION_GENERIC = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function."

    SYSTEM_MESSAGE_INST_CLLAMA = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function.\
 The user has written a input for the testcase.\
 You will calculate the output of the testcase and \
 write out the complete assertion statement between [PYTHON] and [/PYTHON] tags."

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example: 
```python 
# YOUR CODE HERE
```"""

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_MESSAGE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


def truncate_io(io):
    if len(str(io)) > 1000:
        io = str(io)[:1000] + "...."
        print(io)
    return io


def format_testcase_func_name_input(function_name, testcase):
    """
    use the form of "assert func_name(input) == "
    """
    # TODO should there be a space after the == ?
    input_str = ", ".join(testcase.split("\n"))
    return f"assert {function_name}({input_str}) == # TODO"


def parse_function_name_from_starter_code(starter_code):
    """
    starter_code : str
    """
    import ast

    tree = ast.parse(starter_code)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert fn is None
            fn = node.name
    return fn


def get_generic_question_template_test_completion(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"Problem:\n{question.question_content}"
    prompt += f"Function:\n```\n{question.starter_code}\n```\n"

    # parse function name from starter_code
    func_name = parse_function_name_from_starter_code(question.starter_code)
    prompt += "Please complete the following test case:\n\n"
    prompt += (
        f"```\n{format_testcase_func_name_input(func_name, testcase_input)}\n```\n"
    )

    return prompt


def get_cllama_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"### Question\n"
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Answer\n"
    return prompt


def get_deepseekcode_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"### Instruction: {PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n\n"
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Response:\n\n"
    return prompt


def get_magicoder_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    # prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt = f"Question:\n"
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"@@ Response \n"
    return prompt


def get_mixtral_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = get_generic_question_template_test_completion(question, testcase_input)
    return prompt


def get_wizard_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"""### Instruction: {PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"""
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Response:\n"
    return prompt


def get_phind_question_template_answer(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"\n\n### Assistant"
    return prompt

def get_qwen_question_template_answer(question: TestOutputPredictionProblem, testcase_input: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "abacusai/Dracarys-72B-Instruct", padding_side="left", use_fast=False
    )

    prompt = f"""### Instruction: {PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"""
    prompt += get_generic_question_template_test_completion(question, testcase_input)
    prompt += f"### Response:\n"

    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt

def format_prompt_test_output(
    question: TestOutputPredictionProblem, LanguageModelStyle: LMStyle
) -> str:
    testcase_input = question.test[0].input
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        return chat_messages
    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    elif LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n\n"
        prompt += f"{get_generic_question_template_test_completion(question, testcase_input).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ).rstrip(),
            }
        ]
        return system, prompt
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"
        prompt += (
            f"{get_generic_question_template_test_completion(question, testcase_input)}"
        )
        return prompt

    elif LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"
        prompt += (
            f"{get_generic_question_template_test_completion(question, testcase_input)}"
        )
        return prompt

    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = (
            f"{get_deepseekcode_question_template_answer(question, testcase_input)}"
        )
        return prompt
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_INST_CLLAMA}\n<</SYS>>\n\n"
        prompt += (
            f"{get_cllama_question_template_answer(question, testcase_input)}\n[/INST]"
        )
        return prompt
    elif LanguageModelStyle == LMStyle.MagiCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"
        prompt += f"{get_magicoder_question_template_answer(question, testcase_input)}"
        return prompt
    elif LanguageModelStyle == LMStyle.WizardCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_WIZARD}\n\n{get_wizard_question_template_answer(question, testcase_input)}"
        return prompt
    elif LanguageModelStyle == LMStyle.Phind:
        prompt = f"### System Prompt\n\n{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n\n### User Message\n\n{get_phind_question_template_answer(question, testcase_input)}"
        return prompt
    elif LanguageModelStyle == LMStyle.OC:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC}\n"
        prompt += (
            f"{get_generic_question_template_test_completion(question, testcase_input)}"
        )
        return prompt
    elif LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        return chat_messages
    elif (
        LanguageModelStyle == LMStyle.DracarysQwen
    ):
        prompt = f"{get_qwen_question_template_answer(question, testcase_input)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DracarysLlama:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "abacusai/Dracarys-Llama-3.1-70B-Instruct", padding_side="right", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )
