# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from recipe.collabllm.utils import extract_json, parse_messages

INTERACTIVITY_PROMPT = '''You are a helpful and meticulous conversation evaluator. \
Your task is to evaluate the interactivity of the responses provided by an AI assistant \
to user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's engagement, clarity, and ability to understand the user's needs. \
Give a float number between 0 and 1. 

Scoring Criteria:
- Let U = user understanding & response clarity ∈ [0,1]  
  - 1.0 = Fully understands the user's intent and gives a clear answer.  
  - 0.7 = Mostly understands and the answer is generally clear.  
  - 0.3 = Partially misunderstands or the answer is hard to follow.  
  - 0.0 = Misunderstands the intent and gives an unclear or irrelevant answer.
- Let Q = clarification in [0,1]
  - 1.0 = Asks precise, necessary clarifying questions when needed.
  - 0.7 = Asks somewhat helpful but incomplete clarifications.
  - 0.3 = Only asks generic questions (e.g., “Does that help?”).
  - 0.0 = Asks no clarifying questions when needed.
- Let S = suggestion helpfulness in [0,1]
  - 1.0 = Provides useful, actionable suggestions.
  - 0.7 = Suggestions are somewhat helpful but limited.
  - 0.3 = Suggestions are vague or generic.
  - 0.0 = No suggestions when they would clearly help.
score = average([U, Q, S])

Output format (JSON):
{{
    "thought": "<How interactive is the assistant?>",
    "interactivity": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. \
Use " or """ to wrap up the thought. You should not use other triple quotes inside the "thought" field. \
Instead you should use single quotes to avoid JSON escape issues.

Your evaluation:
'''


async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # Check if litellm is available, fallback to openai if not
    try:
        import litellm

        use_litellm = True
    except ImportError:
        # litellm not found, falling back to openai
        import openai

        use_litellm = False

    chat_history = parse_messages(messages, strip_sys_prompt=True)
    prompt = INTERACTIVITY_PROMPT.format(chat_history=chat_history)

    if use_litellm:
        full_response = (
            (
                await litellm.acompletion(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )
    else:
        client = openai.AsyncOpenAI()  # Assumes API key is set in environment
        full_response = (
            (
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )

    full_response = extract_json(full_response)

    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {"interactivity", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )

    interactivity = full_response.pop("interactivity")
    return float(interactivity)
