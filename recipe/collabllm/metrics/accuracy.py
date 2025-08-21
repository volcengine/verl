from recipe.collabllm.utils import parse_messages, extract_json


ACCURACY_PROMPT = '''You are a helpful and meticulous evaluator. Your task is to \
evaluate the *accuracy* of an AI model's answer to a target question. \
You will be given the target question, the ground truth answer, and the conversation between the AI and the user.

Provided Information:

<|The Start of Target Question and Ground Truth Answer|>
Target Question: {single_turn_prompt}
Ground Truth Answer: {ground_truth}
<|The End of Target Question and Ground Truth Answer|>

<|The Start of The Conversation|>
{chat_history}
<|The End of The Conversation|>

You should determine whether the model's final response to the target question is \
factually correct and consistent with the provided ground truth.

Rating criteria (binary):
  • 1 = Correct   — the response matches the ground truth.
  • 0 = Incorrect — the response contradicts or misses the ground truth.

Output format (JSON):
{{
    "thought": "<your reasoning here>",
    "accuracy": <0 or 1>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

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
    prompt = ACCURACY_PROMPT.format(
      single_turn_prompt=extra_info["interaction_kwargs"]['single_turn_prompt'],
      ground_truth=ground_truth,
      chat_history=chat_history
    )
    
    if use_litellm:
        full_response = (
            await litellm.acompletion(
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        ).choices[0].message.content
    else:
        client = openai.AsyncOpenAI()  # Assumes API key is set in environment
        full_response = (
          await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
          )
        ).choices[0].message.content

    full_response = extract_json(full_response)
    
    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {'accuracy', 'thought'}.issubset(full_response.keys()), \
        f"Expected keys not found from {full_response.keys()}"

    accuracy = full_response.pop('accuracy')
    return float(accuracy)

