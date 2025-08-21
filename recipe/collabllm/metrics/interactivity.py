from recipe.collabllm.utils import parse_messages, extract_json

INTERACTIVITY_PROMPT = '''You are a helpful and meticulous conversation evaluator. \
Your task is to evaluate the *interactivity* of the responses provided by an AI assistant \
to user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's engagement, clarity, and ability to understand the user's needs. \
Give a float number between 0 and 1, where:
    1 = Highly interactive: The assistant is very engaging, asks all relevant questions, and significantly enhances understanding and problem-solving.
     - Example: The assistant thoroughly understands the user's question, asks for necessary clarifications, such as "It sounds like you're asking about the causes of climate change. Are you looking for specific examples or a general overview?"
    0.5 = Moderately interactive: The assistant is engaging, asks some relevant questions, but can be substantially improved.
     - Example: The assistant asks some relevant questions about the user's inquiry but misses key details, such as "Are you asking about the effects of climate change?" but does not probe further for clarification.
    0 = Low interactivity: The assistant shows low engagement, asks few relevant questions, and barely try to understand the user's needs.
     - Example: The assistant provides a vague or incomplete response without fully understanding the user's intent, such as "Climate change is bad," without asking any follow-up questions or providing detailed information.


Output format (JSON):
{{
    "thought": "<How interactive is the assistant?>",
    "interactivity": <score>
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
    prompt = INTERACTIVITY_PROMPT.format(chat_history=chat_history)
    
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
    assert {'interactivity', 'thought'}.issubset(full_response.keys()), \
        f"Expected keys not found from {full_response.keys()}"

    interactivity = full_response.pop('interactivity')
    return float(interactivity)

