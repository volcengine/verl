from bigcodebench.eval import untrusted_check
from recipe.collabllm.utils import parse_messages, extract_json


EXTRACT_MULTITURN_COMPLETION_PROMPT = '''You are a thorough and diligent conversation analyzer. Your task is to extract the final and complete version of a code solution that was generated during a multiturn conversation between a user and a chat assistant. The extracted content should reflect the final and comprehensive response provided by the assistant based on the user’s request.

You will be provided with the conversation:

<|The Start of The Conversation|>
{chat_history}
<|The End of The Conversation|>

Instructions for Extraction:

1. Identify the Most Update-to-Date Contents: Review the entire conversation to identify the most updated parts of the content provided by the assistant. This may include:
   - Different parts of the code snippet, function, class, or script.

2. Integrate Revisions: If the assistant made revisions, updates, or added sections throughout the conversation, ensure that these changes are fully integrated into the final content. The goal is to extract a single, cohesive output that incorporates all modifications and additions made during the conversation. For example, if the assistant writes a function at the beginning and changes a part, the final output should take the modification into account.

3. Focus on Completeness:
   - For code: Extract a complete and functional code snippet, including all necessary components such as imports, functions, classes, and any other essential elements. The code should be runnable, but you do not need to include any testing examples including the contents after `if __name__ == "__main__":`. Only the function code is required.

You should output a JSON object with two entries:
- "thought" (str): Output your thought process when extracting the final content. 
   1. How do different parts of the conversation contribute to the final output?
   2. How do you make sure you included the most updated and complete information?
   3. How do you make sure you did not include any information that is not necessary?
- "final_completion" (str): The final and complete version of the code extracted from the conversation.

Note: 
1. If there are multiple lines, you should use triple quotes (""") to wrap the content. For example, "final_completion": """first line. 
   second line.""" or "thought": """first line;
   second line.""".
2. In the "final_completion" entry, replace all double quotes (") with single quotes (') to prevent JSON formatting issues. For example, you can output "final_completion": "'Hello World' is a common phrase." 

Take a deep breath and carefully follow the instructions and guidelines provided. 
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
   prompt = EXTRACT_MULTITURN_COMPLETION_PROMPT.format(chat_history=chat_history)
   
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
   assert {'final_completion', 'thought'}.issubset(full_response.keys()), \
      f"Expected keys not found from {full_response.keys()}"

   final_completion = full_response.pop('final_completion')
   metadata = extra_info["single_turn_metadata"]
   res = untrusted_check(
            final_completion,
            metadata["test"],
            metadata["entry_point"],
            max_as_limit=300 * 1024,
            max_data_limit=300 * 1024,
            max_stack_limit=300 * 1024,
            min_time_limit=60,
            gt_time_limit=60
   )
   passed, info = res[0] == "pass", res[1]
   return float(passed)

