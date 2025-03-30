from verl.utils.agent_tasks.swedev import *

def default_prompt_generator(row):
    return row["prompt"]

def default_preprocess_dataset(dataframe):
    return dataframe

def preprocess_gsm8k_dataset(dataframe, tokenizer=None, max_prompt_length=1024):
    # print("\n[Stage 1] Dataset Preprocessing:")
    # sample = dataframe.iloc[0]
    # print(f"Raw data sample:", sample)
    
    if 'data_source' not in dataframe.columns:
        dataframe['data_source'] = 'openai/gsm8k'
    
    if 'reward_model' not in dataframe.columns:
        dataframe['reward_model'] = dataframe.apply(
            lambda row: {'ground_truth': str(row.get('answer', '')), 'style': 'rule'}, 
            axis=1
        )
    
    if tokenizer is not None:
        def check_length(doc):
            try:
                messages = generate_gsm8k_prompt(doc)
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )
                length = len(encoded)
                return length <= max_prompt_length
            except Exception as e:
                return False
                
        dataframe = dataframe[dataframe.apply(check_length, axis=1)]

    return dataframe

def generate_gsm8k_prompt(row):
    """
    Example messages:
    [
        {
            'role': 'system', 
            'content': "You are a helpful math assistant. Solve each problem step by step, explain your reasoning, and provide ONLY the final numeric answer after '####'. Do not include units or extra text after '####'."}, 
        {
            'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".', 
            'role': 'user'}, 
        {
            'role': 'assistant', 
            'content': "Let me solve this step by step and end with a single number after '####'."}
    ]
    """
    question = row["prompt"]
    # ground_truth = row.get('reward_model', {}).get('ground_truth', 'N/A')
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and accurate math tutor who solves grade school-level math word problems step by step. Always reason logically, break down the problem clearly, and compute answers precisely. Do not skip steps, and always explain the reasoning behind each step in simple terms.\nYou must output the final answer in the format: '#### <answer>'"
        },
        question[0], 
        {
            "role": "assistant",
            "content": "Let's solve this step by step.\nFirst, identify what the problem is asking for.\nThen, extract and write down all known values or facts.\nNext, perform the necessary arithmetic operations clearly, showing the logic behind each step.\nFinally, write the answer in the required format: '#### <answer>'"
        }
    ]
    
    # print("\n[Stage 2] Prompt Generation:")
    # print(f"Input question: {question}")
    # print(f"Ground truth: {ground_truth}")
    # print(f"Generated messages: {messages}")
    
    return messages


PROMPT_GENERATOR = {
    "swedev": swedev_prompt_generator,
    "default": default_prompt_generator,
    "gsm8k": generate_gsm8k_prompt
}

PREPROCESS_DATASET = {
    "swedev": swedev_preprocess_dataset,
    "default": default_preprocess_dataset,
    "gsm8k": preprocess_gsm8k_dataset
}

# SPECIFIC_TENSOR_LIST = {
#     "swedev": ["instance_id"],
#     "default": [],
#     "gsm8k": [] 
# }