from verl.utils.agent_tasks.swedev import *


def default_prompt_generator(row):
    return row["prompt"]


def default_preprocess_dataset(dataframe):
    return dataframe


def preprocess_gsm8k_dataset(dataframe, tokenizer=None, max_prompt_length=1024):
    # print("\n[Stage 1] Dataset Preprocessing:")
    # sample = dataframe.iloc[0]
    # print(f"Raw data sample:\n", sample)

    if "data_source" not in dataframe.columns:
        dataframe["data_source"] = "openai/gsm8k"

    if "reward_model" not in dataframe.columns:
        dataframe["reward_model"] = dataframe.apply(
            lambda row: {"ground_truth": str(row.get("answer", "")), "style": "rule"},
            axis=1,
        )
    if "index" not in dataframe.columns:
        dataframe = dataframe.reset_index(drop=True)
        dataframe["index"] = dataframe.index

    if tokenizer is not None:

        def check_length(doc):
            try:
                messages = generate_gsm8k_prompt(doc)
                encoded = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                length = len(encoded)
                return length <= max_prompt_length
            except Exception as e:
                return False

        dataframe = dataframe[dataframe.apply(check_length, axis=1)]
    print(f"Dataframe:\n", dataframe.iloc[0])

    return dataframe


def generate_gsm8k_prompt(row):
    """
    Example question:
        {
            'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer (a number) after "####".',
            'role': 'user'
        }
    """
    question = row["prompt"]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and accurate math tutor who solves grade school-level math word problems step by step. Provide clear reasoning, and only use '####' in the final answer, in the format '#### <answer>'.",
        },
        {
            "role": "user",
            "content": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        },
        {
            "role": "assistant",
            "content": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
        },
        question[0],
    ]
    
    # ground_truth = row.get('reward_model', {}).get('ground_truth', 'N/A')
    # print("\n[Stage 2] Prompt Generation:")
    # print(f"Input question: {question}")
    # print(f"Ground truth: {ground_truth}")
    # print(f"Generated messages: {messages}")

    return messages


PROMPT_GENERATOR = {
    "swedev": swedev_prompt_generator,
    "default": default_prompt_generator,
    "gsm8k": generate_gsm8k_prompt,
}

PREPROCESS_DATASET = {
    "swedev": swedev_preprocess_dataset,
    "default": default_preprocess_dataset,
    "gsm8k": preprocess_gsm8k_dataset,
}

# SPECIFIC_TENSOR_LIST = {
#     "swedev": ["instance_id"],
#     "default": [],
#     "gsm8k": []
# }
