from verl.utils.agent_tasks.swedev import *

def default_prompt_generator(row):
    return row["prompt"]

def default_preprocess_dataset(dataframe):
    return dataframe

PROMPT_GENERATOR = {
    "swedev": swedev_prompt_generator,
    "default": default_prompt_generator
}

PREPROCESS_DATASET = {
    "swedev": swedev_preprocess_dataset,
    "default": default_preprocess_dataset
}

SPECIFIC_TENSOR_LIST = {
    "swedev": ["instance_id"],
    "default": []
}