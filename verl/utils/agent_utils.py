from verl.utils.agent_tasks.swedev import *

def default_prompt_generator(row):
    return row["prompt"]

PROMPT_GENERATOR = {
    "swedev": swedev_prompt_generator,
    "default": default_prompt_generator
}