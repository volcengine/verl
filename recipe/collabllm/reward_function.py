import torch


def conversation_level_reward_func(data_source, messsages, ground_truth, extra_info) -> float:
    """
    Used in verl/workers/reward_manager/collabllm.py

    Apply conversation-level reward function to the future interactions between the user simulator 
    and policy model, which are generated from `verl/interactions/collabllm_interation.py`
    """

    # As a demonstration, apply token penalty here
    original_prompt = [item["prompt"] for item in extra_info]

    rewards = []

    for prompt, messsage in zip(original_prompt, messsages):

        # Calculate the token penalty based on the length of the prompt
        message_lst = messsage["messages"]
        future_conv = message_lst[len(original_prompt):]
        
        total_tokens = sum(len(m.content.split()) for m in future_conv)
        penalty = - min(0.001 * total_tokens, 1)
        rewards.append(penalty)

    # TODO: Add more metrics and apply weighted average

    return torch.tensor(rewards, dtype=torch.float32)
        

