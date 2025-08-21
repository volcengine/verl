def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    prompt = extra_info["prompt"]
    
    # Calculate the token penalty based on the length of the prompt
    future_conv = messages[len(prompt):]
    
    total_tokens = sum(len(m.content.split()) for m in future_conv)

    return total_tokens
