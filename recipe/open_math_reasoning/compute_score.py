

def compute_score_data_source(data_source, response, ground_truth):
    from verl.utils.reward_score.math_reward import compute_score
    if data_source in ['aime24', 'aime25']:
        return compute_score(response, ground_truth)
    else:
        raise ValueError(f"Unknown data source: {data_source}")