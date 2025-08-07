import requests
import json

def custom_compute_score(data_source, prompt_str, solution_str, ground_truth, extra_info=None):
    """
    Custom function to compute the score based on the data source, solution string,
    and ground truth.
    """
    try:
        payload = {
            "instruction": prompt_str,
            "response": solution_str,
            "ground_truth": ground_truth,
            
            "agents_to_force_enable": None,
            "agents_to_force_disable": None
        }
        
        response = requests.post(
            "http://localhost:8000/invoke_agentic_reward_system",
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        
        result = response.json()

        return result["final_eval_score"]

    except Exception as e:
        print(f"Error calling reward API: {e}")
        return 0.0