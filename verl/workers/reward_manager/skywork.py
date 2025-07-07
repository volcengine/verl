from concurrent.futures import ProcessPoolExecutor

import torch
import traceback

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
def parallel_compute_score(evaluation_func, response_str, ground_truth, data_sources, timeout=6, max_workers=64):

    with tqdm(total=len(response_str)) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluation_func, data_sources[index], response_str[index], ground_truth[index]): index
                for index in range(len(response_str))
            }
            results = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    return [results[i] for i in range(len(response_str))]


class SkyworkRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        
    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        ground_truth = [x.tolist() if isinstance(x, np.ndarray) else x for x in ground_truth]
        data_sources = data.non_tensor_batch['data_source']

        assert len(response_str) == len(ground_truth) == len(data_sources)


        scores = []
        try:
            for i in range(0, len(response_str), 1024):
                cur_response_str = response_str[i:i+1024]
                cur_ground_truth = ground_truth[i:i+1024]
                cur_data_sources = data_sources[i:i+1024]

                cur_scores = parallel_compute_score(
                        self.compute_score,
                        cur_response_str,
                        cur_ground_truth,
                        cur_data_sources,
                    )

                scores += cur_scores
            assert len(scores) == len(response_str)

        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            traceback.print_exc()
            scores = [0. for _ in range(len(response_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        #     if data_source not in already_print_data_sources:
        #         already_print_data_sources[data_source] = 0

        #     if already_print_data_sources[data_source] < self.num_examine:
        #         already_print_data_sources[data_source] += 1
        #         print("[response]", response_str[i])

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor