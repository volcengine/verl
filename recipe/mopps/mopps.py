
import torch
import os
import numpy as np


class PosteriorSampler:
    def __init__(self, args, total_num_samples, prior_alpha=1.0, prior_beta=1.0, init=False, init_dir=None):
        self.args = args
        self.real_batch_size = self.args.data.train_batch_size
        self.num_samples = total_num_samples
        self.alpha = {}
        self.beta = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay_ratio = args.tasksampler.bandit_decay_ratio
        self.target_mean = args.tasksampler.bandit_target_mean
        self.lower_bound = args.tasksampler.bandit_lower_bound
        self.upper_bound = args.tasksampler.bandit_upper_bound
        self.sampling_strategy = args.tasksampler.bandit_sample_strategy
        self.no_update = args.tasksampler.bandit_no_update
        
        if init:
            self.initialize_from_json(init_dir)

    def sample_batch(self, batch_candidates_dict):
        candidate_indices = batch_candidates_dict['index']
        m = len(candidate_indices)
        assert self.real_batch_size <= m, "batch_size must be <= number of candidates"

        target_mu = self.target_mean

        local_alpha = []
        local_beta = []
        for index in candidate_indices:
            index = str(index)
            if index not in self.alpha.keys():
                self.alpha[index] = self.prior_alpha
            local_alpha.append(self.alpha[index])
            if index not in self.beta.keys():
                self.beta[index] = self.prior_beta
            local_beta.append(self.beta[index])
        local_alpha = np.array(local_alpha)
        local_beta = np.array(local_beta)
        sampled_r = np.random.beta(local_alpha, local_beta)

        if self.sampling_strategy == 'uniform':
            sampled_index = np.random.choice(m, size=self.real_batch_size, replace=False)
        elif self.sampling_strategy == 'topk':
            distances = (sampled_r - target_mu) ** 2
            sampled_index = np.argsort(distances)[:self.real_batch_size]
        elif self.sampling_strategy == 'threshold':
            in_range_mask = (sampled_r >= self.lower_bound) & (sampled_r <= self.upper_bound)
            in_range_indices = np.where(in_range_mask)[0]
            if len(in_range_indices) >= self.real_batch_size:
                np.random.shuffle(in_range_indices)
                sampled_index = in_range_indices[:self.real_batch_size]
            else:
                scores = np.zeros_like(sampled_r)
                too_low = sampled_r < self.lower_bound
                too_high = sampled_r > self.upper_bound
                scores[too_low] = self.lower_bound - sampled_r[too_low] 
                scores[too_high] = sampled_r[too_high] - self.upper_bound 
                
                sampled_index = np.argsort(scores)[:self.real_batch_size]
    
        batch_candidates_dict = {k: v[sampled_index] for k, v in batch_candidates_dict.items()}
            
        return batch_candidates_dict, torch.tensor(sampled_r[sampled_index])#.to('cuda')

    def train(self, batch_candidates_dict, y):
        if self.no_update:
            return None, None, None
        indices = batch_candidates_dict['index']
        for idx, s in zip(indices, y):
            idx = str(idx)
            self.alpha[idx] = self.alpha[idx]*self.decay_ratio + self.prior_alpha * (1-self.decay_ratio) + s * self.args.actor_rollout_ref.rollout.n if self.args.actor_rollout_ref.rollout.n > 1 else 8
            self.beta[idx] = self.beta[idx]*self.decay_ratio + self.prior_beta * (1-self.decay_ratio) + (1 - s)  * self.args.actor_rollout_ref.rollout.n if self.args.actor_rollout_ref.rollout.n > 1 else 8
        return None, None, None
    
    
    def initialize_from_json(self, json_path = None):
        import json
        if json_path is None:
            json_path = f"{os.path.dirname(os.path.abspath(__file__))}/scripts/math/index_score.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for key, results in data.items():
            idx = key
            successes = sum(results)
            failures = len(results) - successes
            self.alpha[idx] = successes * 3 + self.prior_alpha
            self.beta[idx] = failures * 3 + self.prior_beta

    def save(self, save_path):
        import json
        import os
        data = {}
        for index in self.alpha.keys():
            data[index] = [float(self.alpha[index]), float(self.beta[index])]
        with open(os.path.join(save_path, 'index_score.json'), 'w') as f:
            json.dump(data, f)

    def load(self, load_path):
        try:
            import json
            import os
            with open(os.path.join(load_path, 'index_score.json'), 'r') as f:
                data = json.load(f)
            
            for key, results in data.items():
                # idx = int(key)
                idx = key
                self.alpha[idx] = results[0]
                self.beta[idx] = results[1]
            print(f'Posterior Sampler load from {load_path}')
        except:
            pass

