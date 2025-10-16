import numpy as np
from collections import defaultdict, deque
import random
import logging
from verl import DataProto
import random
import torch
import torch.nn.functional as F
from copy import deepcopy
import uuid
from collections import defaultdict

logger = logging.getLogger(__file__)


def is_finished(item, config, tokenizer, max_window_rounds):
    max_response_length = config.data.get('window_response_length', None)
    max_prompt_length = config.data.max_prompt_length + max_window_rounds * max_response_length
    actual_prompt_length = max_prompt_length - item.batch['window_rounds'].item() * max_response_length
    prompt_ids = item.batch['input_ids'][:actual_prompt_length]
    response_ids = item.batch['input_ids'][actual_prompt_length:]
    response_length = response_ids.shape[-1]
    valid_response_length = item.batch['attention_mask'][actual_prompt_length:].sum().item()
    valid_response_ids = response_ids[:valid_response_length]
    is_trunc = (response_length == valid_response_length) and (valid_response_ids[valid_response_length-1].item() != tokenizer.eos_token_id)
    is_last_round = item.batch['window_rounds'].item() == config.data.max_response_length // config.data.window_response_length - 1
    is_finished = (not is_trunc) or is_last_round
    return is_finished


class SamplePool:
    name = "sample_pool"

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.num_bon = config.actor_rollout_ref.rollout.get("num_bon", 1)
        self.sample_pool = []
        self.prompt_list = []
        self.batch_keys = []
        self.non_tensor_batch_keys = []
        self.meta_info_keys = []
        self.pool_with_grad = defaultdict(list)
        self.pool_with_unfinished = []
        self.id2acc = defaultdict(list)

    def rearrange_sample_pool(self):
        new_sample_list = []
        for v in self.prompt_list:
            new_sample_list.append(deepcopy(v))
        self.prompt_list = []
        for idx, item in enumerate(self.sample_pool):
            new_sample_list.append(deepcopy(item))
        self.sample_pool = [i for i in new_sample_list]

    def fill_sample_pool(self, batch):
        batch_lst = batch.chunk(len(batch))
        max_prompt_length = self.config.data.max_prompt_length
        self.batch_keys = list(batch.batch.keys()) + ['left_pad_len', 'actual_prompt_len', 'window_rounds']
        if 'values' not in self.batch_keys: self.batch_keys += ['values']
        if 'answer_input_ids' in self.batch_keys: self.batch_keys.remove('answer_input_ids')
        if 'answer_attention_mask' in self.batch_keys: self.batch_keys.remove('answer_attention_mask')
        self.non_tensor_batch_keys = list(batch.non_tensor_batch.keys()) + ['rollout_id']
        self.meta_info_keys = batch.meta_info.keys()
        for item in batch_lst:
            item.batch['window_rounds'] = torch.tensor([0], device=item.batch['attention_mask'].device)
            item.batch['actual_prompt_len'] = item.batch['attention_mask'][:, :max_prompt_length].sum(-1)
            item.batch['left_pad_len'] = max_prompt_length - item.batch['actual_prompt_len']
            item.batch['values'] = torch.zeros_like(item.batch['attention_mask'], dtype=torch.float32)[:, :0]
            item.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4())], dtype=object)
            for _ in range(self.num_bon):
                self.sample_pool.append(deepcopy(item))
        print("[SamplePool] fill_batch:", len(batch_lst), "sample_pool size:", len(self.sample_pool))

    def get_gen_batch(self, return_batch_size):
        return_batch = []
        window_round = 0
        padded_size = 0
        while len(return_batch) < return_batch_size:
            item = self.sample_pool[0]
            self.sample_pool = self.sample_pool[1:]
            window_round = max(window_round, item.batch['window_rounds'].item())
            padded_size = max(padded_size, item.batch['input_ids'].size(1))
            new_item = item.select(batch_keys=self.batch_keys, non_tensor_batch_keys=self.non_tensor_batch_keys, meta_info_keys=self.meta_info_keys)
            return_batch.append(deepcopy(new_item))
        for idx, item in enumerate(return_batch):
            if item.batch['window_rounds'].item() == 0:
                pad_size = (window_round - item.batch['window_rounds'].item()) * self.config.data.window_response_length
                item.batch['left_pad_len'] += pad_size
                item.batch['input_ids'] = F.pad(item.batch['input_ids'], (pad_size, 0), value=self.tokenizer.pad_token_id)
                item.batch['attention_mask'] = F.pad(item.batch['attention_mask'], (pad_size, 0), value=0)
                item.batch['values'] = F.pad(item.batch['values'], (pad_size, 0), value=0)
                # item.batch['answer_input_ids'] = F.pad(item.batch['answer_input_ids'], (pad_size, 0), value=self.tokenizer.pad_token_id)
                # item.batch['answer_attention_mask'] = F.pad(item.batch['answer_attention_mask'], (pad_size, 0), value=0)
            elif item.batch['input_ids'].size(1) < padded_size:
                pad_size = padded_size - item.batch['input_ids'].size(1)
                item.batch['input_ids'] = F.pad(item.batch['input_ids'], (pad_size, 0), value=self.tokenizer.pad_token_id)
                item.batch['attention_mask'] = F.pad(item.batch['attention_mask'], (pad_size, 0), value=0)
                item.batch['values'] = F.pad(item.batch['values'], (pad_size, 0), value=0)
                item.batch['left_pad_len'] += pad_size
        return DataProto.concat(return_batch)

    def update_multi_round_pool(self, batch):
        batch_lst = batch.chunk(len(batch))
        for item in batch_lst:
            is_finished = item.batch['is_finished']
            # uid = item.non_tensor_batch['index'][0]
            # prompt = item.batch['prompts']
            # response = item.batch['responses']
            if (not is_finished) and item.batch['window_rounds'].item() < self.config.data.max_response_length // self.config.data.window_response_length - 1:
                start_idx = torch.nonzero(item.batch['attention_mask'].flatten())[0].item()
                real_len = item.batch['attention_mask'].sum(-1).item()
                max_prompt_length = self.config.data.max_prompt_length + (item.batch['window_rounds'].item() + 1) * self.config.data.window_response_length
                prompt_ids = F.pad(item.batch['input_ids'][:, start_idx:start_idx + real_len], (max_prompt_length - real_len, 0), value=self.tokenizer.pad_token_id)
                prompt_attention_mask = F.pad(item.batch['attention_mask'][:, start_idx:start_idx + real_len], (max_prompt_length - real_len, 0), value=0)
                new_item = deepcopy(item.select(batch_keys=self.batch_keys, non_tensor_batch_keys=self.non_tensor_batch_keys, meta_info_keys=self.meta_info_keys))
                new_item.batch['left_pad_len'] = torch.tensor([max_prompt_length - real_len], device=prompt_ids.device) 
                new_item.batch['input_ids'] = prompt_ids
                new_item.batch['attention_mask'] = prompt_attention_mask
                new_item.batch['values'] = new_item.batch['values'][:, :(item.batch['window_rounds'].item() + 1) * self.config.data.window_response_length]
                # new_item.batch['answer_input_ids'] = prompt_ids
                # new_item.batch['answer_attention_mask'] = prompt_attention_mask
                new_item.batch['window_rounds'] += 1
                self.prompt_list.append(deepcopy(new_item))

    def fill_rollout_pool_grad(self, batch):
        batch_lst = batch.chunk(len(batch))
        self.id2data = defaultdict(list)
        self.id2finish = defaultdict(list)
        self.id2unfinish = defaultdict(list)
        # get acc
        for item in batch_lst:
            score = item.batch['token_level_scores'].sum(-1).item()
            is_finished = item.batch['is_finished']
            rollout_id = item.non_tensor_batch['rollout_id'][0]
            if is_finished:
                self.id2acc[rollout_id].append(score)
                self.id2finish[rollout_id].append(item)
            else:
                self.id2unfinish[rollout_id].append(item)
            self.id2data[rollout_id].append(item)
        for k, v in self.id2acc.items():
            if np.mean(v) != self.config.algorithm.rollout_pool.min_score and np.mean(v) != self.config.algorithm.rollout_pool.max_score:
                if self.config.algorithm.rollout_pool.strategy == 'v2':
                    for item in self.id2finish[k]:
                        self.pool_with_grad[k].append(item)
                else:
                    for item in self.id2data[k]:
                        self.pool_with_grad[k].append(item)
        print("[SamplePool/fill_rollout_pool_grad] fill_batch:", len(batch_lst), "pool_size after fill:", len(self.pool_with_grad))


    def get_train_batch_grad(self, return_batch_size):
        return_batch = []
        k_lst = list(self.pool_with_grad.keys())
        random.shuffle(k_lst)
        while len(return_batch) < return_batch_size:
            if len(k_lst) == 0: break
            k = k_lst[0]
            k_lst = k_lst[1:]
            for item in self.pool_with_grad[k]:
                return_batch.append(item)
        k_lst = sorted(self.id2unfinish.keys(), key = lambda k: len(self.id2unfinish[k]), reverse=True)
        if self.config.algorithm.rollout_pool.strategy == 'v2': 
            k_lst = list(filter(lambda k: len(self.id2unfinish[k]) >= self.config.actor_rollout_ref.rollout.num_bon // 8, k_lst))
        while len(return_batch) < return_batch_size:
            if len(k_lst) == 0: break
            k = k_lst[0]
            k_lst = k_lst[1:]
            if self.config.algorithm.rollout_pool.strategy == 'v2':
                for item in self.id2unfinish[k]:
                    return_batch.append(item)
            elif k not in self.pool_with_grad.keys():
                for item in self.id2data[k]:
                    return_batch.append(item)
        k_lst = [k for k in self.id2data.keys() if ((k not in self.pool_with_grad.keys()) and (k not in self.id2unfinish.keys()))]
        while len(return_batch) < return_batch_size:
            if len(k_lst) == 0: break
            k = k_lst[0]
            k_lst = k_lst[1:]
            for item in self.id2data[k]:
                return_batch.append(item)
        if len(return_batch) < return_batch_size:
            return_batch.extend([random.choice(return_batch) for _ in range(return_batch_size - len(return_batch))])
        return_batch = return_batch[:return_batch_size]
        self.pool_with_grad = defaultdict(list) 
        return DataProto.concat(return_batch)
