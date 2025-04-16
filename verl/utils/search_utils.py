import os
import re
import torch
import requests
from verl import DataProto
from tensordict import TensorDict
from typing import List, Dict, Any, Tuple
from verl.utils.torch_functional import pad_sequence_to_length

supported_tools = {}

def register_tools(func):
    name = func.__name__
    assert name not in supported_tools
    supported_tools[name] = func
    return func


@register_tools
def search(response_str):
    return response_str

def batch_search( queries):
    """
    Batchified search for queries.
    Args:
        queries: queries to call the search engine
    Returns:
        search results which is concatenated into a string
    """
    return queries    
    search_url = "http://10.171.10.166:8080/search"
    payload = {
        "queries": queries,
        "topk": 1,
        "return_scores": True
    }
    return requests.post(search_url, json=payload).json()

    
def passages2string( retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


class SearchUtils:
    def __init__(self, tokenizer, tools_list):
        self.tokenizer = tokenizer  
        self.tools_list = tools_list
        self.loop_cnt = 0
        self.max_prompt_length = 512 # TODO 
        self.pad_token_id = tokenizer.pad_token_id

    def execute_predictions(self, predictions: List[str], active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, action in enumerate(cur_actions):
            if action == 'answer':
                next_obs.append('')
                dones.append(True)
                valid_action.append(1)
                is_search.append(0)
            elif action == 'search':
                next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
            else:
                next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                dones.append(False)
                valid_action.append(0)
                is_search.append(0)
        
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents
    
    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids
        
    def postprocess_output(self, output:DataProto):
        '''output: cpu'''
        # print(f'batch {output}')

        prompts_str = self.tokenizer.batch_decode(
                        output.batch.get('prompts'),
                        skip_special_tokens=True,
                    )
        responses_str = self.tokenizer.batch_decode(
            output.batch.get('responses'),
            skip_special_tokens=True,
            )

        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_str = [[] for _ in range(self.batch_size)]
            self.loop_mask_str = [[] for _ in range(self.batch_size)] 
            self.init_prompt_token = output.batch.get('prompts')
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]  
            # print(f'self.init_prompt_token {self.init_prompt_token.shape} {self.init_attention_mask.shape}')
            
            sample_index = list(range(self.batch_size))
            for idx in range(self.batch_size):
                self.loop_responses_str[idx].append(responses_str[idx])
        else:
            sample_index = output.meta_info['index']
            for idx, batch_idx in enumerate(sample_index):
                self.loop_responses_str[batch_idx].append(responses_str[idx])
        
        # rank = os.environ.get('RANK', '0')
        # def print_rank0(msg):
        #     if rank == '0':
        #         print(msg)
        # print_rank0(f'rank {rank} vllm generate_sequence {output.batch.batch_size}')
        # print_rank0(f'rank output.batch {rank} {output.batch}')
        # print_rank0(f'rank {rank},{len(responses_str)} response')

        infos_str, dones, valid_action, is_search = self.execute_predictions(responses_str)
        
        next_prompt = []
        next_idx = []

        for idx, done in enumerate(dones):
            if not done:
                prompt= prompts_str[idx]
                response = responses_str[idx]
                info = infos_str[idx] + '\n'
                next_prompt.append(prompt + response + info)
                batch_idx = sample_index[idx]
                next_idx.append(batch_idx)
                self.loop_responses_str[batch_idx].append(info)
        
        if len(next_prompt) == 0:
            return 
        
        output.meta_info['index'] = next_idx
        # print_rank0(f'next_prompt {next_prompt}')

        next_batch = self.tokenizer(next_prompt, return_tensors='pt', padding=True, padding_side='left', )
        next_input_ids = next_batch.input_ids
        next_attention_mask = next_batch.attention_mask
  
  
        position_ids = (torch.cumsum(next_attention_mask, dim=1) - 1) * next_attention_mask
        
        max_len = self.max_prompt_length
        next_batch = TensorDict(
            {
                'input_ds': next_input_ids[:, -max_len:],
                'position_ids': position_ids[:, -max_len:],
                'attention_mask': next_attention_mask[:, -max_len:]
            },
            batch_size=next_input_ids.shape[0]
        )
        next_batch = DataProto(batch=next_batch)
        next_batch.meta_info.update(output.meta_info)
        next_batch.meta_info.update(do_sample=False) # only step > 0 not do sample
        self.loop_cnt += 1

        return next_batch

    def compose_final_output(self, meta_info, response_length) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        
        for idx, responses in enumerate(self.loop_responses_str):
            prompts_list = []
            loss_masks_list = []
            
            responses_token = self.tokenizer(responses, return_length=True)
            for turn_idx in range(len(responses)):
                length = responses_token['length'][turn_idx]
                prompts_list.extend(responses_token['input_ids'][turn_idx])
                loss_masks_list.extend([turn_idx % 2] * length)
                print(f"search util response {responses_token['input_ids'][turn_idx]}")

            # print(f'prompts_list {prompts_list}')
            input_ids = torch.tensor(prompts_list, dtype=torch.int64)
            loss_masks = torch.tensor(loss_masks_list)
            input_ids_list.append(input_ids)
            loss_mask_list.append(loss_masks)
            length_list.append(len(prompts_list))
        
        length_list.append(response_length)
        length_max = max(length_list)
        device = self.init_prompt_token.device
        response_token = torch.ones((self.batch_size, length_max), device=device, dtype=torch.long) * self.pad_token_id
        response_attention_mask = torch.zeros((self.batch_size, length_max), device=device, dtype=torch.long)
        response_loss_mask = torch.zeros((self.batch_size, length_max), device=device, dtype=torch.long)
        for idx in range(self.batch_size):
            sample_len = length_list[idx]
            response_token[idx][:sample_len] = input_ids_list[idx]
            response_loss_mask[idx][:sample_len] = loss_masks_list[idx]
            response_attention_mask[idx][:sample_len] = torch.ones(sample_len)
        
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.cumsum(attention_mask, dim=1, dtype=torch.long) - 1
        loss_mask = torch.cat([torch.zeros_like(self.init_attention_mask), response_loss_mask], dim=-1)
        final_batch = TensorDict(
            {
                'prompts': self.init_prompt_token,
                'responses': response_token,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'loss_mask': loss_mask
            },
            batch_size=self.batch_size,
        )  

        final_output = DataProto(batch=final_batch)
        final_output.meta_info.update(meta_info)
        # rank = os.environ.get('RANK', '0')
        # def print_rank0(msg):
        #     if rank == '0':
        #         print(msg)
        # print_rank0(f'final_output {final_output} init_attention_mask {self.init_attention_mask.shape} position_ids{position_ids.shape}')
        
        return final_output
    

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    model_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimai-rank/lixiaoguang12/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer(['你好 蓝色养乐多', '你好'], 
                       return_tensors='pt', padding=True, padding_side='left', return_token_type_ids=True)
    response = tokenizer(['verl测试', 'search utils run'], return_tensors='pt', padding=True, padding_side='right')
    samples = ['你好', '蓝色养乐多', '你好' ]
    sampled_id = tokenizer(samples, padding=True, padding_side='left', add_special_tokens=True)
    print(f', eos_token_id {tokenizer.eos_token_id}')
    print(f'sampled_id {sampled_id}')

    batch = TensorDict(
        {
            'prompts': prompt['input_ids'],
            'responses': response['input_ids'],
            'attention_mask': prompt['attention_mask'],
        },
        batch_size=2,
    )  
    batch = DataProto(batch=batch)
    print(f'meta_info {batch.meta_info}')

    su = SearchUtils(tokenizer, [])
    next_batch = su.postprocess_output(batch)
    print(f'next_batch {next_batch}')
    final_batch = su.compose_final_output(batch.meta_info, 1024)
    print(f'final_batch {final_batch}')

