import re
import torch
import itertools
import numpy as np
from verl import DataProto
from tensordict import TensorDict
from typing import List, Dict, Any, Tuple
from .tools import supported_tools


class RolloutProcesor:
    """
    Process the rollout data for multi-turn conversation. compose the multi-turn conversation into a single batch.
    """
    def __init__(self, tokenizer, meta_info, config):
        self.tokenizer = tokenizer  
        self.final_str = config.stop[-1] if config.stop else 'answer' # stop list
        self.config_prompt_length = config.prompt_length
        self.config_response_length = config.response_length
        self.pad_token_id = meta_info.get('pad_token_id')
        self.eos_token_id = meta_info.get('eos_token_id')[0]
        self.meta_info = meta_info
        self.loop_cnt = 0

    def execute_predictions(self, predictions: List[str]) -> List[str]:
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
        next_obs, dones, valid_action = [], [], []
        
        for i, action in enumerate(cur_actions):
            if action == self.final_str:
                next_obs.append('')
                dones.append(True)
                valid_action.append(1)
            elif action in supported_tools:
                tool_func = supported_tools[action]
                info_str, done = tool_func(contents[i])
                next_obs.append(info_str)
                dones.append(done)
                valid_action.append(1)
            else:
                next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                dones.append(False)
                valid_action.append(0)
        
            
        return next_obs, dones, valid_action

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
    

    def postprocess_output(self, output:DataProto, active_idxs: List[int]):
        '''output: cpu'''
        if self.loop_cnt == 0:
            # extract init prompt token and attention mask
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]  
            active_idxs = list(range(self.batch_size))
            for idx in active_idxs:
                prompt_token = self.init_prompt_token[idx]
                prompt_token_list = prompt_token[prompt_token != self.pad_token_id].tolist()
                self.loop_responses_token[idx].append(prompt_token_list)

        # extract rollout response token
        responses = output.batch.get('responses')
        for idx, batch_idx in enumerate(active_idxs):
            response_token = responses[idx] 
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            self.loop_responses_token[batch_idx].append(response_token_list)

        # decode rollout response token and execute predictions
        responses_str = self.tokenizer.batch_decode(
            output.batch.get('responses'),
            skip_special_tokens=True,
            )        
        infos_str, dones, valid_action = self.execute_predictions(responses_str)
        infos_str = [info_str + '\n' for info_str in infos_str]
        info_tokens = self.tokenizer(infos_str).input_ids
        
        next_prompt_token = []
        next_prompt_length = []
        next_active_idxs = []
        for idx, batch_idx in enumerate(active_idxs):
            if not dones[idx]:
                info_token_list = info_tokens[idx]
                self.loop_responses_token[batch_idx].append(info_token_list)
                next_active_idxs.append(batch_idx)
                promt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
                next_prompt_token.append(promt_token)
                next_prompt_length.append(len(promt_token))
        
        if len(next_prompt_token) == 0:
            return None, None
        
        # left pad, compose next turn prompts batch
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        position_ids = (torch.cumsum(next_attention_mask, dim=1) - 1) * next_attention_mask
        
        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:],
                'position_ids': position_ids[:, -max_len:],
                'attention_mask': next_attention_mask[:, -max_len:]
            },
            batch_size=next_input_ids.shape[0]
        )
        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        raw_prompt_ids[:] = [np.array(x[-max_len:]) for x in next_prompt_token]

        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids})
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['do_sample'] = False # step > 0 does not do sample
        self.loop_cnt += 1

        return next_data, next_active_idxs

    def compose_final_output(self) -> DataProto:
        """Compose final generation output.

        final_batch: (
            prompts: init_prompt
            responses: [reponse_token_1, info_token_1, response_token_2, info_token_2, ....]
            input_ids: [init_prompt, response_token_1, info_token_1, response_token_2, info_token_2....]
            attention_mask: [init_attention_mask, response_attention_mask_1, info_attention_mask_1, response_attention_mask_2,...]
            position_ids: [init_position_ids, response_position_ids..]
        )
        """
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        
        for idx, responses in enumerate(self.loop_responses_token):
            loss_mask = [[0] * len(responses[0])] # init_prompt loss
            prompts_list = list(itertools.chain.from_iterable(responses[1:]))
            # responses_token: [prompt_token, reponse_token_1, info_token_1, response_token_2....]
            for turn_idx in range(len(responses[1:])): 
                length = len(responses[turn_idx])
                loss_mask.extend([turn_idx % 2] * length)
            input_ids_list.append(prompts_list)
            loss_mask_list.append(loss_mask)
            length_list.append(len(prompts_list))
        
        max_len = max(max(length_list), self.config_response_length)
        
        # right pad
        input_ids = []
        loss_mask = []
        for idx, input_ids in enumerate(input_ids_list):
            input_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            loss_mask = loss_mask + [0] * (max_len - len(loss_mask))
            input_ids_list[idx] = input_ids
            loss_mask_list[idx] = loss_mask
   
        max_len = self.config_response_length
        response_token = torch.tensor(input_ids_list, dtype=torch.int64)[:,:max_len]
        response_loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)[:,:max_len]

        response_attention_mask = (response_token != self.pad_token_id).long()
        
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.cumsum(attention_mask, dim=1, dtype=torch.long) - 1
        loss_mask = torch.cat([torch.zeros_like(self.init_attention_mask, dtype=torch.float32), response_loss_mask], dim=-1)
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
        return final_output
    

if __name__ == '__main__':
    from transformers import AutoTokenizer
    class Config():
        stop = ['answer']
        prompt_length = 512
        response_length = 1024
    
    model_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimai-rank/lixiaoguang12/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    meta_info = {
        'pad_token_id': tokenizer.pad_token_id, 
        'eos_token_id': [tokenizer.eos_token_id, tokenizer.eos_token_id]
        }

    config = Config()
    su = RolloutProcesor(tokenizer, meta_info, config)

    # step 1 data
    prompt = tokenizer(['蓝色养乐多', '1+1'], return_tensors='pt', padding=True, padding_side='left', return_token_type_ids=True)
    response = tokenizer(['<search>蓝色养乐多产品信息</search>', '<answer>1+1=2</answer>'], return_tensors='pt', padding=True, padding_side='right')

    batch = TensorDict(
        {
            'prompts': prompt.input_ids,
            'responses': response.input_ids,
            'attention_mask': torch.cat([prompt.attention_mask, response.attention_mask],dim=-1),
        },
        batch_size=2,
    )  
    batch = DataProto(batch=batch)
    active_batch_idxs = list(range(batch.batch.batch_size[0]))

    next_batch, next_active_batch = su.postprocess_output(batch, active_batch_idxs)
    prompts_str = tokenizer.batch_decode(batch.batch.get('prompts'), skip_special_tokens=True)
    print(f'step 1 batch: {prompts_str}')

    # step 2 data
    prompts_str = tokenizer.batch_decode(next_batch.batch.get('input_ids'), skip_special_tokens=True)
    print(f'step 1 prompts_str: {prompts_str}\n')
    prompt = tokenizer(['蓝色养乐多<search>蓝色养乐多产品信息</search>',], return_tensors='pt', padding=True, padding_side='left', return_token_type_ids=True)
    response = tokenizer(['<answer>蓝色养乐多通常指的是一种低糖版本的养乐多饮料。</answer>',], return_tensors='pt', padding=True, padding_side='right')
    batch = TensorDict(
        {
            'prompts': prompt.input_ids,
            'responses': response.input_ids,
            'attention_mask': torch.cat([prompt.attention_mask, response.attention_mask],dim=-1),
        },
        batch_size=1,
    )  
    batch = DataProto(batch=batch)
    next_batch, next_active_batch = su.postprocess_output(batch, next_active_batch)
    if next_batch is not None:
        prompts_str = tokenizer.batch_decode(next_batch.batch.get('input_ids'), skip_special_tokens=True)
        print(f'step 2 prompts_str: {prompts_str}\n')
    
    # final output
    next_batch = su.compose_final_output()
    prompts_str = tokenizer.batch_decode(next_batch.batch.get('input_ids'), skip_special_tokens=True)
    print(f'final multi-turn result: {prompts_str}')
