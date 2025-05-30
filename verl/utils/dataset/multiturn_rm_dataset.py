# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn Reward Model dataset that returns token-level rewards.
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.trainer.ppo.core_algos import compute_td_returns

class MultiTurnRMDataset(MultiTurnSFTDataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(
            self,
            parquet_files: Union[str, List[str]],
            tokenizer,
            config
        ):
        multiturn_config = config.get('multiturn', {})
        assert multiturn_config.get('enable', False), "Multi-turn dataset is not enabled. This is required for RM dataset."
        super().__init__(parquet_files, tokenizer=tokenizer, config=config)

        self.td_returns_gamma = config.get('td_returns_gamma', 0.99)
        self.last_action_only = config.get('last_action_only', False)
        self.special_token_id = config.get('special_token_id', None)
        print(f'RM Dataset: {self.last_action_only=}; {self.td_returns_gamma=}; {self.special_token_id=} (actual token: {self.tokenizer.decode([self.special_token_id]) if self.special_token_id is not None else "None"})')
        assert (self.last_action_only and self.td_returns_gamma == 1.0) or not self.last_action_only, "TD returns gamma must 1.0 if last_action_only is True"
        assert self.truncation == 'error', "Truncation must be error for RM dataset"

    def _read_files_and_process(self):
        super()._read_files_and_process()

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls
        self.task_rewards = self.dataframe['task_reward'].apply(series_to_item).tolist()

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        task_reward: int = self.task_rewards[item]

        # First, get the full conversation tokens
        full_tokens = tokenizer.apply_chat_template(messages,
                                                    tokenize=True,
                                                    return_tensors='pt',
                                                    add_generation_prompt=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)

        # We are trying to train a value function that predicts Q(s, a)
        # This mask the last token of each action (i.e., "assistant" role response)
        # We will use this mask to calculate final loss
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        returns = torch.zeros_like(input_ids, dtype=torch.float32)

        # Process each message to find assistant responses
        current_length = 0
        assistant_indices = []  # Track all assistant message indices
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            prefix_messages = messages[:i + 1]
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages,
                                                          tokenize=True,
                                                          return_tensors='pt',
                                                          add_generation_prompt=False)

            # Get tokens for messages up to previous point
            prev_tokens = tokenizer.apply_chat_template(
                messages[:i], tokenize=True, return_tensors='pt', add_generation_prompt=False) if i > 0 else None

            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]

            # If this is an assistant message, track it
            if msg['role'] == 'assistant':
                if self.special_token_id is None:
                    assistant_indices.append((i, end_pos - 1))  # Store message index and end position
                else:
                    # Find the index of special token within span of start_pos and end_pos
                    special_token_index = (input_ids[start_pos:end_pos] == self.special_token_id).nonzero()
                    assert len(special_token_index) > 0, f'Expected special token {self.special_token_id} (actual token: {self.tokenizer.decode([self.special_token_id])}) in the span of {start_pos=} and {end_pos=}'
                    assistant_indices.append((i, special_token_index[0].item() + start_pos))

        assert len(assistant_indices) > 0, "No assistant actions found. This should not happen for Critic training."

        # Apply mask based on last_action_only setting
        # NOTE: in this case, we only calculate the loss for the last action
        if self.last_action_only:
            # Only mark the last assistant message
            _, last_pos = assistant_indices[-1]
            loss_mask[last_pos] = 1

            # Directly use the task reward as the return
            returns[last_pos] = task_reward
        else:
            # Mark all assistant messages
            for _, end_pos in assistant_indices:
                loss_mask[end_pos] = 1

            # print(f'{loss_mask=}')
            
            # Calculate TD returns
            # Get only rewards for the action steps (where loss_mask == 1)
            action_positions = loss_mask.nonzero().squeeze(-1)
            num_actions = len(action_positions)
            # print(f'{num_actions=}, {action_positions=}')
            # tokens for those action positions
            action_tokens = input_ids[action_positions]
            action_tokens_decoded = tokenizer.batch_decode(action_tokens)
            # print(f'{action_tokens=}, {action_tokens_decoded=}')
            assert num_actions > 0, "No assistant actions found. This should not happen for Critic training."

            # Create rewards only for action steps
            step_level_rewards = torch.zeros(num_actions, dtype=torch.float32)
            # Only the last action gets the task reward
            step_level_rewards[-1] = task_reward
            # print(f'{step_level_rewards=}')
            # Compute TD returns for these actions
            step_level_returns = compute_td_returns(step_level_rewards, gamma=self.td_returns_gamma)
            # print(f'{step_level_returns=}')
            returns[action_positions] = step_level_returns

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)
            padded_returns = torch.zeros(size=(self.max_length - sequence_length,), dtype=returns.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
            returns = torch.cat((returns, padded_returns))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
                returns = returns[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                returns = returns[:self.max_length]
            elif self.truncation == 'error':
                raise ValueError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise ValueError(f'Unknown truncation method {self.truncation}')

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'returns': returns
        }

