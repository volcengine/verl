"""
Task description:
Given a random word and a random char, count the number of occurrence of char in the word.

Create CoT dataset that split the word into separate char. Then list the char and count the occurrence.

The word set comes from shakespeare
"""
import os.path
import random

with open('/home/chi/experiments/char_count/input.txt', 'r') as f:
    words = f.read()

# 25670 unique word
word_set = list(set(words.split()))

prompt_template = 'How many {} are there in word {}?'


def create_prompt_response(word):
    char_set = set(word)
    char_lst = list(word)
    # TODO: add chars not in char_set

    output = []

    for target_char in char_set:
        prompt = prompt_template.format(target_char, word)
        final_answer = []
        answer = f'Let count the number of {target_char} in {word} step by step.'

        final_answer.append(answer)
        # cot
        number = 0
        for i, char in enumerate(char_lst):
            cot = f'The #{i + 1} char in {word} is {char}, which is '
            if char != target_char:
                cot += 'not '
            else:
                number += 1
            cot += f'equal to {target_char}.'

            final_answer.append(cot)

        conclusion = f'Thus, in total, there are \\boxed{{{number}}} {target_char} in {word}.'

        final_answer.append(conclusion)

        final_answer = '\n'.join(final_answer)

        output.append((prompt, final_answer))

    return output


full_output = []
for word in word_set:
    # normalize word
    word = ''.join([i for i in word if i.isalpha()])
    if len(word) > 15:
        print(f'Ignore word {word}')
        continue
    output = create_prompt_response(word)
    full_output.extend(output)

# random reorder
random.shuffle(full_output)

# split for train and test
train_split_len = int(0.99 * len(full_output))
train_outputs = full_output[:train_split_len]
test_output = full_output[train_split_len:]

sft_train_dataset = {
    'prompt': [],
    'response': []
}

for o in train_outputs:
    sft_train_dataset['prompt'].append(o[0])
    sft_train_dataset['response'].append(o[1])

sft_test_dataset = {
    'prompt': [],
    'response': []
}

for o in test_output:
    sft_test_dataset['prompt'].append(o[0])
    sft_test_dataset['response'].append(o[1])

import pandas as pd

sft_train_dataset = pd.DataFrame(data=sft_train_dataset)
sft_test_dataset = pd.DataFrame(data=sft_test_dataset)

folder = os.path.expanduser('~/data/char_count/sft')

os.makedirs(folder, exist_ok=True)

sft_train_dataset.to_parquet(os.path.join(folder, 'train.parquet'))
sft_test_dataset.to_parquet(os.path.join(folder, 'test.parquet'))

# build RL dataset
rl_train_dataset = {
    'prompt': [],
    'data_source': [],
    'ability': [],
    'reward_model': [],
    'extra_info': []
}

rl_test_dataset = {
    'prompt': [],
    'data_source': [],
    'ability': [],
    'reward_model': [],
    'extra_info': []
}

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

for o in train_outputs:
    prompt = o[0]
    response = o[1]
    prompt_with_template = [{
        "role": "user",
        "content": prompt,
    }]

    rl_train_dataset['prompt'].append(prompt_with_template)
    rl_train_dataset['data_source'].append('char_count')
    rl_train_dataset['ability'].append('other')
    rl_train_dataset['reward_model'].append({
        'style': 'rule',
        'ground_truth': remove_boxed(last_boxed_only_string(response))
    })
    rl_train_dataset['extra_info'].append({
        'response': response
    })

for o in test_output:
    prompt = o[0]
    response = o[1]
    prompt_with_template = [{
        "role": "user",
        "content": prompt,
    }]

    rl_test_dataset['prompt'].append(prompt_with_template)
    rl_test_dataset['data_source'].append('char_count')
    rl_test_dataset['ability'].append('other')
    rl_test_dataset['reward_model'].append({
        'style': 'rule',
        'ground_truth': remove_boxed(last_boxed_only_string(response))
    })
    rl_test_dataset['extra_info'].append({
        'response': response
    })

rl_train_dataset = pd.DataFrame(data=rl_train_dataset)
rl_test_dataset = pd.DataFrame(data=rl_test_dataset)

folder = os.path.expanduser('~/data/char_count/rl')

os.makedirs(folder, exist_ok=True)

rl_train_dataset.to_parquet(os.path.join(folder, 'train.parquet'))
rl_test_dataset.to_parquet(os.path.join(folder, 'test.parquet'))