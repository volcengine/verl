"""
Task description:
Given a random word and a random char, count the number of occurrence of char in the word.

Create CoT dataset that split the word into separate char. Then list the char and count the occurrence.

The word set comes from shakespeare
"""

with open('input.txt', 'r') as f:
    words = f.read()

# 25670 unique word
word_set = list(set(words.split()))

prompt_template = 'How many {} are there in word {}?'

def create_prompt_response(word):
    word = ''.join([i for i in word if i.isalpha()])

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

# for word in word_set:


output = create_prompt_response(word_set[0])


