set -x

python -m recipe.r1.main_eval \
    data.path=$SAVE_PATH \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=verl/utils/reward_score/reward_score.py \
    custom_reward_function.name=reward_func
