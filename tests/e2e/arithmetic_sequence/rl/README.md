# Digit completion

This is an example of solving a digit completion problem. The problem is defined as below:

The prompt is a sequence of numbers with fixed difference. The agent's goal is to complete the next N numbers.
If the max number is reached, the next number should be modulo with max number.

For example,
- prompt = [1, 2, 3]
- N = 5
- max_number = 6

The response should be [4, 5, 6, 7%6, 8%6] = [4, 5, 6, 0, 1].

# Environment definition

The core definition of the task is defined in verl/envs/digit_completion/task.py

It is highly recommended to take a look at it for better understanding.



# Run experiments

The users are required to specify the config path and config name (and the relative model config path to the current working directory)

```bash
# cd examples/arithmetic_sequence/rl

# Specify the config path and config name (current working dir)
python3 -m verl.trainer.ppo.ray_megatron_train_synchronous --config-path=$(pwd)/config --config-name='ray_megatron'

# The default relative path of model config is 'config/model_config', if you want to change it, you can rewrite it in ray_megatron.yaml or using:
python3 -m verl.trainer.ppo.ray_megatron_train_synchronous --config-path=$(pwd)/config --config-name='ray_megatron' ++model.base_path=config/model_config

```

