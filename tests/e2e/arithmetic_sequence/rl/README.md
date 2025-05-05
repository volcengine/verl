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

The core definition of the task is defined in tests/e2e/envs/digit_completion/task.py

It is highly recommended to take a look at it for better understanding.



# Run experiments

An example of running the task is provided in `tests/e2e/run_ray_trainer.sh`.

```bash
bash tests/e2e/run_ray_trainer.sh
```

