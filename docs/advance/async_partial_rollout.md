# Recipe: Async Partial Rollout Trainer

**Group:**  `Tencent Data & Computation Platform Department`

**Author:** Yue Wang*, Zhipeng Ma*, Yi Yan, Hang Xu, Yang Li, Bo Qian, Peng Chen 

Last updated: 01/15/2026.

## 1. Introduction

### 1.1 Background

During synchronous reinforcement learning training in verl, we observe that the training dataset exhibits significant length imbalance, with a small fraction of exceptionally long samples. As illustrated in Figure 1, the maximum response length in the dataset reaches 160k tokens, while approximately 97% of responses are shorter than 80k tokens. Consequently, the minority of long-tail samples (3%) significantly slows down the training of the majority (97%) of the data. Moreover, these long-tail samples often correspond to more challenging cases, which are essential for effectively enhancing the model’s reasoning capabilities. Therefore, they cannot be removed without compromising training effectiveness.


![Response Length Distribution across the RL Training Dataset](
https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Response_Length_Distribution_across_the_RL_Training_Dataset.png)


### 1.2 Solution

We enhance the partial-rollout mechanism by introducing **sample supplementation** and **interruption techniques**. Since response lengths are unknown at inference time, **inference bubbles** are inevitable. We leverage sample supplementation to effectively utilize this otherwise unavoidable idle GPU time. Specifically, when a GPU worker completes its inference workload earlier than others, we supplement it with additional samples until the total number of samples returned by all GPU workers meets the training requirement. Once this requirement is satisfied, some GPU workers may still be processing ongoing inference tasks. To better utilize these partially processed samples, we **cache unfinished samples** and reuse them in the subsequent inference round.

![Comparison of GPU Execution Timelines between Standard Synchronous Training and the Proposed Async Partial Rollout](
https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Comparison_of_GPU_Execution_Timelines_between_Standard_Synchronous_Training_and_the_Proposed_Async_Partial_Rollout.png)
> reference: [APRIL: ACTIVE PARTIAL ROLLOUTS IN REINFORCEMENT LEARNING TO TAME LONG-TAIL GENERATION](
> https://arxiv.org/pdf/2509.18521)


Our core contributions include:

1. **Sample Supplementation and Interruption Mechanisms**:
   Introducing sample supplementation and interruption mechanisms to enable dynamic sample replenishment and automated scheduling of inference tasks.

2. **Rollout Caching**:
   Using a prompt manager to resume partial rollouts, managing complete and partial samples in the buffer based on sample staleness.


### 1.3 Experimental Results

- **Machine Configuration**: 2 nodes with 8 H20 GPUs
- **Model**: Qwen3-4B
- **Rollout Configuration**:
- **Max Response Length**: 18384 tokens (for DAPO-MATH17k), 1024 tokens (for GSM8K)
- **Algorithm**: GRPO
- **Rollout Engine**: vLLM

#### GSM8K
On the GSM8K dataset, our method achieves comparable convergence and tangible performance gains compared to the baseline. Upon completing the **full dataset** training, it reduces total training time by <span style="color:red">11.7%</span> and improves average GPU utilization by <span style="color:red">5.93%</span>. 

| Training mode          | Engine        | Step | Total Time       |Acc/mean@1   | GPU Avg Utilization |
|------------------------|---------------|------|------------------|---------------|---------------|
| GRPO+noPR              | VLLM+Megatron | 290  | 4h59m            | 94.99         |71.54 |
| GRPO+PR                | VLLM+Megatron | 280  | 4h24m <span style="color:red"> (-35m)  </span> | 94.08         |77.47|


> source data: https://swanlab.cn/@allenzpma/verl_exp_partial-rollout_gsm8k/runs

#### DAPO-MATH17k
Furthermore, on the DAPO-math dataset, our approach facilitates **full dataset** training with a <span style="color:red">51.1%</span> reduction in end-to-end execution time and an <span style="color:red">8.77%</span> boost in GPU utilization. And, our method achieve comparable convergence to the baseline.

| Training Mode | Engine | Step | Total Time |Acc/best@32/mean | Acc/maj@32/mean |GPU Avg Utilization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GRPO+noPR | VLLM+Megatron | 200 |67h34m | 79.94 | 73.33 |74.64|
| GRPO+PR | VLLM+Megatron | 110 | 33h02m <span style="color:red"> (-34h32m) </span> | 82.90 | 73.41 |83.41|


> source data: https://swanlab.cn/@allenzpma/verl_exp_partial-rollout_dapo-math/runs


## 2. Implementation

### 2.1 Sample Supplementation and Interruption Mechanisms (SSIM)

The main components of the SSIM mechanism are as follows:
<!-- ![Architectural Design of the SSIM Mechanism](
https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Architectural_Design_of_the_SSIM_Mechanism.png) -->

<img src="https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Architectural_Design_of_the_SSIM_Mechanism.png" width="60%">

The event interaction logic of the SSIM mechanism is as follows:
![The Event Interaction Logic of the SSIM Mechanism](
https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/The_Event_Interaction_Logic_of_the_SSIM_Mechanism.png)


### 2.2 Rollout Caching
The rollout caching mechanism is implemented using a prompt manager. The prompt manager uses a queue to control the order of sample resumption, with prompt priority defined by the **get_scheduling_priority** function.

```python
class PromptsManager:
    """
    PromptsManager is used to manage the prompts queue.
    """
    def __init__(
        self,
        global_config,
        train_dataloader : StatefulDataLoader,
        sampling_num : int,
        rollout_manager_obj,
        trained_prompts_index: set[int] = set(),
    ):
        """
        Args:
            global_config: the global config
            train_dataloader: the train dataloader from `ray_trainer.py`
            sampling_num: the number of samples to generate for each prompt
            rollout_manager_obj: the rollout manager object
            trained_prompts_index: the prompts that have been trained, used to skip the prompts that have been trained
        """
        self.global_config = global_config
        self.sampling_num = sampling_num
        self.prompt_queue = PromptsQueue()
        self.trained_prompts_index = trained_prompts_index

        # init dataloader_iter
        self.dataloader_iter = iter(train_dataloader)
        self.dataloader_iter_exhausted = False
        self.filter_cnt = 0
        self.model_version = 0


    # Sort Priority (for each prompt)
    def get_scheduling_priority(self, ignored_samples: set[Sample] = set()) -> tuple[int, float, int]:
        """
        Return a priority key for prompt scheduling.

        The tuple is ordered so that it can be directly used in `sort(key=...)`:
            (
                unfinished_samples_num,
                finished_mean_response_length (1e9 if no finished samples),
                max_staleness
            )
        """
        unfinished_samples = set(self.get_unfinished_samples()) - set(ignored_samples)
        finished_samples = self.get_finished_samples()

        # 1. unfinished samples number
        unfinished_num = len(unfinished_samples)
        # 2. mean response length of finished samples
        finished_mean_resp_len = (
            np.mean([sample.get_responses_length() for sample in finished_samples])
            if finished_samples
            else 1e9
        )
        # 3. max staleness
        max_staleness = np.max(
            [sample.get_staleness(expected_version=self.expected_model_version)
            for sample in self.samples]
        )

        return unfinished_num, finished_mean_resp_len, max_staleness
```

### 2.3 Off-Policy Correctness
To ensure the correctness of the PPO algorithm, PPO importance sampling is performed using **rollout log probs** with a decoupled trick, which preserves algorithmic correctness under interruptible generation and policy updates.

$$
J(\theta)=\mathbb{E}_{q \sim \mathcal{D}, a_t \sim \pi_{\text {behav}}^{\text{rollout}}}[\sum_{t=1}^H \min (\frac{\pi_{\theta}^{\text{train}}}{\pi_{\text {behav}}^{\text{rollout}}} \hat{A}_t, \frac{\pi_{\text {prox }}^{\text{rollout}}}{\pi_{\text {behav }}^{\text{rollout}}} \operatorname{clip}\left(\frac{\pi_{\theta}^{\text{train}}}{\pi_{\text {prox }}^{\text{rollout}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t)] \\
$$
> reference: [AREAL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](
> https://arxiv.org/pdf/2505.24298)

### 2.4 AgentLoop
In the current implementation, we use AgentLoop mode, which also supports multi-turn tool calling.

## 3.Usage
### GSM8K Configuration Example
```shell
bash recipe/partial_rollout/run_gsm8k_nopr_4b_bs128.sh
bash recipe/partial_rollout/run_gsm8k_pr_4b_bs128.sh
```

### DAPO_MATH Configuration Example
```shell
bash recipe/partial_rollout/run_dapo_math17k_nopr_4b_2node.sh
bash recipe/partial_rollout/run_dapo_math17k_pr_4b_2node.sh
```

## 4. Functional Support


| Category           | Support Situation                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------------------------|
| train engine       | FSDP2  <br/> Megatron                                                                                           |
| rollout engine     | vLLM                                                                                                            |
| AdvantageEstimator | GRPO <br/> GSPO <br/> SAPO <br/> GRPO_PASSK <br/> REINFORCE_PLUS_PLUS <br/> RLOO <br/> OPO <br/> REINFORCE_PLUS_PLUS_BASELINE<br/>GPG |
| Reward             | all                                                                                                             |
