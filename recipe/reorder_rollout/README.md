# Recipe: Reorder Rollout with TPPO

This recipe is under development.

## Introduction

### Background

当然，以下是更加正式、适合用于技术文档或论文的英文翻译：

1.	In the rollout phase, a conventional approach involves sending an entire batch of data to the inference engine and proceeding to subsequent stages only after the inference for the entire batch has been completed.

2.	During this process, a small subset of samples may produce significantly longer token sequences, resulting in extended inference times for those samples. These outliers delay the overall completion of the batch and are commonly referred to as long-tail samples.

### Related Work

To mitigate the overall throughput degradation caused by long-tail samples, the community has proposed several solutions, including:
	•	Partial Rollout: Long-tail samples are truncated by setting a maximum sequence length (max_seq_length), and the remaining portions are deferred to subsequent rollout batches for continued inference.
	•	Async/Split Rollout: A portion of the current batch—defined as a percentage of the target batch size—is sent for training once inference is complete, while the remaining long-tail samples continue inference and are later combined with the next batch for training.

These approaches introduce a concept of staleness, where the samples consumed during training may not originate from the current model’s inference output, but rather from earlier iterations. This raises certain algorithmic concerns, particularly regarding the consistency and timeliness of the training signal relative to the model’s current parameters.

## Core Idea: Non-Intrusive Long-Tail Mitigation via Data Reordering

The key motivation is to alleviate the long-tail performance bottleneck without introducing algorithm-level modifications or asynchronous processing, by addressing the issue purely at the data scheduling layer.

Reordering Strategy
	1.	Token-Aware Grouping:
The idea is to identify samples that are likely to generate longer token sequences, and proactively group them into the same batch. By aligning such long-tail samples together, we can significantly improve batch-level throughput (e.g., minimizing idle time caused by stragglers).
(An open question remains: how do we estimate or predict token length beforehand?)
	2.	Fine-Grained Request-Level Parallelism:
In current rollout implementations, batch-level inputs are typically split into request-level units for inference, and later aggregated back into a training batch. This request-level granularity opens up opportunities for smarter scheduling.
	3.	Dynamic Batch Selection:
Instead of performing static batching prior to rollout, we propose a strategy that saturates the inference engine with as many requests as possible, and then selects the first N completed requests (where N equals the target batch size) to form the training batch.
	4.	Synchronous Behavior, No Asynchrony:
To preserve strict synchronization semantics, any requests not included in the fast batch are simply discarded for the current round and recycled into the next rollout. This avoids the introduction of stale data or stale model outputs.

### Synchronous Batch Refinement

In addition, we propose a variant of sync batching that introduces bounded delay constraints — i.e., samples can only be postponed for up to X rollout rounds before being forced into training. This addresses convergence instability observed in earlier schemes with high staleness bias.

Overall, this strategy does not alter the core training algorithm, but simply reshuffles the sample presentation order to front-load short-output samples and group long-output samples together, thereby improving rollout efficiency while retaining algorithmic correctness.


### Theoretical Benefit Analysis

Our objective is to mitigate the performance bottleneck caused by long-tail samples. We consider a simplified scenario where each batch contains exactly one long-tail sample, with the following assumptions:
	•	Inference time for long-tail samples: t_max
	•	Inference time for regular samples: t_mean
	•	Total number of batches in the dataset: num_b

Then:
	1.	Baseline Total Rollout Time
Since every batch is blocked by the long-tail sample, the total rollout time is:
T_{\text{baseline}} = t_{\text{max}} \times \text{num}_b
	2.	Reorder Rollout Time
In reorder-mode (or token-aware reordering), we allow early-exiting of short samples and defer long-tail samples to subsequent rounds. Only one batch is bottlenecked by the long-tail case, and the rest complete at t_mean:
T_{\text{reorder}} = t_{\text{mean}} \times (\text{num}b - 1) + t{\text{max}}
	3.	Average Rollout Time Comparison
When num_b >> 1, the average rollout time per batch becomes:
	•	Baseline: ~t_max
	•	Stream-mode: ~t_mean

This shows a significant throughput gain, especially when there’s a large gap between t_max and t_mean.

For example, if we assume t_max = 120s and t_mean = 30s, and num_b = 100, then:
	•	Baseline total time ≈ 120 * 100 = 12,000s
	•	Stream-mode total time ≈ 30 * 99 + 120 = 3,090s
	•	Theoretical speedup ≈ 3.9×

## Experiment
see https://github.com/volcengine/verl/pull/2200 for more details

### Rollout Bench

We evaluate our rollout reordering strategy on the Eurus-2-RL-Data dataset, focusing on code-related samples. The first 5,000 samples are selected for rollout using the following configuration:
•	Model: Qwen2-7B
•	Hardware: 4 × H20 GPUs
•	Max Output Length: 16K tokens
•	Batch Size: 1024
•	Number of Mini-batches: 5
Run the experiment with the following command:

```bash
pytest ./recipe/reorder_rollout/test/test_reorder_scheduler.py -s
```

### E2E Bench
dataset: GSM8k

max_output_length = 16k

model: qwen2-7b

8*H20，dp=4

Run the experiment with the following command:

```bash
bash ./recipe/reorder_rollout/run_qwen2.5-7b_gsm8k_reorder.sh
```

Baseline can be run with the following command:

```bash
bash ./recipe/reorder_rollout/run_qwen2.5-7b_gsm8k_baseline.sh
```

we also provide a sync batch script:

```bash
bash ./recipe/reorder_rollout/run_qwen2.5-7b_gsm8k_sync.sh
```

## Future Plan
1. Partial Rollout with Drop Policy: currently we drop all the partial generation result to avoid the staleness issue, we will implement a drop policy to keep the partial generation result, which might improve the performance.
2. implement TPPO algorithm: https://arxiv.org/pdf/2506.15050 with Partial Rollout.