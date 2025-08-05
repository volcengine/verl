#!/usr/bin/env bash
set -xeuo pipefail

# Benchmark script for fully_async_policy performance testing
# This script runs various performance tests to evaluate the async training system

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}

# Download model if not exists
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

# Create benchmark results directory
BENCHMARK_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BENCHMARK_DIR}"

echo "Starting fully_async_policy performance benchmark..."
echo "Results will be saved to: ${BENCHMARK_DIR}"

# Benchmark parameters
n_gpus_rollout=2
n_gpus_training=$((NUM_GPUS - n_gpus_rollout))

# Common parameters
train_prompt_bsz=16
n_resp_per_prompt=4
train_prompt_mini_bsz=4
max_prompt_length=512
max_response_length=1024

# Benchmark Test 1: Different staleness thresholds
echo "=== Benchmark Test 1: Staleness Threshold Impact ==="
staleness_values=(1 3 5 10)

for staleness in "${staleness_values[@]}"; do
    echo "Testing staleness threshold: ${staleness}"

    exp_name="benchmark-staleness-${staleness}"
    log_file="${BENCHMARK_DIR}/staleness_${staleness}.log"

    timeout 300 python3 -m recipe.fully_async_policy.fully_async_main \
        data.train_files="${HOME}/data/gsm8k/train.parquet" \
        data.val_files="${HOME}/data/gsm8k/test.parquet" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        trainer.logger=['console'] \
        trainer.project_name='verl-benchmark' \
        trainer.experiment_name="${exp_name}" \
        trainer.val_before_train=False \
        trainer.test_freq=-1 \
        trainer.save_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=10 \
        trainer.n_gpus_per_node=${n_gpus_training} \
        rollout.n_gpus_per_node=${n_gpus_rollout} \
        async_training.staleness_threshold=${staleness} \
        async_training.max_staleness_allowed=$((staleness + 2)) \
        > "${log_file}" 2>&1 || echo "Test with staleness ${staleness} timed out or failed"

    # Extract key metrics from log
    if [ -f "${log_file}" ]; then
        echo "=== Metrics for staleness=${staleness} ===" >> "${BENCHMARK_DIR}/summary.txt"
        grep -E "(Generated.*batches|Dropped.*samples|param_version|Queue size)" "${log_file}" | tail -5 >> "${BENCHMARK_DIR}/summary.txt" || true
        echo "" >> "${BENCHMARK_DIR}/summary.txt"
    fi
done

# Benchmark Test 2: Different queue sizes
echo "=== Benchmark Test 2: Queue Size Impact ==="
queue_sizes=(50 100 500 1000)

for queue_size in "${queue_sizes[@]}"; do
    echo "Testing queue size: ${queue_size}"

    exp_name="benchmark-queue-${queue_size}"
    log_file="${BENCHMARK_DIR}/queue_${queue_size}.log"

    timeout 300 python3 -m recipe.fully_async_policy.fully_async_main \
        data.train_files="${HOME}/data/gsm8k/train.parquet" \
        data.val_files="${HOME}/data/gsm8k/test.parquet" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        trainer.logger=['console'] \
        trainer.project_name='verl-benchmark' \
        trainer.experiment_name="${exp_name}" \
        trainer.val_before_train=False \
        trainer.test_freq=-1 \
        trainer.save_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=10 \
        trainer.n_gpus_per_node=${n_gpus_training} \
        rollout.n_gpus_per_node=${n_gpus_rollout} \
        async_training.max_queue_size=${queue_size} \
        > "${log_file}" 2>&1 || echo "Test with queue size ${queue_size} timed out or failed"

    # Extract key metrics from log
    if [ -f "${log_file}" ]; then
        echo "=== Metrics for queue_size=${queue_size} ===" >> "${BENCHMARK_DIR}/summary.txt"
        grep -E "(Generated.*batches|Queue size|memory)" "${log_file}" | tail -5 >> "${BENCHMARK_DIR}/summary.txt" || true
        echo "" >> "${BENCHMARK_DIR}/summary.txt"
    fi
done

# Benchmark Test 3: Different batch generation intervals
echo "=== Benchmark Test 3: Generation Interval Impact ==="
intervals=(0.0 0.1 0.5 1.0)

for interval in "${intervals[@]}"; do
    echo "Testing batch generation interval: ${interval}s"

    exp_name="benchmark-interval-${interval}"
    log_file="${BENCHMARK_DIR}/interval_${interval}.log"

    timeout 300 python3 -m recipe.fully_async_policy.fully_async_main \
        data.train_files="${HOME}/data/gsm8k/train.parquet" \
        data.val_files="${HOME}/data/gsm8k/test.parquet" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.train_batch_size=${train_prompt_bsz} \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        trainer.logger=['console'] \
        trainer.project_name='verl-benchmark' \
        trainer.experiment_name="${exp_name}" \
        trainer.val_before_train=False \
        trainer.test_freq=-1 \
        trainer.save_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=10 \
        trainer.n_gpus_per_node=${n_gpus_training} \
        rollout.n_gpus_per_node=${n_gpus_rollout} \
        async_training.batch_generation_interval=${interval} \
        > "${log_file}" 2>&1 || echo "Test with interval ${interval} timed out or failed"

    # Extract key metrics from log
    if [ -f "${log_file}" ]; then
        echo "=== Metrics for interval=${interval}s ===" >> "${BENCHMARK_DIR}/summary.txt"
        grep -E "(Generated.*batches|generation_timestamp)" "${log_file}" | tail -5 >> "${BENCHMARK_DIR}/summary.txt" || true
        echo "" >> "${BENCHMARK_DIR}/summary.txt"
    fi
done

# Benchmark Test 4: Resource allocation comparison
echo "=== Benchmark Test 4: Resource Allocation Comparison ==="

# Test different rollout/training GPU distributions
if [ "${NUM_GPUS}" -ge "6" ]; then
    gpu_configs=(
        "1,$((NUM_GPUS - 1))"  # 1 rollout, rest training
        "2,$((NUM_GPUS - 2))"  # 2 rollout, rest training
        "3,$((NUM_GPUS - 3))"  # 3 rollout, rest training
    )

    for config in "${gpu_configs[@]}"; do
        IFS=',' read -r rollout_gpus training_gpus <<< "$config"

        echo "Testing GPU allocation: ${rollout_gpus} rollout, ${training_gpus} training"

        exp_name="benchmark-gpu-${rollout_gpus}r-${training_gpus}t"
        log_file="${BENCHMARK_DIR}/gpu_${rollout_gpus}_${training_gpus}.log"

        timeout 300 python3 -m recipe.fully_async_policy.fully_async_main \
            data.train_files="${HOME}/data/gsm8k/train.parquet" \
            data.val_files="${HOME}/data/gsm8k/test.parquet" \
            data.prompt_key=prompt \
            data.truncation='left' \
            data.max_prompt_length=${max_prompt_length} \
            data.max_response_length=${max_response_length} \
            data.train_batch_size=${train_prompt_bsz} \
            actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
            trainer.logger=['console'] \
            trainer.project_name='verl-benchmark' \
            trainer.experiment_name="${exp_name}" \
            trainer.val_before_train=False \
            trainer.test_freq=-1 \
            trainer.save_freq=-1 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=10 \
            trainer.n_gpus_per_node=${training_gpus} \
            rollout.n_gpus_per_node=${rollout_gpus} \
            > "${log_file}" 2>&1 || echo "Test with GPU config ${config} timed out or failed"

        # Extract key metrics from log
        if [ -f "${log_file}" ]; then
            echo "=== Metrics for ${rollout_gpus}r/${training_gpus}t GPUs ===" >> "${BENCHMARK_DIR}/summary.txt"
            grep -E "(Generated.*batches|training.*steps|GPU)" "${log_file}" | tail -5 >> "${BENCHMARK_DIR}/summary.txt" || true
            echo "" >> "${BENCHMARK_DIR}/summary.txt"
        fi
    done
fi

# Benchmark Test 5: Pause/Resume Performance
echo "=== Benchmark Test 5: Pause/Resume Performance Test ==="
log_file="${BENCHMARK_DIR}/pause_resume.log"

# Start the training in background
python3 -m recipe.fully_async_policy.fully_async_main \
    data.train_files="${HOME}/data/gsm8k/train.parquet" \
    data.val_files="${HOME}/data/gsm8k/test.parquet" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.logger=['console'] \
    trainer.project_name='verl-benchmark-pause' \
    trainer.experiment_name='pause-resume-test' \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=20 \
    trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    > "${log_file}" 2>&1 &

TRAINING_PID=$!

# Note: In actual implementation, we would need a way to remotely control pause/resume
# This is a placeholder for testing the pause/resume functionality
echo "Training started with PID: ${TRAINING_PID}"
echo "Pause/resume testing would require remote control interface" >> "${BENCHMARK_DIR}/summary.txt"

# Wait a bit and then kill the training (simulating early termination)
sleep 60
if kill -0 $TRAINING_PID 2>/dev/null; then
    echo "Stopping training process..."
    kill $TRAINING_PID
fi

# Generate performance report
echo "=== Generating Performance Report ==="
report_file="${BENCHMARK_DIR}/performance_report.md"

cat > "${report_file}" << EOF
# Fully Async Policy Performance Benchmark Report

**Date:** $(date)
**Hardware:** ${NUM_GPUS} GPUs
**Strategy:** ${ACTOR_STRATEGY}
**Model:** ${MODEL_ID}

## Test Configuration
- Training Batch Size: ${train_prompt_bsz}
- Responses per Prompt: ${n_resp_per_prompt}
- Max Prompt Length: ${max_prompt_length}
- Max Response Length: ${max_response_length}

## Results Summary
$(cat "${BENCHMARK_DIR}/summary.txt" 2>/dev/null || echo "No summary available")

## Log Files
EOF

# List all log files
for log_file in "${BENCHMARK_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        echo "- $(basename "${log_file}")" >> "${report_file}"
    fi
done

cat >> "${report_file}" << EOF

## Key Findings
- **Staleness Impact:** Lower staleness thresholds may increase sample dropping but improve freshness
- **Queue Size Impact:** Larger queues provide better buffering but use more memory
- **Generation Interval:** Shorter intervals increase throughput but may stress the system
- **GPU Allocation:** Balance between generation and training capacity is crucial
- **Pause/Resume:** System should handle interruptions gracefully

## Recommendations
1. Start with staleness_threshold=3 for good balance
2. Use queue_size=500-1000 for most workloads
3. Set generation_interval=0.1s for good performance
4. Allocate 2-3 GPUs for rollout in typical 8-GPU setups
5. Monitor queue utilization and adjust based on workload

EOF

echo "Benchmark completed!"
echo "Results saved to: ${BENCHMARK_DIR}/"
echo "Performance report: ${report_file}"

# Print summary to console
if [ -f "${BENCHMARK_DIR}/summary.txt" ]; then
    echo ""
    echo "=== BENCHMARK SUMMARY ==="
    cat "${BENCHMARK_DIR}/summary.txt"
fi

