#settings=ori
SETTINGS=lr
MODEL_NAME=Qwen/Qwen2.5-Math-1.5B
MODEL_SHORT_NAME=qwen2.5-math-1.5b
BASE_DIR=checkpoints/LengthPenalty_Long/math-grpo-qwen3-4b-base-8h20-lr

DATASET=math
export DATA_PATH=data/${DATASET}/test.parquet

export SAVE_PATH=results/${DATASET}/math-train-math-test-${MODEL_SHORT_NAME}-${SETTINGS}.parquet
MODEL_SAVE_DIR="${BASE_DIR}"/merged
STEPS=(300 )
for step in "${STEPS[@]}"; do
    step_dir=global_step_${step}
    export MODEL_DIR="${MODEL_SAVE_DIR}"/"${step_dir}"

#    python scripts/model_merger.py \
#        --backend fsdp \
#        --hf_model_path "${MODEL_NAME}" \
#        --local_dir "${MODEL_DIR}"/actor \
#        --target_dir "${MODEL_SAVE_DIR}"/"${step_dir}"

#    bash scripts/math_generation.sh

    bash scripts/math_eval.sh

done
