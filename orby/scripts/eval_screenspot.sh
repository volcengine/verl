set -e

# How to run:
# bash orby/scripts/eval_screenspot.sh --version screenspot
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro

# Default values
DATASET_VERSION="screenspot"
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
REWARD_FILE=orby/reward/screenspot.py
REWARD_FN=reward_func
OUTPUT_FILE=result-test-output-1.parquet

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            DATASET_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set dataset-specific variables
case $DATASET_VERSION in
    "screenspot")
        DATA_PATH=~/data/screenspot
        PARQUET_PATTERN="test.parquet"
        ;;
    "screenspot_v2")
        DATA_PATH=~/data/screenspot_v2
        PARQUET_PATTERN="test.parquet"
        ;;
    "screenspot_pro")
        DATA_PATH=~/data/screenspot_pro
        PARQUET_PATTERN="test.parquet"
        ;;
    *)
        echo "Invalid dataset version: $DATASET_VERSION"
        echo "Available versions: screenspot, screenspot_v2, screenspot_pro"
        exit 1
        ;;
esac

echo "Using dataset version: $DATASET_VERSION"
echo "Data path: $DATA_PATH"

# Check if parquet files already exist
if ls $DATA_PATH/$PARQUET_PATTERN 1> /dev/null 2>&1; then
    echo "Parquet files already exist, skipping conversion..."
else
    echo "Converting dataset..."
    case $DATASET_VERSION in
        "screenspot")
            python3 -m orby.data.convert_screenspot
            ;;
        "screenspot_v2")
            huggingface-cli download OS-Copilot/ScreenSpot-v2 --repo-type dataset --local-dir=$DATA_PATH
            cd $DATA_PATH
            unzip screenspotv2_image.zip
            cd -
            python orby/data/convert_screenspot_v2.py --image_dir=$DATA_PATH/screenspotv2_image/
            ;;
        "screenspot_pro")
            huggingface-cli download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir=$DATA_PATH
            python orby/data/convert_screenspot_pro.py
            ;;
    esac
fi

# Generation
# Screenspot pro has example with more than 16k tokens.
python3 -m orby.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/$PARQUET_PATTERN \
    data.prompt_key=prompt \
    data.batch_size=256 \
    +data.max_prompt_length=20000 \
    +data.image_key=images \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/$OUTPUT_FILE \
    model.path=$MODEL_PATH \
    rollout.temperature=0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=20000 \
    rollout.response_length=256 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$DATA_PATH/$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN
