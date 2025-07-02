#!/bin/bash
# Script to regenerate ScreenSpot data with correct image format

set -x

echo "Regenerating ScreenSpot data with correct image format..."

# Remove old data
rm -rf ~/data/arc_vision/screenspot/*.parquet

# Regenerate with corrected format
python3 prepare_screenspot_data.py \
    --local_dir ~/data/arc_vision/screenspot \
    --max_samples 1200 \
    --split_test_data

echo "Data regeneration complete!"
echo "Now run: bash run_arc_vision_3b_fixed.sh"