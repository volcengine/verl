# GPU Instance Setup Guide for Arc Vision Training

## Quick Fix for Current Error

The error `ModuleNotFoundError: No module named 'qwen_vl_utils'` occurs because VERL needs to be installed with the "geo" extras for vision-language model support.

### Immediate Solution

On your GPU instance, run:

```bash
pip install qwen-vl-utils
```

### Complete Installation (Recommended)

If you continue to have issues, reinstall VERL with all required extras:

```bash
# Navigate to VERL root
cd /root/verl

# Install with vision-language support
pip install -e ".[geo,sglang]"

# Verify installation
python -c "import qwen_vl_utils; print('qwen_vl_utils installed successfully')"
```

## Full Setup Checklist

1. **Install Required Dependencies**:
   ```bash
   # Core dependencies for vision-language models
   pip install qwen-vl-utils
   
   # Optional: Install all extras if needed
   pip install -e ".[geo,sglang,gpu]"
   ```

2. **Verify Data Files**:
   ```bash
   # Check data files exist
   ls -la /root/data/arc_vision/screenspot/
   # Should show: train.parquet, validation.parquet, test.parquet
   ```

3. **Pull Latest Code**:
   ```bash
   git pull origin main
   ```

4. **Run Training**:
   ```bash
   cd examples/arc_vision
   bash run_arc_vision_3b.sh
   ```

## Dependencies Explained

- **qwen-vl-utils**: Required for processing images/videos with Qwen2.5-VL models
- **av**: Video processing (installed with qwen-vl-utils)
- **sglang**: Required for multi-turn tool interactions
- **geo extras**: Includes mathruler, torchvision, and qwen_vl_utils

## Common Issues

1. **ModuleNotFoundError**: Install missing module with pip
2. **CUDA/GPU errors**: Ensure CUDA toolkit matches PyTorch version
3. **Memory errors**: Reduce batch size or GPU memory utilization
4. **Data errors**: Verify parquet files are not corrupted

## Next Steps

After installing `qwen-vl-utils`, your training should start successfully. Monitor GPU memory usage and adjust parameters if needed.