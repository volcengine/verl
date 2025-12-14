## Summary

Add `exclude_fully_masked_seq` option to control whether fully masked sequences are excluded from the denominator when computing seq-mean loss aggregation.

**Changes:**
- Add `exclude_fully_masked_seq: Optional[bool] = None` parameter to `agg_loss()` function in `core_algos.py`
- Add corresponding field to `ActorConfig` for user configuration
- Propagate the option through `global_batch_info` in `losses.py`, `dp_actor.py`, and `megatron_actor.py`
- Fix pre-existing issue where `dp_actor.py` and `megatron_actor.py` did not populate `global_batch_info` for `agg_loss` calls (entropy and kl_loss)
- **Fix bug**: Remove `* dp_size` when using local count fallback (when `global_batch_size` is None)

**Behavior:**
- When `global_batch_size` is provided (from data): Uses `local_sum / global_batch_size * dp_size` for correct global mean
- When `global_batch_size` is None (fallback):
  - `exclude_fully_masked_seq=None` or `False`: Uses `local_sum / local_batch_size` (local mean)
  - `exclude_fully_masked_seq=True`: Uses `local_sum / local_non_masked_count` (local mean excluding masked)

**Bug Fix Explanation:**
The previous code used `local_sum / local_count * dp_size` when `global_batch_size` was None. This is incorrect because:
- With dp_size=2, workers compute: `loss = local_sum / local_count * 2`
- After FSDP averaging: `final = (loss0 + loss1) / 2 = local_sum / local_count`
- This gives 2x the intended local mean

The fix removes `* dp_size` when using local counts, so FSDP averaging correctly gives the average of local means.

**Example:**
```python
# 4 sequences per worker, 1 fully masked, dp_size=2
# With global_batch_size provided (e.g., 8):
#   loss = local_sum / 8 * 2 → after FSDP avg → global_sum / 8 (correct global mean)
# With local fallback (global_batch_size=None):
#   loss = local_sum / 4 → after FSDP avg → avg of local means (correct)
# With local fallback + exclude_fully_masked_seq=True:
#   loss = local_sum / 3 → after FSDP avg → avg of local means (correct)
```

## Test Plan

- [ ] Test with `global_batch_size` provided (standard path via losses.py)
- [ ] Test with `global_batch_size=None` fallback (dp_actor.py, megatron_actor.py paths)
- [ ] Test with `exclude_fully_masked_seq=True` on batches with fully masked sequences
- [ ] Verify distributed training correctness with multiple DP workers
