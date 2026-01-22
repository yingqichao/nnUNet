# Potential Issues and Suggestions for nnUNetTrainer_WithTuningSet

Analysis performed by Cursor AI on the following files:
- `nnUNetTrainer_with_tuning_set.py`
- `run_training.py`
- `README_TUNING_SET_TRAINER.md`
- `continue_training_with_ckpt.sh`

---

## Issue Status Legend

- ✅ **FIXED**: Issue has been resolved in code
- 📝 **DOC FIXED**: Documentation updated to match code behavior
- ⚠️ **OPEN**: Issue remains and may need attention
- 💡 **SUGGESTION**: Enhancement idea, not a bug

---

## 1. Documentation vs Implementation Mismatches

### 1.1 Data Split Ratio Discrepancy 📝 DOC FIXED

**Original Issue**: Documentation said 80/10/10 split but code uses `tuning_size = test_size`.

**Resolution**: README updated to accurately describe the split:
- Test: Original nnUNet validation fold (e.g., 20% in 5-fold CV)
- Tuning: Same size as test, carved from training fold
- Training: Remaining samples (~60% in 5-fold CV)

### 1.2 MAX_SYNTHETIC_RATIO Documentation 📝 DOC FIXED

**Original Issue**: README showed `MAX_SYNTHETIC_RATIO = 1.0` but default is `0.5`.

**Resolution**: README updated to show correct default `max_synthetic_ratio = 0.5` (50% cap).

---

## 2. Shell Script vs Expected Behavior

### 2.1 CHECKPOINT_PATH Uses `checkpoint_best` with `--c` ⚠️ OPEN

**File**: `continue_training_with_ckpt.sh` (line 6)

```bash
CHECKPOINT_PATH=...checkpoint_best_nnUNetResEncUNetLPlans.pth
```

**Concern**: Using `checkpoint_best` (e.g., from epoch 50) instead of `checkpoint_final` (epoch 1000) means training progress beyond the best checkpoint is lost.

**Impact**: Medium - May be intentional but could cause confusion.

**Suggestion**: Document this behavior explicitly or consider using `checkpoint_final` for continue training.

### 2.2 `--skip_val` Not Documented 📝 DOC FIXED

**Resolution**: Added `--skip_val` to README usage examples and command-line arguments section.

---

## 3. GPU Configuration Issues

### 3.1 CUDA_VISIBLE_DEVICES Timing Window ⚠️ OPEN (Low Priority)

**File**: `nnUNetTrainer_with_tuning_set.py` (lines 2350-2389)

**Issue**: Brief window where parent's `CUDA_VISIBLE_DEVICES` is modified before subprocess spawn.

```python
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(subprocess_gpu_id)  # Window opens
    self._async_eval_process = ctx.Process(...)
    self._async_eval_process.start()
finally:
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible   # Window closes
```

**Impact**: Low - Race condition is theoretically possible but unlikely in practice.

**Suggestion**: Could be improved by using subprocess environment passing instead of modifying parent's environment, but current approach works.

### 3.2 No Validation That backup_gpu_id ≠ main_gpu_id ⚠️ OPEN

**File**: `run_training.py`

**Issue**: If user sets `--main_gpu_id 0 --backup_gpu_id 0`, both processes compete for the same GPU.

**Impact**: Medium - Could cause OOM errors.

**Suggestion**: Add warning in `run_training.py`:
```python
if main_gpu_id is not None and backup_gpu_id is not None and main_gpu_id == backup_gpu_id:
    print("WARNING: --main_gpu_id and --backup_gpu_id are the same. "
          "This may cause GPU memory issues during async evaluation.")
```

---

## 4. Async Subprocess Lifecycle Issues

### 4.1 Daemon Process May Be Killed Prematurely ✅ FIXED

**Original Issue**: Daemon subprocess could be killed before saving checkpoint when training completes.

**Resolution**: Added `on_train_end()` method that waits for any running async evaluation:
```python
def on_train_end(self):
    if self._async_eval_process is not None and self._async_eval_process.is_alive():
        result = self._wait_for_async_eval(timeout=ASYNC_EVAL_TIMEOUT)
        if result:
            self._handle_async_eval_result(result)
```

### 4.2 Orphaned Files After Crash ⚠️ OPEN (Mitigated)

**Issue**: If main process crashes, temp checkpoint files remain.

**Mitigation**: `_cleanup_orphaned_temp_files()` is called during `on_train_start()` to clean up files from previous runs.

**Remaining Concern**: No validation that orphaned checkpoint belongs to the same training configuration.

---

## 5. Checkpoint Management Issues

### 5.1 `top_k_dices` Always Cleared on Load ⚠️ OPEN (By Design)

**File**: `nnUNetTrainer_with_tuning_set.py` (line ~4900)

```python
# Keep top_k_dices empty - it will be populated by scanning files in on_epoch_end
self.top_k_dices = []
```

**Design Decision**: Rankings are reconstructed by scanning existing checkpoint files rather than trusting checkpoint data. This ensures consistency even if files were manually moved/deleted.

**Trade-off**: If checkpoint files are deleted, ranking information is lost.

### 5.2 `_skip_existing_best_comparison` Flag Timing ⚠️ OPEN (Edge Case)

**Issue**: Flag is used on first evaluation epoch after loading. If `current_epoch` is loaded as epoch 100 and `eval_every_n_epochs = 5`, the skip happens at epoch 100 or 105.

**Impact**: Low - Edge case, behavior is acceptable but could be confusing.

---

## 6. Naming/Terminology Confusion

### 6.1 `_best_ema` vs `_best_dice` Confusion ⚠️ OPEN

**Issue**: Code maintains both `_best_ema` (legacy) and `_best_dice` (new) for backward compatibility.

**Current State**:
- Checkpoint selection uses single-epoch tumor Dice (not EMA)
- Both fields are saved/loaded for compatibility
- Log output says "Pseudo dice" but value is single-epoch Dice

**Suggestion**: Consider deprecating `_best_ema` in future version and clarifying log output.

### 6.2 "Pseudo dice" Log Terminology ⚠️ OPEN

**Issue**: Log output says "Pseudo dice" but this trainer uses single-epoch tumor Dice, not EMA-based pseudo Dice like parent trainer.

**Impact**: Low - Could confuse users comparing logs from different trainers.

**Suggestion**: Change log message to "Tumor Dice" or "Test Dice" for clarity.

---

## 7. Sliding Window Cache Issue

### 7.1 Cache Key Doesn't Include Patch Size ⚠️ OPEN (Low Priority)

**Issue**: `_sliding_window_box_cache` uses sample key but not patch size.

**Impact**: Low - Patch size doesn't change within a training run.

**Edge Case**: If trainer is reused with different configuration (different patch size), cache would be invalid.

**Suggestion**: Include patch size in cache key for robustness:
```python
cache_key = f"{key}_{tuple(patch_size)}"
```

---

## 8. Missing Validation/Warnings

### 8.1 Pattern Regex Not Validated ✅ FIXED

**Original Issue**: Invalid regex in `--pattern_original_samples` failed with cryptic error.

**Resolution**: Added try/except with helpful error message:
```python
try:
    return bool(re.match(pattern, key))
except re.error as e:
    raise ValueError(
        f"Invalid regex pattern in --pattern_original_samples: '{self.pattern_original_samples}'\n"
        f"Regex error: {e}\n"
        f"Please provide a valid Python regex pattern."
    ) from e
```

### 8.2 No Warning When All Samples Excluded ⚠️ OPEN

**Issue**: If sanity check excludes all samples (all lack tumor), training would fail later with unclear error.

**Suggestion**: Add early check after sanity check:
```python
if len(valid_keys) == 0:
    raise RuntimeError(
        "All samples were excluded by sanity check (no tumor label 2 found). "
        "Please verify your dataset contains tumor labels."
    )
```

---

## Summary Table

| ID | Issue | Status | Priority | Impact |
|----|-------|--------|----------|--------|
| 1.1 | Split ratio documentation | 📝 DOC FIXED | - | - |
| 1.2 | Synthetic ratio documentation | 📝 DOC FIXED | - | - |
| 2.1 | checkpoint_best with --c | ⚠️ OPEN | Low | Medium |
| 2.2 | --skip_val documentation | 📝 DOC FIXED | - | - |
| 3.1 | CUDA_VISIBLE_DEVICES window | ⚠️ OPEN | Low | Low |
| 3.2 | backup_gpu == main_gpu | ⚠️ OPEN | Medium | Medium |
| 4.1 | Daemon killed prematurely | ✅ FIXED | - | - |
| 4.2 | Orphaned files after crash | ⚠️ OPEN | Low | Low |
| 5.1 | top_k_dices cleared on load | ⚠️ OPEN | Low | Low |
| 5.2 | _skip_existing_best timing | ⚠️ OPEN | Low | Low |
| 6.1 | _best_ema vs _best_dice | ⚠️ OPEN | Low | Low |
| 6.2 | "Pseudo dice" terminology | ⚠️ OPEN | Low | Low |
| 7.1 | Cache key missing patch size | ⚠️ OPEN | Low | Low |
| 8.1 | Regex pattern validation | ✅ FIXED | - | - |
| 8.2 | All samples excluded warning | ⚠️ OPEN | Medium | Medium |

---

## Recommended Priority Fixes

### High Priority (Should Fix)
1. **3.2**: Add warning when `backup_gpu_id == main_gpu_id`
2. **8.2**: Add early error when all samples excluded by sanity check

### Medium Priority (Nice to Have)
1. **6.2**: Change "Pseudo dice" to "Tumor Dice" in log output
2. **2.1**: Document or reconsider `checkpoint_best` usage in shell script

### Low Priority (Future Consideration)
1. **7.1**: Include patch size in cache key
2. **6.1**: Deprecate `_best_ema` in favor of `_best_dice`

---

## Notes

- Analysis performed on code state as of January 2026
- Some "issues" are design decisions that may be intentional
- Always test changes thoroughly before deploying to production training
