# nnUNetTrainer_WithTuningSet

A custom nnUNet trainer that implements **3-way data splitting** (train/tuning/test), **dynamic patch sampling**, and **flexible evaluation modes** for improved tumor-focused segmentation training.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Splitting](#data-splitting)
- [Dynamic Patch Sampling Strategy](#dynamic-patch-sampling-strategy)
- [Evaluation Modes](#evaluation-modes)
- [Async vs Sync Test Evaluation](#async-vs-sync-test-evaluation)
- [Top-K Checkpoint Management](#top-k-checkpoint-management)
- [Configuration Options](#configuration-options)
- [Usage](#usage)
- [Integration Test Mode](#integration-test-mode)

---

## Overview

This trainer extends the base `nnUNetTrainer` with:

1. **3-Way Split**: Divides data into train, tuning, and test sets
2. **Dynamic Sampling**: Adapts patch centering probabilities based on tuning set performance
3. **Dual Evaluation Modes**: Supports both dual-crop (fast) and sliding-window (accurate) evaluation
4. **Async Test Evaluation**: Runs test evaluation in a subprocess to avoid blocking training

---

## Key Features

| Feature | Description |
|---------|-------------|
| 3-Way Split | Train / Tuning / Test with tuning size = test size (deterministic) |
| Dynamic Sampling | Adjusts random/anatomy/tumor centering probabilities during training |
| Tuning Set | Fast dual-crop evaluation for adaptive strategy decisions |
| Test Set | Accurate sliding-window evaluation for checkpoint selection |
| Async Evaluation | Test evaluation runs in parallel subprocess (non-blocking) |
| Top-K Checkpoints | Saves top 5 best models based on tumor Dice |
| Perfect Anatomy Simulation | GT masking for tumor-only metric computation |
| Input Validation | Regex patterns validated with helpful error messages |

---

## Data Splitting

The trainer creates a 3-way split where **tuning set size = test set size** for balanced evaluation:

```
┌─────────────────────────────────────────────────────────────┐
│              Original nnUNet Split (e.g., 5-fold CV)         │
├─────────────────────────────────────────┬───────────────────┤
│           Training Fold (80%)           │  Val Fold (20%)   │
│                                         │    = Test Set     │
├─────────────────────────────────────────┼───────────────────┤
│                                         │                   │
│   ┌─────────────────┬──────────────┐    │                   │
│   │ Remaining Train │ Tuning Set   │    │                   │
│   │     (~60%)      │ (= test size)│    │                   │
│   │                 │   (~20%)     │    │                   │
│   └─────────────────┴──────────────┘    │                   │
└─────────────────────────────────────────┴───────────────────┘
```

**Split Details**:
- **Test Set**: Original nnUNet validation fold (untouched)
- **Tuning Set**: Carved from training fold, size = test set size, original samples only
- **Training Set**: Remaining samples (original + synthetic if enabled)

**Example with 5-fold CV (100 samples)**:
- Test: 20 samples (one fold)
- Tuning: 20 samples (from remaining 80, original only)
- Training: ~60 samples (40 original + up to 20 synthetic)

**Important**: Samples without tumor (label 2) are automatically excluded via sanity check.

---

## Dynamic Patch Sampling Strategy

### The Problem

Standard nnUNet uses fixed foreground oversampling (33% foreground-centered patches). This may not be optimal for:
- Small tumors (need more tumor-centered patches)
- High false positive rates (need more context/anatomy patches)

### The Solution: Adaptive 3-Way Centering

The trainer uses **three centering modes** for patch extraction during training:

| Mode | Description | Controls |
|------|-------------|----------|
| **Random** | Center on any voxel | General context, background learning |
| **Anatomy** | Center on organ voxel (label 1) | False positive reduction |
| **Tumor** | Center on tumor voxel (label 2) | Recall improvement |

### Probability Relationship

The strategy uses a **simplified model** where only `p_anatomy` is actively adjusted:

```
p_random = 0.5 × p_anatomy    (random is always half of anatomy)
p_tumor  = 1 - 1.5 × p_anatomy (tumor gets the rest)
```

**Bounds**:
- `p_anatomy ∈ [0.10, 0.50]`
- When `p_anatomy = 0.10`: random=0.05, tumor=0.85 (max lesion focus)
- When `p_anatomy = 0.50`: random=0.25, tumor=0.25 (balanced)

### Trend Detection and Adaptation

The strategy monitors **EMA (Exponential Moving Average)** of metrics from the tuning set:

| Trend | Condition | Action |
|-------|-----------|--------|
| `improving` | Dice EMA increasing | Continue current direction |
| `degrading` | Dice EMA decreasing | Reverse adjustment direction |
| `stable` | Dice EMA converged | Small decay toward balance |
| `recall_dropping` | Recall drops > 3% | **Increase tumor focus** (priority) |
| `precision_dropping` | Precision drops > 3% | **Increase context** |

### Best Dice Tracking with Reversion

To ensure overall improvement:

1. **Track best**: Remember `best_dice_ema` and the `p_anatomy` that achieved it
2. **Explore margin**: Only adjust when within 2% of best Dice
3. **Revert on drop**: If Dice drops > 5% from best, revert to `best_p_anatomy`

```
┌─────────────────────────────────────────────────────────────┐
│                   Epoch N Evaluation                        │
├─────────────────────────────────────────────────────────────┤
│  1. Evaluate tuning set (fast dual-crop)                    │
│  2. Update EMA metrics (dice, recall, precision)            │
│  3. Detect trend (improving/degrading/stable/recall_drop)   │
│  4. Calculate new p_anatomy based on trend                  │
│  5. Check best Dice tracking, revert if needed              │
│  6. Apply new probabilities to training dataloader          │
└─────────────────────────────────────────────────────────────┘
```

---

## Evaluation Modes

### Behavioral Differences

| Aspect | Training | Tuning Set Eval | Test Set Eval |
|--------|----------|-----------------|---------------|
| **Purpose** | Learn features | Guide adaptive strategy | Select best checkpoint |
| **Mode** | N/A | Dual-crop (default) | Sliding-window (default) |
| **Speed** | N/A | Fast (~2-5s) | Slow (~2-5min) |
| **Accuracy** | N/A | Approximate | High |
| **Frequency** | Every epoch | Every N epochs | Every N epochs |
| **Blocking** | N/A | Sync (main process) | Async (subprocess) |

### Dual-Crop Evaluation (Fast)

For each sample, extracts **two patches**:
1. **Anatomy-centered**: Evaluates false positive control
2. **Tumor-centered**: Evaluates tumor detection (recall)

```python
# Example: 25 samples → 50 patches evaluated
tuning_eval_mode = 'dual_crop'  # Default for tuning set
```

**Pros**: Very fast, good for frequent evaluation
**Cons**: Limited coverage, may miss edge cases

### Sliding-Window Evaluation (Accurate)

Full-volume inference using sliding window with box filtering:

1. **Pre-compute** all possible sliding window boxes for the volume
2. **Classify** boxes by foreground content (tumor / anatomy-only / background)
3. **Infer** only on foreground-containing boxes
4. **Aggregate** predictions with Gaussian weighting
5. **Compute metrics** on the union of processed regions

```python
# Example: 478×470×470 volume with 192³ patches
# Total boxes: 64, Foreground: 37 (58%), Background: 27 (skipped)
test_eval_mode = 'sliding_window'  # Default for test set
```

**Pros**: Accurate, realistic evaluation
**Cons**: Time-consuming (mitigated by async)

---

## Async vs Sync Test Evaluation

### Async Mode (Default)

The trainer supports **multi-subprocess evaluation** where multiple test evaluations can run concurrently on separate GPUs.

**Key Points**:
- Training continues while test evaluation runs in background
- Uses separate GPUs (configurable via `--backup_gpu_ids`, comma-separated)
- Up to `MAX_CONCURRENT_SUBPROCESSES` (default: 3) can run in parallel
- Subprocesses are daemon: auto-terminate if main process crashes
- **Training completion**: `on_train_end()` waits for all running async evals to finish before exiting
- **REQUIRED**: You must set `--backup_gpu_ids` to use async evaluation

For detailed architecture and diagrams, see [MULTIPROCESS_EVALUATION_MECHANISM.md](./MULTIPROCESS_EVALUATION_MECHANISM.md).

### Sync Mode

```python
ASYNC_TEST_EVAL = False  # Set in tuning_set_trainer/constants.py
```

Test evaluation runs in the main process, blocking training until complete.
Use this if you have only one GPU or prefer simpler debugging.

---

## Top-K Checkpoint Management

Saves the **top 5 best checkpoints** based on single-epoch tumor Dice:

```
checkpoint_best_nnUNetResEncUNetLPlans.pth      # Rank 1 (best)
checkpoint_best_nnUNetResEncUNetLPlans.pth.2    # Rank 2
checkpoint_best_nnUNetResEncUNetLPlans.pth.3    # Rank 3
checkpoint_best_nnUNetResEncUNetLPlans.pth.4    # Rank 4
checkpoint_best_nnUNetResEncUNetLPlans.pth.5    # Rank 5
```

When a new checkpoint qualifies:
1. Find insertion rank
2. Shift lower-ranked checkpoints down (rank 5 is deleted)
3. Save new checkpoint at its rank

---

## Configuration Options

### Module-Level Constants (in `tuning_set_trainer/constants.py`)

These constants control trainer behavior and can be modified in the constants file:

```python
# Pre-training evaluation (debugging)
ENABLE_PRE_TRAINING_EVAL = True            # Test set pre-training eval
ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET = True  # Tuning set pre-training eval
PRE_TRAINING_EVAL_MAX_SAMPLES = 5          # Limit samples for sanity check

# Async test evaluation
ASYNC_TEST_EVAL = True                     # Enable async subprocess
MAX_CONCURRENT_SUBPROCESSES = 3            # Max parallel subprocesses
SUBPROCESS_TIMEOUT_HOURS = 4.0             # Hard timeout per subprocess

# Error handling
ASYNC_EVAL_EXIT_ON_FAILURE = False         # Continue on subprocess failure

# Sanity check parallelization
SANITY_CHECK_NUM_WORKERS = 16              # Workers for tumor presence check
```

### Instance-Level Defaults (in `__init__`)

These can be modified after trainer creation or via checkpoint loading:

```python
# Evaluation modes (separate for tuning vs test)
self.tuning_eval_mode = 'dual_crop'      # Fast for adaptive decisions
self.test_eval_mode = 'sliding_window'   # Accurate for checkpoints
self.sliding_window_step_size = 0.5      # 50% overlap

# Evaluation frequency
self.eval_every_n_epochs = 5             # Run eval every N epochs
self.tuning_max_patches_per_sample = 5   # Limit for fast tuning eval
self.test_max_patches_per_sample = None  # Full evaluation for test

# Checkpoint management
self.top_k_checkpoints = 5               # Number of best checkpoints

# Synthetic data handling (set via --ignore_synthetic or directly)
self.max_synthetic_ratio = 0.5           # Default 50% cap
```

### Command-Line Arguments

```bash
nnUNetv2_train DATASET CONFIG FOLD \
    -tr nnUNetTrainer_WithTuningSet \
    --main_gpu_id 0 \              # GPU for main training process
    --backup_gpu_ids "1,2" \       # GPUs for async test evaluation (comma-separated, REQUIRED)
    --pattern_original_samples "liver_\\d+" \  # Regex to identify original samples
    --ignore_synthetic \           # Exclude synthetic data (sets max_synthetic_ratio=0)
    --ignore_existing_best \       # Reset checkpoint ranking
    --skip_val \                   # Skip final validation after training
    --integration_test             # Run quick verification with minimal samples
```

**Important Notes**:
- `--backup_gpu_ids` is **REQUIRED** when using async evaluation (the default). Without it, training will raise an error.
- `--pattern_original_samples` must be a valid Python regex. Invalid patterns will raise a helpful error message.

---

## Usage

### Basic Training

```bash
# Note: --backup_gpu_ids is REQUIRED when ASYNC_TEST_EVAL=True (default)
nnUNetv2_train 602 3d_fullres 0 \
    -tr nnUNetTrainer_WithTuningSet \
    -p nnUNetResEncUNetLPlans \
    --backup_gpu_ids "1"
```

### With GPU Assignment (Required for Async Eval)

```bash
nnUNetv2_train 602 3d_fullres 0 \
    -tr nnUNetTrainer_WithTuningSet \
    -p nnUNetResEncUNetLPlans \
    --main_gpu_id 0 \
    --backup_gpu_ids "1,2"
```

### With Synthetic Data Filtering

```bash
nnUNetv2_train 602 3d_fullres 0 \
    -tr nnUNetTrainer_WithTuningSet \
    -p nnUNetResEncUNetLPlans \
    --pattern_original_samples 'liver_\d+' \
    --ignore_synthetic
```

### Continue Training (Load Weights, Reset Checkpoint Ranking)

```bash
nnUNetv2_train 602 3d_fullres 0 \
    -tr nnUNetTrainer_WithTuningSet \
    -p nnUNetResEncUNetLPlans \
    --c \
    --checkpoint_path /path/to/checkpoint_best_nnUNetResEncUNetLPlans.pth \
    --ignore_existing_best
```

### Skip Final Validation

Use `--skip_val` when you plan to run validation separately (e.g., on external test set):

```bash
nnUNetv2_train 602 3d_fullres 0 \
    -tr nnUNetTrainer_WithTuningSet \
    -p nnUNetResEncUNetLPlans \
    --skip_val
```

---

## Integration Test Mode

Quick verification of the entire pipeline with minimal resources:

```bash
nnUNetv2_train 602 3d_fullres 0 \
    -tr nnUNetTrainer_WithTuningSet \
    --integration_test
```

**What it does**:
- Limits samples to 5 per split (train/tuning/test)
- Reduces epochs to 10
- Reduces iterations per epoch to 10
- Sets evaluation every 2 epochs
- Creates separate output folder with `_integration_test` suffix
- Removes existing integration test folder before starting

**Use for**: Verifying code changes work end-to-end before full training.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     nnUNetTrainer_WithTuningSet                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  Training Set   │    │   Tuning Set    │    │    Test Set     │     │
│  │   (remaining)   │    │  (= test size)  │    │  (val fold)     │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ 3-Way Centering │    │   Dual-Crop     │    │ Sliding Window  │     │
│  │   DataLoader    │    │   Evaluation    │    │   Evaluation    │     │
│  │                 │    │   (SYNC)        │    │   (ASYNC)       │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           │                      ▼                      ▼               │
│           │             ┌─────────────────┐    ┌─────────────────┐     │
│           │             │    Dynamic      │    │    Top-K        │     │
│           │             │   Sampling      │    │   Checkpoint    │     │
│           │             │   Strategy      │    │   Management    │     │
│           │             └────────┬────────┘    └─────────────────┘     │
│           │                      │                                      │
│           │                      ▼                                      │
│           │             ┌─────────────────┐                            │
│           └────────────▶│ Update p_random │                            │
│                         │ p_anatomy       │                            │
│                         │ p_tumor         │                            │
│                         └─────────────────┘                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### CUDA Out of Memory During Evaluation

- Metrics are computed on CPU by default
- Reduce `sliding_window_step_size` for fewer boxes
- Use `tuning_max_patches_per_sample = 5` (default) to limit tuning patches
- Ensure `--backup_gpu_ids` contains GPUs different from `--main_gpu_id`

### Subprocess Not Starting or "backup_gpu_ids not set" Error

- You **MUST** set `--backup_gpu_ids` when using async evaluation (default)
- Example: `--backup_gpu_ids "1,2"` (comma-separated GPU IDs)
- Verify `ASYNC_TEST_EVAL = True` in `tuning_set_trainer/constants.py`
- Check for orphaned temp files in output folder (auto-cleaned on startup)

### Training Appears Stuck

- Check if async subprocess is still running (look for `[MP]` prefix in logs)
- Verify training tqdm progress bar is updating
- Non-eval epochs use cached metrics (no evaluation runs)

### Invalid Regex Pattern Error

If you see an error like:
```
ValueError: Invalid regex pattern in --pattern_original_samples: '...'
Regex error: ...
```

- Check your regex syntax (Python regex, not glob patterns)
- Common issues: unescaped special characters, unbalanced brackets
- Example valid patterns: `liver_\d+`, `case_[0-9]+`, `patient\d{3}`

### Training Ends But Async Eval Still Running

This is expected behavior. The trainer will:
1. Detect async eval is still running
2. Wait for it to complete (up to `ASYNC_EVAL_TIMEOUT`)
3. Handle the result and save checkpoint if qualified
4. Then exit cleanly

Look for: `WAITING FOR ASYNC TEST EVALUATION TO COMPLETE` in logs.

---

## References

- Base trainer: `nnUNetTrainer` in `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- Sliding window inference: `predict_from_raw_data.py`
- Label conventions: 0=background, 1=anatomy (organ), 2=tumor, 3+=other (trained but not centered)
