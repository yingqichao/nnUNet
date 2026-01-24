# Multiprocess Evaluation Mechanism

This document describes the asynchronous multi-subprocess evaluation architecture used in `nnUNetTrainer_WithTuningSet` for parallel test set evaluation during training.

## Overview

The trainer uses a **multi-subprocess architecture** where:
- **Main Process**: Handles training, tuning evaluation, checkpoint management, and visualization folder management
- **Evaluation Subprocesses**: Run sliding window inference on the test set in parallel, generate metrics and visualization figures

This design allows training to continue while multiple test evaluations run concurrently on separate GPUs.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN PROCESS                                    │
│                         (Training + Checkpoint Mgmt)                         │
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │  Training   │──▶│   Tuning    │──▶│   Spawn     │──▶│    Poll     │     │
│  │    Loop     │   │    Eval     │   │ Subprocess  │   │  Results    │     │
│  │  (GPU 0)    │   │ (dual-crop) │   │             │   │             │     │
│  └─────────────┘   └─────────────┘   └──────┬──────┘   └──────┬──────┘     │
│                                             │                  │            │
│                                             ▼                  ▼            │
│                                    ┌─────────────────────────────────┐      │
│                                    │   Checkpoint Ranking & Saving   │      │
│                                    │   (Top-K management in MAIN)    │      │
│                                    └─────────────────────────────────┘      │
└────────────────────────────────────────────┬────────────────────────────────┘
                                             │
            ┌────────────────────────────────┼────────────────────────────────┐
            │                                │                                │
            ▼                                ▼                                ▼
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│   SUBPROCESS 1        │    │   SUBPROCESS 2        │    │   SUBPROCESS 3        │
│   (GPU 1)             │    │   (GPU 2)             │    │   (GPU 3)             │
│                       │    │                       │    │                       │
│  ┌─────────────────┐  │    │  ┌─────────────────┐  │    │  ┌─────────────────┐  │
│  │ Load Checkpoint │  │    │  │ Load Checkpoint │  │    │  │ Load Checkpoint │  │
│  │  from temp file │  │    │  │  from temp file │  │    │  │  from temp file │  │
│  └────────┬────────┘  │    │  └────────┬────────┘  │    │  └────────┬────────┘  │
│           ▼           │    │           ▼           │    │           ▼           │
│  ┌─────────────────┐  │    │  ┌─────────────────┐  │    │  ┌─────────────────┐  │
│  │ Sliding Window  │  │    │  │ Sliding Window  │  │    │  │ Sliding Window  │  │
│  │   Inference     │  │    │  │   Inference     │  │    │  │   Inference     │  │
│  │ (all test keys) │  │    │  │ (all test keys) │  │    │  │ (all test keys) │  │
│  └────────┬────────┘  │    │  └────────┬────────┘  │    │  └────────┬────────┘  │
│           ▼           │    │           ▼           │    │           ▼           │
│  ┌─────────────────┐  │    │  ┌─────────────────┐  │    │  ┌─────────────────┐  │
│  │  Create Vis     │  │    │  │  Create Vis     │  │    │  │  Create Vis     │  │
│  │   Figures       │  │    │  │   Figures       │  │    │  │   Figures       │  │
│  │ (10 samples,    │  │    │  │ (10 samples,    │  │    │  │ (10 samples,    │  │
│  │  8 slices each) │  │    │  │  8 slices each) │  │    │  │  8 slices each) │  │
│  └────────┬────────┘  │    │  └────────┬────────┘  │    │  └────────┬────────┘  │
│           ▼           │    │           ▼           │    │           ▼           │
│  ┌─────────────────┐  │    │  ┌─────────────────┐  │    │  ┌─────────────────┐  │
│  │  Write Result   │  │    │  │  Write Result   │  │    │  │  Write Result   │  │
│  │   JSON File     │  │    │  │   JSON File     │  │    │  │   JSON File     │  │
│  │  (metrics +     │  │    │  │  (metrics +     │  │    │  │  (metrics +     │  │
│  │ timestamp + vis)│  │    │  │ timestamp + vis)│  │    │  │ timestamp + vis)│  │
│  └─────────────────┘  │    │  └─────────────────┘  │    │  └─────────────────┘  │
└───────────────────────┘    └───────────────────────┘    └───────────────────────┘
```

## Key Design Principles

### 1. Main Process Handles ALL Checkpoint Management

To avoid race conditions and file locking issues:
- Subprocesses **only compute metrics** and write results to JSON
- Main process **reads results** and decides if checkpoint qualifies for top-K
- Main process **saves/renames/deletes** checkpoint files
- Timestamp in result JSON ensures ordering when multiple results arrive

### 2. GPU Isolation via Environment Variables

```
Before Spawn:          After Spawn:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Main Process   │    │  Main Process   │    │   Subprocess    │
│                 │    │                 │    │                 │
│ CUDA_VISIBLE=0  │───▶│ CUDA_VISIBLE=0  │    │ CUDA_VISIBLE=1  │
│ (original)      │    │ (restored)      │    │ (inherited)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                                              ▲
       │  1. Save original CUDA_VISIBLE_DEVICES       │
       │  2. Set CUDA_VISIBLE_DEVICES=1               │
       │  3. Spawn subprocess ─────────────────────────
       │  4. Restore original CUDA_VISIBLE_DEVICES
       ▼
```

### 3. Daemon Processes for Clean Shutdown

All subprocesses are spawned with `daemon=True`:
- If main process exits/crashes, subprocesses are automatically terminated
- Prevents orphaned GPU processes consuming resources

## Timeline: Typical Training Run with Multi-Subprocess Evaluation

```
TIME    MAIN PROCESS (GPU 0)              SUBPROCESS 1 (GPU 1)    SUBPROCESS 2 (GPU 2)
─────   ──────────────────────────────    ────────────────────    ────────────────────
T+0     on_train_start()
        ├─ Cleanup orphaned temp files
        ├─ Pre-training tuning eval
        └─ Pre-training test eval (optional)

T+1     EPOCH 0: Training loop
        └─ 250 iterations...

T+2     on_validation_epoch_end()
        ├─ Poll subprocesses (none yet)
        ├─ Tuning eval (dual-crop)
        ├─ Save temp checkpoint (epoch 0)
        └─ Spawn subprocess for epoch 0 ──────▶ START
                                               ├─ Load temp ckpt
                                               └─ Begin sliding window...

T+3     EPOCH 1: Training loop
        └─ 250 iterations...                   [RUNNING]

T+4     on_validation_epoch_end()
        ├─ Poll subprocesses
        │   └─ Epoch 0 not done yet
        ├─ Tuning eval (dual-crop)
        ├─ Save temp checkpoint (epoch 1)
        └─ Spawn subprocess for epoch 1 ────────────────────────▶ START
                                                                  ├─ Load temp ckpt
                                                                  └─ Begin sliding...

T+5     EPOCH 2: Training loop                 [RUNNING]          [RUNNING]
        └─ 250 iterations...

T+6     on_validation_epoch_end()
        ├─ Poll subprocesses
        │   ├─ Epoch 0 DONE! ◀──────────────── Write result JSON + vis folder
        │   │   └─ _process_subprocess_result()
        │   │       ├─ Read metrics from JSON
        │   │       ├─ Check if qualifies for top-K
        │   │       ├─ Save checkpoint (if qualified)
        │   │       ├─ Rename vis folder to vis_best_* (if qualified)
        │   │       ├─ Update top_k_dices
        │   │       └─ Cleanup temp files (incl. vis folder if not qualified)
        │   └─ Epoch 1 not done yet
        ├─ Tuning eval (dual-crop)
        ├─ Save temp checkpoint (epoch 2)
        └─ Spawn subprocess for epoch 2 ──────▶ START            [RUNNING]
                                               ├─ Load temp ckpt
                                               └─ Begin sliding...

        ... (continues) ...

T+END   on_train_end()
        ├─ Wait for all running subprocesses
        ├─ Process all remaining results
        └─ Final cleanup
```

## Process Communication

### Main → Subprocess (at spawn time)

Information passed via `kwargs` to `_async_test_eval_worker`:

```python
{
    'temp_ckpt_path': '/path/to/checkpoint_eval_temp_epoch5.pth',
    'result_path': '/path/to/eval_result_epoch5.json',
    'preprocessed_folder': '/path/to/preprocessed/',
    'test_keys': ['case_001', 'case_002', ...],
    'eval_config': {
        'patch_size': [128, 128, 128],
        'step_size': 0.5,
        'timeout_hours': 4.0,
        'temp_ckpt_path': '/path/to/checkpoint_eval_temp_epoch5.pth',
        ...
    },
    'gpu_id': 1,  # For logging only
    ...
}
```

### Subprocess → Main (via result JSON file)

Result JSON written by subprocess:

```json
{
    "success": true,
    "epoch": 5,
    "dice_per_class": [0.95, 0.72],
    "recall_per_class": [98.5, 85.2],
    "precision_per_class": [97.1, 78.3],
    "tumor_dice": 0.72,
    "total_samples": 50,
    "successful_samples": 50,
    "success_rate": 1.0,
    "temp_ckpt_path": "/path/to/checkpoint_eval_temp_epoch5.pth",
    "vis_folder_path": "/path/to/vis_eval_temp_epoch5",
    "timestamp": 1706054321.123456
}
```

### Subprocess Tracking in Main Process

```python
@dataclass
class SubprocessInfo:
    process: mp.Process      # The process object
    pid: int                 # Process ID
    epoch: int               # Which epoch this evaluates
    gpu_id: int              # Which GPU it's using
    result_path: str         # Where result JSON will be written
    temp_ckpt_path: str      # Path to temp checkpoint
    start_time: float        # When subprocess was spawned

# Main process tracks all running subprocesses
self._running_subprocesses: List[SubprocessInfo] = []
```

## Key Methods

### Main Process Methods

| Method | Purpose |
|--------|---------|
| `_spawn_async_test_eval()` | Save temp ckpt, spawn subprocess, track in list |
| `_poll_running_subprocesses()` | Check all subprocesses, collect completed results |
| `_process_subprocess_result()` | Handle result, manage checkpoints, cleanup |
| `_process_pending_eval_checkpoints()` | Spawn queued evaluations when slots become available |
| `_get_available_gpu()` | Find unused GPU from backup_gpu_ids |
| `_wait_for_subprocess_slot()` | Block until subprocess slot available |
| `_cleanup_subprocess_files()` | Remove temp checkpoint and result JSON |

### Subprocess Function

| Function | Purpose |
|----------|---------|
| `_async_test_eval_worker()` | Module-level function that runs sliding window inference |

## Configuration

### Command Line Arguments

```bash
nnUNetv2_train DATASET CONFIG FOLD \
    -tr nnUNetTrainer_WithTuningSet \
    --main_gpu_id 0 \
    --backup_gpu_ids "1,2,3"
```

### Constants (in `tuning_set_trainer/constants.py`)

```python
ASYNC_TEST_EVAL = True                    # Enable async evaluation
MAX_CONCURRENT_SUBPROCESSES = 3           # Max parallel subprocesses
SUBPROCESS_TIMEOUT_HOURS = 4.0            # Hard timeout per subprocess
ASYNC_EVAL_EXIT_ON_FAILURE = False        # Continue training on failure
SANITY_CHECK_NUM_WORKERS = 16             # Workers for tumor presence check
```

**Important**: `--backup_gpu_ids` is **REQUIRED** when `ASYNC_TEST_EVAL = True`. The trainer will raise an error if backup GPUs are not configured.

## Error Handling

### Subprocess Timeout

- Each subprocess has a **4-hour hard timeout**
- Must complete ALL test samples within this limit
- If timeout exceeded, subprocess writes failure result and exits
- Main process logs error and optionally continues training

### Subprocess Crash

- If subprocess crashes without writing result JSON:
  - `_poll_running_subprocesses()` detects `process.is_alive() == False` without result file
  - Reports as failure with "Process finished without result file"

### Main Process Exit

- All subprocesses are `daemon=True`
- When main process exits, all subprocesses are automatically terminated
- Prevents orphaned GPU processes

## File Artifacts

### Temporary Files (auto-cleaned)

| File Pattern | Created By | Cleaned By |
|--------------|------------|------------|
| `checkpoint_eval_temp_epoch{N}.pth` | Main (before spawn) | Main (after result processed) |
| `eval_result_epoch{N}.json` | Subprocess (after eval) | Main (after result processed) |
| `vis_eval_temp_epoch{N}/` | Subprocess (during eval) | Main (after result processed, if not top-k) |

### Permanent Files (checkpoints & visualizations)

| File Pattern | Created By | Notes |
|--------------|------------|-------|
| `checkpoint_best_{plans}.pth` | Main (rank 1 checkpoint) | |
| `checkpoint_best_{plans}.pth.{2-5}` | Main (rank 2-5 checkpoints) | |
| `vis_best_{plans}/` | Main (rank 1 vis folder) | Renamed from temp |
| `vis_best_{plans}.{2-5}/` | Main (rank 2-5 vis folders) | Renamed from temp |

## Comparison: Single vs Multi-Subprocess

| Aspect | Single Subprocess (Old) | Multi-Subprocess (New) |
|--------|------------------------|------------------------|
| Concurrent evaluations | 1 | Up to 3 |
| GPU utilization | 1 backup GPU | Multiple backup GPUs |
| Training blocking | Often waits for eval | Rarely waits |
| Checkpoint management | Subprocess saved ckpts | Main process only |
| Race conditions | Possible | Eliminated |
| Complexity | Lower | Higher |

## Visualization

### Overview

Each evaluation subprocess generates visualization figures for a random subset of test samples. This helps understand model predictions at each checkpoint.

### Visualization Settings (in `tuning_set_trainer/multiprocess_eval.py`)

```python
VIS_NUM_SAMPLES = 10           # Number of samples to visualize
VIS_MAX_SLICES_PER_SAMPLE = 8  # Max slices with tumor per sample
VIS_TUMOR_LABEL = 2            # Label for tumor class
```

### Figure Format

For each selected sample, a multi-row, 3-column figure is created:

| Column | Content |
|--------|---------|
| 1 | Original slice (grayscale) |
| 2 | Predicted tumor mask (blue overlay, α=0.5) |
| 3 | GT tumor mask (green overlay, α=0.5) |

**Note**: Only tumor masks are overlaid (not anatomy predictions/GT).

### Visualization Flow

```
SUBPROCESS                          MAIN PROCESS
──────────                          ────────────
1. Select 10 random samples
2. For each sample:
   - Find slices with tumor GT
   - Select up to 8 slices
   - Create 3-column figure
   - Save as PNG
3. Report vis_folder_path in JSON
                                    4. Check if checkpoint qualifies for top-k
                                    5. IF qualified:
                                       - Rename vis folder to vis_best_*
                                       - Keep alongside checkpoint
                                    6. IF NOT qualified:
                                       - Delete temp vis folder
```

### File Naming

| Temp Folder | Final Folder (if top-k) |
|-------------|-------------------------|
| `vis_eval_temp_epoch5/` | `vis_best_{plans}` (rank 1) |
| `vis_eval_temp_epoch5/` | `vis_best_{plans}.2` (rank 2) |
| ... | ... |

Inside each folder:
```
vis_best_plans/
├── case_001.png      # 8-row × 3-column figure
├── case_002.png
├── ...
└── case_010.png      # Up to 10 samples
```

## Troubleshooting

### "All subprocess slots in use, waiting..."

- All 3 subprocesses are running
- Main process polls every 10 seconds for completion
- If persists > 1 hour, consider: more GPUs, faster eval, or reduce eval frequency

### "Subprocess crashed without result file"

- Check subprocess GPU memory (OOM?)
- Check CUDA errors in subprocess output
- Verify temp checkpoint is valid

### Orphaned temp files on startup

- `_cleanup_orphaned_temp_files()` runs in `on_train_start()`
- Cleans up files from previous crashed runs
- Patterns:
  - `checkpoint_eval_temp_epoch*.pth`
  - `eval_result_epoch*.json`
  - `vis_eval_temp_epoch*/` (folders)

### Visualization folder not appearing

- Check subprocess logs for `[VIS]` messages
- Verify matplotlib is available in subprocess environment
- Check if any samples have tumor GT (required for visualization)
