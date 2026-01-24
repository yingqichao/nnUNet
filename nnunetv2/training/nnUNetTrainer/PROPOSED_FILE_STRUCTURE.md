# Proposed File Structure for nnUNetTrainer_WithTuningSet

The current `nnUNetTrainer_with_tuning_set.py` (~5600 lines) can be split into modular components for better maintainability.

## Current Structure Analysis

| Line Range | Content | Lines |
|------------|---------|-------|
| 1-167 | Module docstring, imports, constants, SubprocessInfo | ~170 |
| 170-610 | `_async_test_eval_worker()` function | ~440 |
| 613-690 | `_check_single_sample_for_tumor()` function | ~80 |
| 669-855 | Metric utility functions (5 functions) | ~190 |
| 856-906 | `SamplingConfig` dataclass | ~50 |
| 908-1354 | `DynamicSamplingStrategy` class | ~450 |
| 1356-1604 | `nnUNetDataLoader3WayCentering` class | ~250 |
| 1605-1787 | `nnUNetDataLoaderDualCropEval` class | ~180 |
| 1788-5589 | `nnUNetTrainer_WithTuningSet` class | ~3800 |

## Proposed New Structure

```
nnUNet/nnunetv2/training/nnUNetTrainer/
├── nnUNetTrainer.py                           # Original base trainer (unchanged)
├── nnUNetTrainer_with_tuning_set.py           # Main trainer class (~2000 lines after split)
│
├── tuning_set_trainer/                        # NEW: Package for helper modules
│   ├── __init__.py                            # Exports all public classes/functions
│   ├── constants.py                           # Configuration flags and constants
│   ├── multiprocess_eval.py                   # Async evaluation subprocess worker
│   ├── metric_utils.py                        # Metric computation utilities
│   ├── dynamic_sampling.py                    # Adaptive sampling strategy
│   └── custom_dataloaders.py                  # 3-way centering & dual-crop dataloaders
│
├── MULTIPROCESS_EVALUATION_MECHANISM.md       # Documentation (already created)
├── PROPOSED_FILE_STRUCTURE.md                 # This file
└── README_TUNING_SET_TRAINER.md               # User-facing documentation
```

## Detailed File Breakdown

### 1. `tuning_set_trainer/constants.py` (~50 lines)

```python
"""Configuration constants for nnUNetTrainer_WithTuningSet."""

# Pre-training evaluation flags
ENABLE_PRE_TRAINING_EVAL = True
ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET = True
PRE_TRAINING_EVAL_MAX_SAMPLES = 5

# Async test evaluation
ASYNC_TEST_EVAL = True
MAX_CONCURRENT_SUBPROCESSES = 3
SUBPROCESS_TIMEOUT_HOURS = 4.0
ASYNC_EVAL_TIMEOUT_HOURS = SUBPROCESS_TIMEOUT_HOURS
ASYNC_EVAL_EXIT_ON_FAILURE = False

# Legacy constants (backward compatibility)
ASYNC_EVAL_GPU_ID = None
ASYNC_EVAL_TIMEOUT = 3600
ASYNC_EVAL_MIN_SUCCESS_RATE = 1.0
ASYNC_EVAL_TIMEOUT_MINUTES = 240
```

### 2. `tuning_set_trainer/multiprocess_eval.py` (~550 lines)

```python
"""
Multiprocess evaluation worker functions for async test set evaluation.

These functions are at module level to be picklable for multiprocessing.
"""
from dataclasses import dataclass
from typing import Any

@dataclass
class SubprocessInfo:
    """Information about a running evaluation subprocess."""
    process: Any
    pid: int
    epoch: int
    gpu_id: int
    result_path: str
    temp_ckpt_path: str
    start_time: float

def _async_test_eval_worker(
    temp_ckpt_path: str,
    result_path: str,
    preprocessed_folder: str,
    test_keys: list,
    top_k_dices: list,
    top_k_checkpoints: int,
    output_folder: str,
    plans_identifier: str,
    eval_config: dict,
    gpu_id: int = None,
):
    """
    Worker function for async test evaluation subprocess.
    Must be at module level to be picklable.
    """
    # ... implementation ...

def _check_single_sample_for_tumor(args: tuple) -> tuple:
    """
    Worker function for parallel tumor sanity check.
    Used by multiprocessing.Pool.
    """
    # ... implementation ...
```

### 3. `tuning_set_trainer/metric_utils.py` (~200 lines)

```python
"""
Metric computation utilities for evaluation.

Pure functions for computing Dice scores, TP/FP/FN/TN, etc.
"""
import numpy as np
from typing import Dict, Optional, Tuple

def compute_dice_score_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score between two binary numpy arrays."""
    # ... implementation ...

def compute_tp_fp_fn_tn_np(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute confusion matrix components."""
    # ... implementation ...

def compute_class_metrics_np(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    class_idx: int,
    simulate_perfect_anatomy: bool = False
) -> Dict[str, Optional[float]]:
    """Compute all metrics for a single class."""
    # ... implementation ...

def compute_anatomy_dice_np(pred_seg: np.ndarray, gt_seg: np.ndarray) -> float:
    """Compute Dice for anatomy class (label 1)."""
    # ... implementation ...

def soft_pred_to_segmentation_np(
    soft_pred: np.ndarray,
    apply_softmax: bool = False
) -> np.ndarray:
    """Convert soft predictions to hard segmentation."""
    # ... implementation ...
```

### 4. `tuning_set_trainer/dynamic_sampling.py` (~500 lines)

```python
"""
Dynamic sampling strategy for adaptive patch centering.

Adjusts the probability distribution of patch centering modes
(random/anatomy/tumor) based on training performance metrics.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

@dataclass
class SamplingConfig:
    """Configuration for dynamic sampling strategy."""
    min_anatomy_prob: float = 0.10
    max_anatomy_prob: float = 0.50
    min_context_prob: float = 0.15
    alpha: float = 0.7
    convergence_threshold: float = 0.005
    convergence_window: int = 5
    adjustment_rate: float = 0.05
    decay_rate: float = 0.02
    smoothing_factor: float = 0.5
    recall_drop_threshold: float = 0.03
    precision_drop_threshold: float = 0.03
    dice_revert_threshold: float = 0.05
    exploration_margin: float = 0.02

class DynamicSamplingStrategy:
    """
    Adaptive sampling strategy that adjusts centering probabilities
    based on tuning set performance metrics.
    """
    def __init__(self, config: Optional[SamplingConfig] = None, 
                 log_fn: Optional[Callable[[str], None]] = None):
        # ... implementation ...
    
    def update(self, metrics_dict: Dict[str, float]) -> Tuple[float, float]:
        """Update probabilities based on new metrics."""
        # ... implementation ...
    
    # ... other methods ...
```

### 5. `tuning_set_trainer/custom_dataloaders.py` (~450 lines)

```python
"""
Custom DataLoaders for 3-way centering and dual-crop evaluation.
"""
import numpy as np
from typing import List, Tuple, Union, Dict

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager

class nnUNetDataLoader3WayCentering(nnUNetDataLoader):
    """
    DataLoader with configurable 3-way patch centering probabilities.
    
    For each sample in batch, randomly choose centering mode:
    - Random: Center on any random voxel
    - Anatomy: Center on random anatomy voxel (label 1)
    - Tumor: Center on random tumor voxel (label 2)
    """
    BACKGROUND_LABEL = 0
    ANATOMY_LABEL = 1
    PRIMARY_TUMOR_LABEL = 2
    
    def __init__(self, data, batch_size, patch_size, final_patch_size,
                 label_manager, prob_random=0.0, prob_anatomy=0.67,
                 prob_tumor=0.33, sampling_probabilities=None,
                 pad_sides=None, transforms=None):
        # ... implementation ...
    
    def update_probabilities(self, prob_random, prob_anatomy, prob_tumor):
        """Update centering probabilities dynamically."""
        # ... implementation ...
    
    def generate_train_batch(self):
        """Generate batch with 3-way centering."""
        # ... implementation ...

class nnUNetDataLoaderDualCropEval(nnUNetDataLoader):
    """
    Evaluation DataLoader that creates TWO patches per sample.
    
    For each sample:
    - Patch 1: Centered on anatomy voxel → evaluates FP control
    - Patch 2: Centered on tumor voxel → evaluates tumor detection
    """
    ANATOMY_LABEL = 1
    TUMOR_LABEL = 2
    
    def __init__(self, data, batch_size, patch_size, final_patch_size,
                 label_manager, sampling_probabilities=None,
                 pad_sides=None, transforms=None):
        # ... implementation ...
    
    def generate_train_batch(self):
        """Generate dual-crop evaluation batch."""
        # ... implementation ...
```

### 6. `tuning_set_trainer/__init__.py` (~30 lines)

```python
"""
Tuning set trainer helper modules.

This package contains helper classes and functions for
nnUNetTrainer_WithTuningSet.
"""

from .constants import (
    ASYNC_TEST_EVAL,
    MAX_CONCURRENT_SUBPROCESSES,
    SUBPROCESS_TIMEOUT_HOURS,
    ASYNC_EVAL_TIMEOUT_HOURS,
    ASYNC_EVAL_EXIT_ON_FAILURE,
    ENABLE_PRE_TRAINING_EVAL,
    ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET,
    PRE_TRAINING_EVAL_MAX_SAMPLES,
)

from .multiprocess_eval import (
    SubprocessInfo,
    _async_test_eval_worker,
    _check_single_sample_for_tumor,
)

from .metric_utils import (
    compute_dice_score_np,
    compute_tp_fp_fn_tn_np,
    compute_class_metrics_np,
    compute_anatomy_dice_np,
    soft_pred_to_segmentation_np,
)

from .dynamic_sampling import (
    SamplingConfig,
    DynamicSamplingStrategy,
)

from .custom_dataloaders import (
    nnUNetDataLoader3WayCentering,
    nnUNetDataLoaderDualCropEval,
)
```

### 7. Updated `nnUNetTrainer_with_tuning_set.py` (~2000 lines)

```python
"""
nnUNet Trainer with separate tuning set for adaptive training strategies.
"""
import numpy as np
import torch
# ... other imports ...

# Import from helper modules
from nnunetv2.training.nnUNetTrainer.tuning_set_trainer import (
    # Constants
    ASYNC_TEST_EVAL,
    MAX_CONCURRENT_SUBPROCESSES,
    SUBPROCESS_TIMEOUT_HOURS,
    ASYNC_EVAL_EXIT_ON_FAILURE,
    ENABLE_PRE_TRAINING_EVAL,
    ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET,
    PRE_TRAINING_EVAL_MAX_SAMPLES,
    
    # Classes
    SubprocessInfo,
    SamplingConfig,
    DynamicSamplingStrategy,
    nnUNetDataLoader3WayCentering,
    nnUNetDataLoaderDualCropEval,
    
    # Functions
    _async_test_eval_worker,
    _check_single_sample_for_tumor,
    compute_dice_score_np,
    compute_class_metrics_np,
)

class nnUNetTrainer_WithTuningSet(nnUNetTrainer):
    """
    nnUNet trainer with 3-way data split and adaptive sampling.
    
    Now much cleaner with helper modules extracted!
    """
    # ... implementation (only the trainer class methods) ...
```

## Benefits of This Split

| Benefit | Description |
|---------|-------------|
| **Readability** | Each file has a single responsibility |
| **Testability** | Pure functions in `metric_utils.py` are easy to unit test |
| **Reusability** | DataLoaders and metrics can be used by other trainers |
| **Maintainability** | Changes to sampling strategy don't affect evaluation code |
| **Code Review** | Smaller files are easier to review |
| **Import Speed** | Only import what you need |

## Migration Steps

1. Create `tuning_set_trainer/` directory
2. Create `__init__.py` with empty content
3. Move constants to `constants.py`
4. Move multiprocess functions to `multiprocess_eval.py`
5. Move metric functions to `metric_utils.py`
6. Move sampling classes to `dynamic_sampling.py`
7. Move dataloader classes to `custom_dataloaders.py`
8. Update imports in main trainer file
9. Test with integration test (`--integration_test`)

## Estimated Effort

| Task | Time |
|------|------|
| Create directory structure | 5 min |
| Extract constants | 10 min |
| Extract multiprocess functions | 20 min |
| Extract metric functions | 15 min |
| Extract sampling classes | 20 min |
| Extract dataloader classes | 20 min |
| Update imports | 30 min |
| Test and fix import issues | 30 min |
| **Total** | **~2.5 hours** |

## Notes

- All module-level functions (`_async_test_eval_worker`, `_check_single_sample_for_tumor`) 
  must remain in their own file for multiprocessing pickle support
- The main trainer class stays in `nnUNetTrainer_with_tuning_set.py` to maintain 
  compatibility with nnUNet's trainer discovery mechanism
- Consider adding `py.typed` marker for better IDE support
