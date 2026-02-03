"""
Tuning set trainer helper modules.

This package contains helper classes and functions for
nnUNetTrainer_WithTuningSet.

Modules:
- constants: Configuration flags and constants
- multiprocess_eval: Async evaluation subprocess worker functions
- metric_utils: Metric computation utilities
- dynamic_sampling: Adaptive sampling strategy
- custom_dataloaders: 3-way centering and dual-crop dataloaders
"""

# Configuration constants
from .constants import (
    # Pre-training evaluation
    ENABLE_PRE_TRAINING_EVAL,
    ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET,
    PRE_TRAINING_EVAL_MAX_SAMPLES,
    # Async test evaluation
    ASYNC_TEST_EVAL,
    MAX_CONCURRENT_SUBPROCESSES,
    SUBPROCESS_TIMEOUT_HOURS,
    ASYNC_EVAL_TIMEOUT_HOURS,
    ASYNC_EVAL_EXIT_ON_FAILURE,
    # Sanity check
    SANITY_CHECK_NUM_WORKERS,
    # Legacy constants
    ASYNC_EVAL_GPU_ID,
    ASYNC_EVAL_TIMEOUT,
    ASYNC_EVAL_MIN_SUCCESS_RATE,
    ASYNC_EVAL_TIMEOUT_MINUTES,
    MAX_SYNTHETIC_RATIO,
)

# Multiprocess evaluation
from .multiprocess_eval import (
    SubprocessInfo,
    _async_test_eval_worker,
    _check_single_sample_for_tumor,
    _create_visualization_figures,
    VIS_NUM_SAMPLES,
    VIS_MAX_SLICES_PER_SAMPLE,
    VIS_TUMOR_LABEL,
)

# Metric utilities
from .metric_utils import (
    compute_dice_score_np,
    compute_tp_fp_fn_tn_np,
    compute_class_metrics_np,
    compute_anatomy_dice_np,
    soft_pred_to_segmentation_np,
)

# Dynamic sampling strategy
from .dynamic_sampling import (
    SamplingConfig,
    DynamicSamplingStrategy,
)

# Custom dataloaders
from .custom_dataloaders import (
    nnUNetDataLoader3WayCentering,
    nnUNetDataLoaderDualCropEval,
)

# SwinUNETR builder (for --model_name swinunetr)
from .swinunetr_builder import (
    build_swinunetr,
    validate_patch_size_for_swinunetr,
    adjust_patch_size_for_swinunetr,
    get_swinunetr_default_config,
)


__all__ = [
    # Constants
    'ENABLE_PRE_TRAINING_EVAL',
    'ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET',
    'PRE_TRAINING_EVAL_MAX_SAMPLES',
    'ASYNC_TEST_EVAL',
    'MAX_CONCURRENT_SUBPROCESSES',
    'SUBPROCESS_TIMEOUT_HOURS',
    'ASYNC_EVAL_TIMEOUT_HOURS',
    'ASYNC_EVAL_EXIT_ON_FAILURE',
    'ASYNC_EVAL_GPU_ID',
    'ASYNC_EVAL_TIMEOUT',
    'ASYNC_EVAL_MIN_SUCCESS_RATE',
    'ASYNC_EVAL_TIMEOUT_MINUTES',
    # Multiprocess
    'SubprocessInfo',
    'SANITY_CHECK_NUM_WORKERS',
    '_async_test_eval_worker',
    '_check_single_sample_for_tumor',
    # Visualization
    '_create_visualization_figures',
    'VIS_NUM_SAMPLES',
    'VIS_MAX_SLICES_PER_SAMPLE',
    'VIS_TUMOR_LABEL',
    # Metrics
    'compute_dice_score_np',
    'compute_tp_fp_fn_tn_np',
    'compute_class_metrics_np',
    'compute_anatomy_dice_np',
    'soft_pred_to_segmentation_np',
    # Dynamic sampling
    'SamplingConfig',
    'DynamicSamplingStrategy',
    # Dataloaders
    'nnUNetDataLoader3WayCentering',
    'nnUNetDataLoaderDualCropEval',
    # SwinUNETR builder
    'build_swinunetr',
    'validate_patch_size_for_swinunetr',
    'adjust_patch_size_for_swinunetr',
    'get_swinunetr_default_config',
]
