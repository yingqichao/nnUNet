"""
Configuration constants for nnUNetTrainer_WithTuningSet.

These constants control the behavior of the trainer and can be modified
before training to customize the pipeline.
"""

# =============================================================================
# PRE-TRAINING EVALUATION
# =============================================================================
# Run evaluation before first epoch to catch bugs early

# Test set pre-training eval (sliding window)
ENABLE_PRE_TRAINING_EVAL = True

# Tuning set pre-training eval (dual-crop)
ENABLE_PRE_TRAINING_EVAL_ON_TUNING_SET = True

# Limit samples for faster pre-training sanity check
PRE_TRAINING_EVAL_MAX_SAMPLES = 5


# =============================================================================
# ASYNC TEST EVALUATION
# =============================================================================
# Run test set evaluation in subprocess(es) to allow training to continue

# Set to False to use synchronous evaluation
ASYNC_TEST_EVAL = True

# Maximum concurrent evaluation subprocesses
# Actual max is min(MAX_CONCURRENT_SUBPROCESSES, len(backup_gpu_ids))
MAX_CONCURRENT_SUBPROCESSES = 3

# Subprocess timeout (hours) - HARD limit, must complete ALL samples
# If exceeded, subprocess terminates and reports failure
SUBPROCESS_TIMEOUT_HOURS = 4.0

# Alias for backward compatibility
ASYNC_EVAL_TIMEOUT_HOURS = SUBPROCESS_TIMEOUT_HOURS


# =============================================================================
# ERROR HANDLING
# =============================================================================

# Whether to exit main process when subprocess evaluation fails
# If True, main process will raise RuntimeError and exit gracefully
# If False, main process logs error and continues training
ASYNC_EVAL_EXIT_ON_FAILURE = False


# =============================================================================
# SANITY CHECK SETTINGS
# =============================================================================

# Number of parallel workers for checking tumor presence in samples
SANITY_CHECK_NUM_WORKERS = 32


# =============================================================================
# LEGACY CONSTANTS (kept for backward compatibility)
# =============================================================================
# These are deprecated and not used in multi-subprocess mode

# Deprecated: use --backup_gpu_ids instead
ASYNC_EVAL_GPU_ID = None

# Deprecated: use SUBPROCESS_TIMEOUT_HOURS instead
ASYNC_EVAL_TIMEOUT = 3600

# Must complete ALL samples (no partial results)
ASYNC_EVAL_MIN_SUCCESS_RATE = 1.0

# Deprecated: use SUBPROCESS_TIMEOUT_HOURS instead
ASYNC_EVAL_TIMEOUT_MINUTES = 240
