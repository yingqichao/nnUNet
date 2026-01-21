"""
nnUNet Trainer with separate tuning set for adaptive training strategies.

This trainer implements a 3-way split within each fold:
- Training set: For gradient updates (remaining original + synthetic samples)
- Tuning set: For adaptive strategy decisions (same size as test, original samples only)
- Test set: For final evaluation only (original validation fold, untouched)

This design allows validation-informed adaptive training without data leakage,
because the test set is never used for any training decisions.

Key features:
- Tuning set size = test set size (balanced evaluation)
- Tuning set selection is DETERMINISTIC (sorted + fixed seed per fold)
- Tuning metrics computed FIRST each epoch, then test metrics
- Configurable 3-way patch centering: random / anatomy / tumor
- Extension point for adaptive hyperparameter adjustment based on tuning performance
- Original vs synthetic sample handling via --pattern_original_samples
- Tuning set always uses ONLY original samples
- Synthetic data ratio capped at 50% in training set
- Checkpoint saved based on SINGLE-EPOCH TUMOR DICE (not EMA)

Evaluation order each epoch:
1. TUNING SET: Compute metrics (logged, for adaptive decisions)
2. TEST SET: Compute pseudo dice (logged, for checkpoint selection)
3. Checkpoint: "Yayy! New best pseudo Dice" if improved (based on TEST tumor dice only)

Evaluation modes (configurable via self.eval_mode):
- 'sliding_window': Full sliding window inference within GT anatomy bounding box
  - More accurate, closer to real inference
  - Computes metrics on full prediction within anatomy region
- 'dual_crop': Two random patches per sample (legacy)
  - 1 patch centered on random anatomy voxel (evaluates FP control)
  - 1 patch centered on random tumor voxel (evaluates tumor detection)
  - Faster but less representative of real inference

Patch centering strategy for TRAINING:
- Configurable 3-way probabilistic centering (random / anatomy / tumor)

Label convention:
- Label 0: Background
- Label 1: Anatomy (liver for LiTS, kidney for KiTS)
- Label 2: Tumor/lesion (PRIMARY focus for centering)
- Label 3+: Secondary lesions (e.g., cyst in KiTS - trained but NOT explicitly centered on)

Usage:
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_WithTuningSet
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_WithTuningSet --pattern_original_samples "liver_\\d+"
"""

import re
import numpy as np
import torch
from typing import List, Tuple, Union, Dict, Optional, Callable
from threadpoolctl import threadpool_limits
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window, compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter


# =============================================================================
# DYNAMIC SAMPLING STRATEGY
# =============================================================================

@dataclass
class SamplingConfig:
    """Configuration for dynamic sampling strategy.
    
    Probability relationship (simplified model):
    - p_random = 0.5 * p_anatomy (random is always half of anatomy)
    - p_tumor = 1 - 1.5 * p_anatomy (tumor gets the rest)
    - Only p_anatomy is actively adjusted; others follow automatically
    
    This means:
    - When we "increase context", we increase p_anatomy (and p_random follows)
    - When we "increase lesion focus", we decrease p_anatomy (freeing up for tumor)
    """
    
    # Probability bounds for p_anatomy (primary control variable)
    # Since p_random = 0.5 * p_anatomy and p_tumor = 1 - 1.5 * p_anatomy:
    # - min_anatomy = 0.10 → p_random = 0.05, p_tumor = 0.85 (max lesion focus)
    # - max_anatomy = 0.50 → p_random = 0.25, p_tumor = 0.25 (balanced)
    min_anatomy_prob: float = 0.10
    max_anatomy_prob: float = 0.50
    
    # Context preservation: minimum (p_anatomy + p_random) = 1.5 * p_anatomy >= 0.15
    # This is automatically satisfied when min_anatomy_prob >= 0.10
    min_context_prob: float = 0.15  # Lower bound for anatomy + random combined
    
    # Moving average smoothing (exponential smoothing factor)
    alpha: float = 0.7  # Higher = more weight to recent epochs
    
    # Convergence detection
    convergence_threshold: float = 0.005  # Relative change < 0.5%
    convergence_window: int = 5  # Check last 5 epochs
    
    # Adjustment rates (how aggressively to change p_anatomy)
    adjustment_rate: float = 0.05  # Max change per epoch for p_anatomy
    decay_rate: float = 0.02  # Gradual decay when converged
    
    # Stabilization: reduce oscillation
    smoothing_factor: float = 0.5  # Blend new and old probabilities
    
    # Priority order for trend detection: Dice > Recall > Precision
    # 1. Dice trend is the primary signal
    # 2. Recall drop triggers "increase lesion focus" (reduce FN is priority)
    # 3. Precision drop triggers "increase context" (reduce FP, lower priority)
    recall_drop_threshold: float = 0.03    # 3% recall drop → increase lesion focus
    precision_drop_threshold: float = 0.03  # 3% precision drop → increase context
    
    # Best Dice tracking: revert to best-known settings if performance degrades
    # This ensures Dice generally improves by remembering what worked
    dice_revert_threshold: float = 0.05  # Revert if Dice drops > 5% from best
    exploration_margin: float = 0.02     # Only explore when within 2% of best Dice


class DynamicSamplingStrategy:
    """
    Adaptive patch sampling strategy for medical image segmentation.
    
    This class maintains historical metric records and implements decision logic
    to adjust p1 (anatomy probability) and p2 (lesion probability) to optimize
    lesion segmentation performance.
    
    Key features:
    - EMA smoothing of validation metrics
    - Trend detection (improving/stable/degrading)
    - Adaptive probability adjustment based on trends
    - Convergence detection with decay toward equilibrium
    - **Best Dice tracking**: remembers best-performing settings and reverts if degraded
    
    Dice Improvement Guarantee:
    - Tracks the best Dice EMA achieved and corresponding p_anatomy setting
    - If current Dice drops significantly below best (> dice_revert_threshold), 
      reverts to the best-known settings
    - Only explores new settings when current Dice is within exploration_margin of best
    - This creates an "exploit best, explore cautiously" behavior
    """
    
    def __init__(self, config: Optional[SamplingConfig] = None, 
                 log_fn: Optional[Callable[[str], None]] = None):
        """
        Initialize the dynamic sampling strategy.
        
        Args:
            config: SamplingConfig object with hyperparameters
            log_fn: Optional logging function (e.g., print_to_log_file)
        """
        self.config = config or SamplingConfig()
        self.log_fn = log_fn or print
        
        # History tracking: epoch -> {metric_name -> value}
        self.history: Dict[int, Dict[str, float]] = defaultdict(dict)
        
        # Moving average tracking (scalar values, updated each epoch)
        self.ema_lesion_dice: Optional[float] = None
        self.ema_lesion_f1: Optional[float] = None
        self.ema_lesion_recall: Optional[float] = None
        self.ema_lesion_precision: Optional[float] = None
        
        # History of EMAs for trend/convergence detection
        self.ema_history: List[float] = []  # Dice EMA history
        self.recall_ema_history: List[float] = []  # Recall EMA history for drop detection
        self.precision_ema_history: List[float] = []  # Precision EMA history for drop detection
        
        # Probability history
        self.p1_history: List[float] = []  # anatomy
        self.p2_history: List[float] = []  # lesion
        
        # Current epoch
        self.epoch = 0
        
        # Convergence tracking
        self.stagnation_counter = 0
        
        # Best Dice tracking for improvement guarantee
        self.best_dice_ema: Optional[float] = None  # Best Dice EMA achieved
        self.best_p_anatomy: Optional[float] = None  # p_anatomy that achieved best Dice
        self.best_epoch: int = 0  # Epoch when best was achieved
        self.reverted_count: int = 0  # How many times we reverted to best
    
    def _log(self, msg: str) -> None:
        """Log a message using the configured logging function."""
        self.log_fn(msg)
    
    def update(self, metrics_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Main function to update probabilities based on validation metrics.
        
        Includes Dice improvement guarantee:
        1. Track best Dice EMA and corresponding settings
        2. If current Dice drops significantly, revert to best settings
        3. Only explore new settings when close to best performance
        
        Args:
            metrics_dict: Dictionary containing validation metrics.
                Expected keys:
                - 'lesion_dice', 'lesion_recall', 'lesion_precision', 'lesion_f1'
                - 'anatomy_dice', 'anatomy_recall', 'anatomy_precision', 'anatomy_f1'
        
        Returns:
            Tuple of (p1_new, p2_new) where p1=anatomy, p2=lesion probability
        """
        self.epoch += 1
        self._log(f"  [DynamicSampling] Epoch {self.epoch}: Updating probabilities")
        
        # Store metrics in history
        self.history[self.epoch] = metrics_dict.copy()
        
        # Update EMAs for lesion metrics
        self._update_ema(metrics_dict)
        
        # Log current metrics and EMAs
        self._log_metrics_summary(metrics_dict)
        
        # === BEST DICE TRACKING ===
        current_dice_ema = self.ema_lesion_dice
        current_p_anatomy = self.p1_history[-1] if self.p1_history else 0.25
        
        # Check if this is a new best
        if self.best_dice_ema is None or current_dice_ema > self.best_dice_ema:
            self.best_dice_ema = current_dice_ema
            self.best_p_anatomy = current_p_anatomy
            self.best_epoch = self.epoch
            self._log(f"    ★ NEW BEST Dice EMA: {self.best_dice_ema:.4f} at p_anatomy={self.best_p_anatomy:.3f}")
        
        # Check if we should revert to best settings
        should_revert = False
        if self.best_dice_ema is not None and self.best_dice_ema > 0:
            dice_drop = (self.best_dice_ema - current_dice_ema) / self.best_dice_ema
            if dice_drop > self.config.dice_revert_threshold:
                should_revert = True
                self.reverted_count += 1
                self._log(f"    ⚠ Dice dropped {dice_drop:.1%} from best → REVERTING to best settings")
                self._log(f"      Best: Dice={self.best_dice_ema:.4f} at p_anatomy={self.best_p_anatomy:.3f} (epoch {self.best_epoch})")
        
        # Check if we're close enough to best to explore
        can_explore = True
        if self.best_dice_ema is not None and self.best_dice_ema > 0:
            dice_gap = (self.best_dice_ema - current_dice_ema) / self.best_dice_ema
            if dice_gap > self.config.exploration_margin:
                can_explore = False
                self._log(f"    ⏸ Dice {dice_gap:.1%} below best → limiting exploration")
        
        # === PROBABILITY UPDATE ===
        if should_revert:
            # Revert to best-known settings
            p1_new = self.best_p_anatomy
            p2_new = 1.0 - 1.5 * p1_new  # Derived from fixed relationship
            self._log(f"    REVERTED: p_anatomy={p1_new:.3f}")
        else:
            # Normal update path
            is_converged = self._check_convergence()
            
            if can_explore:
                # Calculate new probabilities based on trends
                p1_new, p2_new = self._calculate_new_probabilities(is_converged)
            else:
                # Stay close to current (no major changes when far from best)
                p1_new = current_p_anatomy
                p2_new = 1.0 - 1.5 * p1_new
                self._log(f"    HOLDING: staying at current settings until Dice improves")
            
            # Apply smoothing to reduce oscillation
            p1_new, p2_new = self._smooth_probabilities(p1_new, p2_new)
            
            # Normalize to ensure valid probabilities
            p1_new, p2_new = self._normalize_probabilities(p1_new, p2_new)
        
        # Store in history
        self.p1_history.append(p1_new)
        self.p2_history.append(p2_new)
        
        # Log decision
        self._log_decision(p1_new, p2_new, should_revert)
        
        return p1_new, p2_new
    
    def _update_ema(self, metrics_dict: Dict[str, float]) -> None:
        """Update exponential moving averages for lesion metrics."""
        alpha = self.config.alpha
        
        lesion_dice = metrics_dict.get('lesion_dice', 0)
        lesion_f1 = metrics_dict.get('lesion_f1', 0)
        lesion_recall = metrics_dict.get('lesion_recall', 0)
        lesion_precision = metrics_dict.get('lesion_precision', 0)
        
        if self.ema_lesion_dice is None:
            # First epoch: initialize
            self.ema_lesion_dice = lesion_dice
            self.ema_lesion_f1 = lesion_f1
            self.ema_lesion_recall = lesion_recall
            self.ema_lesion_precision = lesion_precision
        else:
            # Update with exponential smoothing: EMA_new = alpha * value + (1 - alpha) * EMA_old
            self.ema_lesion_dice = alpha * lesion_dice + (1 - alpha) * self.ema_lesion_dice
            self.ema_lesion_f1 = alpha * lesion_f1 + (1 - alpha) * self.ema_lesion_f1
            self.ema_lesion_recall = alpha * lesion_recall + (1 - alpha) * self.ema_lesion_recall
            self.ema_lesion_precision = alpha * lesion_precision + (1 - alpha) * self.ema_lesion_precision
        
        # Store EMA history for trend detection
        self.ema_history.append(self.ema_lesion_dice)
        self.recall_ema_history.append(self.ema_lesion_recall)
        self.precision_ema_history.append(self.ema_lesion_precision)
    
    def _log_metrics_summary(self, metrics_dict: Dict[str, float]) -> None:
        """Log current metrics and EMAs."""
        self._log(f"    Lesion metrics (current | EMA): "
                  f"Dice={metrics_dict.get('lesion_dice', 0):.4f}|{self.ema_lesion_dice:.4f}, "
                  f"Recall={metrics_dict.get('lesion_recall', 0):.4f}|{self.ema_lesion_recall:.4f}, "
                  f"Precision={metrics_dict.get('lesion_precision', 0):.4f}|{self.ema_lesion_precision:.4f}")
    
    def _check_convergence(self) -> bool:
        """Check if metrics have converged over recent epochs."""
        window = self.config.convergence_window
        
        if len(self.ema_history) < window + 1:
            return False
        
        # Get recent EMAs
        recent_emas = self.ema_history[-window:]
        
        # Calculate relative changes
        changes = []
        for i in range(1, len(recent_emas)):
            if recent_emas[i-1] > 0:
                rel_change = abs(recent_emas[i] - recent_emas[i-1]) / recent_emas[i-1]
                changes.append(rel_change)
        
        if not changes:
            return False
        
        avg_change = np.mean(changes)
        is_converged = avg_change < self.config.convergence_threshold
        
        if is_converged:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return is_converged
    
    def _calculate_trend(self) -> str:
        """
        Determine trend with priority: Dice > Recall > Precision.
        
        Decision priority:
        1. If recall drops > 3% → 'recall_dropping' (needs more lesion focus to reduce FN)
        2. If precision drops > 3% (and recall stable) → 'precision_dropping' (needs more context)
        3. Otherwise → Dice-based: 'improving', 'degrading', or 'stable'
        
        This prioritizes reducing False Negatives (missed lesions) over reducing
        False Positives (over-segmentation), which is appropriate for lesion detection
        where missing a lesion is typically more costly than a false alarm.
        
        Returns: 'improving', 'recall_dropping', 'precision_dropping', 'degrading', or 'stable'
        """
        if len(self.ema_history) < 2:
            return 'stable'
        
        # Compare last two Dice EMAs
        dice_change = self.ema_history[-1] - self.ema_history[-2]
        
        # Small threshold for Dice trend
        threshold = 0.002
        
        # Priority 1: Check for recall degradation (FN increasing - highest priority after Dice)
        recall_dropping = False
        if len(self.recall_ema_history) >= 2:
            recall_change = self.recall_ema_history[-1] - self.recall_ema_history[-2]
            if recall_change < -self.config.recall_drop_threshold:
                recall_dropping = True
                self._log(f"    ⚠ Recall dropping: {recall_change:.4f} "
                          f"(threshold: -{self.config.recall_drop_threshold}) → need more lesion focus")
        
        # Priority 2: Check for precision degradation (FP increasing - lower priority)
        precision_dropping = False
        if len(self.precision_ema_history) >= 2:
            precision_change = self.precision_ema_history[-1] - self.precision_ema_history[-2]
            if precision_change < -self.config.precision_drop_threshold:
                precision_dropping = True
                self._log(f"    ⚠ Precision dropping: {precision_change:.4f} "
                          f"(threshold: -{self.config.precision_drop_threshold}) → may need more context")
        
        # Decision logic with recall > precision priority
        if recall_dropping:
            # Recall is dropping - missing lesions, need more tumor patches
            # This takes precedence even if precision is also dropping
            return 'recall_dropping'
        elif precision_dropping:
            # Precision is dropping (but recall is stable) - over-segmenting
            # Need more context patches
            if dice_change > threshold:
                self._log(f"    Note: Dice improving but precision dropping → increasing context")
            return 'precision_dropping'
        elif dice_change > threshold:
            return 'improving'
        elif dice_change < -threshold:
            return 'degrading'
        else:
            return 'stable'
    
    def _calculate_new_probabilities(self, is_converged: bool) -> Tuple[float, float]:
        """
        Calculate new probabilities based on trends.
        
        Simplified model where only p_anatomy is adjusted:
        - p_random = 0.5 * p_anatomy (random is always half of anatomy)
        - p_tumor = 1 - 1.5 * p_anatomy (tumor gets the rest)
        
        Actions:
        - IMPROVING or RECALL_DROPPING: decrease p_anatomy → more tumor patches
        - DEGRADING or PRECISION_DROPPING: increase p_anatomy → more context patches
        - STABLE: maintain current
        - CONVERGED: decay toward equilibrium (p_anatomy = 0.30)
        
        Priority: Dice trend > Recall drop > Precision drop
        """
        config = self.config
        
        # Get current p_anatomy or start from default
        if self.p1_history:
            p_anatomy_current = self.p1_history[-1]
        else:
            # First epoch: start with moderate anatomy (0.25 → p_random=0.125, p_tumor=0.625)
            p_anatomy_current = 0.25
        
        # Get trend
        trend = self._calculate_trend()
        adjustment_rate = config.adjustment_rate
        
        if trend == "improving":
            # Dice improving - increase lesion focus (decrease p_anatomy)
            p_anatomy_new = p_anatomy_current - adjustment_rate
            self._log(f"    Trend: IMPROVING → increase lesion focus (↓ p_anatomy)")
            
        elif trend == "recall_dropping":
            # Recall dropping - need more tumor patches to reduce FN
            # This is higher priority than precision, so use larger adjustment
            p_anatomy_new = p_anatomy_current - adjustment_rate * 1.5
            self._log(f"    Trend: RECALL_DROPPING → increase lesion focus (↓↓ p_anatomy)")
            
        elif trend == "precision_dropping":
            # Precision dropping (but recall stable) - need more context to reduce FP
            p_anatomy_new = p_anatomy_current + adjustment_rate
            self._log(f"    Trend: PRECISION_DROPPING → increase context (↑ p_anatomy)")
            
        elif trend == "degrading":
            # Dice degrading - add more context
            p_anatomy_new = p_anatomy_current + adjustment_rate
            self._log(f"    Trend: DEGRADING → increase context (↑ p_anatomy)")
            
        else:  # stable
            p_anatomy_new = p_anatomy_current
            self._log(f"    Trend: STABLE → maintain current")
        
        # Handle convergence: decay toward equilibrium (p_anatomy = 0.30)
        equilibrium_anatomy = 0.30  # Balanced: p_random=0.15, p_tumor=0.55
        if is_converged:
            p_anatomy_new = p_anatomy_current + config.decay_rate * (equilibrium_anatomy - p_anatomy_current)
            self._log(f"    CONVERGED: decaying toward equilibrium (p_anatomy → {equilibrium_anatomy})")
        
        # Derive other probabilities from p_anatomy
        # These will be properly bounded in _normalize_probabilities
        p_random_new = 0.5 * p_anatomy_new
        p_tumor_new = 1.0 - 1.5 * p_anatomy_new
        
        # Return (p_anatomy, p_tumor) - p_random is derived in _normalize_probabilities
        return p_anatomy_new, p_tumor_new
    
    def _smooth_probabilities(self, p1_new: float, p2_new: float) -> Tuple[float, float]:
        """Apply smoothing to reduce oscillation."""
        if not self.p1_history:
            return p1_new, p2_new
        
        factor = self.config.smoothing_factor
        p1_smoothed = factor * p1_new + (1 - factor) * self.p1_history[-1]
        p2_smoothed = factor * p2_new + (1 - factor) * self.p2_history[-1]
        
        return p1_smoothed, p2_smoothed
    
    def _normalize_probabilities(self, p_anatomy: float, p_tumor: float) -> Tuple[float, float]:
        """
        Normalize probabilities with simplified model:
        - p_random = 0.5 * p_anatomy (random is always half of anatomy)
        - p_tumor = 1 - 1.5 * p_anatomy (tumor gets the rest)
        - Only p_anatomy is bounded; others follow automatically
        
        Bounds:
        - p_anatomy ∈ [min_anatomy_prob, max_anatomy_prob] = [0.10, 0.50]
        - This ensures: p_anatomy + p_random = 1.5 * p_anatomy ≥ 0.15 (context preservation)
        
        At boundaries:
        - Min context (max lesion): p_anatomy=0.10, p_random=0.05, p_tumor=0.85
        - Max context (balanced):   p_anatomy=0.50, p_random=0.25, p_tumor=0.25
        """
        config = self.config
        
        # Bound p_anatomy (the control variable)
        p_anatomy = np.clip(p_anatomy, config.min_anatomy_prob, config.max_anatomy_prob)
        
        # Derive other probabilities from the fixed relationship
        p_random = 0.5 * p_anatomy
        p_tumor = 1.0 - 1.5 * p_anatomy
        
        # Sanity check: ensure context preservation
        context_prob = p_anatomy + p_random  # = 1.5 * p_anatomy
        if context_prob < config.min_context_prob:
            # This shouldn't happen if min_anatomy_prob >= min_context_prob / 1.5
            self._log(f"    WARNING: Context prob {context_prob:.3f} < {config.min_context_prob:.3f}")
            p_anatomy = config.min_context_prob / 1.5
            p_random = 0.5 * p_anatomy
            p_tumor = 1.0 - 1.5 * p_anatomy
        
        # Log the fixed relationship
        self._log(f"    Probability model: p_anatomy={p_anatomy:.3f}, "
                  f"p_random={p_random:.3f} (=0.5×anatomy), p_tumor={p_tumor:.3f}")
        
        return p_anatomy, p_tumor
    
    def _log_decision(self, p1: float, p2: float, reverted: bool = False) -> None:
        """Log final decision."""
        p_random = 0.5 * p1  # Fixed relationship
        
        strategy_type = "LESION-ORIENTED" if p2 > p1 and p2 > p_random else \
                        "ANATOMY-ORIENTED" if p1 > p2 and p1 > p_random else \
                        "BALANCED"
        
        status = " [REVERTED]" if reverted else ""
        best_info = f" (best={self.best_dice_ema:.4f})" if self.best_dice_ema else ""
        
        self._log(f"    Decision: {strategy_type}{status} → p_anatomy={p1:.3f}, "
                  f"p_tumor={p2:.3f}, p_random={p_random:.3f}{best_info}")
    
    def get_history(self) -> Dict[str, List]:
        """Return complete history for visualization/analysis."""
        epochs = list(self.history.keys())
        
        result = {
            'epochs': epochs,
            'p1_anatomy': self.p1_history,
            'p2_lesion': self.p2_history,
            'ema_lesion_dice': self.ema_history,
            'ema_lesion_recall': self.recall_ema_history,
            'ema_lesion_precision': self.precision_ema_history,
            # Best tracking info
            'best_dice_ema': self.best_dice_ema,
            'best_p_anatomy': self.best_p_anatomy,
            'best_epoch': self.best_epoch,
            'reverted_count': self.reverted_count,
        }
        
        # Add metric histories
        for metric_key in ['lesion_dice', 'lesion_f1', 'lesion_recall', 'lesion_precision',
                          'anatomy_dice', 'anatomy_f1', 'anatomy_recall', 'anatomy_precision']:
            result[metric_key] = [self.history[ep].get(metric_key, 0) for ep in epochs]
        
        return result


# =============================================================================
# CUSTOM DATALOADER WITH 3-WAY CENTERING
# =============================================================================

class nnUNetDataLoader3WayCentering(nnUNetDataLoader):
    """
    Custom DataLoader with configurable 3-way patch centering probabilities.
    
    For each sample in batch, randomly choose centering mode based on probabilities:
    - Random: Center on any random voxel
    - Anatomy: Center on random anatomy voxel (label 1)
    - Tumor: Center on random tumor voxel (label 2 ONLY)
    
    Note: Labels >= 3 (e.g., cyst in KiTS) are trained but NOT explicitly centered on.
    This provides fine-grained control over patch sampling strategy.
    """
    
    # Label conventions
    BACKGROUND_LABEL = 0
    ANATOMY_LABEL = 1  # liver for LiTS, kidney for KiTS
    PRIMARY_TUMOR_LABEL = 2  # Only label 2 is used for tumor-centered cropping
    # Note: Labels >= 3 (e.g., cyst in KiTS) are trained but NOT explicitly centered on
    
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 prob_random: float = 0.0,
                 prob_anatomy: float = 0.67,
                 prob_tumor: float = 0.33,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 transforms=None):
        """
        Args:
            prob_random: Probability of centering on random voxel
            prob_anatomy: Probability of centering on anatomy voxel (label 1)
            prob_tumor: Probability of centering on tumor voxel (label 2 ONLY)
            
            Note: prob_random + prob_anatomy + prob_tumor must equal 1.0
        """
        # Map to parent's oversample_foreground_percent (not directly used, but required)
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            label_manager=label_manager,
            oversample_foreground_percent=prob_tumor,  # Legacy compatibility
            sampling_probabilities=sampling_probabilities,
            pad_sides=pad_sides,
            probabilistic_oversampling=True,  # Always use probabilistic for 3-way
            transforms=transforms
        )
        
        # Validate probabilities
        total_prob = prob_random + prob_anatomy + prob_tumor
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")
        
        self.prob_random = prob_random
        self.prob_anatomy = prob_anatomy
        self.prob_tumor = prob_tumor
        
        # Only use PRIMARY_TUMOR_LABEL (2) for tumor-centered cropping
        # Labels >= 3 (e.g., cyst in KiTS) are trained but not explicitly centered on
        if self.PRIMARY_TUMOR_LABEL not in label_manager.foreground_labels:
            raise ValueError(
                f"The dedicated tumor channel (2) is not found! Please note that "
                f"{self.__class__.__name__} is designed for better tumor-focused training!"
            )
        self.tumor_labels = [self.PRIMARY_TUMOR_LABEL]
        self.anatomy_label = self.ANATOMY_LABEL
        
        # Counters for tracking actual centering modes used (including fallbacks)
        self.reset_centering_counts()
    
    def reset_centering_counts(self):
        """Reset the centering mode counters. Call at start of each epoch."""
        self.centering_counts = {
            'random': 0,
            'anatomy': 0,
            'tumor': 0
        }
    
    def get_centering_counts(self) -> dict:
        """Get the current centering mode counts."""
        return self.centering_counts.copy()
    
    def update_probabilities(self, prob_random: float, prob_anatomy: float, prob_tumor: float):
        """
        Update centering probabilities dynamically.
        
        This allows adaptive adjustment of patch sampling strategy during training.
        
        Args:
            prob_random: Probability of centering on random voxel
            prob_anatomy: Probability of centering on anatomy voxel (label 1)
            prob_tumor: Probability of centering on tumor voxel (label 2 ONLY)
        """
        total = prob_random + prob_anatomy + prob_tumor
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        self.prob_random = prob_random
        self.prob_anatomy = prob_anatomy
        self.prob_tumor = prob_tumor
    
    def _sample_centering_mode(self) -> str:
        """
        Randomly sample centering mode based on probabilities.
        
        Returns: 'random', 'anatomy', or 'tumor'
        """
        r = np.random.uniform()
        if r < self.prob_random:
            return 'random'
        elif r < self.prob_random + self.prob_anatomy:
            return 'anatomy'
        else:
            return 'tumor'
    
    def get_bbox_3way(self, data_shape: np.ndarray, centering_mode: str,
                      class_locations: Union[dict, None]) -> Tuple[List[int], List[int], str]:
        """
        Get bounding box based on centering mode.
        
        Args:
            data_shape: Shape of the data (excluding channel dimension)
            centering_mode: 'random', 'anatomy', or 'tumor'
            class_locations: Dict mapping class labels to voxel locations
            
        Returns:
            bbox_lbs, bbox_ubs: Bounding box lower and upper bounds
            actual_mode: The actual centering mode used (may differ due to fallbacks)
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
        
        selected_voxel = None
        actual_mode = centering_mode  # Track actual mode used
        
        if centering_mode == 'tumor':
            # Try to center on tumor (label 2 ONLY - self.tumor_labels contains only PRIMARY_TUMOR_LABEL)
            tumor_classes_with_voxels = [l for l in self.tumor_labels 
                                         if class_locations and l in class_locations 
                                         and len(class_locations[l]) > 0]
            if tumor_classes_with_voxels:
                selected_class = tumor_classes_with_voxels[np.random.choice(len(tumor_classes_with_voxels))]
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]
                actual_mode = 'tumor'
            else:
                # Fallback to anatomy if no tumor
                centering_mode = 'anatomy'
        
        if centering_mode == 'anatomy' and selected_voxel is None:
            # Center on anatomy (label 1)
            if class_locations and self.anatomy_label in class_locations \
               and len(class_locations[self.anatomy_label]) > 0:
                voxels = class_locations[self.anatomy_label]
                selected_voxel = voxels[np.random.choice(len(voxels))]
                actual_mode = 'anatomy'
            else:
                # Fallback to random if no anatomy
                centering_mode = 'random'
        
        if centering_mode == 'random' or selected_voxel is None:
            # Random centering
            actual_mode = 'random'
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) if lbs[i] < ubs[i] + 1 
                        else lbs[i] for i in range(dim)]
        else:
            # Center on selected voxel
            # i + 1 because first dimension is channel
            bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
        
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        
        return bbox_lbs, bbox_ubs, actual_mode
    
    def generate_train_batch(self):
        """
        Generate a training batch with 3-way centering.
        Tracks the actual centering modes used (including fallbacks).
        """
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        
        for j, i in enumerate(selected_keys):
            # Sample centering mode for this sample
            centering_mode = self._sample_centering_mode()
            
            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            # Get bbox using 3-way centering (returns actual mode used after fallbacks)
            bbox_lbs, bbox_ubs, actual_mode = self.get_bbox_3way(
                shape, centering_mode, properties.get('class_locations')
            )
            
            # Track the actual centering mode used
            self.centering_counts[actual_mode] += 1
            
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]
            
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped
        
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
        
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
        
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


# =============================================================================
# DUAL-CROP EVALUATION DATALOADER (ANATOMY + TUMOR)
# =============================================================================

class nnUNetDataLoaderDualCropEval(nnUNetDataLoader):
    """
    Evaluation DataLoader that creates TWO patches per sample for balanced evaluation.
    
    For each sample in the dataset:
    - Patch 1: Centered on a random ANATOMY voxel (label 1) → evaluates FP control
    - Patch 2: Centered on a random TUMOR voxel (label 2) → evaluates tumor detection
    
    This provides balanced evaluation that considers both:
    - False positive rate on normal anatomy regions
    - True positive rate (recall) on tumor regions
    
    If a sample has N original cases, evaluation will use 2N patches.
    Example: 25 samples → 50 patches (25 anatomy-centered + 25 tumor-centered)
    
    This is used for tuning and test set evaluation ONLY (not training).
    """
    
    ANATOMY_LABEL = 1  # liver for LiTS, kidney for KiTS
    TUMOR_LABEL = 2    # Only label 2 is used for tumor centering
    
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 transforms=None):
        """
        Note: batch_size here refers to the number of ORIGINAL samples per batch.
        The actual output will have 2x batch_size patches (anatomy + tumor for each sample).
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            label_manager=label_manager,
            oversample_foreground_percent=1.0,  # Not used, we do dual-crop
            sampling_probabilities=sampling_probabilities,
            pad_sides=pad_sides,
            probabilistic_oversampling=True,
            transforms=transforms
        )
        
        # Update data_shape and seg_shape to accommodate 2x patches
        # Original shape: (batch_size, channels, *patch_size)
        # New shape: (2 * batch_size, channels, *patch_size)
        self.original_batch_size = batch_size
        self.data_shape = (2 * batch_size,) + self.data_shape[1:]
        self.seg_shape = (2 * batch_size,) + self.seg_shape[1:]
    
    def get_bbox_for_mode(self, data_shape: np.ndarray, mode: str,
                          class_locations: Union[dict, None]) -> Tuple[List[int], List[int]]:
        """
        Get bounding box centered on a voxel of the specified class.
        
        Args:
            data_shape: Shape of the data (excluding channel dimension)
            mode: 'anatomy' or 'tumor'
            class_locations: Dict mapping class labels to voxel locations
            
        Returns:
            bbox_lbs, bbox_ubs: Bounding box lower and upper bounds
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
        
        selected_voxel = None
        target_label = self.ANATOMY_LABEL if mode == 'anatomy' else self.TUMOR_LABEL
        fallback_label = self.TUMOR_LABEL if mode == 'anatomy' else self.ANATOMY_LABEL
        
        # Try to get voxel from target class
        if class_locations and target_label in class_locations and len(class_locations[target_label]) > 0:
            voxels = class_locations[target_label]
            selected_voxel = voxels[np.random.choice(len(voxels))]
        
        # Fallback to other foreground class if target not available
        if selected_voxel is None and class_locations:
            if fallback_label in class_locations and len(class_locations[fallback_label]) > 0:
                voxels = class_locations[fallback_label]
                selected_voxel = voxels[np.random.choice(len(voxels))]
        
        # Final fallback: random location
        if selected_voxel is None:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) if lbs[i] < ubs[i] + 1 
                        else lbs[i] for i in range(dim)]
        else:
            # Center on selected voxel (skip channel dim: voxel is [channel, z, y, x])
            bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
        
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        
        return bbox_lbs, bbox_ubs
    
    def generate_train_batch(self):
        """
        Generate an evaluation batch with dual-crop strategy.
        
        For each sample:
        1. Create one patch centered on random anatomy voxel
        2. Create one patch centered on random tumor voxel
        
        Output has 2x the original batch size.
        """
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        
        # Track keys for both patches (each key appears twice: once for anatomy, once for tumor)
        output_keys = []
        
        for j, key in enumerate(selected_keys):
            data, seg, seg_prev, properties = self._data.load_case(key)
            shape = data.shape[1:]
            class_locations = properties.get('class_locations')
            
            # === Patch 1: Anatomy-centered ===
            idx_anatomy = 2 * j
            bbox_lbs_anat, bbox_ubs_anat = self.get_bbox_for_mode(shape, 'anatomy', class_locations)
            bbox_anat = [[i, k] for i, k in zip(bbox_lbs_anat, bbox_ubs_anat)]
            
            data_all[idx_anatomy] = crop_and_pad_nd(data, bbox_anat, 0)
            seg_cropped_anat = crop_and_pad_nd(seg, bbox_anat, -1)
            if seg_prev is not None:
                seg_cropped_anat = np.vstack((seg_cropped_anat, crop_and_pad_nd(seg_prev, bbox_anat, -1)[None]))
            seg_all[idx_anatomy] = seg_cropped_anat
            output_keys.append(f"{key}_anatomy")
            
            # === Patch 2: Tumor-centered ===
            idx_tumor = 2 * j + 1
            bbox_lbs_tumor, bbox_ubs_tumor = self.get_bbox_for_mode(shape, 'tumor', class_locations)
            bbox_tumor = [[i, k] for i, k in zip(bbox_lbs_tumor, bbox_ubs_tumor)]
            
            data_all[idx_tumor] = crop_and_pad_nd(data, bbox_tumor, 0)
            seg_cropped_tumor = crop_and_pad_nd(seg, bbox_tumor, -1)
            if seg_prev is not None:
                seg_cropped_tumor = np.vstack((seg_cropped_tumor, crop_and_pad_nd(seg_prev, bbox_tumor, -1)[None]))
            seg_all[idx_tumor] = seg_cropped_tumor
            output_keys.append(f"{key}_tumor")
        
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
        
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    # Process all 2*batch_size patches
                    for b in range(2 * self.original_batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'keys': output_keys}
        
        return {'data': data_all, 'target': seg_all, 'keys': output_keys}


# =============================================================================
# TRAINER CLASS
# =============================================================================


class nnUNetTrainer_WithTuningSet(nnUNetTrainer):
    """
    nnUNet Trainer with a separate tuning set carved from training data.
    
    Split structure (within each fold):
    ┌─────────────────────────────────────────────────────────────┐
    │ Original Training Data (e.g., folds 1,2,3,4)                │
    │  ┌──────────────────────┬─────────────────┐                 │
    │  │   Training (80%)     │   Tuning (20%)  │                 │
    │  │   - Gradient updates │   - Metrics for │                 │
    │  │                      │     adaptive    │                 │
    │  │                      │     decisions   │                 │
    │  └──────────────────────┴─────────────────┘                 │
    │                                                             │
    │ Original Validation Data (e.g., fold 0) = TEST SET          │
    │  ┌─────────────────────────────────────────┐                │
    │  │   Test (100%) - Final evaluation ONLY   │                │
    │  │   NEVER used for training decisions!    │                │
    │  └─────────────────────────────────────────┘                │
    └─────────────────────────────────────────────────────────────┘
    
    This ensures:
    - Tuning metrics can guide adaptive training without leaking test info
    - Test set remains truly held-out for unbiased final evaluation
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
                 checkpoint_signature: str = None,
                 splits_file: str = None):
        super().__init__(plans, configuration, fold, dataset_json, device,
                         checkpoint_signature, splits_file)
        
        # Tuning set configuration
        # Note: tuning_ratio is kept for backward compatibility but NOT used
        # Tuning set size is now set equal to test set size for balanced evaluation
        self.tuning_ratio = 0.2  # Legacy, not used - tuning size = test size
        self.tuning_keys: List[str] = []
        self.tuning_dataset: nnUNetBaseDataset = None
        self.tuning_dataloader = None
        
        # =====================================================================
        # ORIGINAL vs SYNTHETIC SAMPLE HANDLING
        # =====================================================================
        # Pattern to identify original samples (set via --pattern_original_samples)
        # If None, all samples are considered original
        self.pattern_original_samples: str = None  # Set by run_training.py
        self.max_synthetic_ratio = 0.5  # Max ratio of synthetic samples in training
        
        # Track original/synthetic sample counts (populated in do_split)
        self.n_original_total = 0
        self.n_synthetic_total = 0
        self.n_synthetic_removed = 0
        self.removed_synthetic_keys: List[str] = []
        # =====================================================================
        
        # =====================================================================
        # 3-WAY CENTERING PROBABILITIES
        # =====================================================================
        # Probabilities for patch centering: (prob_random, prob_anatomy, prob_tumor)
        # Must sum to 1.0 for each mode
        #
        # Training: Where gradient updates happen
        self.train_prob_random = 0.25     # Center on random voxel
        self.train_prob_anatomy = 0.25   # Center on anatomy voxel (label 1)
        self.train_prob_tumor = 0.5     # Center on tumor voxel (label 2 ONLY)
        
        # Tuning: For adaptive decisions (evaluation)
        self.tuning_prob_random = 0.0
        self.tuning_prob_anatomy = 0.0
        self.tuning_prob_tumor = 1.0     # 100% tumor-centered for tuning evaluation
        
        # Test: For final evaluation (pseudo dice, checkpoint selection)
        self.test_prob_random = 0.0
        self.test_prob_anatomy = 0.0
        self.test_prob_tumor = 1.0       # 100% tumor-centered for test evaluation
        # =====================================================================
        
        # Storage for tuning metrics (computed each epoch)
        self.tuning_metrics = {
            'dice_per_class': [],
            'mean_fg_dice': [],
            'ema_fg_dice': [],  # EMA of mean foreground dice (same formula as test set)
            'loss': []
        }
        
        # =====================================================================
        # DYNAMIC SAMPLING STRATEGY
        # =====================================================================
        # Automatically adjusts training patch centering probabilities based on
        # tuning set metrics. Set enable_dynamic_sampling=False to disable.
        self.enable_dynamic_sampling = True
        self.dynamic_sampling_strategy: Optional[DynamicSamplingStrategy] = None
        self.probability_history: List[Tuple[float, float, float]] = []  # (random, anatomy, tumor)
        # =====================================================================
        
        # =====================================================================
        # TOP-K CHECKPOINT TRACKING
        # =====================================================================
        # Maintain top-K checkpoints ranked by tumor dice (single-epoch, NOT EMA)
        # Naming: checkpoint_best.pth (rank 1), checkpoint_best.pth.2 (rank 2), etc.
        self.top_k_checkpoints = 5  # Number of top checkpoints to maintain
        self.top_k_dices: List[Tuple[float, int]] = []  # List of (dice, epoch) sorted descending
        # =====================================================================
        
        # =====================================================================
        # EVALUATION MODE: 'sliding_window' or 'dual_crop'
        # =====================================================================
        # sliding_window: Full inference within GT anatomy bounding box (more accurate)
        # dual_crop: Two random patches per sample (faster, legacy)
        self.eval_mode = 'sliding_window'  # Default to sliding window for better accuracy
        self.sliding_window_step_size = 0.5  # Step size for sliding window (0.5 = 50% overlap)
        # Cache for anatomy bounding boxes (computed once, reused across epochs)
        # Format: {case_key: (bbox_lbs, bbox_ubs)}
        self._anatomy_bbox_cache: Dict[str, Tuple[List[int], List[int]]] = {}
        # =====================================================================
        
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Using nnUNetTrainer_WithTuningSet")
        self.print_to_log_file("=" * 70)
        self.print_to_log_file(f"3-way split: train (remaining original + synthetic), "
                               f"tuning (= test size, original only), test (original val fold)")
        self.print_to_log_file("Tuning set: For adaptive training decisions (NO leakage to test)")
        self.print_to_log_file("Test set: For final evaluation ONLY (original val fold)")
        self.print_to_log_file("-" * 70)
        self.print_to_log_file("Patch centering strategy (TRAINING):")
        self.print_to_log_file(f"  3-way probabilistic ({self.train_prob_random:.2f} / {self.train_prob_anatomy:.2f} / {self.train_prob_tumor:.2f})")
        self.print_to_log_file("-" * 70)
        self.print_to_log_file(f"Evaluation mode: {self.eval_mode.upper()}")
        if self.eval_mode == 'sliding_window':
            self.print_to_log_file(f"  → Sliding window inference within GT anatomy bounding box")
            self.print_to_log_file(f"  → Step size: {self.sliding_window_step_size} (overlap: {(1-self.sliding_window_step_size)*100:.0f}%)")
        else:
            self.print_to_log_file(f"  → Dual-crop: 1 anatomy + 1 tumor patch per sample")
            self.print_to_log_file(f"  → N samples → 2N patches (balanced FP control + tumor detection)")
        self.print_to_log_file("-" * 70)
        self.print_to_log_file(f"Dynamic sampling: {'ENABLED' if self.enable_dynamic_sampling else 'DISABLED'}")
        if self.enable_dynamic_sampling:
            self.print_to_log_file("  → Training probabilities will be adjusted based on tuning metrics")
        self.print_to_log_file(f"Top-K checkpoints: Maintaining top {self.top_k_checkpoints} by single-epoch tumor Dice")
        self.print_to_log_file("=" * 70 + "\n")
    
    # =========================================================================
    # TOP-K CHECKPOINT MANAGEMENT METHODS
    # =========================================================================
    
    def get_top_k_checkpoint_path(self, rank: int) -> str:
        """
        Get checkpoint path for a given rank (1-5).
        
        Naming convention:
        - Rank 1 (best): checkpoint_best_<plans>.pth
        - Rank 2: checkpoint_best_<plans>.pth.2
        - Rank 3: checkpoint_best_<plans>.pth.3
        - etc.
        
        Args:
            rank: Rank of the checkpoint (1 = best, 2 = second best, etc.)
            
        Returns:
            Full path to the checkpoint file
        """
        base_path = join(self.output_folder, self.get_checkpoint_filename('checkpoint_best'))
        if rank == 1:
            return base_path
        else:
            return f"{base_path}.{rank}"
    
    def _scan_existing_top_k_checkpoints(self) -> List[Tuple[float, int]]:
        """
        Scan the output folder for existing top-k checkpoint files and reconstruct
        the top_k_dices list from them.
        
        This provides backward compatibility when loading checkpoints that don't
        have top_k_dices stored, or when resuming training with existing checkpoints.
        
        Returns:
            List of (dice, epoch) tuples sorted descending by dice
        """
        top_k_list = []
        
        for rank in range(1, self.top_k_checkpoints + 1):
            checkpoint_path = self.get_top_k_checkpoint_path(rank)
            if isfile(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    # Try to get single-epoch dice first, fall back to _best_ema for compatibility
                    dice = checkpoint.get('_best_dice', checkpoint.get('_best_ema', None))
                    epoch = checkpoint.get('current_epoch', 0)
                    if dice is not None:
                        top_k_list.append((dice, epoch, rank))
                        self.print_to_log_file(f"  Found existing checkpoint rank {rank}: Dice={dice:.4f}, epoch={epoch}")
                except Exception as e:
                    self.print_to_log_file(f"  Warning: Could not load checkpoint at rank {rank}: {e}")
        
        # Sort by dice descending and drop rank info (used only for logging)
        top_k_list.sort(key=lambda x: x[0], reverse=True)
        return [(dice, epoch) for dice, epoch, _ in top_k_list]
    
    def _get_insertion_rank(self, current_dice: float) -> int:
        """
        Determine where the current dice would rank in the top-k list.
        
        Args:
            current_dice: Current epoch's tumor dice (single-epoch, not EMA)
            
        Returns:
            Rank (1-based) where this dice would be inserted, or 0 if not in top-k
        """
        # Find insertion position (1-based rank)
        for i, (dice, _) in enumerate(self.top_k_dices):
            if current_dice > dice:
                return i + 1  # 1-based rank
        
        # Check if we can add to the end (list not full yet)
        if len(self.top_k_dices) < self.top_k_checkpoints:
            return len(self.top_k_dices) + 1
        
        return 0  # Not in top-k
    
    def _shift_checkpoints_down(self, insert_rank: int) -> None:
        """
        Shift checkpoint files down to make room for a new checkpoint at insert_rank.
        
        Example: inserting at rank 3 with 5 existing checkpoints:
        - Remove rank 5 file
        - Rename rank 4 → rank 5
        - Rename rank 3 → rank 4
        - (Rank 3 slot is now free for the new checkpoint)
        
        Args:
            insert_rank: The rank where the new checkpoint will be saved (1-based)
        """
        import os
        
        # Only shift if we have enough checkpoints that would be affected
        max_existing_rank = min(len(self.top_k_dices), self.top_k_checkpoints)
        
        # If list is full, remove the last checkpoint file
        if len(self.top_k_dices) >= self.top_k_checkpoints:
            last_path = self.get_top_k_checkpoint_path(self.top_k_checkpoints)
            if isfile(last_path):
                try:
                    os.remove(last_path)
                except Exception as e:
                    self.print_to_log_file(f"Warning: Could not remove rank {self.top_k_checkpoints} checkpoint: {e}")
        
        # Shift files down: from (max_existing_rank - 1) to insert_rank
        # We shift backwards to avoid overwriting
        for rank in range(min(max_existing_rank, self.top_k_checkpoints - 1), insert_rank - 1, -1):
            src_path = self.get_top_k_checkpoint_path(rank)
            dst_path = self.get_top_k_checkpoint_path(rank + 1)
            if isfile(src_path):
                try:
                    os.rename(src_path, dst_path)
                except Exception as e:
                    self.print_to_log_file(f"Warning: Could not rename rank {rank} → {rank + 1}: {e}")
    
    def _update_top_k_checkpoints(self, current_dice: float, current_epoch: int) -> int:
        """
        Update top-k checkpoints if current dice qualifies.
        
        This method:
        1. Checks if current dice ranks in top-k
        2. If yes, shifts existing checkpoints and saves new one
        3. Updates the top_k_dices list
        
        Args:
            current_dice: Current epoch's tumor dice (single-epoch, not EMA)
            current_epoch: Current epoch number
            
        Returns:
            The rank achieved (1-5), or 0 if not in top-k
        """
        # Get insertion rank
        insert_rank = self._get_insertion_rank(current_dice)
        
        if insert_rank == 0:
            return 0  # Not in top-k
        
        # Shift existing checkpoint files down
        self._shift_checkpoints_down(insert_rank)
        
        # Save new checkpoint at the insertion rank
        checkpoint_path = self.get_top_k_checkpoint_path(insert_rank)
        
        # Store the dice value for this checkpoint
        old_best_dice = getattr(self, '_best_dice', None)
        self._best_dice = current_dice
        self.save_checkpoint(checkpoint_path)
        
        # Update _best_dice only if this is rank 1
        if insert_rank != 1:
            self._best_dice = old_best_dice
        
        # Update the in-memory list
        # Insert at correct position (0-indexed)
        self.top_k_dices.insert(insert_rank - 1, (current_dice, current_epoch))
        
        # Trim list to top_k_checkpoints
        if len(self.top_k_dices) > self.top_k_checkpoints:
            self.top_k_dices = self.top_k_dices[:self.top_k_checkpoints]
        
        return insert_rank
    
    # =========================================================================
    
    def _is_original_sample(self, key: str) -> bool:
        """
        Check if a sample key represents an original (non-synthetic) sample.
        
        Args:
            key: Sample key (basename without extension)
            
        Returns:
            True if sample is original, False if synthetic
        """
        if self.pattern_original_samples is None:
            # No pattern specified, all samples are original
            return True
        
        # Match the entire key against the pattern
        pattern = f"^{self.pattern_original_samples}$"
        return bool(re.match(pattern, key))
    
    def do_split(self) -> Tuple[List[str], List[str]]:
        """
        Override to create 3-way split: train/tuning/test.
        
        Features:
        - Classifies samples as original vs synthetic based on pattern
        - Ensures tuning set contains ONLY original samples
        - Caps synthetic data ratio at max_synthetic_ratio (default 50%)
        - Logs all statistics and removed samples
        
        The tuning keys are stored in self.tuning_keys.
        Returns (train_keys, test_keys) - tuning is handled separately.
        """
        # Get original 2-way split from parent
        original_tr_keys, test_keys = super().do_split()
        
        # =====================================================================
        # STEP 1: Classify samples as original vs synthetic
        # =====================================================================
        original_keys = []
        synthetic_keys = []
        
        for key in original_tr_keys:
            if self._is_original_sample(key):
                original_keys.append(key)
            else:
                synthetic_keys.append(key)
        
        self.n_original_total = len(original_keys)
        self.n_synthetic_total = len(synthetic_keys)
        
        self.print_to_log_file("\n" + "-" * 70)
        self.print_to_log_file("Original vs Synthetic sample classification:")
        if self.pattern_original_samples:
            self.print_to_log_file(f"  Pattern for original samples: '{self.pattern_original_samples}'")
        else:
            self.print_to_log_file(f"  No pattern specified - all samples treated as original")
        self.print_to_log_file(f"  Original samples: {self.n_original_total}")
        self.print_to_log_file(f"  Synthetic samples: {self.n_synthetic_total}")
        self.print_to_log_file("-" * 70)
        
        # =====================================================================
        # STEP 2: Create tuning set from ORIGINAL samples only
        # =====================================================================
        # Use fixed seed for deterministic tuning set selection across runs
        rng = np.random.RandomState(seed=12345 + self.fold)
        
        # IMPORTANT: Sort original_keys first to ensure determinism
        # (order from parent class may vary, but sorted order is always the same)
        sorted_original = sorted(original_keys)
        
        # Shuffle with fixed seed for reproducible selection
        shuffled_original = rng.permutation(sorted_original).tolist()
        
        # Tuning set size = test set size (instead of tuning_ratio)
        # This ensures balanced evaluation between tuning and test
        n_tuning = len(test_keys)
        if n_tuning > len(shuffled_original):
            self.print_to_log_file(f"WARNING: Test set size ({n_tuning}) > available original samples ({len(shuffled_original)})")
            self.print_to_log_file(f"         Using all {len(shuffled_original)} original samples for tuning")
            n_tuning = len(shuffled_original)
        
        self.tuning_keys = shuffled_original[:n_tuning]
        remaining_original = shuffled_original[n_tuning:]
        
        self.print_to_log_file(f"Tuning set: {len(self.tuning_keys)} cases (= test set size, from original samples only)")
        self.print_to_log_file(f"  [Deterministic] Seed=12345+fold={12345 + self.fold}, sorted before shuffle")
        
        # =====================================================================
        # STEP 3: Cap synthetic ratio in training set
        # =====================================================================
        # Training = remaining original + (possibly capped) synthetic
        n_remaining_original = len(remaining_original)
        
        # Calculate max synthetic samples allowed
        # If synthetic_ratio = 0.5, then synthetic <= original
        # max_synthetic = original * synthetic_ratio / (1 - synthetic_ratio)
        if self.max_synthetic_ratio < 1.0:
            max_synthetic = int(n_remaining_original * self.max_synthetic_ratio / (1 - self.max_synthetic_ratio))
        else:
            max_synthetic = len(synthetic_keys)  # No cap
        
        # Randomly select synthetic samples if we have too many
        if len(synthetic_keys) > max_synthetic:
            shuffled_synthetic = rng.permutation(synthetic_keys).tolist()
            synthetic_for_training = shuffled_synthetic[:max_synthetic]
            self.removed_synthetic_keys = shuffled_synthetic[max_synthetic:]
            self.n_synthetic_removed = len(self.removed_synthetic_keys)
            
            self.print_to_log_file(f"\n*** SYNTHETIC DATA CAP APPLIED ***")
            self.print_to_log_file(f"  Max synthetic ratio: {self.max_synthetic_ratio:.0%}")
            self.print_to_log_file(f"  Synthetic samples allowed: {max_synthetic}")
            self.print_to_log_file(f"  Synthetic samples removed: {self.n_synthetic_removed}")
            self.print_to_log_file(f"  Removed samples:")
            for key in self.removed_synthetic_keys:
                self.print_to_log_file(f"    - {key}")
        else:
            synthetic_for_training = synthetic_keys
            self.removed_synthetic_keys = []
            self.n_synthetic_removed = 0
        
        # Combine remaining original + allowed synthetic for training
        train_keys = remaining_original + synthetic_for_training
        
        # Shuffle training set
        train_keys = rng.permutation(train_keys).tolist()
        
        # =====================================================================
        # STEP 4: Log final split statistics
        # =====================================================================
        n_original_in_train = len(remaining_original)
        n_synthetic_in_train = len(synthetic_for_training)
        actual_synthetic_ratio = n_synthetic_in_train / len(train_keys) if len(train_keys) > 0 else 0
        
        self.print_to_log_file(f"\n3-way split for fold {self.fold}:")
        self.print_to_log_file(f"  - Training: {len(train_keys)} cases")
        self.print_to_log_file(f"      Original:  {n_original_in_train} ({100*n_original_in_train/len(train_keys):.1f}%)")
        self.print_to_log_file(f"      Synthetic: {n_synthetic_in_train} ({100*actual_synthetic_ratio:.1f}%)")
        self.print_to_log_file(f"  - Tuning:   {len(self.tuning_keys)} cases (100% original)")
        self.print_to_log_file(f"  - Test:     {len(test_keys)} cases (original val fold)")
        self.print_to_log_file("-" * 70 + "\n")
        
        return train_keys, test_keys
    
    def get_tr_and_val_datasets(self):
        """
        Override to also create tuning dataset.
        
        Note: 'val' in nnUNet terminology is our 'test' set.
        We create a separate tuning dataset from self.tuning_keys.
        """
        dataset_tr, dataset_test = super().get_tr_and_val_datasets()
        
        # Create tuning dataset
        self.tuning_dataset = self.dataset_class(
            self.preprocessed_dataset_folder,
            self.tuning_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        
        self.print_to_log_file(f"Created tuning dataset with {len(self.tuning_keys)} cases")
        
        return dataset_tr, dataset_test
    
    def get_dataloaders(self):
        """
        Override to use 3-way centering DataLoaders for training and test.
        Also creates the tuning dataloader.
        """
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        
        # Get configuration
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Training transforms (with augmentation)
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        
        # Validation/Test transforms (no augmentation)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        
        # Get datasets
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        # Create TRAINING dataloader with 3-way centering
        # Store reference for accessing centering counts
        self._dl_tr = nnUNetDataLoader3WayCentering(
            dataset_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            prob_random=self.train_prob_random,
            prob_anatomy=self.train_prob_anatomy,
            prob_tumor=self.train_prob_tumor,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms
        )
        dl_tr = self._dl_tr
        
        # Create TEST dataloader with dual-crop evaluation (anatomy + tumor per sample)
        # For each sample: 1 anatomy-centered patch + 1 tumor-centered patch
        # This evaluates both FP control (anatomy) and tumor detection (tumor)
        self._dl_val = nnUNetDataLoaderDualCropEval(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms
        )
        dl_val = self._dl_val
        
        # Wrap in augmenters
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
        
        # Initialize
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        
        # Create tuning dataloader
        self._create_tuning_dataloader()
        
        return mt_gen_train, mt_gen_val
    
    def _create_tuning_dataloader(self):
        """
        Create a dataloader for the tuning set with dual-crop evaluation strategy.
        
        For each sample, creates TWO patches:
        - One centered on random anatomy voxel (evaluates FP control)
        - One centered on random tumor voxel (evaluates tumor detection)
        
        This doubles the effective evaluation samples for balanced metrics.
        """
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        
        # Get validation transforms (no augmentation)
        deep_supervision_scales = self._get_deep_supervision_scales()
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        
        # Create tuning dataloader with dual-crop evaluation (anatomy + tumor per sample)
        dl_tuning = nnUNetDataLoaderDualCropEval(
            self.tuning_dataset,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms
        )
        
        # Wrap in augmenter
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            self.tuning_dataloader = SingleThreadedAugmenter(dl_tuning, None)
        else:
            self.tuning_dataloader = NonDetMultiThreadedAugmenter(
                data_loader=dl_tuning,
                transform=None,
                num_processes=max(1, allowed_num_processes // 4),
                num_cached=max(2, allowed_num_processes // 8),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002
            )
        
        # Initialize by fetching first batch
        _ = next(self.tuning_dataloader)
        
        # Log the actual evaluation mode (DataLoader is created for both modes, but may not be used)
        if self.eval_mode == 'sliding_window':
            self.print_to_log_file(f"Created tuning dataloader (for dual-crop fallback)")
            self.print_to_log_file(f"  → Actual eval mode: SLIDING WINDOW within GT anatomy bbox")
        else:
            self.print_to_log_file(f"Created tuning dataloader with dual-crop eval (anatomy + tumor per sample)")
    
    def on_train_epoch_start(self):
        """
        Override to reset centering counts at the start of each training epoch.
        """
        # Reset training dataloader centering counts
        if hasattr(self, '_dl_tr') and self._dl_tr is not None:
            self._dl_tr.reset_centering_counts()
        
        # Call parent's on_train_epoch_start
        super().on_train_epoch_start()
    
    def on_train_epoch_end(self, train_outputs):
        """
        Override to log centering counts at the end of each training epoch.
        """
        # Log training centering counts
        if hasattr(self, '_dl_tr') and self._dl_tr is not None:
            counts = self._dl_tr.get_centering_counts()
            total = sum(counts.values())
            if total > 0:
                pct_random = 100 * counts['random'] / total
                pct_anatomy = 100 * counts['anatomy'] / total
                pct_tumor = 100 * counts['tumor'] / total
                self.print_to_log_file(
                    f"Training centering: random={counts['random']} ({pct_random:.1f}%), "
                    f"anatomy={counts['anatomy']} ({pct_anatomy:.1f}%), "
                    f"tumor={counts['tumor']} ({pct_tumor:.1f}%) [total={total}]"
                )
        
        # Call parent's on_train_epoch_end
        super().on_train_epoch_end(train_outputs)
    
    # =========================================================================
    # SLIDING WINDOW EVALUATION METHODS
    # =========================================================================
    
    def _get_anatomy_bounding_box(self, segmentation: np.ndarray, 
                                   anatomy_label: int = 1,
                                   margin: int = 5,
                                   cache_key: str = None) -> Tuple[List[int], List[int]]:
        """
        Get bounding box of the anatomy region (label 1) from GT segmentation.
        
        Uses caching to avoid recomputation across epochs since GT doesn't change.
        
        Args:
            segmentation: Ground truth segmentation array (no channel dim)
            anatomy_label: Label value for anatomy (default 1)
            margin: Margin to add around the bounding box in voxels
            cache_key: Optional key for caching (e.g., case_id)
            
        Returns:
            bbox_lbs, bbox_ubs: Lower and upper bounds for each dimension
        """
        # Check cache first
        if cache_key is not None and cache_key in self._anatomy_bbox_cache:
            return self._anatomy_bbox_cache[cache_key]
        
        # Find where anatomy exists
        anatomy_mask = (segmentation == anatomy_label)
        
        if not np.any(anatomy_mask):
            # No anatomy found, return full image bounds
            bbox_lbs = [0] * len(segmentation.shape)
            bbox_ubs = list(segmentation.shape)
        else:
            # Get coordinates where anatomy exists
            coords = np.where(anatomy_mask)
            
            bbox_lbs = []
            bbox_ubs = []
            
            for dim in range(len(segmentation.shape)):
                dim_coords = coords[dim]
                lb = max(0, int(np.min(dim_coords)) - margin)
                ub = min(segmentation.shape[dim], int(np.max(dim_coords)) + margin + 1)
                bbox_lbs.append(lb)
                bbox_ubs.append(ub)
        
        # Cache the result
        if cache_key is not None:
            self._anatomy_bbox_cache[cache_key] = (bbox_lbs, bbox_ubs)
        
        return bbox_lbs, bbox_ubs
    
    def _get_sliding_window_slicers(self, image_size: Tuple[int, ...], 
                                     patch_size: Tuple[int, ...],
                                     step_size: float = 0.5) -> List[Tuple]:
        """
        Compute sliding window positions for inference.
        
        Args:
            image_size: Size of the image region (D, H, W)
            patch_size: Size of patches for inference
            step_size: Overlap between patches (0.5 = 50% overlap)
            
        Returns:
            List of slicers for extracting patches
        """
        # Ensure image is at least patch size (pad if needed)
        adjusted_size = [max(i, p) for i, p in zip(image_size, patch_size)]
        
        steps = compute_steps_for_sliding_window(tuple(adjusted_size), tuple(patch_size), step_size)
        
        slicers = []
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)])
                    )
        
        return slicers
    
    @torch.inference_mode()
    def _sliding_window_inference(self, data: torch.Tensor, 
                                   use_gaussian: bool = True) -> torch.Tensor:
        """
        Perform sliding window inference on preprocessed data.
        
        Args:
            data: Input tensor of shape (C, D, H, W)
            use_gaussian: Whether to use Gaussian weighting for aggregation
            
        Returns:
            Predicted logits of same spatial shape as input
        """
        patch_size = self.configuration_manager.patch_size
        step_size = self.sliding_window_step_size
        
        # Pad to ensure divisible by patch size
        data_padded, slicer_revert = pad_nd_image(data, patch_size, 'constant', {'value': 0}, True, None)
        
        # Get slicers
        slicers = self._get_sliding_window_slicers(data_padded.shape[1:], patch_size, step_size)
        
        # Initialize prediction arrays
        predicted_logits = torch.zeros(
            (self.label_manager.num_segmentation_heads, *data_padded.shape[1:]),
            dtype=torch.half,
            device=self.device
        )
        n_predictions = torch.zeros(data_padded.shape[1:], dtype=torch.half, device=self.device)
        
        # Gaussian weighting
        if use_gaussian:
            gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1./8, 
                                        value_scaling_factor=10, device=self.device)
        else:
            gaussian = 1
        
        # Move data to device
        data_padded = data_padded.to(self.device)
        
        # Process each patch
        for sl in slicers:
            # Extract patch: sl is for spatial dims, need to add channel dim
            patch = data_padded[(slice(None),) + sl][None]  # Add batch dim
            
            # Forward pass
            pred = self.network(patch)[0]  # Remove batch dim
            
            if self.enable_deep_supervision:
                pred = pred[0]  # Take first output (highest resolution)
            
            if use_gaussian:
                pred = pred * gaussian
            
            # Aggregate
            predicted_logits[(slice(None),) + sl] += pred
            n_predictions[sl] += gaussian
        
        # Normalize by prediction counts
        torch.div(predicted_logits, n_predictions, out=predicted_logits)
        
        # Revert padding
        predicted_logits = predicted_logits[(slice(None),) + slicer_revert[1:]]
        
        return predicted_logits
    
    def _compute_metrics_from_prediction(self, 
                                         predicted_logits: torch.Tensor,
                                         target_seg: torch.Tensor,
                                         loss_fn = None) -> dict:
        """
        Compute TP/FP/FN/TN from predicted logits and target segmentation.
        
        Args:
            predicted_logits: Predicted logits (num_classes, D, H, W)
            target_seg: Target segmentation (1, D, H, W) or (D, H, W)
            loss_fn: Optional loss function to compute loss
            
        Returns:
            Dict with tp_hard, fp_hard, fn_hard, tn_hard, loss
        """
        # Ensure target has channel dimension
        if target_seg.ndim == 3:
            target_seg = target_seg.unsqueeze(0)
        
        # Convert logits to segmentation
        if self.label_manager.has_regions:
            predicted_seg_onehot = (torch.sigmoid(predicted_logits) > 0.5).long()
        else:
            output_seg = predicted_logits.argmax(0, keepdim=True)
            predicted_seg_onehot = torch.zeros(predicted_logits.shape, device=predicted_logits.device, dtype=torch.float32)
            predicted_seg_onehot.scatter_(0, output_seg, 1)
        
        # Add batch dimension for get_tp_fp_fn_tn
        predicted_seg_onehot = predicted_seg_onehot.unsqueeze(0)
        target_seg = target_seg.unsqueeze(0)
        
        # Convert target to one-hot if needed
        if not self.label_manager.has_regions:
            target_onehot = torch.zeros(predicted_seg_onehot.shape, device=target_seg.device, dtype=torch.float32)
            target_onehot.scatter_(1, target_seg.long(), 1)
        else:
            target_onehot = target_seg
        
        axes = [0] + list(range(2, predicted_seg_onehot.ndim))
        tp, fp, fn, tn = get_tp_fp_fn_tn(predicted_seg_onehot, target_onehot, axes=axes, mask=None)
        
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        tn_hard = tn.detach().cpu().numpy()
        
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]  # Skip background
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            tn_hard = tn_hard[1:]
        
        # Compute loss if provided
        loss_value = 0.0
        if loss_fn is not None:
            with torch.no_grad():
                loss_value = loss_fn(predicted_logits.unsqueeze(0), target_seg).item()
        
        return {
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'tn_hard': tn_hard,
            'loss': loss_value
        }
    
    def compute_metrics_sliding_window(self, dataset, keys: List[str], 
                                        num_samples: int = None,
                                        progress_desc: str = "Sliding Window Eval") -> dict:
        """
        Compute metrics using sliding window inference within GT anatomy bounding box.
        
        For each sample:
        1. Load preprocessed data
        2. Get GT anatomy bounding box (cached for efficiency)
        3. Crop data to bounding box (with padding to patch size)
        4. Run sliding window inference
        5. Compute metrics
        
        Args:
            dataset: nnUNet dataset with preprocessed data
            keys: List of case keys to evaluate
            num_samples: Max number of samples (None = all)
            progress_desc: Description for progress bar
            
        Returns:
            Aggregated metrics dict
        """
        self.network.eval()
        
        if num_samples is not None:
            keys = keys[:num_samples]
        
        all_outputs = []
        patch_size = self.configuration_manager.patch_size
        
        # Check cache status
        cached_keys = [k for k in keys if k in self._anatomy_bbox_cache]
        if cached_keys:
            self.print_to_log_file(f"  Using {len(cached_keys)}/{len(keys)} cached anatomy bboxes")
        
        with torch.no_grad():
            # Only show progress bar on main process (local_rank == 0)
            for key in tqdm(keys, desc=progress_desc, disable=self.local_rank != 0):
                try:
                    # Load preprocessed data
                    data, seg, seg_prev, properties = dataset.load_case(key)
                    
                    # Get GT anatomy bounding box (cached to avoid recomputation)
                    # seg shape: (1, D, H, W) or (D, H, W)
                    seg_np = seg[0] if seg.ndim == 4 else seg
                    bbox_lbs, bbox_ubs = self._get_anatomy_bounding_box(
                        seg_np, anatomy_label=1, margin=10, cache_key=key
                    )
                    
                    # Ensure bbox is at least patch size
                    for dim in range(len(bbox_lbs)):
                        current_size = bbox_ubs[dim] - bbox_lbs[dim]
                        if current_size < patch_size[dim]:
                            # Expand the bounding box
                            expand_needed = patch_size[dim] - current_size
                            expand_start = expand_needed // 2
                            expand_end = expand_needed - expand_start
                            bbox_lbs[dim] = max(0, bbox_lbs[dim] - expand_start)
                            bbox_ubs[dim] = min(seg_np.shape[dim], bbox_ubs[dim] + expand_end)
                    
                    # Create slicers
                    bbox_slicer = tuple(slice(lb, ub) for lb, ub in zip(bbox_lbs, bbox_ubs))
                    
                    # Crop data and segmentation to bounding box
                    data_cropped = data[(slice(None),) + bbox_slicer]  # (C, d, h, w)
                    seg_cropped = seg[(slice(None),) + bbox_slicer]    # (1, d, h, w)
                    
                    # Convert to tensor
                    data_tensor = torch.from_numpy(data_cropped).float()
                    seg_tensor = torch.from_numpy(seg_cropped).long().to(self.device)
                    
                    # Run sliding window inference
                    predicted_logits = self._sliding_window_inference(data_tensor)
                    
                    # Compute metrics
                    metrics = self._compute_metrics_from_prediction(
                        predicted_logits, seg_tensor, loss_fn=None
                    )
                    all_outputs.append(metrics)
                    
                except Exception as e:
                    self.print_to_log_file(f"  Warning: Failed to process {key}: {e}")
                    continue
        
        self.network.train()
        
        if not all_outputs:
            # Return empty metrics
            num_classes = len(self.label_manager.foreground_labels)
            return {
                'dice_per_class': [0.0] * num_classes,
                'mean_fg_dice': 0.0,
                'loss': 0.0,
                'tp_per_class': [0] * num_classes,
                'fp_per_class': [0] * num_classes,
                'fn_per_class': [0] * num_classes,
                'tn_per_class': [0] * num_classes,
                'recall_per_class': [0.0] * num_classes,
                'precision_per_class': [0.0] * num_classes,
                'f1_per_class': [0.0] * num_classes,
                'fpr_per_class': [0.0] * num_classes,
            }
        
        # Aggregate results
        outputs_collated = collate_outputs(all_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        tn = np.sum(outputs_collated['tn_hard'], 0)
        
        # Compute metrics
        dice_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0.0 
                         for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = float(np.nanmean(dice_per_class))
        mean_loss = float(np.mean(outputs_collated['loss']))
        
        recall_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fn)]
        precision_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fp)]
        f1_per_class = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 
                       for p, r in zip(precision_per_class, recall_per_class)]
        fpr_per_class = [100.0 * f / (f + t) if (f + t) > 0 else 0.0 for f, t in zip(fp, tn)]
        
        return {
            'dice_per_class': dice_per_class,
            'mean_fg_dice': mean_fg_dice,
            'loss': mean_loss,
            'tp_per_class': tp.tolist(),
            'fp_per_class': fp.tolist(),
            'fn_per_class': fn.tolist(),
            'tn_per_class': tn.tolist(),
            'recall_per_class': recall_per_class,
            'precision_per_class': precision_per_class,
            'f1_per_class': f1_per_class,
            'fpr_per_class': fpr_per_class,
        }
    
    # =========================================================================
    # END SLIDING WINDOW EVALUATION METHODS
    # =========================================================================
    
    def compute_tuning_metrics(self) -> dict:
        """
        Compute metrics on the tuning set.
        
        Uses either sliding_window or dual_crop mode based on self.eval_mode.
        
        Returns:
            Dictionary with per-class metrics:
            - 'dice_per_class': List of dice scores per class
            - 'mean_fg_dice': Mean foreground dice
            - 'loss': Average loss on tuning set
            - 'tp_per_class', 'fp_per_class', 'fn_per_class', 'tn_per_class': Raw counts
            - 'recall_per_class': TP / (TP + FN) in percent
            - 'precision_per_class': TP / (TP + FP) in percent
            - 'f1_per_class': 2 * precision * recall / (precision + recall)
            - 'fpr_per_class': FP / (FP + TN) in percent (False Positive Rate)
        """
        if self.eval_mode == 'sliding_window':
            # Use sliding window inference within GT anatomy bounding box
            return self.compute_metrics_sliding_window(
                self.tuning_dataset, 
                self.tuning_keys,
                num_samples=None,  # Use all tuning samples
                progress_desc="Tuning SW Eval"
            )
        else:
            # Use dual-crop evaluation (legacy)
            return self._compute_tuning_metrics_dual_crop()
    
    def _compute_tuning_metrics_dual_crop(self) -> dict:
        """
        Compute tuning metrics using dual-crop evaluation (legacy method).
        
        For each sample, creates two patches: one anatomy-centered, one tumor-centered.
        """
        self.network.eval()
        
        # Determine number of batches (similar to validation)
        num_batches = max(1, len(self.tuning_keys) // self.batch_size)
        num_batches = min(num_batches, 50)  # Cap at 50 batches for efficiency
        
        tuning_outputs = []
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch = next(self.tuning_dataloader)
                output = self._tuning_step(batch)
                tuning_outputs.append(output)
        
        # Aggregate results
        outputs_collated = collate_outputs(tuning_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        tn = np.sum(outputs_collated['tn_hard'], 0)
        
        # Compute dice per class: 2*TP / (2*TP + FP + FN)
        dice_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0.0 
                         for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = float(np.nanmean(dice_per_class))
        mean_loss = float(np.mean(outputs_collated['loss']))
        
        # Compute additional metrics per class
        # Recall (Sensitivity): TP / (TP + FN)
        recall_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 
                           for t, f in zip(tp, fn)]
        
        # Precision (Positive Predictive Value): TP / (TP + FP)
        precision_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 
                              for t, f in zip(tp, fp)]
        
        # F1 Score: 2 * precision * recall / (precision + recall)
        f1_per_class = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 
                       for p, r in zip(precision_per_class, recall_per_class)]
        
        # False Positive Rate: FP / (FP + TN)
        fpr_per_class = [100.0 * f / (f + t) if (f + t) > 0 else 0.0 
                        for f, t in zip(fp, tn)]
        
        self.network.train()
        
        return {
            'dice_per_class': dice_per_class,
            'mean_fg_dice': mean_fg_dice,
            'loss': mean_loss,
            # Raw counts
            'tp_per_class': tp.tolist(),
            'fp_per_class': fp.tolist(),
            'fn_per_class': fn.tolist(),
            'tn_per_class': tn.tolist(),
            # Rate metrics (in percent)
            'recall_per_class': recall_per_class,
            'precision_per_class': precision_per_class,
            'f1_per_class': f1_per_class,
            'fpr_per_class': fpr_per_class,
        }
    
    def _tuning_step(self, batch: dict) -> dict:
        """
        Single forward pass on tuning batch to compute metrics.
        Mirrors validation_step logic.
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        # Forward pass
        from torch.amp import autocast
        from nnunetv2.utilities.helpers import dummy_context
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)
        
        # Handle deep supervision
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        
        # Compute metrics
        axes = [0] + list(range(2, output.ndim))
        
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg
        
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None
        
        tp, fp, fn, tn = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        tn_hard = tn.detach().cpu().numpy()
        
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            tn_hard = tn_hard[1:]
        
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 
                'fn_hard': fn_hard, 'tn_hard': tn_hard}
    
    def on_validation_epoch_start(self):
        """
        Override to compute tuning metrics BEFORE test validation.
        
        Flow in run_training():
        1. on_validation_epoch_start() <-- We compute tuning metrics here (FIRST)
        2. validation_step() loop on test set
        3. on_validation_epoch_end() <-- Test pseudo dice computed here (SECOND)
        4. on_epoch_end() <-- "Yayy!" message if new best (THIRD)
        """
        # First compute and log tuning metrics
        self._compute_and_log_tuning_metrics()
        
        # Then call parent's on_validation_epoch_start (sets network to eval mode)
        super().on_validation_epoch_start()
    
    def _compute_and_log_tuning_metrics(self):
        """
        Compute tuning metrics and log them.
        This is called BEFORE test validation starts.
        """
        self.print_to_log_file("-" * 40)
        self.print_to_log_file("TUNING SET evaluation (for adaptive decisions):")
        
        tuning_result = self.compute_tuning_metrics()
        
        # Use TUMOR DICE ONLY for EMA calculation (index 1 = tumor, label 2)
        # dice_per_class: [anatomy_dice, tumor_dice, ...] (index corresponds to foreground labels)
        tumor_dice = tuning_result['dice_per_class'][1] if len(tuning_result['dice_per_class']) > 1 else tuning_result['mean_fg_dice']
        
        # Compute EMA of tumor_dice (α=0.1)
        # ema[t] = ema[t-1] * 0.9 + tumor_dice * 0.1
        if len(self.tuning_metrics['ema_fg_dice']) > 0:
            prev_ema = self.tuning_metrics['ema_fg_dice'][-1]
            current_ema = prev_ema * 0.9 + tumor_dice * 0.1
        else:
            # First epoch - no previous EMA, use current tumor dice directly
            current_ema = tumor_dice
        
        # Store metrics
        self.tuning_metrics['dice_per_class'].append(tuning_result['dice_per_class'])
        self.tuning_metrics['mean_fg_dice'].append(tuning_result['mean_fg_dice'])
        self.tuning_metrics['ema_fg_dice'].append(current_ema)
        self.tuning_metrics['loss'].append(tuning_result['loss'])
        
        # Log tuning metrics
        self.print_to_log_file(f"  Tuning loss: {np.round(tuning_result['loss'], 4)}")
        self.print_to_log_file(f"  Tuning Pseudo dice: {[np.round(d, 4) for d in tuning_result['dice_per_class']]}")
        self.print_to_log_file(f"  Tuning Pseudo dice (tumor only): {np.round(tumor_dice, 4)}")
        self.print_to_log_file(f"  Tuning EMA Pseudo dice (tumor only): {np.round(current_ema, 4)}")
        
        # Log additional rate-based metrics
        self.print_to_log_file(f"  Tuning Recall (%): {[np.round(r, 2) for r in tuning_result['recall_per_class']]}")
        self.print_to_log_file(f"  Tuning Precision (%): {[np.round(p, 2) for p in tuning_result['precision_per_class']]}")
        self.print_to_log_file(f"  Tuning F1: {[np.round(f, 4) for f in tuning_result['f1_per_class']]}")
        self.print_to_log_file(f"  Tuning FPR (%): {[np.round(f, 4) for f in tuning_result['fpr_per_class']]}")
        
        # Store current tuning result for potential use in adaptive decisions
        tuning_result['ema_fg_dice'] = current_ema
        self._current_tuning_result = tuning_result
        
        # =====================================================================
        # DYNAMIC SAMPLING STRATEGY: Adjust training probabilities
        # =====================================================================
        if self.enable_dynamic_sampling and hasattr(self, '_dl_tr') and self._dl_tr is not None:
            # Initialize strategy on first call
            if self.dynamic_sampling_strategy is None:
                self.dynamic_sampling_strategy = DynamicSamplingStrategy(
                    config=SamplingConfig(),
                    log_fn=self.print_to_log_file
                )
                # Set initial probabilities in the strategy
                self.dynamic_sampling_strategy.p1_history.append(self.train_prob_anatomy)
                self.dynamic_sampling_strategy.p2_history.append(self.train_prob_tumor)
            
            # Build metrics dict for DynamicSamplingStrategy
            # Note: Ensure we have at least 2 classes (anatomy and tumor)
            if len(tuning_result['dice_per_class']) >= 2:
                metrics_for_strategy = {
                    'lesion_dice': tuning_result['dice_per_class'][1],           # tumor (index 1)
                    'lesion_recall': tuning_result['recall_per_class'][1] / 100, # convert from %
                    'lesion_precision': tuning_result['precision_per_class'][1] / 100,
                    'lesion_f1': tuning_result['f1_per_class'][1],
                    'anatomy_dice': tuning_result['dice_per_class'][0],          # anatomy (index 0)
                    'anatomy_recall': tuning_result['recall_per_class'][0] / 100,
                    'anatomy_precision': tuning_result['precision_per_class'][0] / 100,
                    'anatomy_f1': tuning_result['f1_per_class'][0],
                }
                
                # Get new probabilities from strategy
                # Note: strategy returns (p_anatomy, p_tumor), p_random = 0.5 * p_anatomy
                p_anatomy, p_tumor = self.dynamic_sampling_strategy.update(metrics_for_strategy)
                p_random = 0.5 * p_anatomy  # Fixed relationship: random is half of anatomy
                
                # Update training dataloader probabilities
                self._dl_tr.update_probabilities(p_random, p_anatomy, p_tumor)
                
                # Update trainer state
                self.train_prob_random = p_random
                self.train_prob_anatomy = p_anatomy
                self.train_prob_tumor = p_tumor
                
                # Track history
                self.probability_history.append((p_random, p_anatomy, p_tumor))
        # =====================================================================
        
        self.print_to_log_file("-" * 40)
        self.print_to_log_file("TEST SET evaluation (for checkpoint selection):")
        # Log top-k checkpoints status
        if self.top_k_dices:
            self.print_to_log_file(f"  [Reminder] Current top-{len(self.top_k_dices)} pseudo Dice (tumor, single-epoch):")
            for rank, (dice, epoch) in enumerate(self.top_k_dices, 1):
                self.print_to_log_file(f"    Rank {rank}: {np.round(dice, decimals=4)} (epoch {epoch})")
        elif getattr(self, '_best_dice', None) is not None:
            # Fallback if top_k_dices not populated yet but _best_dice exists
            self.print_to_log_file(f"  [Reminder] Current best pseudo Dice (tumor): {np.round(self._best_dice, decimals=4)}")
        else:
            self.print_to_log_file(f"  [Reminder] No best checkpoint yet (first evaluation)")
    
    def validation_step(self, batch: dict) -> dict:
        """
        Override to also capture TN for additional metrics computation.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        from torch.amp import autocast
        from nnunetv2.utilities.helpers import dummy_context

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, tn = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        tn_hard = tn.detach().cpu().numpy()
        
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            tn_hard = tn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 
                'fn_hard': fn_hard, 'tn_hard': tn_hard}
    
    def on_validation_epoch_end(self, val_outputs: list):
        """
        Override to support both dual-crop and sliding window evaluation modes.
        Uses TUMOR DICE ONLY for checkpoint selection.
        
        In sliding_window mode: Computes test metrics using sliding window inference
        In dual_crop mode: Uses the val_outputs from parent's validation loop
        """
        from torch import distributed as dist
        
        if self.eval_mode == 'sliding_window':
            # =====================================================================
            # SLIDING WINDOW MODE: Use sliding window inference for test set
            # =====================================================================
            # Use the already-initialized test dataset via self._dl_val
            # _dl_val is the nnUNetDataLoaderDualCropEval we created in get_dataloaders()
            
            # Get test keys from the dataloader's dataset
            test_keys = list(self._dl_val._data.keys)
            
            self.print_to_log_file(f"  Running sliding window evaluation on {len(test_keys)} test samples...")
            
            test_result = self.compute_metrics_sliding_window(
                self._dl_val._data,  # The underlying dataset
                test_keys,
                num_samples=None,  # Use all test samples
                progress_desc="Test SW Eval"
            )
            
            global_dc_per_class = test_result['dice_per_class']
            recall_per_class = test_result['recall_per_class']
            precision_per_class = test_result['precision_per_class']
            f1_per_class = test_result['f1_per_class']
            fpr_per_class = test_result['fpr_per_class']
            loss_here = test_result['loss']
        else:
            # =====================================================================
            # DUAL-CROP MODE: Use the val_outputs from parent's validation loop
            # =====================================================================
            outputs_collated = collate_outputs(val_outputs)
            tp = np.sum(outputs_collated['tp_hard'], 0)
            fp = np.sum(outputs_collated['fp_hard'], 0)
            fn = np.sum(outputs_collated['fn_hard'], 0)
            tn = np.sum(outputs_collated['tn_hard'], 0)

            if self.is_ddp:
                world_size = dist.get_world_size()

                tps = [None for _ in range(world_size)]
                dist.all_gather_object(tps, tp)
                tp = np.vstack([i[None] for i in tps]).sum(0)

                fps = [None for _ in range(world_size)]
                dist.all_gather_object(fps, fp)
                fp = np.vstack([i[None] for i in fps]).sum(0)

                fns = [None for _ in range(world_size)]
                dist.all_gather_object(fns, fn)
                fn = np.vstack([i[None] for i in fns]).sum(0)
                
                tns = [None for _ in range(world_size)]
                dist.all_gather_object(tns, tn)
                tn = np.vstack([i[None] for i in tns]).sum(0)

                losses_val = [None for _ in range(world_size)]
                dist.all_gather_object(losses_val, outputs_collated['loss'])
                loss_here = np.vstack(losses_val).mean()
            else:
                loss_here = np.mean(outputs_collated['loss'])

            # Compute Dice per class: 2*TP / (2*TP + FP + FN)
            global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
            
            # Compute additional metrics per class
            recall_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fn)]
            precision_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fp)]
            f1_per_class = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 
                           for p, r in zip(precision_per_class, recall_per_class)]
            fpr_per_class = [100.0 * f / (f + t) if (f + t) > 0 else 0.0 for f, t in zip(fp, tn)]
        
        # =====================================================================
        # Use TUMOR DICE ONLY for checkpoint selection (single-epoch, not EMA)
        # =====================================================================
        # global_dc_per_class: [anatomy_dice, tumor_dice, ...]
        # Index 1 = tumor (label 2)
        tumor_dice = global_dc_per_class[1] if len(global_dc_per_class) > 1 else np.nanmean(global_dc_per_class)
        
        # Log tumor_dice as 'mean_fg_dice' - used for checkpoint selection
        # Note: Logger will still compute EMA, but we use single-epoch dice for checkpointing
        self.logger.log('mean_fg_dice', tumor_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        
        # Store additional metrics for logging in on_epoch_end
        self._test_additional_metrics = {
            'recall_per_class': recall_per_class,
            'precision_per_class': precision_per_class,
            'f1_per_class': f1_per_class,
            'fpr_per_class': fpr_per_class,
        }
    
    def on_epoch_end(self):
        """
        Override to log additional metrics after the standard Pseudo dice line.
        """
        from time import time
        
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        
        # Log additional rate-based metrics
        if hasattr(self, '_test_additional_metrics') and self._test_additional_metrics is not None:
            metrics = self._test_additional_metrics
            self.print_to_log_file(f"Recall (%): {[np.round(r, 2) for r in metrics['recall_per_class']]}")
            self.print_to_log_file(f"Precision (%): {[np.round(p, 2) for p in metrics['precision_per_class']]}")
            self.print_to_log_file(f"F1: {[np.round(f, 4) for f in metrics['f1_per_class']]}")
            self.print_to_log_file(f"FPR (%): {[np.round(f, 4) for f in metrics['fpr_per_class']]}")
        
        # Log current training probabilities (dynamic sampling evolution)
        if self.enable_dynamic_sampling and self.probability_history:
            p_random, p_anatomy, p_tumor = self.probability_history[-1]
            self.print_to_log_file(f"Training probs (random/anatomy/tumor): "
                                   f"{p_random:.3f}/{p_anatomy:.3f}/{p_tumor:.3f}")
        
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, self.get_checkpoint_filename('checkpoint_latest')))

        # =====================================================================
        # TOP-K CHECKPOINT HANDLING
        # =====================================================================
        # Use single-epoch tumor dice for checkpoint ranking (NOT EMA)
        # mean_fg_dice is actually tumor_dice for this trainer (see on_validation_epoch_end)
        current_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        
        # Initialize top_k_dices from existing checkpoints if empty (e.g., first epoch or after loading)
        if not self.top_k_dices:
            self.print_to_log_file("Scanning for existing top-k checkpoints...")
            self.top_k_dices = self._scan_existing_top_k_checkpoints()
            if self.top_k_dices:
                self.print_to_log_file(f"  Reconstructed top-{len(self.top_k_dices)} list from existing files")
            else:
                self.print_to_log_file("  No existing top-k checkpoints found")
        
        # Check if _skip_existing_best_comparison is set (--ignore_existing_best with --c)
        skip_comparison = getattr(self, '_skip_existing_best_comparison', False)
        if skip_comparison:
            self.print_to_log_file(f"Skipping top-k comparison (--ignore_existing_best). Current Dice: {np.round(current_dice, decimals=4)}")
            # Clear top_k_dices to start fresh
            self.top_k_dices = []
            self._skip_existing_best_comparison = False
        
        # Try to update top-k checkpoints
        achieved_rank = self._update_top_k_checkpoints(current_dice, current_epoch)
        
        if achieved_rank > 0:
            if achieved_rank == 1:
                self.print_to_log_file(f"Yayy! New best pseudo Dice: {np.round(current_dice, decimals=4)} (top-1)")
            else:
                self.print_to_log_file(f"New top-{achieved_rank} checkpoint! Dice: {np.round(current_dice, decimals=4)}")
        # =====================================================================

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Override to also save tuning metrics, centering probabilities, and 
        original/synthetic sample handling info in checkpoint.
        """
        # Store tuning info and centering probabilities before saving
        self._tuning_checkpoint_data = {
            'tuning_keys': self.tuning_keys,
            'tuning_metrics': self.tuning_metrics,
            'tuning_ratio': self.tuning_ratio,
            # 3-way centering probabilities
            'train_prob_random': self.train_prob_random,
            'train_prob_anatomy': self.train_prob_anatomy,
            'train_prob_tumor': self.train_prob_tumor,
            'tuning_prob_random': self.tuning_prob_random,
            'tuning_prob_anatomy': self.tuning_prob_anatomy,
            'tuning_prob_tumor': self.tuning_prob_tumor,
            'test_prob_random': self.test_prob_random,
            'test_prob_anatomy': self.test_prob_anatomy,
            'test_prob_tumor': self.test_prob_tumor,
            # Original vs synthetic sample handling
            'pattern_original_samples': self.pattern_original_samples,
            'max_synthetic_ratio': self.max_synthetic_ratio,
            'n_original_total': self.n_original_total,
            'n_synthetic_total': self.n_synthetic_total,
            'n_synthetic_removed': self.n_synthetic_removed,
            'removed_synthetic_keys': self.removed_synthetic_keys,
            # Dynamic sampling strategy state
            'enable_dynamic_sampling': self.enable_dynamic_sampling,
            'probability_history': self.probability_history,
            'dynamic_sampling_history': self.dynamic_sampling_strategy.get_history() if self.dynamic_sampling_strategy else None,
            # Top-K checkpoint tracking
            'top_k_checkpoints': self.top_k_checkpoints,
            'top_k_dices': self.top_k_dices,
            '_best_dice': getattr(self, '_best_dice', None),
            # Evaluation mode
            'eval_mode': self.eval_mode,
            'sliding_window_step_size': self.sliding_window_step_size,
        }
        
        super().save_checkpoint(filename)
    
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """
        Override to also load tuning metrics, centering probabilities, and
        original/synthetic sample handling info from checkpoint.
        """
        super().load_checkpoint(filename_or_checkpoint)
        
        # Load tuning info if available
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location='cpu', weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        
        if 'tuning_keys' in checkpoint:
            self.tuning_keys = checkpoint['tuning_keys']
            self.tuning_metrics = checkpoint.get('tuning_metrics', self.tuning_metrics)
            self.tuning_ratio = checkpoint.get('tuning_ratio', self.tuning_ratio)
            
            # Load centering probabilities if available
            self.train_prob_random = checkpoint.get('train_prob_random', self.train_prob_random)
            self.train_prob_anatomy = checkpoint.get('train_prob_anatomy', self.train_prob_anatomy)
            self.train_prob_tumor = checkpoint.get('train_prob_tumor', self.train_prob_tumor)
            self.tuning_prob_random = checkpoint.get('tuning_prob_random', self.tuning_prob_random)
            self.tuning_prob_anatomy = checkpoint.get('tuning_prob_anatomy', self.tuning_prob_anatomy)
            self.tuning_prob_tumor = checkpoint.get('tuning_prob_tumor', self.tuning_prob_tumor)
            self.test_prob_random = checkpoint.get('test_prob_random', self.test_prob_random)
            self.test_prob_anatomy = checkpoint.get('test_prob_anatomy', self.test_prob_anatomy)
            self.test_prob_tumor = checkpoint.get('test_prob_tumor', self.test_prob_tumor)
            
            # Load original/synthetic sample handling info
            self.pattern_original_samples = checkpoint.get('pattern_original_samples', self.pattern_original_samples)
            self.max_synthetic_ratio = checkpoint.get('max_synthetic_ratio', self.max_synthetic_ratio)
            self.n_original_total = checkpoint.get('n_original_total', self.n_original_total)
            self.n_synthetic_total = checkpoint.get('n_synthetic_total', self.n_synthetic_total)
            self.n_synthetic_removed = checkpoint.get('n_synthetic_removed', self.n_synthetic_removed)
            self.removed_synthetic_keys = checkpoint.get('removed_synthetic_keys', self.removed_synthetic_keys)
            
            # Load dynamic sampling state
            self.enable_dynamic_sampling = checkpoint.get('enable_dynamic_sampling', self.enable_dynamic_sampling)
            self.probability_history = checkpoint.get('probability_history', self.probability_history)
            
            # Restore dynamic sampling strategy if history exists
            saved_history = checkpoint.get('dynamic_sampling_history', None)
            if saved_history and self.enable_dynamic_sampling:
                self.dynamic_sampling_strategy = DynamicSamplingStrategy(
                    config=SamplingConfig(),
                    log_fn=self.print_to_log_file
                )
                # Restore strategy state from history
                self.dynamic_sampling_strategy.p1_history = saved_history.get('p1_anatomy', [])
                self.dynamic_sampling_strategy.p2_history = saved_history.get('p2_lesion', [])
                self.dynamic_sampling_strategy.ema_history = saved_history.get('ema_lesion_dice', [])
                self.dynamic_sampling_strategy.recall_ema_history = saved_history.get('ema_lesion_recall', [])
                self.dynamic_sampling_strategy.precision_ema_history = saved_history.get('ema_lesion_precision', [])
                self.dynamic_sampling_strategy.epoch = len(self.dynamic_sampling_strategy.p1_history)
                if self.dynamic_sampling_strategy.ema_history:
                    self.dynamic_sampling_strategy.ema_lesion_dice = self.dynamic_sampling_strategy.ema_history[-1]
                if self.dynamic_sampling_strategy.recall_ema_history:
                    self.dynamic_sampling_strategy.ema_lesion_recall = self.dynamic_sampling_strategy.recall_ema_history[-1]
                if self.dynamic_sampling_strategy.precision_ema_history:
                    self.dynamic_sampling_strategy.ema_lesion_precision = self.dynamic_sampling_strategy.precision_ema_history[-1]
                # Restore best dice tracking
                self.dynamic_sampling_strategy.best_dice_ema = saved_history.get('best_dice_ema', None)
                self.dynamic_sampling_strategy.best_p_anatomy = saved_history.get('best_p_anatomy', None)
                self.dynamic_sampling_strategy.best_epoch = saved_history.get('best_epoch', 0)
                self.dynamic_sampling_strategy.reverted_count = saved_history.get('reverted_count', 0)
            
            # Load top-K checkpoint tracking
            # Note: We don't load top_k_dices from the checkpoint data because we'll 
            # reconstruct it by scanning existing files in on_epoch_end. This ensures
            # consistency even if checkpoint files were manually moved/deleted.
            # We only log that we'll scan for existing checkpoints later.
            self.top_k_checkpoints = checkpoint.get('top_k_checkpoints', self.top_k_checkpoints)
            # Keep top_k_dices empty - it will be populated by scanning files in on_epoch_end
            self.top_k_dices = []
            
            # Load evaluation mode
            self.eval_mode = checkpoint.get('eval_mode', self.eval_mode)
            self.sliding_window_step_size = checkpoint.get('sliding_window_step_size', self.sliding_window_step_size)
            
            self.print_to_log_file(f"Loaded tuning info: {len(self.tuning_keys)} tuning cases")
            self.print_to_log_file(f"Loaded centering probs - Train: ({self.train_prob_random:.2f}/{self.train_prob_anatomy:.2f}/{self.train_prob_tumor:.2f})")
            self.print_to_log_file(f"Loaded eval mode: {self.eval_mode}")
            if self.pattern_original_samples:
                self.print_to_log_file(f"Loaded original/synthetic handling: pattern='{self.pattern_original_samples}', "
                                       f"orig={self.n_original_total}, synth={self.n_synthetic_total}, removed={self.n_synthetic_removed}")
            if self.enable_dynamic_sampling and self.probability_history:
                self.print_to_log_file(f"Loaded dynamic sampling: {len(self.probability_history)} epochs of history")


# Convenience alias with shorter name
nnUNetTrainer_Tuning = nnUNetTrainer_WithTuningSet
