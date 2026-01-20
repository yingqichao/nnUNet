"""
nnUNet Trainer with separate tuning set for adaptive training strategies.

This trainer implements a 3-way split within each fold:
- Training set: For gradient updates (80% of original train)
- Tuning set: For adaptive strategy decisions (20% of original train)
- Test set: For final evaluation only (original validation fold, untouched)

This design allows validation-informed adaptive training without data leakage,
because the test set is never used for any training decisions.

Key features:
- Automatic 20% tuning split from training data (no JSON modification needed)
- Tuning metrics computed FIRST each epoch, then test metrics
- Configurable 3-way patch centering: random / anatomy / tumor
- Extension point for adaptive hyperparameter adjustment based on tuning performance
- Original vs synthetic sample handling via --pattern_original_samples
- Tuning set always uses ONLY original samples
- Synthetic data ratio capped at 50% in training set
- EMA pseudo dice based on TUMOR DICE ONLY (not mean of all foreground classes)

Evaluation order each epoch:
1. TUNING SET: Compute metrics (logged, for adaptive decisions)
2. TEST SET: Compute pseudo dice (logged, for checkpoint selection)
3. Checkpoint: "Yayy! New best EMA pseudo Dice" if improved (based on TEST tumor dice only)

Patch centering probabilities (configurable per mode):
- prob_random: Center on random voxel (anywhere)
- prob_anatomy: Center on random anatomy voxel (label 1)
- prob_tumor: Center on random tumor voxel (label >= 2)
(Probabilities must sum to 1.0)

Label convention:
- Label 0: Background
- Label 1: Anatomy (liver for LiTS, kidney for KiTS)
- Label 2+: Tumor/lesion

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
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

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
    - Tumor: Center on random tumor voxel (label >= 2)
    
    This provides fine-grained control over patch sampling strategy.
    """
    
    # Label conventions
    BACKGROUND_LABEL = 0
    ANATOMY_LABEL = 1  # liver for LiTS, kidney for KiTS
    # Labels >= 2 are considered tumor/lesion
    
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
            prob_tumor: Probability of centering on tumor voxel (label >= 2)
            
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
        
        # Get tumor labels (all labels >= 2)
        self.tumor_labels = [l for l in label_manager.foreground_labels if l >= 2]
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
            prob_tumor: Probability of centering on tumor voxel (label >= 2)
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
            # Try to center on tumor (label >= 2)
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
        self.tuning_ratio = 0.2  # 20% of original training for tuning
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
        self.train_prob_tumor = 0.5     # Center on tumor voxel (label >= 2)
        
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
        
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Using nnUNetTrainer_WithTuningSet")
        self.print_to_log_file("=" * 70)
        self.print_to_log_file(f"3-way split: {(1-self.tuning_ratio)*100:.0f}% train, "
                               f"{self.tuning_ratio*100:.0f}% tuning, test=original val fold")
        self.print_to_log_file("Tuning set: For adaptive training decisions (NO leakage to test)")
        self.print_to_log_file("Test set: For final evaluation ONLY (original val fold)")
        self.print_to_log_file("-" * 70)
        self.print_to_log_file("Patch centering probabilities (random / anatomy / tumor):")
        self.print_to_log_file(f"  Training: ({self.train_prob_random:.2f} / {self.train_prob_anatomy:.2f} / {self.train_prob_tumor:.2f})")
        self.print_to_log_file(f"  Tuning:   ({self.tuning_prob_random:.2f} / {self.tuning_prob_anatomy:.2f} / {self.tuning_prob_tumor:.2f})")
        self.print_to_log_file(f"  Test:     ({self.test_prob_random:.2f} / {self.test_prob_anatomy:.2f} / {self.test_prob_tumor:.2f})")
        self.print_to_log_file("-" * 70)
        self.print_to_log_file(f"Dynamic sampling: {'ENABLED' if self.enable_dynamic_sampling else 'DISABLED'}")
        if self.enable_dynamic_sampling:
            self.print_to_log_file("  → Training probabilities will be adjusted based on tuning metrics")
        self.print_to_log_file("=" * 70 + "\n")
    
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
        rng = np.random.RandomState(seed=12345 + self.fold)
        
        # Shuffle original samples for tuning selection
        shuffled_original = rng.permutation(original_keys).tolist()
        
        # Calculate tuning set size (from original samples only)
        n_tuning = int(len(shuffled_original) * self.tuning_ratio)
        self.tuning_keys = shuffled_original[:n_tuning]
        remaining_original = shuffled_original[n_tuning:]
        
        self.print_to_log_file(f"Tuning set: {len(self.tuning_keys)} cases (original samples only)")
        
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
        
        # Create TEST dataloader with 3-way centering
        # Store reference for accessing centering counts
        self._dl_val = nnUNetDataLoader3WayCentering(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            prob_random=self.test_prob_random,
            prob_anatomy=self.test_prob_anatomy,
            prob_tumor=self.test_prob_tumor,
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
        Create a dataloader for the tuning set with 3-way centering.
        Uses validation transforms (no augmentation) for consistent metrics.
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
        
        # Create tuning dataloader with 3-way centering
        dl_tuning = nnUNetDataLoader3WayCentering(
            self.tuning_dataset,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            prob_random=self.tuning_prob_random,
            prob_anatomy=self.tuning_prob_anatomy,
            prob_tumor=self.tuning_prob_tumor,
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
        
        self.print_to_log_file(f"Created tuning dataloader with 3-way centering")
    
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
    
    def compute_tuning_metrics(self) -> dict:
        """
        Compute metrics on the tuning set.
        
        This is similar to validation_step but on the tuning set.
        These metrics can be used for adaptive training decisions.
        
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
        
        # Use TUMOR DICE ONLY for EMA calculation (index 1 = tumor, label >= 2)
        # dice_per_class: [anatomy_dice, tumor_dice]
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
        Override to use TUMOR DICE ONLY for EMA calculation instead of mean foreground dice.
        Also computes and logs additional metrics: Recall, Precision, F1, FPR.
        
        dice_per_class_or_region: [anatomy_dice, tumor_dice]
        We use index 1 (tumor) for the EMA that determines checkpoint selection.
        """
        from torch import distributed as dist
        
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
        
        # =====================================================================
        # KEY CHANGE: Use TUMOR DICE ONLY for EMA calculation
        # =====================================================================
        # global_dc_per_class: [anatomy_dice, tumor_dice]
        # Index 1 = tumor (label >= 2)
        tumor_dice = global_dc_per_class[1] if len(global_dc_per_class) > 1 else np.nanmean(global_dc_per_class)
        
        # Log tumor_dice as 'mean_fg_dice' - the logger will compute EMA from this
        # This makes the checkpoint selection based on tumor dice EMA
        self.logger.log('mean_fg_dice', tumor_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        
        # Log additional rate-based metrics for test set (will appear after "Pseudo dice" in on_epoch_end)
        # Store for logging in on_epoch_end
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

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        current_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
        if self._best_ema is None or current_ema > self._best_ema:
            # Use the new method that checks existing checkpoint before overwriting
            if self.save_best_checkpoint_if_improved(current_ema):
                self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")

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
            
            self.print_to_log_file(f"Loaded tuning info: {len(self.tuning_keys)} tuning cases")
            self.print_to_log_file(f"Loaded centering probs - Train: ({self.train_prob_random:.2f}/{self.train_prob_anatomy:.2f}/{self.train_prob_tumor:.2f})")
            if self.pattern_original_samples:
                self.print_to_log_file(f"Loaded original/synthetic handling: pattern='{self.pattern_original_samples}', "
                                       f"orig={self.n_original_total}, synth={self.n_synthetic_total}, removed={self.n_synthetic_removed}")
            if self.enable_dynamic_sampling and self.probability_history:
                self.print_to_log_file(f"Loaded dynamic sampling: {len(self.probability_history)} epochs of history")


# Convenience alias with shorter name
nnUNetTrainer_Tuning = nnUNetTrainer_WithTuningSet
