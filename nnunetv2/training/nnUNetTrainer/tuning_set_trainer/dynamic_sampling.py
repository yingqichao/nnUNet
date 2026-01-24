"""
Dynamic sampling strategy for adaptive patch centering.

Adjusts the probability distribution of patch centering modes
(random/anatomy/tumor) based on training performance metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


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
