"""
Metric computation utilities for evaluation.

Pure functions for computing Dice scores, TP/FP/FN/TN, and other metrics.
These functions work with numpy arrays and don't require GPU.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.special import softmax as scipy_softmax


def compute_dice_score_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dice score between binary prediction and ground truth (numpy).
    
    Args:
        pred: Binary prediction array (any shape)
        gt: Binary ground truth array (same shape as pred)
    
    Returns:
        Dice score (float between 0 and 1)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if pred_sum + gt_sum == 0:
        return 1.0  # Both empty, perfect match
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return float(dice)


def compute_tp_fp_fn_tn_np(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN between binary prediction and ground truth (numpy).
    
    Args:
        pred: Binary prediction array
        gt: Binary ground truth array
    
    Returns:
        Tuple of (TP, FP, FN, TN) as integers
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    
    return int(tp), int(fp), int(fn), int(tn)


def compute_class_metrics_np(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    class_idx: int,
    simulate_perfect_anatomy: bool = False
) -> Dict[str, Optional[float]]:
    """
    Compute metrics for a specific class using numpy arrays.
    
    This function is shared between:
    - nnUNetTrainer_with_tuning_set.py (for training evaluation)
    - eval_with_soft_predictions.py (for post-hoc evaluation)
    
    Args:
        pred_seg: Predicted segmentation (D, H, W) with class indices
        gt_seg: Ground truth segmentation (1, D, H, W) or (D, H, W) with class indices
            May contain -1 as ignore label (will be treated as background)
        class_idx: Class index to evaluate (e.g., 2 for tumor)
        simulate_perfect_anatomy: If True, zero out predictions where GT is background (label 0).
            This simulates "perfect anatomy detection" and computes tumor metrics only
            within the GT anatomy region, reducing FP from outside anatomy.
    
    Returns:
        Dictionary with:
        - "Dice": Dice score (None if undefined)
        - "IoU": Intersection over Union (None if undefined)
        - "TP", "FP", "FN", "TN": Raw counts
        - "Recall": TP / (TP + FN) in percent
        - "Precision": TP / (TP + FP) in percent
        - "n_pred": Number of predicted voxels
        - "n_ref": Number of reference (GT) voxels
    """
    # Handle channel dimension for both gt and pred
    if gt_seg.ndim == 4:
        gt_seg = gt_seg[0]
    if pred_seg.ndim == 4:
        pred_seg = pred_seg[0]
    
    # Handle ignore label (-1): treat as background for evaluation
    # nnUNet uses -1 for regions to exclude (padding, uncertain regions)
    if np.any(gt_seg < 0):
        gt_seg = gt_seg.copy()
        gt_seg[gt_seg < 0] = 0
    
    # Apply postprocessing: zero predictions outside GT anatomy
    if simulate_perfect_anatomy:
        gt_anatomy_mask = (gt_seg != 0)  # GT is not background
        pred_seg_pp = pred_seg.copy()
        pred_seg_pp[~gt_anatomy_mask] = 0  # Zero predictions outside GT anatomy
    else:
        pred_seg_pp = pred_seg
    
    # Binary masks for the target class
    pred_mask = (pred_seg_pp == class_idx)
    gt_mask = (gt_seg == class_idx)
    
    # Compute metrics
    tp, fp, fn, tn = compute_tp_fp_fn_tn_np(pred_mask, gt_mask)
    
    # Dice and IoU
    if tp + fp + fn == 0:
        dice = None
        iou = None
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    
    # Recall and Precision
    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return {
        "Dice": float(dice) if dice is not None else None,
        "IoU": float(iou) if iou is not None else None,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Recall": recall,
        "Precision": precision,
        "n_pred": tp + fp,
        "n_ref": tp + fn
    }


def compute_anatomy_dice_np(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray
) -> float:
    """
    Compute Dice score for anatomy prediction (non-background).
    
    Args:
        pred_seg: Predicted segmentation (D, H, W)
        gt_seg: Ground truth segmentation (1, D, H, W) or (D, H, W)
    
    Returns:
        Dice score for anatomy segmentation (foreground vs background)
    """
    if gt_seg.ndim == 4:
        gt_seg = gt_seg[0]
    
    # Anatomy = anywhere not background (class != 0)
    pred_anatomy = (pred_seg != 0).astype(np.uint8)
    gt_anatomy = (gt_seg != 0).astype(np.uint8)
    
    return compute_dice_score_np(pred_anatomy, gt_anatomy)


def soft_pred_to_segmentation_np(
    soft_pred: np.ndarray,
    apply_softmax: bool = False
) -> np.ndarray:
    """
    Convert soft predictions (logits) to hard segmentation using argmax.
    
    Args:
        soft_pred: Soft prediction array with shape (num_classes, D, H, W)
                   These are logits (pre-softmax) from nnUNet
        apply_softmax: Whether to apply softmax first (doesn't change argmax result,
                       but included for consistency)
    
    Returns:
        Hard segmentation array with shape (D, H, W) containing class indices
    """
    if apply_softmax:
        probs = scipy_softmax(soft_pred, axis=0)
    else:
        probs = soft_pred
    
    # Argmax across class dimension
    segmentation = np.argmax(probs, axis=0)
    
    return segmentation
