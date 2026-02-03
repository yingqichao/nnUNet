"""
Multiprocess evaluation worker functions for async test set evaluation.

These functions are at module level to be picklable for multiprocessing.
They handle:
- Async test evaluation in subprocesses
- Sanity check for tumor presence in samples
- Visualization of predictions vs GT

GPU CONFIGURATION FOR SUBPROCESS:
We set CUDA_VISIBLE_DEVICES in the parent process BEFORE spawning. The 'spawn'
context creates a fresh Python interpreter that inherits the environment at
spawn time. After spawning, we immediately restore the parent's original
CUDA_VISIBLE_DEVICES. This ensures:
1. The subprocess uses the configured GPU (backup_gpu_ids[i])
2. The parent process continues using its original GPU
3. No CUDA context is shared between processes (avoiding crashes)
"""

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

from .constants import SANITY_CHECK_NUM_WORKERS

# Visualization settings
VIS_NUM_SAMPLES = 10           # Number of samples to visualize
VIS_MAX_SLICES_PER_SAMPLE = 8  # Max slices with tumor per sample
VIS_TUMOR_LABEL = 2            # Label for tumor class


def _convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, numpy array/scalar, or primitive)
        
    Returns:
        JSON-serializable version of the object
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_json_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@dataclass
class SubprocessInfo:
    """Information about a running evaluation subprocess."""
    process: Any  # mp.Process object
    pid: int
    epoch: int
    gpu_id: int
    result_path: str
    temp_ckpt_path: str
    start_time: float  # time.time() when spawned


def _create_visualization_figures(
    vis_data: List[dict],
    output_folder: str,
    epoch: int,
    plans_identifier: str,
    _mp_log_fn=None
) -> Optional[str]:
    """
    Create visualization figures for prediction results.
    
    For each sample in vis_data, creates a multi-row, 3-column figure:
    - Column 1: Original slice (grayscale)
    - Column 2: Original + predicted tumor mask (blue overlay, alpha=0.5)
    - Column 3: Original + GT tumor mask (green overlay, alpha=0.5)
    
    Only tumor masks are overlaid (not anatomy).
    
    Args:
        vis_data: List of dicts with keys 'key', 'data', 'pred_seg', 'gt_seg'
        output_folder: Base output folder for checkpoints
        epoch: Current epoch number
        plans_identifier: Plans name for folder naming
        _mp_log_fn: Logging function (optional)
    
    Returns:
        Path to the visualization folder, or None if failed
    """
    import os
    import numpy as np
    
    if _mp_log_fn is None:
        _mp_log_fn = print
    
    if not vis_data:
        _mp_log_fn("[VIS] No data for visualization")
        return None
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for subprocess
        import matplotlib.pyplot as plt
        
        # Create temp folder for figures (matches checkpoint naming without .pth)
        # e.g., checkpoint_eval_temp_epoch5 -> vis_eval_temp_epoch5
        vis_folder_name = f"vis_eval_temp_epoch{epoch}"
        vis_folder = os.path.join(output_folder, vis_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        
        _mp_log_fn(f"[VIS] Creating visualization figures in {vis_folder}")
        
        for sample_data in vis_data:
            key = sample_data['key']
            data = sample_data['data']  # Shape: (C, D, H, W) or (D, H, W)
            pred_seg = sample_data['pred_seg']  # Shape: (D, H, W)
            gt_seg = sample_data['gt_seg']  # Shape: (D, H, W)
            
            # Get channel 0 if multi-channel
            if data.ndim == 4:
                img_3d = data[0]  # Take first channel
            else:
                img_3d = data
            
            # Ensure numpy arrays
            img_3d = np.asarray(img_3d)
            pred_seg = np.asarray(pred_seg)
            gt_seg = np.asarray(gt_seg)
            
            # Find slices with tumor in GT (label == VIS_TUMOR_LABEL)
            tumor_slices = []
            for z in range(gt_seg.shape[0]):
                if np.any(gt_seg[z] == VIS_TUMOR_LABEL):
                    tumor_slices.append(z)
            
            if not tumor_slices:
                _mp_log_fn(f"[VIS] Sample {key}: No tumor slices found, skipping")
                continue
            
            # Randomly select up to VIS_MAX_SLICES_PER_SAMPLE slices
            rng = np.random.default_rng(seed=42)
            if len(tumor_slices) > VIS_MAX_SLICES_PER_SAMPLE:
                selected_slices = sorted(rng.choice(tumor_slices, size=VIS_MAX_SLICES_PER_SAMPLE, replace=False))
            else:
                selected_slices = tumor_slices
            
            num_rows = len(selected_slices)
            fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
            
            # Handle single-row case
            if num_rows == 1:
                axes = axes.reshape(1, -1)
            
            for row_idx, z in enumerate(selected_slices):
                # Get the slice data
                img_slice = img_3d[z]
                pred_slice = pred_seg[z]
                gt_slice = gt_seg[z]
                
                # Normalize image for display
                img_min, img_max = img_slice.min(), img_slice.max()
                if img_max > img_min:
                    img_norm = (img_slice - img_min) / (img_max - img_min)
                else:
                    img_norm = np.zeros_like(img_slice)
                
                # Convert to RGB for overlay
                img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
                
                # Create tumor masks (only tumor, not anatomy)
                pred_tumor_mask = (pred_slice == VIS_TUMOR_LABEL)
                gt_tumor_mask = (gt_slice == VIS_TUMOR_LABEL)
                
                # Column 1: Original image
                axes[row_idx, 0].imshow(img_norm, cmap='gray')
                axes[row_idx, 0].set_title(f'Slice {z}' if row_idx == 0 else f'Slice {z}')
                axes[row_idx, 0].axis('off')
                
                # Column 2: Predicted tumor (blue overlay)
                img_pred = img_rgb.copy()
                if np.any(pred_tumor_mask):
                    # Blue overlay: increase B channel, decrease R and G
                    alpha = 0.5
                    img_pred[pred_tumor_mask, 0] = img_pred[pred_tumor_mask, 0] * (1 - alpha)  # R
                    img_pred[pred_tumor_mask, 1] = img_pred[pred_tumor_mask, 1] * (1 - alpha)  # G
                    img_pred[pred_tumor_mask, 2] = img_pred[pred_tumor_mask, 2] * (1 - alpha) + alpha  # B
                axes[row_idx, 1].imshow(img_pred)
                axes[row_idx, 1].set_title('Pred Tumor (blue)' if row_idx == 0 else '')
                axes[row_idx, 1].axis('off')
                
                # Column 3: GT tumor (green overlay)
                img_gt = img_rgb.copy()
                if np.any(gt_tumor_mask):
                    # Green overlay: increase G channel, decrease R and B
                    alpha = 0.5
                    img_gt[gt_tumor_mask, 0] = img_gt[gt_tumor_mask, 0] * (1 - alpha)  # R
                    img_gt[gt_tumor_mask, 1] = img_gt[gt_tumor_mask, 1] * (1 - alpha) + alpha  # G
                    img_gt[gt_tumor_mask, 2] = img_gt[gt_tumor_mask, 2] * (1 - alpha)  # B
                axes[row_idx, 2].imshow(img_gt)
                axes[row_idx, 2].set_title('GT Tumor (green)' if row_idx == 0 else '')
                axes[row_idx, 2].axis('off')
            
            # Set overall title
            fig.suptitle(f'Sample: {key} (Epoch {epoch})', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(vis_folder, f'{key}.png')
            plt.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        _mp_log_fn(f"[VIS] Created {len(vis_data)} visualization figures")
        return vis_folder
        
    except Exception as e:
        _mp_log_fn(f"[VIS] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _async_test_eval_worker(
    temp_ckpt_path: str,
    result_path: str,
    preprocessed_folder: str,
    test_keys: List[str],
    top_k_dices: List[Tuple[float, int]],
    top_k_checkpoints: int,
    output_folder: str,
    plans_identifier: str,
    eval_config: dict,
    gpu_id: Optional[int] = None,
):
    """
    Standalone worker function for async test evaluation.
    
    This function runs in a separate process and:
    1. Sets up the GPU environment (via initializer function)
    2. Loads the model from temp checkpoint
    3. Runs sliding window evaluation on test set
    4. Writes result JSON to result_path
    
    IMPORTANT: CUDA_VISIBLE_DEVICES is set by parent process BEFORE spawn,
    ensuring correct GPU selection.
    
    The main process handles ALL checkpoint management:
    - Determines if result qualifies for top-k
    - Saves/renames checkpoints
    - Updates top_k_dices list
    
    Args:
        temp_ckpt_path: Path to temporary checkpoint with model weights
        result_path: Path to write result JSON
        preprocessed_folder: Path to preprocessed dataset
        test_keys: List of test sample keys
        top_k_dices: Current top-k dice values (for reference only)
        top_k_checkpoints: Number of top checkpoints to maintain
        output_folder: Folder to save checkpoints
        plans_identifier: Identifier for checkpoint naming
        eval_config: Dict with evaluation configuration
        gpu_id: GPU ID (for logging only - already set by initializer)
    """
    # Import everything inside the function to ensure clean subprocess environment
    # This avoids inheriting any CUDA context from parent process
    import os
    import sys
    import json
    import traceback
    from datetime import datetime
    
    # Helper for timestamped logging in subprocess
    def _mp_log(msg: str):
        """Print with timestamp and [MP] prefix for subprocess logs."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp}: [MP] {msg}", flush=True)
    
    # Verify GPU setting (for debugging)
    actual_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    _mp_log(f"CUDA_VISIBLE_DEVICES={actual_gpu}")
    
    try:
        # All imports happen AFTER CUDA_VISIBLE_DEVICES is set by initializer
        import torch
        import torch._dynamo as _dynamo
        import numpy as np
        from tqdm import tqdm
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window, compute_gaussian
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        from nnunetv2.utilities.collate_outputs import collate_outputs
        from batchgenerators.utilities.file_and_folder_operations import join, isfile
        
        # Suppress dynamo errors to avoid shape mismatch in residual connections
        # This allows dynamo to fall back to eager mode when shapes are dynamic
        _dynamo.config.suppress_errors = True
        
        # Now initialize CUDA - this creates a fresh CUDA context for this subprocess
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _mp_log(f"Using device: {device}, torch.cuda.device_count()={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        
        # Load checkpoint
        checkpoint = torch.load(temp_ckpt_path, map_location='cpu', weights_only=False)
        
        # Reconstruct model from init_args
        init_args = checkpoint['init_args']
        plans = init_args['plans']
        configuration = init_args['configuration']
        dataset_json = init_args['dataset_json']
        
        # Get model architecture
        plans_manager = PlansManager(plans)
        config_manager = plans_manager.get_configuration(configuration)
        label_manager = plans_manager.get_label_manager(dataset_json)
        
        # Determine number of input channels
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
        num_output_channels = label_manager.num_segmentation_heads
        
        # Build network - supports both nnunet and swinunetr architectures
        model_name = checkpoint.get('model_name', 'nnunet')  # Default to nnunet for backwards compatibility
        
        if model_name == 'swinunetr':
            # Build SwinUNETR (deep supervision is always disabled)
            from .swinunetr_builder import build_swinunetr
            # Get feature_size from checkpoint (default to 48 for backwards compatibility)
            swinunetr_feature_size = checkpoint.get('swinunetr_feature_size', 48)
            _mp_log_fn(f"[Subprocess] Using SwinUNETR architecture (feature_size={swinunetr_feature_size}, deep supervision disabled)")
            network = build_swinunetr(
                num_input_channels=num_input_channels,
                num_output_channels=num_output_channels,
                patch_size=tuple(config_manager.patch_size),
                feature_size=swinunetr_feature_size,
                use_checkpoint=False,  # Don't need gradient checkpointing for inference
                spatial_dims=3,
            )
            # Override enable_deep_supervision for SwinUNETR
            eval_config['enable_deep_supervision'] = False
        else:
            # Build standard nnUNet network
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
            
            network = get_network_from_plans(
                arch_class_name=config_manager.network_arch_class_name,
                arch_kwargs=config_manager.network_arch_init_kwargs,
                arch_kwargs_req_import=config_manager.network_arch_init_kwargs_req_import,
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                allow_init=True,
                deep_supervision=eval_config.get('enable_deep_supervision', True)
            )
        
        network.load_state_dict(checkpoint['network_weights'])
        network = network.to(device)
        network.eval()
        
        # Create dataset
        dataset_class = infer_dataset_class(preprocessed_folder)
        test_dataset = dataset_class(
            preprocessed_folder,
            test_keys,
            folder_with_segs_from_previous_stage=None
        )
        
        # Extract eval config
        patch_size = tuple(eval_config['patch_size'])
        step_size = eval_config['step_size']
        simulate_perfect_anatomy = eval_config.get('simulate_perfect_anatomy', True)
        max_patches_per_sample = eval_config.get('max_patches_per_sample', None)
        current_epoch = eval_config['current_epoch']
        num_classes = label_manager.num_segmentation_heads
        enable_deep_supervision = eval_config.get('enable_deep_supervision', True)
        timeout_hours = eval_config.get('timeout_hours', 4.0)  # Default 4 hour HARD timeout
        
        # Get temp checkpoint path from eval_config (for result reporting)
        temp_ckpt_path_for_result = eval_config.get('temp_ckpt_path', temp_ckpt_path)
        
        # Run sliding window evaluation with HARD timeout (must complete ALL samples)
        import time
        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        
        all_outputs = []
        failed_samples = []  # Track failed sample keys
        total_samples = len(test_keys)
        
        # Select samples for visualization (random subset)
        vis_rng = np.random.default_rng(seed=current_epoch)  # Reproducible per epoch
        if total_samples <= VIS_NUM_SAMPLES:
            vis_sample_keys = set(test_keys)
        else:
            vis_sample_keys = set(vis_rng.choice(test_keys, size=VIS_NUM_SAMPLES, replace=False))
        vis_data = []  # Will store {'key', 'data', 'pred_seg', 'gt_seg'} for visualization
        
        _mp_log(f"Starting evaluation: {total_samples} samples, timeout={timeout_hours}h")
        _mp_log(f"[VIS] Will visualize {len(vis_sample_keys)} samples")
        
        with torch.no_grad():
            for sample_idx, key in enumerate(tqdm(test_keys, desc=f"[MP] Test SW Eval (epoch {current_epoch})")):
                # Check for HARD timeout - if exceeded, FAIL the entire evaluation
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout_seconds:
                    error_msg = f"TIMEOUT: Exceeded {timeout_hours}h limit after {sample_idx}/{total_samples} samples"
                    _mp_log(error_msg)
                    result = {
                        'success': False,
                        'error': error_msg,
                        'epoch': current_epoch,
                        'total_samples': total_samples,
                        'processed_samples': sample_idx,
                        'temp_ckpt_path': temp_ckpt_path_for_result,
                        'timestamp': time.time(),
                    }
                    with open(result_path, 'w') as f:
                        json.dump(_convert_to_json_serializable(result), f, indent=2)
                    return  # Exit subprocess
                
                try:
                    data, seg, seg_prev, properties = test_dataset.load_case(key)
                    
                    # Get original segmentation for foreground classification
                    seg_np_orig = seg[0] if seg.ndim == 4 else seg
                    original_shape = tuple(seg_np_orig.shape)
                    
                    # PAD data and seg to at least patch_size (critical for UNet residual connections)
                    # This mirrors what nnUNet predictor does in predict_sliding_window_return_logits
                    # NOTE: pad_nd_image uses np.pad for numpy arrays (expects 'constant_values')
                    #       and torch.nn.functional.pad for tensors (expects 'value')
                    from acvl_utils.cropping_and_padding.padding import pad_nd_image
                    data_padded, slicer_revert = pad_nd_image(
                        np.asarray(data), patch_size, 'constant', {'constant_values': 0}, True, None
                    )
                    seg_padded, _ = pad_nd_image(
                        np.asarray(seg), patch_size, 'constant', {'constant_values': 0}, True, None
                    )
                    
                    # Use padded shapes for sliding window computation
                    seg_np = seg_padded[0] if seg_padded.ndim == 4 else seg_padded
                    volume_shape = tuple(seg_np.shape)
                    
                    # Compute sliding window boxes on PADDED volume
                    steps = compute_steps_for_sliding_window(volume_shape, patch_size, step_size)
                    
                    slicers = []
                    for sx in steps[0]:
                        for sy in steps[1]:
                            for sz in steps[2]:
                                slicers.append(
                                    tuple([slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)])
                                )
                    
                    # Classify boxes by foreground
                    tumor_indices = []
                    anatomy_only_indices = []
                    
                    for idx, sl in enumerate(slicers):
                        box_seg = seg_np[sl]
                        has_tumor = np.any(box_seg == 2)
                        has_anatomy = np.any(box_seg == 1)
                        
                        if has_tumor:
                            tumor_indices.append(idx)
                        elif has_anatomy:
                            anatomy_only_indices.append(idx)
                    
                    foreground_indices = tumor_indices + anatomy_only_indices
                    
                    # Priority sampling if max_patches set
                    if max_patches_per_sample is not None and max_patches_per_sample > 0:
                        selected = []
                        rng = np.random.default_rng(seed=42)
                        
                        if len(tumor_indices) > 0:
                            if len(tumor_indices) <= max_patches_per_sample:
                                selected.extend(tumor_indices)
                            else:
                                selected.extend(rng.choice(tumor_indices, size=max_patches_per_sample, replace=False).tolist())
                        
                        remaining = max_patches_per_sample - len(selected)
                        if remaining > 0 and len(anatomy_only_indices) > 0:
                            if len(anatomy_only_indices) <= remaining:
                                selected.extend(anatomy_only_indices)
                            else:
                                selected.extend(rng.choice(anatomy_only_indices, size=remaining, replace=False).tolist())
                        
                        foreground_indices = selected
                    
                    if len(foreground_indices) == 0:
                        continue
                    
                    # Run inference on PADDED data
                    data_tensor = torch.from_numpy(np.asarray(data_padded)).float().to(device)
                    
                    predicted_logits = torch.zeros(
                        (num_classes, *volume_shape), dtype=torch.half, device=device
                    )
                    n_predictions = torch.zeros(volume_shape, dtype=torch.half, device=device)
                    gaussian = compute_gaussian(patch_size, sigma_scale=1./8, value_scaling_factor=10, device=device)
                    
                    for box_idx in foreground_indices:
                        sl = slicers[box_idx]
                        patch = data_tensor[(slice(None),) + sl][None]
                        
                        # Forward pass (dynamo errors suppressed, will fall back to eager)
                        pred = network(patch)
                        # Handle deep supervision: output is list of tensors at different scales
                        # For SwinUNETR or nnUNet without DS: output is single tensor
                        if isinstance(pred, (list, tuple)):
                            pred = pred[0]  # Get highest resolution output
                        
                        pred = pred * gaussian
                        predicted_logits[(slice(None),) + sl] += pred
                        n_predictions[sl] += gaussian
                    
                    valid_mask = n_predictions > 0
                    n_predictions_safe = torch.clamp(n_predictions, min=1e-8)
                    predicted_logits = predicted_logits / n_predictions_safe
                    
                    # Move to CPU for metrics
                    predicted_logits = predicted_logits.cpu()
                    valid_mask = valid_mask.cpu()
                    
                    # REVERT PADDING on predictions and valid_mask
                    # slicer_revert is for spatial dims, need to add channel dim for predicted_logits
                    predicted_logits = predicted_logits[(slice(None),) + slicer_revert[1:]]
                    valid_mask = valid_mask[slicer_revert[1:]]
                    
                    # Use ORIGINAL (unpadded) seg for metrics
                    seg_tensor = torch.from_numpy(np.asarray(seg)).long()
                    
                    if seg_tensor.ndim == 3:
                        seg_tensor = seg_tensor.unsqueeze(0)
                    
                    # Handle ignore label
                    if seg_tensor.min() < 0:
                        seg_tensor = seg_tensor.clone()
                        seg_tensor[seg_tensor < 0] = 0
                    if seg_tensor.max() >= num_classes:
                        seg_tensor = torch.clamp(seg_tensor, 0, num_classes - 1)
                    
                    # Convert to segmentation
                    output_seg = predicted_logits.argmax(0, keepdim=True)
                    
                    if simulate_perfect_anatomy:
                        gt_is_background = (seg_tensor == 0)
                        output_seg = output_seg.clone()
                        output_seg[gt_is_background] = 0
                    
                    # Store visualization data for selected samples
                    if key in vis_sample_keys:
                        # Use ORIGINAL (non-padded) data for visualization
                        gt_seg_np = seg_tensor.squeeze().numpy()  # (D, H, W)
                        pred_seg_np = output_seg.squeeze().numpy()  # (D, H, W)
                        vis_data.append({
                            'key': key,
                            'data': np.asarray(data),  # Original data (C, D, H, W) or (D, H, W)
                            'pred_seg': pred_seg_np,
                            'gt_seg': gt_seg_np,
                        })
                    
                    # One-hot encoding
                    pred_onehot = torch.zeros(predicted_logits.shape, dtype=torch.float32)
                    pred_onehot.scatter_(0, output_seg, 1)
                    pred_onehot = pred_onehot.unsqueeze(0)
                    seg_tensor = seg_tensor.unsqueeze(0)
                    
                    target_onehot = torch.zeros(pred_onehot.shape, dtype=torch.bool)
                    target_onehot.scatter_(1, seg_tensor, 1)
                    
                    valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0).float()
                    
                    axes = [0] + list(range(2, pred_onehot.ndim))
                    tp, fp, fn, tn = get_tp_fp_fn_tn(pred_onehot, target_onehot, axes=axes, mask=valid_mask_expanded)
                    
                    tp_hard = tp.numpy()[1:]  # Skip background
                    fp_hard = fp.numpy()[1:]
                    fn_hard = fn.numpy()[1:]
                    tn_hard = tn.numpy()[1:]
                    
                    all_outputs.append({
                        'tp_hard': tp_hard,
                        'fp_hard': fp_hard,
                        'fn_hard': fn_hard,
                        'tn_hard': tn_hard,
                        'loss': 0.0
                    })
                    
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    _mp_log(f"Warning: Failed to process {key}: {e}")
                    traceback.print_exc()
                    failed_samples.append(key)
                    continue
        
        # Check success rate - must complete ALL samples (no partial results)
        successful_samples = total_samples - len(failed_samples)
        success_rate = successful_samples / total_samples if total_samples > 0 else 0.0
        
        elapsed_minutes = (time.time() - start_time) / 60
        _mp_log(f"Completed in {elapsed_minutes:.1f} min. "
                f"Processed {successful_samples}/{total_samples} samples successfully ({success_rate*100:.1f}%)")
        if failed_samples:
            _mp_log(f"Failed samples: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")
        
        # Create visualization figures for selected samples
        vis_folder_path = None
        if vis_data:
            vis_folder_path = _create_visualization_figures(
                vis_data=vis_data,
                output_folder=output_folder,
                epoch=current_epoch,
                plans_identifier=plans_identifier,
                _mp_log_fn=_mp_log
            )
        
        # Aggregate results - main process handles checkpoint management
        if not all_outputs:
            result = {
                'success': False,
                'error': 'No outputs collected - all samples failed',
                'epoch': current_epoch,
                'total_samples': total_samples,
                'successful_samples': 0,
                'failed_samples': failed_samples,
                'temp_ckpt_path': temp_ckpt_path_for_result,
                'vis_folder_path': vis_folder_path,
                'timestamp': time.time(),
            }
        elif len(failed_samples) > 0:
            # Some samples failed - still compute metrics but flag as incomplete
            _mp_log(f"Warning: {len(failed_samples)} samples failed. Computing metrics from {successful_samples} samples.")
            
            outputs_collated = collate_outputs(all_outputs)
            tp = np.sum(outputs_collated['tp_hard'], 0)
            fp = np.sum(outputs_collated['fp_hard'], 0)
            fn = np.sum(outputs_collated['fn_hard'], 0)
            tn = np.sum(outputs_collated['tn_hard'], 0)
            
            dice_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0.0 
                             for i, j, k in zip(tp, fp, fn)]
            recall_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fn)]
            precision_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fp)]
            tumor_dice = dice_per_class[1] if len(dice_per_class) > 1 else np.nanmean(dice_per_class)
            
            result = {
                'success': True,
                'has_failures': True,
                'epoch': current_epoch,
                'dice_per_class': dice_per_class,
                'recall_per_class': recall_per_class,
                'precision_per_class': precision_per_class,
                'tumor_dice': tumor_dice,
                'total_samples': total_samples,
                'successful_samples': successful_samples,
                'failed_samples': len(failed_samples),
                'success_rate': success_rate,
                'temp_ckpt_path': temp_ckpt_path_for_result,
                'vis_folder_path': vis_folder_path,
                'timestamp': time.time(),
            }
        else:
            # All samples processed successfully - compute metrics
            # Main process will handle checkpoint management
            outputs_collated = collate_outputs(all_outputs)
            tp = np.sum(outputs_collated['tp_hard'], 0)
            fp = np.sum(outputs_collated['fp_hard'], 0)
            fn = np.sum(outputs_collated['fn_hard'], 0)
            tn = np.sum(outputs_collated['tn_hard'], 0)
            
            dice_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0.0 
                             for i, j, k in zip(tp, fp, fn)]
            recall_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fn)]
            precision_per_class = [100.0 * t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp, fp)]
            f1_per_class = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 
                           for p, r in zip(precision_per_class, recall_per_class)]
            fpr_per_class = [100.0 * f / (f + t) if (f + t) > 0 else 0.0 for f, t in zip(fp, tn)]
            
            tumor_dice = dice_per_class[1] if len(dice_per_class) > 1 else np.nanmean(dice_per_class)
            
            result = {
                'success': True,
                'epoch': current_epoch,
                'dice_per_class': dice_per_class,
                'recall_per_class': recall_per_class,
                'precision_per_class': precision_per_class,
                'f1_per_class': f1_per_class,
                'fpr_per_class': fpr_per_class,
                'tumor_dice': tumor_dice,
                'total_samples': total_samples,
                'successful_samples': successful_samples,
                'failed_samples': 0,
                'success_rate': 1.0,
                'temp_ckpt_path': temp_ckpt_path_for_result,
                'vis_folder_path': vis_folder_path,
                'timestamp': time.time(),
            }
        
        # Write result (convert numpy types to native Python types for JSON serialization)
        with open(result_path, 'w') as f:
            json.dump(_convert_to_json_serializable(result), f, indent=2)
        
        _mp_log(f"Evaluation complete. Tumor Dice: {result.get('tumor_dice', 'N/A')}")
        
    except Exception as e:
        import time as time_module
        error_result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'epoch': eval_config.get('current_epoch', -1),
            'temp_ckpt_path': eval_config.get('temp_ckpt_path', temp_ckpt_path),
            'timestamp': time_module.time(),
        }
        with open(result_path, 'w') as f:
            json.dump(_convert_to_json_serializable(error_result), f, indent=2)
        _mp_log(f"Evaluation FAILED: {e}")


def _check_single_sample_for_tumor(args: tuple) -> tuple:
    """
    Worker function to check if a single sample contains tumor (label 2).
    
    This function is designed for multiprocessing.Pool.map() and must be
    at module level to be picklable.
    
    Args:
        args: Tuple of (key, dataset_folder, tumor_label)
        
    Returns:
        Tuple of (key, has_tumor, has_ignore, n_ignore, error_msg)
        - key: Sample key
        - has_tumor: True if sample contains tumor_label
        - has_ignore: True if sample contains ignore label (-1)
        - n_ignore: Number of ignore label voxels
        - error_msg: Error message if failed, None otherwise
    """
    key, dataset_folder, tumor_label = args
    
    try:
        # Import inside function to avoid issues with multiprocessing
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        import numpy as np
        
        # Create dataset for this single key
        dataset_class = infer_dataset_class(dataset_folder)
        temp_dataset = dataset_class(dataset_folder, [key], None)
        
        # Load the sample
        data, seg, seg_prev, properties = temp_dataset.load_case(key)
        
        # Get segmentation array
        seg_np = seg[0] if seg.ndim == 4 else seg
        
        # Check for ignore label (-1)
        has_ignore = bool(np.any(seg_np < 0))
        n_ignore = int(np.sum(seg_np < 0)) if has_ignore else 0
        
        # Check if tumor label exists
        has_tumor = bool(np.any(seg_np == tumor_label))
        
        return (key, has_tumor, has_ignore, n_ignore, None)
        
    except Exception as e:
        return (key, False, False, 0, str(e))
