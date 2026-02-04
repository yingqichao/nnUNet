"""
SwinUNETR network builder for integration with nnUNet training pipeline.

This module provides functions to build MONAI's SwinUNETR architecture
in a way that's compatible with nnUNet's training infrastructure.

Key differences from standard nnUNet architecture:
- SwinUNETR does NOT support deep supervision (single output only)
- Patch size must be divisible by window size (typically 32 for 3D)
- torch.compile may cause issues with transformer dynamic shapes

Usage:
    from tuning_set_trainer.swinunetr_builder import build_swinunetr
    
    model = build_swinunetr(
        num_input_channels=1,
        num_output_channels=3,
        patch_size=(128, 128, 128),
        feature_size=48,
    )
"""

from typing import Tuple, Union, List
import torch.nn as nn


def adjust_patch_size_for_swinunetr(
    patch_size: Union[Tuple[int, ...], List[int]],
    min_divisor: int = 32,
) -> Tuple[int, ...]:
    """
    Adjust patch size to be compatible with SwinUNETR architecture.
    
    SwinUNETR uses Swin Transformer which requires input dimensions to be
    divisible by 32 (due to 5-stage 2x downsampling: 2^5 = 32).
    
    This function rounds each dimension to the nearest multiple of min_divisor,
    preferring to round UP to avoid reducing the receptive field too much.
    
    Args:
        patch_size: Patch size tuple (D, H, W) for 3D
        min_divisor: Minimum divisor for patch dimensions (default 32)
        
    Returns:
        Adjusted patch size tuple with all dimensions divisible by min_divisor
    """
    adjusted = []
    for dim in patch_size:
        if dim % min_divisor == 0:
            adjusted.append(dim)
        else:
            # Round to nearest multiple of min_divisor
            lower = (dim // min_divisor) * min_divisor
            upper = lower + min_divisor
            # Prefer rounding up unless it increases by more than 50%
            if upper - dim <= dim - lower:
                adjusted.append(upper)
            else:
                adjusted.append(max(lower, min_divisor))  # Ensure at least min_divisor
    return tuple(adjusted)


def validate_patch_size_for_swinunetr(
    patch_size: Union[Tuple[int, ...], List[int]],
    window_size: int = 7,
    min_divisor: int = 32,
    auto_adjust: bool = True,
) -> Tuple[int, ...]:
    """
    Validate and optionally adjust patch size for SwinUNETR compatibility.
    
    SwinUNETR uses Swin Transformer which requires input dimensions to be
    divisible by the patch embedding stride and window size.
    
    Args:
        patch_size: Patch size tuple (D, H, W) for 3D
        window_size: Swin Transformer window size (default 7)
        min_divisor: Minimum divisor for patch dimensions (default 32)
        auto_adjust: If True, auto-adjust incompatible sizes; if False, raise error
        
    Returns:
        Valid patch size (possibly adjusted)
        
    Raises:
        ValueError: If patch size is incompatible and auto_adjust is False
    """
    needs_adjustment = any(dim % min_divisor != 0 for dim in patch_size)
    
    if needs_adjustment:
        if auto_adjust:
            adjusted = adjust_patch_size_for_swinunetr(patch_size, min_divisor)
            print(f"\n*** SwinUNETR PATCH SIZE ADJUSTMENT ***")
            print(f"  Original patch size: {list(patch_size)}")
            print(f"  Adjusted patch size: {list(adjusted)}")
            print(f"  (All dimensions must be divisible by {min_divisor})")
            print()
            return adjusted
        else:
            invalid_dims = [(i, dim) for i, dim in enumerate(patch_size) if dim % min_divisor != 0]
            raise ValueError(
                f"SwinUNETR requires patch dimensions to be divisible by {min_divisor}. "
                f"Invalid dimensions: {invalid_dims}. "
                f"Consider using patch sizes like 64, 96, 128, 160, 192, etc. "
                f"Or enable auto_adjust=True to automatically adjust."
            )
    
    return tuple(patch_size)


def build_swinunetr(
    num_input_channels: int,
    num_output_channels: int,
    patch_size: Union[Tuple[int, ...], List[int]],
    feature_size: int = 48,
    use_checkpoint: bool = False,
    spatial_dims: int = 3,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    normalize: bool = True,
    use_v2: bool = False,
) -> nn.Module:
    """
    Build a SwinUNETR model compatible with nnUNet training.
    
    SwinUNETR is a transformer-based architecture that combines Swin Transformer
    encoder with a CNN decoder for medical image segmentation.
    
    NOTE: SwinUNETR does NOT support deep supervision. It produces a single
    output tensor, unlike standard nnUNet which can produce multiple outputs
    at different scales.
    
    Args:
        num_input_channels: Number of input channels (e.g., 1 for CT)
        num_output_channels: Number of output classes (including background)
        patch_size: Patch size for training (D, H, W). Used as img_size for SwinUNETR.
                   Must be divisible by 32 (due to 5-stage downsampling: 2^5).
                   Common values: 96, 128, 160, etc.
        feature_size: Feature size for Swin Transformer (24, 48, or 96 typical)
                     - 24: "tiny" ~6M params
                     - 48: "base" ~25M params (default, similar to original BTCV)
                     - 96: "large" ~100M params
        use_checkpoint: Use gradient checkpointing to save memory (slower training)
        spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D)
        drop_rate: Dropout rate (default 0.0, same as original BTCV)
        attn_drop_rate: Attention dropout rate (default 0.0, same as original BTCV)
        dropout_path_rate: Stochastic depth drop path rate (default 0.0)
        normalize: Whether to use instance normalization (default True)
        use_v2: Use SwinUNETR V2 if available (default False)
        
    Returns:
        SwinUNETR model instance
        
    Raises:
        ImportError: If MONAI is not installed
        ValueError: If patch_size is incompatible with SwinUNETR
        
    Note:
        The original BTCV implementation (maglevswin/BTCV/main.py) uses:
        - feature_size=48 (via --feature_size arg)
        - drop_rate=0.0, attn_drop_rate=0.0 (hardcoded)
        - ROI size 96x96x96 for sliding window inference
        - Does NOT pass img_size to SwinUNETR (older MONAI version)
        
        For nnUNet integration, we pass img_size=patch_size to ensure
        proper positional embedding sizing for the training patch size.
    """
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as e:
        raise ImportError(
            "MONAI is required for SwinUNETR. Install it with:\n"
            "  pip install monai\n"
            "or\n"
            "  pip install 'monai[all]'"
        ) from e
    
    # Verify patch size is compatible - auto-adjust if needed as safety net
    # (trainer should already have adjusted, but this handles edge cases like subprocess loading)
    adjusted_patch_size = list(patch_size)
    needs_adjustment = False
    for i, dim in enumerate(adjusted_patch_size):
        if dim % 32 != 0:
            # Round to nearest multiple of 32
            adjusted_patch_size[i] = max(32, round(dim / 32) * 32)
            needs_adjustment = True
    
    if needs_adjustment:
        import warnings
        warnings.warn(
            f"SwinUNETR: Auto-adjusting patch_size from {patch_size} to {tuple(adjusted_patch_size)} "
            f"(must be divisible by 32)"
        )
        patch_size = tuple(adjusted_patch_size)
    
    # Build the model
    # Note: SwinUNETR in this MONAI version does NOT take img_size parameter.
    # It validates input size in forward() via _check_input_size().
    model = SwinUNETR(
        in_channels=num_input_channels,
        out_channels=num_output_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        normalize=normalize,
        use_v2=use_v2,
    )
    
    return model


def get_swinunetr_default_config(
    patch_size: Union[Tuple[int, ...], List[int]],
    model_size: str = 'base',
) -> dict:
    """
    Get default configuration for SwinUNETR based on model size.
    
    Args:
        patch_size: Patch size (for validation)
        model_size: One of 'tiny', 'small', 'base', 'large'
        
    Returns:
        Dictionary with recommended SwinUNETR configuration
    """
    configs = {
        'tiny': {
            'feature_size': 24,
            'use_checkpoint': False,
            'dropout_path_rate': 0.0,
        },
        'small': {
            'feature_size': 36,
            'use_checkpoint': False,
            'dropout_path_rate': 0.1,
        },
        'base': {
            'feature_size': 48,
            'use_checkpoint': True,  # Recommended for memory
            'dropout_path_rate': 0.2,
        },
        'large': {
            'feature_size': 96,
            'use_checkpoint': True,  # Required for memory
            'dropout_path_rate': 0.2,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model_size '{model_size}'. Choose from: {list(configs.keys())}")
    
    config = configs[model_size]
    
    # Validate patch size
    validate_patch_size_for_swinunetr(patch_size)
    
    return config


# Export symbols
__all__ = [
    'build_swinunetr',
    'validate_patch_size_for_swinunetr',
    'get_swinunetr_default_config',
]
