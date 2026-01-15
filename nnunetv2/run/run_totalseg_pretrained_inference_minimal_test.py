"""
Minimal test script for TotalSegmentator pretrained inference debugging.
Usage:
    python -m nnUNet.nnunetv2.run.run_totalseg_pretrained_inference_minimal_test

This script creates a dummy CT volume and runs TotalSegmentator pretrained inference,
useful for testing pdb breakpoints in dynamic_network_architectures or nnunetv2.

NOTE: torch.compile is disabled automatically (nnUNet_compile=false) so that
pdb breakpoints inside the model code will work. Without this, torch.compile
wraps the model in OptimizedModule which bypasses the original Python code.
"""
import os
import argparse
import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import torch


def create_dummy_ct_volume(shape=(128, 128, 64), spacing=(1.5, 1.5, 1.5)):
    """
    Create a dummy CT-like volume wrapped in a Nifti1Image.
    
    Args:
        shape: (x, y, z) voxel dimensions
        spacing: (x, y, z) voxel spacing in mm
    
    Returns:
        Nifti1Image with CT-like intensity values
    """
    # Create a volume with CT-like Hounsfield unit values
    # Background: ~-1000 HU (air)
    # Soft tissue: ~40-80 HU
    # Add some structure to make it more realistic
    
    data = np.ones(shape, dtype=np.float32) * (-1000)  # Air background
    
    # Add a "body" region in the center with soft tissue values
    center = np.array(shape) // 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Distance from center
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < min(shape) // 3:
                    # Soft tissue with some variation
                    data[x, y, z] = 40 + np.random.randn() * 20
    
    # Create affine matrix based on spacing
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    return nib.Nifti1Image(data, affine)


def run_totalseg_minimal_test(
    roi_subset=None,
    device="gpu:0",
    volume_shape=(128, 128, 64),
    spacing=(1.5, 1.5, 1.5),
    fast=True,
    verbose=True
):
    """
    Run TotalSegmentator inference on a dummy volume.
    
    Args:
        roi_subset: List of organs to segment (e.g., ['heart', 'spleen', 'liver'])
        device: Device string (e.g., 'gpu:0', 'cpu')
        volume_shape: Shape of dummy volume
        spacing: Voxel spacing in mm
        fast: Use fast (3mm) model for quicker testing
        verbose: Print verbose output
    """
    # Import here to ensure env vars are set first
    from totalsegmentator.python_api import totalsegmentator
    
    print("="*60)
    print("TotalSegmentator Pretrained Inference Minimal Test")
    print("="*60)
    
    # Default roi_subset similar to run_totalseg.py
    if roi_subset is None:
        roi_subset = ['heart', 'spleen', 'kidney_right', 'kidney_left', 'pancreas', 'liver']
    
    print(f"\nDevice: {device}")
    print(f"Volume shape: {volume_shape}")
    print(f"Spacing: {spacing} mm")
    print(f"ROI subset: {roi_subset}")
    print(f"Fast mode: {fast}")
    
    # Create dummy CT volume
    print("\nCreating dummy CT volume...")
    dummy_img = create_dummy_ct_volume(shape=volume_shape, spacing=spacing)
    print(f"  Shape: {dummy_img.shape}")
    print(f"  Affine:\n{dummy_img.affine}")
    print(f"  Data dtype: {dummy_img.get_data_dtype()}")
    print(f"  Intensity range: [{dummy_img.get_fdata().min():.1f}, {dummy_img.get_fdata().max():.1f}]")
    
    # Run TotalSegmentator inference
    print("\n" + "-"*40)
    print("Running TotalSegmentator inference...")
    print("(Set pdb breakpoints in dynamic_network_architectures or nnunetv2 to debug)")
    print("-"*40 + "\n")
    
    # Call totalsegmentator with:
    # - input: Nifti1Image (dummy volume)
    # - output: None (skip saving)
    # - ml=True: multilabel output
    # - skip_saving=True: don't save intermediate files
    seg_img: Nifti1Image = totalsegmentator(
        input=dummy_img,
        output=None,  # Don't save output
        ml=True,
        device=device,
        roi_subset=roi_subset,
        verbose=verbose,
        fast=fast,  # Use fast model for quicker testing
        quiet=False,
        skip_saving=True
    )
    
    # Analyze results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    seg_data = seg_img.get_fdata()
    print(f"\nSegmentation output:")
    print(f"  Shape: {seg_img.shape}")
    print(f"  Data dtype: {seg_data.dtype}")
    print(f"  Unique labels: {np.unique(seg_data).astype(int).tolist()}")
    
    # Count voxels per label
    unique_labels, counts = np.unique(seg_data, return_counts=True)
    print(f"\n  Label distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = count / seg_data.size * 100
        print(f"    Label {int(label)}: {count:,} voxels ({percentage:.2f}%)")
    
    print("\n" + "="*60)
    print("✓ TotalSegmentator inference test completed successfully!")
    print("="*60)
    
    return seg_img, dummy_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Minimal TotalSegmentator inference test for debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with default settings (fast mode, 6 organs)
    python -m nnUNet.nnunetv2.run.run_totalseg_pretrained_inference_minimal_test

    # Test with specific organs
    python -m nnUNet.nnunetv2.run.run_totalseg_pretrained_inference_minimal_test --roi heart liver spleen

    # Test on CPU
    python -m nnUNet.nnunetv2.run.run_totalseg_pretrained_inference_minimal_test --device cpu

    # Full resolution model (slower but more accurate)
    python -m nnUNet.nnunetv2.run.run_totalseg_pretrained_inference_minimal_test --no-fast

    # Custom volume shape and spacing
    python -m nnUNet.nnunetv2.run.run_totalseg_pretrained_inference_minimal_test --shape 256 256 128 --spacing 1.0 1.0 2.0
        """
    )
    parser.add_argument('--roi', type=str, nargs='+', default=None,
                        help="ROI subset to segment (e.g., heart spleen liver). "
                             "Default: ['heart', 'spleen', 'kidney_right', 'kidney_left', 'pancreas', 'liver']")
    parser.add_argument('--device', type=str, default='gpu:0',
                        help="Device: 'gpu:0', 'gpu:1', 'cpu', etc. Default: gpu:0")
    parser.add_argument('--shape', type=int, nargs=3, default=[128, 128, 64],
                        help="Dummy volume shape (x y z). Default: 128 128 64")
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help="Voxel spacing in mm (x y z). Default: 1.5 1.5 1.5")
    parser.add_argument('--no-fast', action='store_true',
                        help="Use full resolution model instead of fast 3mm model")
    parser.add_argument('--quiet', action='store_true',
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # IMPORTANT: Disable torch.compile for pdb debugging!
    os.environ['nnUNet_compile'] = 'false'
    print("NOTE: torch.compile disabled for debugging (nnUNet_compile=false)\n")
    
    seg_img, dummy_img = run_totalseg_minimal_test(
        roi_subset=args.roi,
        device=args.device,
        volume_shape=tuple(args.shape),
        spacing=tuple(args.spacing),
        fast=not args.no_fast,
        verbose=not args.quiet
    )

