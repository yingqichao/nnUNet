"""
Minimal test script for debugging nnUNet model architecture.
Usage:
    python -m nnUNet.nnunetv2.run.run_training_minimal_test 602 3d_fullres 0
Or:
    python -m nnUNet.nnunetv2.run.run_training_minimal_test 602 3d_fullres 0 -p nnUNetPlans -device cuda

This script initializes the model and runs a dummy forward pass,
useful for testing pdb breakpoints in dynamic_network_architectures.

NOTE: torch.compile is disabled automatically (nnUNet_compile=false) so that
pdb breakpoints inside the model code will work. Without this, torch.compile
wraps the model in OptimizedModule which bypasses the original Python code.
"""
import os
import torch
from typing import Union

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          device: torch.device = torch.device('cuda')):
    """Simplified version of get_trainer_from_args for minimal testing."""
    # load nnunet class
    nnunet_trainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, 'nnunetv2.training.nnUNetTrainer'
    )
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name}')
    
    # handle dataset input
    if isinstance(dataset_name_or_id, str) and not dataset_name_or_id.startswith('Dataset'):
        dataset_name_or_id = int(dataset_name_or_id)

    # load plans and dataset json
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    
    # create trainer
    trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                             dataset_json=dataset_json, device=device)
    return trainer


def run_minimal_test(dataset_name_or_id: str, configuration: str, fold: int,
                     plans_identifier: str = 'nnUNetPlans',
                     device_str: str = 'cuda'):
    """
    Minimal test: initialize model and run dummy inference.
    """
    print("="*60)
    print("nnUNet Minimal Model Test")
    print("="*60)
    
    # Set device
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    # Get trainer
    print(f"\nLoading trainer for Dataset {dataset_name_or_id}, config: {configuration}, fold: {fold}")
    trainer = get_trainer_from_args(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        plans_identifier=plans_identifier,
        device=device
    )
    
    # Initialize (this builds the network)
    print("\nInitializing trainer (building network)...")
    trainer.initialize()
    
    # Get network
    network = trainer.network
    print(f"\nNetwork type: {type(network).__name__}")
    
    # Print network info
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Get patch size from configuration
    patch_size = trainer.configuration_manager.patch_size
    num_input_channels = trainer.num_input_channels
    print(f"\nPatch size from plan: {patch_size}")
    print(f"Number of input channels: {num_input_channels}")
    
    # Create dummy input tensor
    # Shape: (batch_size, channels, *patch_size)
    batch_size = 1
    
    # Use the actual patch size from plans, or fallback to (96, 96, 96)
    if len(patch_size) == 3:
        test_size = tuple(min(p, 96) for p in patch_size)  # Use smaller size for quick test
    else:
        test_size = (96, 96)  # 2D case
    
    dummy_input = torch.ones(batch_size, num_input_channels, *test_size, device=device)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Dummy input device: {dummy_input.device}")
    
    # Run dummy inference
    network.eval()
    with torch.no_grad():
        print("\n" + "-"*40)
        print("Running dummy forward pass...")
        print("(Set pdb breakpoints in dynamic_network_architectures to debug)")
        print("-"*40)
        
        output = network(dummy_input)
        
        if isinstance(output, (list, tuple)):
            # Deep supervision enabled - multiple outputs
            print(f"\nOutput is a list/tuple with {len(output)} elements (deep supervision)")
            for i, o in enumerate(output):
                print(f"  Output[{i}] shape: {o.shape}, dtype: {o.dtype}")
        else:
            print(f"\nOutput shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
    
    print("\n" + "="*60)
    print("✓ Minimal test completed successfully!")
    print("="*60)
    
    return network, output


if __name__ == '__main__':
    os.environ['DYN_UNET_DEBUG'] = '1'
    import argparse
    
    parser = argparse.ArgumentParser(description='Minimal nnUNet model test for debugging')
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID (e.g., 602 or Dataset602_KiTS23)")
    parser.add_argument('configuration', type=str,
                        help="Configuration (e.g., 3d_fullres, 3d_lowres, 2d)")
    parser.add_argument('fold', type=int,
                        help='Fold number (0-4)')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Default: nnUNetPlans')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Device: 'cuda' or 'cpu'. Default: cuda")
    
    args = parser.parse_args()
    
    # Set environment variables for reproducibility
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # IMPORTANT: Disable torch.compile for pdb debugging!
    # torch.compile wraps the model in OptimizedModule which bypasses Python code,
    # making pdb breakpoints inside the model ineffective.
    os.environ['nnUNet_compile'] = 'false'
    print("NOTE: torch.compile disabled for debugging (nnUNet_compile=false)")
    
    network, output = run_minimal_test(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        plans_identifier=args.p,
        device_str=args.device
    )
