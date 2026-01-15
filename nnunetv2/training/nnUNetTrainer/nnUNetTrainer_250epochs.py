"""
Custom nnUNet Trainer with 250 epochs for LiTS baseline experiment.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_250epochs(nnUNetTrainer):
    """
    nnUNet Trainer configured for 250 epochs.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
                 checkpoint_signature: str = None):
        super().__init__(plans, configuration, fold, dataset_json,
                        device, checkpoint_signature)
        # Override the number of epochs
        self.num_epochs = 250
        
        print(f"\n{'='*60}")
        print(f"nnUNetTrainer_250epochs initialized")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"{'='*60}\n")
