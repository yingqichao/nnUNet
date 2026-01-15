#!/usr/bin/env python3
import json
import os
from pathlib import Path

# Paths
splits_file = "/raid/datasets/nnUNet_preprocessed/Dataset604_LiTS_Baseline/splits_final.json"
source_dir = "/raid/datasets/nnUNet_raw/Dataset602_LiTS19/imagesTr"
output_base = "/raid/datasets/nnUNet_raw/Dataset602_LiTS19/folds"

# Load splits
with open(splits_file, 'r') as f:
    splits = json.load(f)

# Create base folds directory
os.makedirs(output_base, exist_ok=True)

# Process each fold
for fold_idx, fold_data in enumerate(splits):
    fold_dir = os.path.join(output_base, f"fold{fold_idx}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create symlinks for training samples
    for sample in fold_data["train"]:
        source_file = os.path.join(source_dir, f"{sample}_0000.nii.gz")
        target_link = os.path.join(train_dir, f"{sample}_0000.nii.gz")
        
        if os.path.exists(source_file):
            if os.path.lexists(target_link):
                os.remove(target_link)
            os.symlink(source_file, target_link)
            print(f"Created symlink: {target_link} -> {source_file}")
        else:
            print(f"Warning: Source file not found: {source_file}")
    
    # Create symlinks for validation samples
    for sample in fold_data["val"]:
        source_file = os.path.join(source_dir, f"{sample}_0000.nii.gz")
        target_link = os.path.join(val_dir, f"{sample}_0000.nii.gz")
        
        if os.path.exists(source_file):
            if os.path.lexists(target_link):
                os.remove(target_link)
            os.symlink(source_file, target_link)
            print(f"Created symlink: {target_link} -> {source_file}")
        else:
            print(f"Warning: Source file not found: {source_file}")
    
    print(f"\nFold {fold_idx}: {len(fold_data['train'])} train samples, {len(fold_data['val'])} val samples")

print(f"\nDone! Created fold structure in {output_base}")

