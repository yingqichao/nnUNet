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

Evaluation order each epoch:
1. TUNING SET: Compute metrics (logged, for adaptive decisions)
2. TEST SET: Compute pseudo dice (logged, for checkpoint selection)
3. Checkpoint: "Yayy! New best EMA pseudo Dice" if improved (based on TEST only)

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
"""

import numpy as np
import torch
from typing import List, Tuple, Union
from threadpoolctl import threadpool_limits

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
        self.print_to_log_file("=" * 70 + "\n")
    
    def do_split(self) -> Tuple[List[str], List[str]]:
        """
        Override to create 3-way split: train/tuning/test.
        
        The tuning keys are stored in self.tuning_keys.
        Returns (train_keys, test_keys) - tuning is handled separately.
        """
        # Get original 2-way split from parent
        original_tr_keys, test_keys = super().do_split()
        
        # Deterministically split training into train + tuning
        # Use fold number in seed for reproducibility across runs
        rng = np.random.RandomState(seed=12345 + self.fold)
        shuffled = rng.permutation(original_tr_keys).tolist()
        
        n_tuning = int(len(shuffled) * self.tuning_ratio)
        self.tuning_keys = shuffled[:n_tuning]
        train_keys = shuffled[n_tuning:]
        
        self.print_to_log_file(f"3-way split for fold {self.fold}:")
        self.print_to_log_file(f"  - Training: {len(train_keys)} cases")
        self.print_to_log_file(f"  - Tuning:   {len(self.tuning_keys)} cases")
        self.print_to_log_file(f"  - Test:     {len(test_keys)} cases (original val fold)")
        
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
            Dictionary with:
            - 'dice_per_class': List of dice scores per class
            - 'mean_fg_dice': Mean foreground dice
            - 'loss': Average loss on tuning set
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
        
        # Compute dice per class
        dice_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0.0 
                         for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = float(np.nanmean(dice_per_class))
        mean_loss = float(np.mean(outputs_collated['loss']))
        
        self.network.train()
        
        return {
            'dice_per_class': dice_per_class,
            'mean_fg_dice': mean_fg_dice,
            'loss': mean_loss
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
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
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
        
        # Compute EMA of mean_fg_dice (same formula as test set: α=0.1)
        # ema[t] = ema[t-1] * 0.9 + mean_fg_dice * 0.1
        mean_fg_dice = tuning_result['mean_fg_dice']
        if len(self.tuning_metrics['ema_fg_dice']) > 0:
            prev_ema = self.tuning_metrics['ema_fg_dice'][-1]
            current_ema = prev_ema * 0.9 + mean_fg_dice * 0.1
        else:
            # First epoch - no previous EMA, use current mean directly
            current_ema = mean_fg_dice
        
        # Store metrics
        self.tuning_metrics['dice_per_class'].append(tuning_result['dice_per_class'])
        self.tuning_metrics['mean_fg_dice'].append(mean_fg_dice)
        self.tuning_metrics['ema_fg_dice'].append(current_ema)
        self.tuning_metrics['loss'].append(tuning_result['loss'])
        
        # Log tuning metrics
        self.print_to_log_file(f"  Tuning loss: {np.round(tuning_result['loss'], 4)}")
        self.print_to_log_file(f"  Tuning Pseudo dice: {[np.round(d, 4) for d in tuning_result['dice_per_class']]}")
        self.print_to_log_file(f"  Tuning EMA Pseudo dice: {np.round(current_ema, 4)}")
        
        # Store current tuning result for potential use in adaptive decisions
        tuning_result['ema_fg_dice'] = current_ema
        self._current_tuning_result = tuning_result
        
        # =====================================================================
        # EXTENSION POINT: Adaptive hyperparameter adjustment
        # =====================================================================
        # This is where you can implement adaptive training based on tuning metrics.
        # Example (NOT IMPLEMENTED - left for future extension):
        #
        # if tuning_result['dice_per_class'][1] < tuning_result['dice_per_class'][0]:
        #     # Tumor dice < anatomy dice, increase tumor sampling
        #     self.oversample_foreground_percent = min(0.8, 
        #         self.oversample_foreground_percent + 0.02)
        #     self.print_to_log_file(f"Adjusted tumor sampling to "
        #         f"{self.oversample_foreground_percent:.2f}")
        #
        # =====================================================================
        
        self.print_to_log_file("-" * 40)
        self.print_to_log_file("TEST SET evaluation (for checkpoint selection):")
    
    # Note: We don't override on_epoch_end() anymore - parent handles:
    # - Logging test metrics (train_loss, val_loss, Pseudo dice)
    # - Checkpoint selection with "Yayy! New best EMA pseudo Dice" message
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Override to also save tuning metrics and centering probabilities in checkpoint.
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
        }
        
        super().save_checkpoint(filename)
    
    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """
        Override to also load tuning metrics and centering probabilities from checkpoint.
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
            
            self.print_to_log_file(f"Loaded tuning info: {len(self.tuning_keys)} tuning cases")
            self.print_to_log_file(f"Loaded centering probs - Train: ({self.train_prob_random:.2f}/{self.train_prob_anatomy:.2f}/{self.train_prob_tumor:.2f})")


# Convenience alias with shorter name
nnUNetTrainer_Tuning = nnUNetTrainer_WithTuningSet
