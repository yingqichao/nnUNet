"""
Custom nnUNet Trainer with anatomy-tumor focused sampling.

This trainer modifies the default patch sampling strategy:
- Default nnUNet: 33% forced foreground (random class), 67% random
- This trainer: 33% centered on tumor (label 2 ONLY), 67% centered on anatomy (label 1)

Label convention assumed:
- Label 0: Background
- Label 1: Anatomy (liver for LiTS, kidney for KiTS)
- Label 2: Tumor/lesion (PRIMARY focus for centering)
- Label 3+: Other lesions like cyst (trained but NOT explicitly centered on)

This ensures:
- 33% of patches are guaranteed to contain tumor (centered on label 2 voxel)
- 67% of patches are guaranteed to contain anatomy (centered on anatomy voxel)
- No purely background patches
- Better representation of small tumor regions in training
- Labels >= 3 (e.g., cyst in KiTS) are included in training but don't get
  explicit centering priority, avoiding excessive focus on secondary lesions

Usage:
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_AnatomyTumorFocused
"""

import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from batchgenerators.dataloading.data_loader import DataLoader
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoaderAnatomyTumorFocused(nnUNetDataLoader):
    """
    Custom DataLoader that implements anatomy-tumor focused sampling.
    
    For each batch:
    - Last (oversample_foreground_percent * batch_size) samples: centered on TUMOR (label 2 ONLY)
    - Remaining samples: centered on ANATOMY (label 1)
    
    This ensures all patches contain meaningful foreground, with explicit control
    over tumor vs anatomy representation. Labels >= 3 (e.g., cyst in KiTS) are 
    trained but NOT explicitly centered on to avoid excessive focus on secondary lesions.
    """
    
    # Class labels for anatomy-tumor focused sampling
    BACKGROUND_LABEL = 0
    ANATOMY_LABEL = 1  # liver for LiTS, kidney for KiTS
    PRIMARY_TUMOR_LABEL = 2  # Only label 2 is used for tumor-centered cropping
    # Note: Labels >= 3 (e.g., cyst in KiTS) are trained but NOT explicitly centered on
    
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.33,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            label_manager=label_manager,
            oversample_foreground_percent=oversample_foreground_percent,
            sampling_probabilities=sampling_probabilities,
            pad_sides=pad_sides,
            probabilistic_oversampling=probabilistic_oversampling,
            transforms=transforms
        )
        
        # Only use PRIMARY_TUMOR_LABEL (2) for tumor-centered cropping
        # Labels >= 3 (e.g., cyst in KiTS) are trained but not explicitly centered on
        if self.PRIMARY_TUMOR_LABEL not in label_manager.foreground_labels:
            raise ValueError(
                f"The dedicated tumor channel (2) is not found! Please note that "
                f"{self.__class__.__name__} is designed for better tumor-focused training!"
            )
        self.tumor_labels = [self.PRIMARY_TUMOR_LABEL]
        self.anatomy_label = self.ANATOMY_LABEL
        
        # Override the oversampling function to use our custom logic
        self.get_do_oversample = self._tumor_oversample_last_XX_percent if not probabilistic_oversampling \
            else self._tumor_probabilistic_oversampling
    
    def _tumor_oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        Returns True if this sample should be centered on TUMOR (label 2 ONLY).
        Returns False if this sample should be centered on ANATOMY (label 1).
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def _tumor_probabilistic_oversampling(self, sample_idx: int) -> bool:
        """Probabilistic version of tumor oversampling."""
        return np.random.uniform() < self.oversample_foreground_percent
    
    def get_bbox_anatomy_tumor_focused(self, data_shape: np.ndarray, force_tumor: bool, 
                                        class_locations: Union[dict, None], verbose: bool = False):
        """
        Modified bbox selection for anatomy-tumor focused sampling.
        
        Args:
            data_shape: Shape of the data (excluding channel dimension)
            force_tumor: If True, center on tumor voxel (label 2 ONLY). If False, center on anatomy voxel.
            class_locations: Dict mapping class labels to voxel locations
            verbose: Whether to print debug info
            
        Returns:
            bbox_lbs, bbox_ubs: Bounding box lower and upper bounds
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
        
        selected_class = None
        
        if force_tumor:
            # Try to center on tumor (label 2 ONLY - self.tumor_labels contains only PRIMARY_TUMOR_LABEL)
            tumor_classes_with_voxels = [l for l in self.tumor_labels 
                                         if l in class_locations and len(class_locations[l]) > 0]
            
            if len(tumor_classes_with_voxels) > 0:
                # Select tumor class (label 2)
                selected_class = tumor_classes_with_voxels[np.random.choice(len(tumor_classes_with_voxels))]
                if verbose:
                    print(f'Selected tumor class: {selected_class}')
            else:
                # No tumor in this image, fall back to anatomy
                if self.anatomy_label in class_locations and len(class_locations[self.anatomy_label]) > 0:
                    selected_class = self.anatomy_label
                    if verbose:
                        print(f'No tumor found, falling back to anatomy class: {selected_class}')
        else:
            # Center on anatomy (label 1)
            if self.anatomy_label in class_locations and len(class_locations[self.anatomy_label]) > 0:
                selected_class = self.anatomy_label
                if verbose:
                    print(f'Selected anatomy class: {selected_class}')
            else:
                # No anatomy in this image (unusual), try tumor as fallback
                tumor_classes_with_voxels = [l for l in self.tumor_labels 
                                             if l in class_locations and len(class_locations[l]) > 0]
                if len(tumor_classes_with_voxels) > 0:
                    selected_class = tumor_classes_with_voxels[np.random.choice(len(tumor_classes_with_voxels))]
                    if verbose:
                        print(f'No anatomy found, falling back to tumor class: {selected_class}')
        
        if selected_class is not None:
            voxels_of_that_class = class_locations[selected_class]
            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
            # Make sure it is within the bounds of lb and ub
            # i + 1 because we have first dimension 0 (channel)!
            bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
        else:
            # If no foreground at all (should be rare), fall back to random
            if verbose:
                print('No foreground found, using random crop')
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        
        return bbox_lbs, bbox_ubs
    
    def generate_train_batch(self):
        """
        Generate a training batch with anatomy-tumor focused sampling.
        
        Overrides the parent method to use our custom bbox selection.
        """
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        
        for j, i in enumerate(selected_keys):
            # Determine if this sample should be centered on tumor or anatomy
            force_tumor = self.get_do_oversample(j)
            
            data, seg, seg_prev, properties = self._data.load_case(i)
            
            shape = data.shape[1:]
            
            # Use our custom bbox selection
            bbox_lbs, bbox_ubs = self.get_bbox_anatomy_tumor_focused(
                shape, force_tumor, properties['class_locations']
            )
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


class nnUNetTrainer_AnatomyTumorFocused(nnUNetTrainer):
    """
    nnUNet Trainer with anatomy-tumor focused sampling strategy.
    
    Modifies the default sampling:
    - 33% of patches: centered on TUMOR voxels (label 2 ONLY)
    - 67% of patches: centered on ANATOMY voxels (label 1)
    
    This ensures:
    - All patches contain meaningful foreground (no pure background patches)
    - Tumor regions get adequate representation despite being small
    - Anatomy context is maintained in majority of patches
    - Labels >= 3 (e.g., cyst in KiTS) are trained but not explicitly centered on
    
    Label convention:
    - Label 0: Background
    - Label 1: Anatomy (liver for LiTS, kidney for KiTS)  
    - Label 2: Tumor/lesion (PRIMARY focus)
    - Label 3+: Secondary lesions (trained but not centered on)
    
    Usage:
        nnUNetv2_train 602 3d_fullres 0 -tr nnUNetTrainer_AnatomyTumorFocused
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
                 checkpoint_signature: str = None,
                 splits_file: str = None):
        super().__init__(plans, configuration, fold, dataset_json, device, 
                        checkpoint_signature, splits_file)
        
        # Log the sampling strategy
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Using nnUNetTrainer_AnatomyTumorFocused")
        self.print_to_log_file(f"TRAINING sampling strategy:")
        self.print_to_log_file(f"  - {self.oversample_foreground_percent*100:.1f}% patches centered on TUMOR (label 2 ONLY)")
        self.print_to_log_file(f"  - {(1-self.oversample_foreground_percent)*100:.1f}% patches centered on ANATOMY (label 1)")
        self.print_to_log_file(f"  - Labels >= 3 (e.g., cyst) are trained but NOT explicitly centered on")
        self.print_to_log_file(f"  - No purely background patches")
        self.print_to_log_file(f"VALIDATION (pseudo dice) sampling strategy:")
        self.print_to_log_file(f"  - 100% patches centered on TUMOR (label 2 ONLY)")
        self.print_to_log_file("=" * 70 + "\n")
    
    def get_dataloaders(self):
        """
        Override to use our custom DataLoader with anatomy-tumor focused sampling.
        """
        if self.dataset_class is None:
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        
        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)
        
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        # Use our custom DataLoader for TRAINING
        # Training uses self.oversample_foreground_percent (default 0.33):
        # - 33% patches centered on tumor
        # - 67% patches centered on anatomy
        dl_tr = nnUNetDataLoaderAnatomyTumorFocused(
            dataset_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        
        # Use our custom DataLoader for VALIDATION with 100% tumor-centered patches
        # This ensures pseudo dice is calculated on tumor-focused patches
        # - 100% patches centered on tumor (for better tumor-focused evaluation)
        dl_val = nnUNetDataLoaderAnatomyTumorFocused(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=1.0,  # 100% tumor-centered for pseudo dice
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


# Also provide a variant with higher tumor sampling rate (50%)
class nnUNetTrainer_AnatomyTumorFocused50(nnUNetTrainer_AnatomyTumorFocused):
    """
    Same as nnUNetTrainer_AnatomyTumorFocused but with 50% tumor sampling.
    
    - 50% of patches: centered on TUMOR voxels (label 2 ONLY)
    - 50% of patches: centered on ANATOMY voxels (label 1)
    - Labels >= 3 (e.g., cyst) are trained but NOT explicitly centered on
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
                 checkpoint_signature: str = None,
                 splits_file: str = None):
        # Call nnUNetTrainer.__init__ directly to skip parent's logging
        nnUNetTrainer.__init__(self, plans, configuration, fold, dataset_json, device,
                               checkpoint_signature, splits_file)
        # Override the oversample percent to 50%
        self.oversample_foreground_percent = 0.5
        
        self.print_to_log_file("\n" + "=" * 70)
        self.print_to_log_file("Using nnUNetTrainer_AnatomyTumorFocused50")
        self.print_to_log_file(f"TRAINING sampling strategy:")
        self.print_to_log_file(f"  - 50% patches centered on TUMOR (label 2 ONLY)")
        self.print_to_log_file(f"  - 50% patches centered on ANATOMY (label 1)")
        self.print_to_log_file(f"  - Labels >= 3 (e.g., cyst) are trained but NOT explicitly centered on")
        self.print_to_log_file(f"  - No purely background patches")
        self.print_to_log_file(f"VALIDATION (pseudo dice) sampling strategy:")
        self.print_to_log_file(f"  - 100% patches centered on TUMOR (label 2 ONLY)")
        self.print_to_log_file("=" * 70 + "\n")
