"""
Custom DataLoaders for 3-way centering and dual-crop evaluation.

These dataloaders provide specialized patch sampling strategies for
medical image segmentation training and evaluation.
"""

import numpy as np
import torch
from typing import List, Tuple, Union

from threadpoolctl import threadpool_limits
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoader3WayCentering(nnUNetDataLoader):
    """
    Custom DataLoader with configurable 3-way patch centering probabilities.
    
    For each sample in batch, randomly choose centering mode based on probabilities:
    - Random: Center on any random voxel
    - Anatomy: Center on random anatomy voxel (label 1)
    - Tumor: Center on random tumor voxel (label 2 ONLY)
    
    Note: Labels >= 3 (e.g., cyst in KiTS) are trained but NOT explicitly centered on.
    This provides fine-grained control over patch sampling strategy.
    """
    
    # Label conventions
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
            prob_tumor: Probability of centering on tumor voxel (label 2 ONLY)
            
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
        
        # Only use PRIMARY_TUMOR_LABEL (2) for tumor-centered cropping
        # Labels >= 3 (e.g., cyst in KiTS) are trained but not explicitly centered on
        if self.PRIMARY_TUMOR_LABEL not in label_manager.foreground_labels:
            raise ValueError(
                f"The dedicated tumor channel (2) is not found! Please note that "
                f"{self.__class__.__name__} is designed for better tumor-focused training!"
            )
        self.tumor_labels = [self.PRIMARY_TUMOR_LABEL]
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
    
    def update_probabilities(self, prob_random: float, prob_anatomy: float, prob_tumor: float):
        """
        Update centering probabilities dynamically.
        
        This allows adaptive adjustment of patch sampling strategy during training.
        
        Args:
            prob_random: Probability of centering on random voxel
            prob_anatomy: Probability of centering on anatomy voxel (label 1)
            prob_tumor: Probability of centering on tumor voxel (label 2 ONLY)
        """
        total = prob_random + prob_anatomy + prob_tumor
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        self.prob_random = prob_random
        self.prob_anatomy = prob_anatomy
        self.prob_tumor = prob_tumor
    
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
            # Try to center on tumor (label 2 ONLY - self.tumor_labels contains only PRIMARY_TUMOR_LABEL)
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
                    # Use np.asarray() to handle NDArray/memmap types
                    data_all = torch.from_numpy(np.asarray(data_all)).float()
                    seg_all = torch.from_numpy(np.asarray(seg_all)).to(torch.int16)
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


class nnUNetDataLoaderDualCropEval(nnUNetDataLoader):
    """
    Evaluation DataLoader that creates TWO patches per sample for balanced evaluation.
    
    For each sample in the dataset:
    - Patch 1: Centered on a random ANATOMY voxel (label 1) → evaluates FP control
    - Patch 2: Centered on a random TUMOR voxel (label 2) → evaluates tumor detection
    
    This provides balanced evaluation that considers both:
    - False positive rate on normal anatomy regions
    - True positive rate (recall) on tumor regions
    
    If a sample has N original cases, evaluation will use 2N patches.
    Example: 25 samples → 50 patches (25 anatomy-centered + 25 tumor-centered)
    
    This is used for tuning and test set evaluation ONLY (not training).
    """
    
    ANATOMY_LABEL = 1  # liver for LiTS, kidney for KiTS
    TUMOR_LABEL = 2    # Only label 2 is used for tumor centering
    
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 transforms=None):
        """
        Note: batch_size here refers to the number of ORIGINAL samples per batch.
        The actual output will have 2x batch_size patches (anatomy + tumor for each sample).
        """
        super().__init__(
            data=data,
            batch_size=batch_size,
            patch_size=patch_size,
            final_patch_size=final_patch_size,
            label_manager=label_manager,
            oversample_foreground_percent=1.0,  # Not used, we do dual-crop
            sampling_probabilities=sampling_probabilities,
            pad_sides=pad_sides,
            probabilistic_oversampling=True,
            transforms=transforms
        )
        
        # Update data_shape and seg_shape to accommodate 2x patches
        # Original shape: (batch_size, channels, *patch_size)
        # New shape: (2 * batch_size, channels, *patch_size)
        self.original_batch_size = batch_size
        self.data_shape = (2 * batch_size,) + self.data_shape[1:]
        self.seg_shape = (2 * batch_size,) + self.seg_shape[1:]
    
    def get_bbox_for_mode(self, data_shape: np.ndarray, mode: str,
                          class_locations: Union[dict, None]) -> Tuple[List[int], List[int]]:
        """
        Get bounding box centered on a voxel of the specified class.
        
        Args:
            data_shape: Shape of the data (excluding channel dimension)
            mode: 'anatomy' or 'tumor'
            class_locations: Dict mapping class labels to voxel locations
            
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
        
        selected_voxel = None
        target_label = self.ANATOMY_LABEL if mode == 'anatomy' else self.TUMOR_LABEL
        fallback_label = self.TUMOR_LABEL if mode == 'anatomy' else self.ANATOMY_LABEL
        
        # Try to get voxel from target class
        if class_locations and target_label in class_locations and len(class_locations[target_label]) > 0:
            voxels = class_locations[target_label]
            selected_voxel = voxels[np.random.choice(len(voxels))]
        
        # Fallback to other foreground class if target not available
        if selected_voxel is None and class_locations:
            if fallback_label in class_locations and len(class_locations[fallback_label]) > 0:
                voxels = class_locations[fallback_label]
                selected_voxel = voxels[np.random.choice(len(voxels))]
        
        # Final fallback: random location
        if selected_voxel is None:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) if lbs[i] < ubs[i] + 1 
                        else lbs[i] for i in range(dim)]
        else:
            # Center on selected voxel (skip channel dim: voxel is [channel, z, y, x])
            bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
        
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        
        return bbox_lbs, bbox_ubs
    
    def generate_train_batch(self):
        """
        Generate an evaluation batch with dual-crop strategy.
        
        For each sample:
        1. Create one patch centered on random anatomy voxel
        2. Create one patch centered on random tumor voxel
        
        Output has 2x the original batch size.
        """
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        
        # Track keys for both patches (each key appears twice: once for anatomy, once for tumor)
        output_keys = []
        
        for j, key in enumerate(selected_keys):
            data, seg, seg_prev, properties = self._data.load_case(key)
            shape = data.shape[1:]
            class_locations = properties.get('class_locations')
            
            # === Patch 1: Anatomy-centered ===
            idx_anatomy = 2 * j
            bbox_lbs_anat, bbox_ubs_anat = self.get_bbox_for_mode(shape, 'anatomy', class_locations)
            bbox_anat = [[i, k] for i, k in zip(bbox_lbs_anat, bbox_ubs_anat)]
            
            data_all[idx_anatomy] = crop_and_pad_nd(data, bbox_anat, 0)
            seg_cropped_anat = crop_and_pad_nd(seg, bbox_anat, -1)
            if seg_prev is not None:
                seg_cropped_anat = np.vstack((seg_cropped_anat, crop_and_pad_nd(seg_prev, bbox_anat, -1)[None]))
            seg_all[idx_anatomy] = seg_cropped_anat
            output_keys.append(f"{key}_anatomy")
            
            # === Patch 2: Tumor-centered ===
            idx_tumor = 2 * j + 1
            bbox_lbs_tumor, bbox_ubs_tumor = self.get_bbox_for_mode(shape, 'tumor', class_locations)
            bbox_tumor = [[i, k] for i, k in zip(bbox_lbs_tumor, bbox_ubs_tumor)]
            
            data_all[idx_tumor] = crop_and_pad_nd(data, bbox_tumor, 0)
            seg_cropped_tumor = crop_and_pad_nd(seg, bbox_tumor, -1)
            if seg_prev is not None:
                seg_cropped_tumor = np.vstack((seg_cropped_tumor, crop_and_pad_nd(seg_prev, bbox_tumor, -1)[None]))
            seg_all[idx_tumor] = seg_cropped_tumor
            output_keys.append(f"{key}_tumor")
        
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
        
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    # Use np.asarray() to handle NDArray/memmap types
                    data_all = torch.from_numpy(np.asarray(data_all)).float()
                    seg_all = torch.from_numpy(np.asarray(seg_all)).to(torch.int16)
                    images = []
                    segs = []
                    # Process all 2*batch_size patches
                    for b in range(2 * self.original_batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'keys': output_keys}
        
        return {'data': data_all, 'target': seg_all, 'keys': output_keys}
