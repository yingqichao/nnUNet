from typing import Tuple, List, Union

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.preprocessing.resampling.no_resampling import no_resampling_hack


class NoResamplePlanner(ExperimentPlanner):
    """
    Planner variant that disables all resampling by returning identity functions
    for data/seg resampling and softmax export. All other behavior remains
    identical to the default planner.
    """

    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'nnUNetPlans_noResample',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = True):
        # Always suppress transpose to preserve original (H, W, D) axis order
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, True)

    @staticmethod
    def _identity_to_shape(
        data,
        new_shape: Union[Tuple[int, ...], List[int]],
        current_spacing: Union[Tuple[float, ...], List[float]],
        new_spacing: Union[Tuple[float, ...], List[float]],
        is_seg: bool = False,
        order: int = 3,
        order_z: int = 0,
        force_separate_z: Union[bool, None] = False,
        separate_z_anisotropy_threshold: float = None,
    ):
        return data

    def determine_resampling(self, *args, **kwargs):
        # Use a globally discoverable no-op resampling so preprocessing can resolve it by name from plans
        resampling_data = no_resampling_hack
        resampling_data_kwargs = {}
        resampling_seg = no_resampling_hack
        resampling_seg_kwargs = {}
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        # Also keep probabilities on the original grid
        resampling_fn = no_resampling_hack
        resampling_fn_kwargs = {}
        return resampling_fn, resampling_fn_kwargs

