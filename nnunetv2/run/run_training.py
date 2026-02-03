import multiprocessing
import os
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
from nnunetv2.utilities.helpers import check_if_proceed


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          device: torch.device = torch.device('cuda'),
                          checkpoint_signature: str = None,
                          splits_file: str = None):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    
    # Support both plans identifier (e.g., "nnUNetPlans") and absolute path (e.g., "/path/to/plans.json")
    # If "/" is in the plans_identifier, treat it as an absolute path
    if '/' in plans_identifier:
        plans_file = plans_identifier
        print(f"Using absolute plans file path: {plans_file}")
    else:
        plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device,
                                    checkpoint_signature=checkpoint_signature,
                                    splits_file=splits_file)
    return nnunet_trainer


def check_everything_before_training(nnunet_trainer: nnUNetTrainer, skip_confirm: bool = False,
                                      ignore_existing_best: bool = False, continue_training: bool = False):
    """
    Display training configuration and prompt user for confirmation before training starts.
    This helps prevent accidentally training in the wrong folder and overwriting good checkpoints.
    
    Items shown to user for verification:
        1. Dataset name - the dataset being trained on
        2. Plans name - the nnUNet plans identifier being used
        3. Configuration - the training configuration (e.g., 3d_fullres)
        4. Fold - the cross-validation fold number
        5. Output folder - where checkpoints and logs will be saved
        6. Splits file - the JSON file defining train/val splits (default or custom)
        7. Existing top-K checkpoints info (if found):
           - Checkpoint file paths and their ranks
           - Best EMA pseudo Dice scores achieved
           - Epochs when checkpoints were saved
           - Note about checkpoint overwrite policy
        8. Warning if --ignore_existing_best is set
    
    Args:
        nnunet_trainer: The initialized nnUNetTrainer instance
        skip_confirm: If True, show info but don't require user confirmation
        ignore_existing_best: If True, show warning about best checkpoint handling
        continue_training: If True, indicates --c flag is set (affects warning message)
    """
    # Determine the splits file path
    if nnunet_trainer.splits_file is not None:
        splits_file_path = join(nnunet_trainer.preprocessed_dataset_folder_base, nnunet_trainer.splits_file)
    else:
        splits_file_path = join(nnunet_trainer.preprocessed_dataset_folder_base, "splits_final.json") + " (default)"
    
    info_lines = [
        f"Dataset: {nnunet_trainer.plans_manager.dataset_name}",
        f"Plans: {nnunet_trainer.plans_manager.plans_name}",
        f"Configuration: {nnunet_trainer.configuration_name}",
        f"Fold: {nnunet_trainer.fold}",
        f"Output folder: {nnunet_trainer.output_folder}",
        f"Splits file: {splits_file_path}",
    ]
    
    # Check for top-K checkpoints (if trainer supports it) or just the best checkpoint
    has_top_k = hasattr(nnunet_trainer, 'get_top_k_checkpoint_path')
    top_k = getattr(nnunet_trainer, 'top_k_checkpoints', 5) if has_top_k else 1
    
    found_checkpoints = []
    
    if has_top_k:
        # Scan for top-K checkpoints
        for rank in range(1, top_k + 1):
            checkpoint_path = nnunet_trainer.get_top_k_checkpoint_path(rank)
            if isfile(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    # Support both _best_dice (new) and _best_ema (legacy)
                    dice = checkpoint.get('_best_dice', checkpoint.get('_best_ema', None))
                    epoch = checkpoint.get('current_epoch', 'unknown')
                    
                    # If dice is None, try to recover from logging history
                    if dice is None and 'logging' in checkpoint:
                        logging_data = checkpoint['logging']
                        if 'mean_fg_dice' in logging_data and len(logging_data['mean_fg_dice']) > 0:
                            dice_history = logging_data['mean_fg_dice']
                            if isinstance(epoch, int) and epoch > 0 and epoch <= len(dice_history):
                                dice = dice_history[epoch - 1]
                            elif len(dice_history) > 0:
                                dice = dice_history[-1]
                    
                    found_checkpoints.append((rank, checkpoint_path, dice, epoch))
                except Exception as e:
                    found_checkpoints.append((rank, checkpoint_path, None, f"Error: {e}"))
    else:
        # Fall back to old behavior - check for single best checkpoint
        checkpoint_candidates = [
            join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_best')),
            join(nnunet_trainer.output_folder, 'checkpoint_best.pth'),
        ]
        for candidate in checkpoint_candidates:
            if isfile(candidate):
                try:
                    checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)
                    # Support both _best_dice (new) and _best_ema (legacy)
                    dice = checkpoint.get('_best_dice', checkpoint.get('_best_ema', None))
                    epoch = checkpoint.get('current_epoch', 'unknown')
                    
                    # If dice is None, try to recover from logging history
                    if dice is None and 'logging' in checkpoint:
                        logging_data = checkpoint['logging']
                        if 'mean_fg_dice' in logging_data and len(logging_data['mean_fg_dice']) > 0:
                            dice_history = logging_data['mean_fg_dice']
                            if isinstance(epoch, int) and epoch > 0 and epoch <= len(dice_history):
                                dice = dice_history[epoch - 1]
                            elif len(dice_history) > 0:
                                dice = dice_history[-1]
                    
                    found_checkpoints.append((1, candidate, dice, epoch))
                except Exception as e:
                    found_checkpoints.append((1, candidate, None, f"Error: {e}"))
                break
    
    if found_checkpoints:
        info_lines.append(f"\n*** EXISTING TOP-{len(found_checkpoints)} CHECKPOINTS FOUND ***")
        for rank, path, dice, epoch in found_checkpoints:
            dice_str = f"{dice:.4f}" if dice is not None else "Not recorded"
            info_lines.append(f"  Rank {rank}: Dice={dice_str}, epoch={epoch}")
            info_lines.append(f"           {path}")
        info_lines.append(f"\nNOTE: New checkpoints will only be saved if they rank in top-{top_k}.")
    else:
        info_lines.append(f"\nNo existing checkpoints found in output folder.")
    
    # Add warning if --ignore_existing_best is set (message depends on --c flag)
    if ignore_existing_best:
        if continue_training:
            # With --c: checkpoint is loaded but best metric is reset (checkpoint file NOT removed)
            info_lines.append(f"\n*** NOTE: --ignore_existing_best with --c: "
                              f"Checkpoint will be LOADED but best metric tracking will be RESET. "
                              f"Model weights loaded, checkpoint ranking starts fresh. ***")
        else:
            # Without --c: all checkpoint files are removed
            num_to_remove = len(found_checkpoints) if found_checkpoints else 0
            info_lines.append(f"\n*** NOTE!!! You have set --ignore_existing_best, and this will accordingly "
                              f"REMOVE all {num_to_remove} checkpoint file(s) above! ***")
    
    info_str = "\n".join(info_lines)
    
    if skip_confirm:
        print("=" * 60)
        print("Training Configuration (--skip_manual_confirm is set):")
        print("=" * 60)
        print(info_str)
        print("=" * 60)
    else:
        check_if_proceed(info_str)


def maybe_remove_existing_best_checkpoint(nnunet_trainer: nnUNetTrainer, ignore_existing_best: bool = False):
    """
    Remove existing best checkpoint(s) if --ignore_existing_best is set.
    
    This allows starting fresh without the accuracy threshold from a previous training run.
    For trainers with top-K checkpoint support, removes all top-K checkpoint files.
    For standard trainers, removes both new naming convention (checkpoint_best_<plans>.pth) 
    and old naming convention (checkpoint_best.pth).
    
    Args:
        nnunet_trainer: The initialized nnUNetTrainer instance
        ignore_existing_best: If True, remove existing best checkpoints
    """
    if not ignore_existing_best:
        return
    
    import os
    
    checkpoint_candidates = []
    
    # Check if trainer supports top-K checkpoints
    has_top_k = hasattr(nnunet_trainer, 'get_top_k_checkpoint_path')
    
    if has_top_k:
        # Collect all top-K checkpoint paths
        top_k = getattr(nnunet_trainer, 'top_k_checkpoints', 5)
        for rank in range(1, top_k + 1):
            checkpoint_candidates.append(nnunet_trainer.get_top_k_checkpoint_path(rank))
    
    # Also check for old naming convention (for backward compatibility)
    checkpoint_candidates.extend([
        join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_best')),
        join(nnunet_trainer.output_folder, 'checkpoint_best.pth'),
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in checkpoint_candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    removed_count = 0
    for checkpoint_path in unique_candidates:
        if isfile(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"*** REMOVED checkpoint: {checkpoint_path}")
                removed_count += 1
            except Exception as e:
                print(f"WARNING: Failed to remove checkpoint {checkpoint_path}: {e}")
    
    # Reset trainer state
    if removed_count > 0:
        # Reset both _best_dice (new) and _best_ema (legacy) for compatibility
        if hasattr(nnunet_trainer, '_best_dice'):
            nnunet_trainer._best_dice = None
        nnunet_trainer._best_ema = None
        # Also reset top_k_dices/top_k_emas if available
        if hasattr(nnunet_trainer, 'top_k_dices'):
            nnunet_trainer.top_k_dices = []
        if hasattr(nnunet_trainer, 'top_k_emas'):
            nnunet_trainer.top_k_emas = []
        print(f"*** REMOVED {removed_count} checkpoint(s). Training will start fresh.")


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None, checkpoint_path: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        # If checkpoint_path is specified, use it directly
        if checkpoint_path is not None:
            if isfile(checkpoint_path):
                expected_checkpoint_file = checkpoint_path
                print(f"Using specified checkpoint for continue training: {expected_checkpoint_file}")
            else:
                # raise FileNotFoundError(f"Specified checkpoint file not found: {checkpoint_path}")
                # Instead of raising an error, ask the user to confirm if they want to continue training without a checkpoint
                print(f"Specified checkpoint file not found: {checkpoint_path}")
                print("Do you want to continue training without a checkpoint? (y/n)")
                answer = input()
                if answer == 'y':
                    expected_checkpoint_file = None
                    print("Continuing training without a checkpoint...")
                else:
                    raise FileNotFoundError(f"Specified checkpoint file not found: {checkpoint_path}")
        else:
            # Try new naming convention first, then fall back to old naming
            checkpoint_candidates = [
                join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_best')),
                join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_latest')),
                join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_final')),
                # Fallback to old naming convention
                join(nnunet_trainer.output_folder, 'checkpoint_best.pth'),
                join(nnunet_trainer.output_folder, 'checkpoint_latest.pth'),
                join(nnunet_trainer.output_folder, 'checkpoint_final.pth'),
            ]
            expected_checkpoint_file = None
            for candidate in checkpoint_candidates:
                if isfile(candidate):
                    expected_checkpoint_file = candidate
                    print(f"Found checkpoint for continue training: {expected_checkpoint_file}")
                    break
            if expected_checkpoint_file is None:
                print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                                   f"continue from. Starting a new training...")
    elif validation_only:
        # Try new naming convention first, then fall back to old naming
        checkpoint_candidates = [
            join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_best')),
            join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_latest')),
            join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_final')),
            # Fallback to old naming convention
            join(nnunet_trainer.output_folder, 'checkpoint_best.pth'),
            join(nnunet_trainer.output_folder, 'checkpoint_latest.pth'),
            join(nnunet_trainer.output_folder, 'checkpoint_final.pth'),
        ]
        expected_checkpoint_file = None
        for candidate in checkpoint_candidates:
            if isfile(candidate):
                expected_checkpoint_file = candidate
                print(f"Found checkpoint file: {expected_checkpoint_file}. Beginning validation...")
                break
            else:
                print(f"Checkpoint not found: {candidate}")
        if expected_checkpoint_file is None:
            raise RuntimeError(f"Cannot run validation because the training is not finished yet! Candidates tried: {checkpoint_candidates}")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size, checkpoint_signature=None, splits_file=None,
            checkpoint_path=None, skip_manual_confirm=False, pattern_original_samples=None,
            ignore_existing_best=False, skip_val=False, ignore_synthetic=False,
            main_gpu_id=None, backup_gpu_ids=None, integration_test=False, specify_val_set_only=False):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                           checkpoint_signature=checkpoint_signature,
                                           splits_file=splits_file)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing
    
    # Set pattern for identifying original samples (used by nnUNetTrainer_WithTuningSet)
    if pattern_original_samples is not None:
        nnunet_trainer.pattern_original_samples = pattern_original_samples
    
    # If ignore_synthetic is set, exclude all synthetic data from training
    if ignore_synthetic and hasattr(nnunet_trainer, 'max_synthetic_ratio'):
        nnunet_trainer.max_synthetic_ratio = 0.0
    
    # Set backup GPUs for async test evaluation (used by nnUNetTrainer_WithTuningSet)
    if hasattr(nnunet_trainer, 'backup_gpu_ids') and backup_gpu_ids is not None:
        gpu_list = [int(x.strip()) for x in backup_gpu_ids.split(',') if x.strip()]
        if len(gpu_list) == 0:
            raise ValueError("--backup_gpu_ids cannot be empty in DDP mode")
        if len(gpu_list) > 3:
            gpu_list = gpu_list[:3]
        nnunet_trainer.backup_gpu_ids = gpu_list
        # Update max concurrent subprocesses based on available GPUs
        if hasattr(nnunet_trainer, '_max_concurrent_subprocesses'):
            nnunet_trainer._max_concurrent_subprocesses = len(gpu_list)
    
    # Set integration test mode (used by nnUNetTrainer_WithTuningSet)
    # Must call _setup_integration_test_mode() to modify output folder BEFORE confirmation
    if integration_test and hasattr(nnunet_trainer, 'integration_test_mode'):
        nnunet_trainer.integration_test_mode = True
        if hasattr(nnunet_trainer, '_setup_integration_test_mode'):
            nnunet_trainer._setup_integration_test_mode()
    
    # Set specify_val_set_only mode (use only fold's val set, train on all other samples)
    if specify_val_set_only:
        nnunet_trainer.specify_val_set_only = True
        if rank == 0:
            print(f"*** specify_val_set_only mode enabled: fold {fold}'s val set only, all other samples for training")

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    # Show existing best checkpoint info and ask for confirmation (only on rank 0, unless validation only)
    if rank == 0 and not val:
        check_everything_before_training(nnunet_trainer, skip_confirm=skip_manual_confirm,
                                         ignore_existing_best=ignore_existing_best,
                                         continue_training=c)
    # Synchronize all ranks after confirmation
    dist.barrier()
    
    # Remove existing best checkpoint if requested (only on rank 0, only for fresh training)
    if rank == 0 and not val and not c:
        maybe_remove_existing_best_checkpoint(nnunet_trainer, ignore_existing_best)
    # Synchronize all ranks after removal
    dist.barrier()

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights, checkpoint_path=checkpoint_path)
    
    # If --ignore_existing_best is set with --c, reset best metrics after loading checkpoint
    # This allows loading model weights but starting fresh for checkpoint selection
    # (useful when evaluation criteria has changed)
    if c and ignore_existing_best:
        # Reset both _best_dice (new) and _best_ema (legacy) for compatibility
        old_best_dice = getattr(nnunet_trainer, '_best_dice', None)
        old_best_ema = nnunet_trainer._best_ema
        old_best = old_best_dice if old_best_dice is not None else old_best_ema
        
        if hasattr(nnunet_trainer, '_best_dice'):
            nnunet_trainer._best_dice = None
        nnunet_trainer._best_ema = None
        # Also set flag to skip comparing with existing checkpoint file on first epoch
        nnunet_trainer._skip_existing_best_comparison = True
        if rank == 0:
            print(f"*** --ignore_existing_best with --c: Reset best metric from {old_best} to None")
            print(f"*** Model weights loaded, but best checkpoint selection starts fresh.")
            print(f"*** Will skip existing checkpoint file comparison on first evaluation.")

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if not skip_val:
        if val_with_best:
            best_checkpoint = join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_best'))
            nnunet_trainer.load_checkpoint(best_checkpoint)
        nnunet_trainer.perform_actual_validation(npz)
    else:
        print("Skipping final validation (--skip_val flag set)")
    cleanup_ddp()


def run_training(
    dataset_name_or_id: Union[str, int],             # args.dataset_name_or_id
    configuration: str,                              # args.configuration
    fold: Union[int, str],                           # args.fold
    trainer_class_name: str = 'nnUNetTrainer',       # args.tr
    plans_identifier: str = 'nnUNetPlans',           # args.p
    pretrained_weights: Optional[str] = None,        # args.pretrained_weights
    num_gpus: int = 1,                              # args.num_gpus
    export_validation_probabilities: bool = False,   # args.npz
    continue_training: bool = False,                 # args.c
    only_run_validation: bool = False,              # args.val
    disable_checkpointing: bool = False,             # args.disable_checkpointing
    val_with_best: bool = False,                    # args.val_best
    skip_val: bool = True,                           # args.skip_val (default True now)
    device: torch.device = torch.device('cuda'),     # args.device
    checkpoint_signature: Optional[str] = None,      # args.signature
    splits_file: Optional[str] = None,               # args.split
    checkpoint_path: Optional[str] = None,           # args.checkpoint_path
    skip_manual_confirm: bool = False,               # args.skip_manual_confirm
    pattern_original_samples: Optional[str] = None,  # args.pattern_original_samples
    ignore_existing_best: bool = False,              # args.ignore_existing_best
    ignore_synthetic: bool = False,                  # args.ignore_synthetic
    main_gpu_id: Optional[int] = None,               # args.main_gpu_id
    backup_gpu_ids: Optional[str] = None,            # args.backup_gpu_ids (comma-separated)
    integration_test: bool = False,                  # args.integration_test
    specify_val_set_only: bool = False                   # args.specify_val_set_only
):
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus,
                     checkpoint_signature,
                     splits_file,
                     checkpoint_path,
                     skip_manual_confirm,
                     pattern_original_samples,
                     ignore_existing_best,
                     skip_val,
                     ignore_synthetic,
                     main_gpu_id,
                     backup_gpu_ids,
                     integration_test,
                     specify_val_set_only),
                 nprocs=num_gpus,
                 join=True)
    else:
        # =====================================================================
        # GPU CONFIGURATION: Set main GPU before creating trainer
        # =====================================================================
        if main_gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(main_gpu_id)
            print(f"*** Set CUDA_VISIBLE_DEVICES={main_gpu_id} for main training process")
        
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, device=device,
                                               checkpoint_signature=checkpoint_signature,
                                               splits_file=splits_file)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing
        
        # Set pattern for identifying original samples (used by nnUNetTrainer_WithTuningSet)
        if pattern_original_samples is not None:
            nnunet_trainer.pattern_original_samples = pattern_original_samples
        
        # If ignore_synthetic is set, exclude all synthetic data from training
        if ignore_synthetic and hasattr(nnunet_trainer, 'max_synthetic_ratio'):
            nnunet_trainer.max_synthetic_ratio = 0.0
        
        # Set backup GPUs for async test evaluation (used by nnUNetTrainer_WithTuningSet)
        if hasattr(nnunet_trainer, 'backup_gpu_ids'):
            if backup_gpu_ids is not None:
                # Parse comma-separated GPU IDs
                gpu_list = [int(x.strip()) for x in backup_gpu_ids.split(',') if x.strip()]
                if len(gpu_list) == 0:
                    raise ValueError("--backup_gpu_ids cannot be empty. Provide at least one GPU ID.")
                if len(gpu_list) > 3:
                    print(f"*** WARNING: More than 3 backup GPUs specified. Using only first 3: {gpu_list[:3]}")
                    gpu_list = gpu_list[:3]
                nnunet_trainer.backup_gpu_ids = gpu_list
                # Update max concurrent subprocesses based on available GPUs
                if hasattr(nnunet_trainer, '_max_concurrent_subprocesses'):
                    nnunet_trainer._max_concurrent_subprocesses = len(gpu_list)
                print(f"*** Set backup GPU IDs={gpu_list} for async test evaluation (max {len(gpu_list)} concurrent subprocesses)")
        
        # Set integration test mode (used by nnUNetTrainer_WithTuningSet)
        # Must call _setup_integration_test_mode() to modify output folder BEFORE confirmation
        if integration_test and hasattr(nnunet_trainer, 'integration_test_mode'):
            nnunet_trainer.integration_test_mode = True
            if hasattr(nnunet_trainer, '_setup_integration_test_mode'):
                nnunet_trainer._setup_integration_test_mode()
        
        # Set specify_val_set_only mode (use only fold's val set, train on all other samples)
        # This is set BEFORE do_split() is called, so the split logic picks it up
        if specify_val_set_only:
            nnunet_trainer.specify_val_set_only = True
            print(f"*** specify_val_set_only mode enabled: fold {fold}'s val set only, all other samples for training")

        assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        # Show existing best checkpoint info and ask for confirmation (unless validation only)
        if not only_run_validation:
            check_everything_before_training(nnunet_trainer, skip_confirm=skip_manual_confirm,
                                             ignore_existing_best=ignore_existing_best,
                                             continue_training=continue_training)
        
        # Remove existing best checkpoint if requested (only for fresh training, not continue or validation)
        if not only_run_validation and not continue_training:
            maybe_remove_existing_best_checkpoint(nnunet_trainer, ignore_existing_best)

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights,
                              checkpoint_path=checkpoint_path)
        
        # If --ignore_existing_best is set with --c, reset best metrics after loading checkpoint
        # This allows loading model weights but starting fresh for checkpoint selection
        # (useful when evaluation criteria has changed)
        if continue_training and ignore_existing_best:
            # Reset both _best_dice (new) and _best_ema (legacy) for compatibility
            old_best_dice = getattr(nnunet_trainer, '_best_dice', None)
            old_best_ema = nnunet_trainer._best_ema
            old_best = old_best_dice if old_best_dice is not None else old_best_ema
            
            if hasattr(nnunet_trainer, '_best_dice'):
                nnunet_trainer._best_dice = None
            nnunet_trainer._best_ema = None
            # Also set flag to skip comparing with existing checkpoint file on first epoch
            nnunet_trainer._skip_existing_best_comparison = True
            print(f"*** --ignore_existing_best with --c: Reset best metric from {old_best} to None")
            print(f"*** Model weights loaded, but best checkpoint selection starts fresh.")
            print(f"*** Will skip existing checkpoint file comparison on first evaluation.")

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if not skip_val:
            if val_with_best:
                best_checkpoint = join(nnunet_trainer.output_folder, nnunet_trainer.get_checkpoint_filename('checkpoint_best'))
                nnunet_trainer.load_checkpoint(best_checkpoint)
            nnunet_trainer.perform_actual_validation(export_validation_probabilities)
        else:
            print("Skipping final validation (--skip_val flag set)")


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Plans identifier (e.g., nnUNetPlans, nnUNetResEncUNetLPlans) or '
                             'absolute path to plans JSON file (if contains "/"). Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('--skip_val', action='store_true', required=False, default=True,
                        help='[DEFAULT=True] Skip the final validation step after training. '
                             'Use --no_skip_val to run final validation.')
    parser.add_argument('--no_skip_val', action='store_true', required=False,
                        help='[OPTIONAL] Run final validation after training (overrides default --skip_val).')
    parser.add_argument('-signature', type=str, required=False, default=None,
                        help='[OPTIONAL] Custom signature to append to checkpoint filenames. '
                             'Checkpoints will be named: checkpoint_<type>_<plans_name>_<signature>.pth. '
                             'If not specified, only plans_name is appended.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    parser.add_argument('--split', type=str, required=False, default=None,
                        help='[OPTIONAL] Custom splits file name (e.g., splits_mix.json). '
                             'File should be in the preprocessed dataset folder. '
                             'If not specified, uses splits_final.json (default nnUNet behavior).')
    parser.add_argument('--checkpoint_path', type=str, required=False, default=None,
                        help='[OPTIONAL] Absolute path to a specific checkpoint file to load when using --c. '
                             'If not specified, searches for checkpoint_final/latest/best in output folder.')
    parser.add_argument('--skip_manual_confirm', action='store_true', required=False,
                        help='[OPTIONAL] Skip the manual confirmation prompt that shows existing best checkpoint '
                             'accuracy before training. Use this for automated/scripted training runs.')
    parser.add_argument('--pattern_original_samples', type=str, required=False, default=None,
                        help='[OPTIONAL] Regex pattern to identify original (non-synthetic) samples by their '
                             'basename (without extension). E.g., "liver_\\d+" for files like liver_0, liver_123. '
                             'Samples not matching this pattern are considered synthetic. '
                             'Only used by nnUNetTrainer_WithTuningSet. If not specified, all samples are original.')
    parser.add_argument('--ignore_existing_best', action='store_true', required=False,
                        help='[OPTIONAL] Reset best checkpoint threshold. Behavior depends on --c flag: '
                             '(1) Without --c: REMOVES existing best checkpoint file(s) before training. '
                             '(2) With --c: Loads checkpoint but RESETS best metric to None after loading. '
                             'This allows loading model weights but starting fresh for checkpoint selection '
                             '(useful when evaluation criteria has changed). Use with caution!')
    parser.add_argument('--ignore_synthetic', action='store_true', required=False,
                        help='[OPTIONAL] If set, all synthetic samples (those NOT matching --pattern_original_samples) '
                             'will be excluded from training. Requires --pattern_original_samples to be set. '
                             'Only used by nnUNetTrainer_WithTuningSet.')
    parser.add_argument('--specify_val_set_only', action='store_true', required=False,
                        help='[OPTIONAL] If set, only use the validation set from splits_final.json for the specified fold. '
                             'ALL other samples (including from other folds) go into training. '
                             'This is applied BEFORE --ignore_synthetic filtering.')
    parser.add_argument('--main_gpu_id', type=int, required=False, default=None,
                        help='[OPTIONAL] GPU ID for main training process. Sets CUDA_VISIBLE_DEVICES before training. '
                             'If not specified, uses current CUDA_VISIBLE_DEVICES setting.')
    parser.add_argument('--backup_gpu_ids', type=str, required=False, default=None,
                        help='[OPTIONAL] Comma-separated GPU IDs for async test evaluation subprocesses (e.g., "0,1,2"). '
                             'Up to 3 concurrent subprocesses can run on these GPUs. '
                             'Only used by nnUNetTrainer_WithTuningSet with async evaluation. '
                             'If not specified, uses same GPU as main training (may cause OOM).')
    parser.add_argument('--integration_test', action='store_true', required=False,
                        help='[OPTIONAL] Run integration test with minimal data (5 samples per split). '
                             'Only used by nnUNetTrainer_WithTuningSet. Output folder gets "_integration_test" suffix. '
                             'Existing integration test folder will be removed.')
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # Handle skip_val: default is True, --no_skip_val overrides to False
    skip_val = args.skip_val and not args.no_skip_val

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                 skip_val, device=device, checkpoint_signature=args.signature, splits_file=args.split,
                 checkpoint_path=args.checkpoint_path, skip_manual_confirm=args.skip_manual_confirm,
                 pattern_original_samples=args.pattern_original_samples,
                 ignore_existing_best=args.ignore_existing_best,
                 ignore_synthetic=args.ignore_synthetic,
                 main_gpu_id=args.main_gpu_id,
                 backup_gpu_ids=args.backup_gpu_ids,
                 integration_test=args.integration_test,
                 specify_val_set_only=args.specify_val_set_only)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
