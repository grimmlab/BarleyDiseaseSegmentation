"""
Hyperparameter optimization (HPO) initialization for barley disease segmentation.

Provides search spaces, parameter suggestion strategies, and experiment setup
for optimizing binary and multiclass segmentation models.
"""

import optuna
from torch.utils.data import DataLoader
import torch

from barley_disease_segmentation.config import *
from barley_disease_segmentation.dataset import BarleyLeafDataset
from barley_disease_segmentation.common import get_augmentations, get_batch_size_config

from .evaluation_inference import SegmentationMetrics
from .loss import FocalDiceLoss
from .model_architecture import FlexibleUNet

__all__ = ['HPOInitializer',
           'setup_experiment_from_params',
           'create_pruning_study']

class HPOInitializer:
    """
    HPO initialization with search spaces for barley disease segmentation.

    Includes both preliminary and optimized search spaces based on analysis
    of 3,892 trials across encoder architectures and tasks.
    """

    # ORIGINAL UNIFIED_SEARCH_SPACE -> preliminary HPO
    UNIFIED_SEARCH_SPACE = {
        'lr_range': [1e-5, 1e-2],
        'batch_sizes': [64, 128, 192],
        'weight_decay_range': [1e-6, 1e-2],
        'bottleneck_dropout_range': [0.1, 0.5],
        'decoder_dropout_range': [0.1, 0.3],
        'dice_weight_range': [0.7, 1.0],
        'focal_alpha_range': [0.6, 0.8],
        'focal_gamma_range': [1.5, 2.8]
    }

    # OPTIMIZED SEARCH SPACES BASED ON 3,892 TRIALS ANALYSIS -> refined HPO
    OPTIMIZED_UNIVERSAL_SPACE = {
        'lr_range': [1e-6, 1e-2],  # Expanded lower bound
        'weight_decay_range': [1e-8, 1e-1],  # Expanded lower bound
        'decoder_dropout_range': [0.05, 0.25],  # Expanded lower bound
        # Fixed parameters based on stability analysis
        'fixed_params': {
            'dice_weight': 0.721,
            'focal_alpha': 0.737,
            'focal_gamma': 1.847,
            'bottleneck_dropout': 0.385
        }
    }

    @staticmethod
    def suggest_parameters_optimized_universal(trial, fixed_task=None, fixed_encoder=None, fixed_batch_size=32):
        """
        Suggest parameters from data-driven universal search space.

        Based on analysis of 3,892 trials across all encoders/tasks.
        Optimizes only 3 key parameters where best values cluster at boundaries.

        Args:
            trial: Optuna trial object
            fixed_task: If provided, fix the task ('multiclass', 'binary_ram', 'binary_rust')
            fixed_encoder: If provided, fix the encoder architecture
            fixed_batch_size: If provided, fix the batch size

        Returns:
            dict: Parameter dictionary for experiment setup
        """
        params = {}

        # Architecture and task (unless fixed)
        if fixed_encoder:
            params['encoder_name'] = fixed_encoder
        else:
            params['encoder_name'] = trial.suggest_categorical(
                'encoder_name',
                ['resnet34', 'efficientnet_b2', 'convnext_tiny']
            )

        if fixed_task:
            params['task'] = fixed_task
        else:
            params['task'] = trial.suggest_categorical(
                'task',
                ['multiclass', 'binary_ram', 'binary_rust']
            )

        # Fixed batch size or use recommended from config
        batch_config = get_batch_size_config(params['encoder_name'])
        batch_choice = batch_config['recommended']
        if fixed_batch_size:
            params['batch_size'] = fixed_batch_size
        else:
            params['batch_size'] = batch_choice

        # Fixed parameters from stability analysis
        params.update(HPOInitializer.OPTIMIZED_UNIVERSAL_SPACE['fixed_params'])

        # Optimize only these 3 key parameters
        params.update({
            'lr': trial.suggest_float(
                'lr',
                *HPOInitializer.OPTIMIZED_UNIVERSAL_SPACE['lr_range'],
                log=True
            ),
            'weight_decay': trial.suggest_float(
                'weight_decay',
                *HPOInitializer.OPTIMIZED_UNIVERSAL_SPACE['weight_decay_range'],
                log=True
            ),
            'decoder_dropout': trial.suggest_float(
                'decoder_dropout',
                *HPOInitializer.OPTIMIZED_UNIVERSAL_SPACE['decoder_dropout_range']
            ),
        })

        return params

    @staticmethod
    def suggest_parameters(trial, fixed_task=None, fixed_encoder=None, fixed_batch_size=32):
        """
        Suggest parameters from preliminary unified search space.

        Args:
            trial: Optuna trial object
            fixed_task: If provided, fix the task
            fixed_encoder: If provided, fix the encoder architecture
            fixed_batch_size: If provided, fix the batch size

        Returns:
            dict: Parameter dictionary for experiment setup
        """
        params = {}

        # Architecture and task (unless fixed)
        if fixed_encoder:
            params['encoder_name'] = fixed_encoder
        else:
            params['encoder_name'] = trial.suggest_categorical(
                'encoder_name',
                ['resnet34', 'efficientnet_b2', 'convnext_tiny']
            )

        if fixed_task:
            params['task'] = fixed_task
        else:
            params['task'] = trial.suggest_categorical(
                'task',
                ['multiclass', 'binary_ram', 'binary_rust']
            )

        # Fixed batch size or use recommended from config
        batch_config = get_batch_size_config(params['encoder_name'])
        batch_choice = batch_config['recommended']
        if fixed_batch_size:
            params['batch_size'] = fixed_batch_size
        else:
            params['batch_size'] = batch_choice

        # Hyperparameters from preliminary search space
        params.update({
            'lr': trial.suggest_float('lr', *HPOInitializer.UNIFIED_SEARCH_SPACE['lr_range'], log=True),
            'batch_size': batch_choice,
            'weight_decay': trial.suggest_float('weight_decay',
                                                *HPOInitializer.UNIFIED_SEARCH_SPACE['weight_decay_range'],
                                                log=True),
            'bottleneck_dropout': trial.suggest_float('bottleneck_dropout',
                                                      *HPOInitializer.UNIFIED_SEARCH_SPACE['bottleneck_dropout_range']),
            'decoder_dropout': trial.suggest_float('decoder_dropout',
                                                   *HPOInitializer.UNIFIED_SEARCH_SPACE['decoder_dropout_range']),
            'dice_weight': trial.suggest_float('dice_weight',
                                               *HPOInitializer.UNIFIED_SEARCH_SPACE['dice_weight_range']),
            'focal_alpha': trial.suggest_float('focal_alpha',
                                               *HPOInitializer.UNIFIED_SEARCH_SPACE['focal_alpha_range']),
            'focal_gamma': trial.suggest_float('focal_gamma',
                                               *HPOInitializer.UNIFIED_SEARCH_SPACE['focal_gamma_range']),
        })

        return params

    @staticmethod
    def get_datasets(task, train_genotypes=TRAIN_GENOTYPES, val_genotypes=VAL_GENOTYPES):
        """
        Initialize datasets for a specific task with genotype-based splits.

        Args:
            task: Task name ('multiclass', 'binary_ram', 'binary_rust')
            train_genotypes: List of genotypes for training
            val_genotypes: List of genotypes for validation

        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Map task names to dataset task parameter
        task_mapping = {
            "multiclass": "multiclass",
            "binary_ram": "ramularia",
            "binary_rust": "brownrust"
        }

        dataset_task = task_mapping.get(task, "multiclass")

        print(f"Initializing datasets for task: {task} -> {dataset_task}")

        train_dataset = BarleyLeafDataset(
            all_genotypes_dir=TRAIN_DATA_DIR,
            genotypes_list=train_genotypes,
            task=dataset_task,
            augmentations=get_augmentations(),
            standardize=True,
            exclude_invalid=True,
            calculate_weights=True
        )

        val_dataset = BarleyLeafDataset(
            all_genotypes_dir=VAL_DATA_DIR,
            genotypes_list=val_genotypes,
            task=dataset_task,
            augmentations=None,
            standardize=True,
            exclude_invalid=True,
            calculate_weights=False
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"Number of classes: {train_dataset.num_classes}")
        print(f"Class weights: {train_dataset.class_weights.tolist()}")

        return train_dataset, val_dataset

    @staticmethod
    def get_dataloaders(train_dataset, val_dataset, batch_size, num_workers=2):
        """
        Create dataloaders for training and validation.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of data loading workers

        Returns:
            tuple: (train_loader, val_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )

        # Compute normalization statistics
        train_dataset.mean, train_dataset.std = train_dataset.compute_mean_and_std(train_loader, 'imagenet')
        val_dataset.mean, val_dataset.std = val_dataset.compute_mean_and_std(val_loader, 'imagenet')

        return train_loader, val_loader

    @staticmethod
    def get_model(encoder_name, task,
                  bottleneck_dropout=0.0, decoder_dropout=0.0):
        """
        Initialize U-Net model with specific encoder and dropout configuration.

        Args:
            encoder_name: Backbone architecture
            task: Task type ('multiclass', 'binary_ram', 'binary_rust')
            bottleneck_dropout: Dropout rate in bottleneck
            decoder_dropout: Dropout rate in decoder

        Returns:
            FlexibleUNet model on appropriate device
        """
        # Map HPO task names to model task_type
        task_mapping = {
            "multiclass": "multiclass",
            "binary_ram": "binary",
            "binary_rust": "binary"
        }

        model_task = task_mapping[task]

        print(f"Initializing {encoder_name} for {task} -> {model_task}")

        if task == "multiclass":
            NUM_CLASSES = 3  # background, ramularia, rust
        else:
            NUM_CLASSES = 2  # binary case

        model = FlexibleUNet(
            encoder_name=encoder_name,
            num_classes=NUM_CLASSES,
            task_type=model_task,
            bottleneck_dropout_rate=bottleneck_dropout,
            decoder_dropout_rate=decoder_dropout
        ).to(DEVICE)

        # Wrap with DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))

        return model

    @staticmethod
    def get_loss_function(dice_weight=0.5, focal_alpha=0.25, focal_gamma=2.0, class_weights=None):
        """
        Initialize FocalDiceLoss with flexible parameters.

        Args:
            dice_weight: Weight for Dice loss component
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            class_weights: Class weights for handling imbalance

        Returns:
            FocalDiceLoss function
        """
        loss_fn = FocalDiceLoss(
            dice_weight=dice_weight,
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights.to(DEVICE) if class_weights is not None else None
        )

        return loss_fn

    @staticmethod
    def get_metrics_object(task):
        """
        Initialize segmentation metrics object for evaluation.

        Args:
            task: Task name

        Returns:
            SegmentationMetrics object
        """
        task_mapping = {
            "multiclass": "multiclass",
            "binary_ram": "binary",
            "binary_rust": "binary"
        }

        if task == "multiclass":
            NUM_CLASSES = 3  # background, ramularia, rust
        else:
            NUM_CLASSES = 2  # binary case

        return SegmentationMetrics(num_classes=NUM_CLASSES, task=task_mapping[task])

    @staticmethod
    def get_optimizer(model, lr, weight_decay):
        """
        Initialize AdamW optimizer.

        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient

        Returns:
            AdamW optimizer
        """
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    @staticmethod
    def get_scheduler(optimizer, patience=2):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule
            patience: Patience for reducing learning rate

        Returns:
            ReduceLROnPlateau scheduler
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience
        )


def setup_experiment_from_params(params):
    """
    Quick setup for complete experiment from HPO parameters.

    Args:
        params: Dictionary from HPOInitializer.suggest_parameters()

    Returns:
        dict: All components needed for training
    """
    initializer = HPOInitializer()

    # Get datasets
    train_dataset, val_dataset = initializer.get_datasets(params['task'])

    # Get dataloaders
    train_loader, val_loader = initializer.get_dataloaders(
        train_dataset, val_dataset, params['batch_size']
    )

    # Get model
    model = initializer.get_model(
        encoder_name=params['encoder_name'],
        task=params['task'],
        bottleneck_dropout=params['bottleneck_dropout'],
        decoder_dropout=params['decoder_dropout']
    )

    # Get loss function with HPO parameters
    loss_fn = initializer.get_loss_function(
        dice_weight=params['dice_weight'],
        focal_alpha=params['focal_alpha'],
        focal_gamma=params['focal_gamma'],
        class_weights=train_dataset.class_weights
    )

    # Get metrics
    metrics_obj = initializer.get_metrics_object(params['task'])

    # Get optimizer and scheduler
    optimizer = initializer.get_optimizer(
        model,
        params['lr'],
        params['weight_decay']
    )

    scheduler = initializer.get_scheduler(optimizer)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'model': model,
        'loss_fn': loss_fn,
        'metrics_obj': metrics_obj,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'params': params  # Keep original params for reference
    }


def create_pruning_study(study_name=None):
    """
    Create Optuna study with checkpointing and pruning.

    Args:
        study_name: Name for the study (optional)

    Returns:
        optuna.Study: Configured study object
    """
    storage = f"sqlite:///hpo_{study_name or 'default'}.db"

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=8,
            reduction_factor=2
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        storage=storage,
        load_if_exists=True  # Resume if study exists
    )

    return study