"""
Utility functions for barley leaf disease segmentation research.
Supports binary and multiclass segmentation of brown rust and ramularia leaf spot.
"""

import random
import numpy as np
import torch
import albumentations as A
import mlflow

__all__ = [
    'get_augmentations',
    'get_batch_size_config',
    'set_seed',
    'extract_sample_metadata',
    'set_initial_mlflow_params',
    'set_initial_mlflow_params_eval'
]

def get_augmentations():
    """
    Image augmentations for 512x512 barley leaf scans.

    Returns:
        A.Compose: Augmentation pipeline with flips, rotations, and random crops.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.Transpose(p=0.5),
        A.RandomSizedCrop(
            min_max_height=(384, 512),
            height=512,
            width=512,
            p=0.25
        )
    ])


def get_batch_size_config(encoder_name):
    """
    Batch size recommendations for different encoders (512x512 images).

    Args:
        encoder_name: Encoder architecture name.

    Returns:
        dict: Batch size configuration for the given encoder.
    """
    base_config = {
        "resnet101": {
            "safe_range": [32, 48, 64],
            "recommended": 48,
            "tested_max": 64,
            "oom_threshold": 192
        },
        "convnext_tiny": {
            "safe_range": [48, 64, 80],
            "recommended": 64,
            "tested_max": 80,
            "oom_threshold": 160
        },
        "resnet34": {
            "safe_range": [64, 96, 128],
            "recommended": 96,
            "tested_max": 128,
            "oom_threshold": 256
        },
        "resnet50": {
            "safe_range": [48, 64, 96],
            "recommended": 64,
            "tested_max": 96,
            "oom_threshold": 192
        },
        "efficientnet_b2": {
            "safe_range": [96, 128, 192],
            "recommended": 128,
            "tested_max": 192,
            "oom_threshold": 384
        },
        "mobilenetv3_large_100": {
            "safe_range": [96, 128, 192],
            "recommended": 128,
            "tested_max": 192,
            "oom_threshold": 384
        }
    }
    return base_config.get(encoder_name, base_config["resnet34"])


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    print(f"Random seed set to {seed}")


def extract_sample_metadata(batch_metadata, sample_index):
    """
    Extract single sample metadata from batched dictionary.

    Args:
        batch_metadata: Batched metadata from dataloader.
        sample_index: Index of sample to extract.

    Returns:
        dict: Metadata for the specified sample.
    """
    sample_metadata = {}

    for key, value in batch_metadata.items():
        if torch.is_tensor(value):
            sample_value = value[sample_index]
            if sample_value.numel() == 1:
                sample_value = sample_value.item()
            sample_metadata[key] = sample_value
        elif isinstance(value, list):
            sample_metadata[key] = value[sample_index]
        else:
            sample_metadata[key] = value

    return sample_metadata


def set_initial_mlflow_params(trial, trial_id, params, run_id):
    """
    Configure MLflow for HPO trials.

    Args:
        trial: Optuna trial object.
        trial_id: Trial identifier.
        params: Hyperparameter dictionary.
        run_id: Parent MLflow run ID.
    """
    mlflow.set_tag("hpo_trial", "true")
    mlflow.set_tag("trial_number", trial.number)
    mlflow.set_tag("encoder", params['encoder_name'])
    mlflow.set_tag("task", params['task'])
    mlflow.set_tag("parent_run_id", run_id)
    mlflow.log_params(params)
    mlflow.log_param("trial_id", trial_id)
    mlflow.log_param("encoder_name", params['encoder_name'])
    mlflow.log_param("task", params['task'])


def set_initial_mlflow_params_eval(trial_id, params, run_id):
    """
    Configure MLflow for evaluation runs.

    Args:
        trial_id: Evaluation run identifier.
        params: Model parameter dictionary.
        run_id: Parent MLflow run ID.
    """
    mlflow.set_tag("encoder", params['encoder_name'])
    mlflow.set_tag("task", params['task'])
    mlflow.set_tag("parent_run_id", run_id)
    mlflow.log_params(params)
    mlflow.log_param("trial_id", trial_id)
    mlflow.log_param("encoder_name", params['encoder_name'])
    mlflow.log_param("task", params['task'])