"""
Model comparison analysis for barley disease segmentation with per-class metrics.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from skimage import measure
import json
from collections import defaultdict
import argparse
import torch
from scipy.optimize import linear_sum_assignment
import warnings
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from barley_disease_segmentation.utils import get_save_path
from barley_disease_segmentation.config import *
from barley_disease_segmentation.evaluator import *


def main():
    """Main entry point for model comparison analysis."""

    def find_common_leaf_ids(model_paths):
        """Find leaf IDs common across all model predictions"""
        all_leaf_sets = []

        for model_path in model_paths:
            labels_path = model_path / "labels"
            if labels_path.exists():
                leaf_ids = set(f.stem for f in labels_path.glob("*.png"))
                all_leaf_sets.append(leaf_ids)
            else:
                print(f"Warning: No labels found in {model_path}")

        if not all_leaf_sets:
            raise ValueError("No valid model paths found")

        common_leaf_ids = set.intersection(*all_leaf_sets)
        print(f"Common leaves across all models: {len(common_leaf_ids)}")

        return sorted(common_leaf_ids)

    parser = argparse.ArgumentParser(description='Model comparison for per-class metrics with F1 threshold analysis')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--min_presence_pixels', type=int, default=10,
                        help='Minimum GT pixels to count as disease presence')
    parser.add_argument('--binary_rust_path', type=str, default=None, help='Path to binary rust predictions')
    parser.add_argument('--binary_ram_path', type=str, default=None, help='Path to binary ram predictions')
    parser.add_argument('--multiclass_path', type=str, default=None, help='Path to multiclass predictions')
    parser.add_argument('--output', type=str, default="Evaluation_metrics", help='Output folder')
    args = parser.parse_args()

    # Set default paths if not provided
    if args.binary_rust_path is None:
        binary_rust_path = get_save_path('convnext_tiny', 'binary_rust', base_dir_type='utils',
                                         subfolder='20251117_1014/saved_predictions')
    else:
        binary_rust_path = args.binary_rust_path

    if args.binary_ram_path is None:
        binary_ram_path = get_save_path('convnext_tiny', 'binary_ram', base_dir_type='utils',
                                        subfolder='20251117_1002/saved_predictions')
    else:
        binary_ram_path = args.binary_ram_path

    if args.multiclass_path is None:
        multiclass_path = get_save_path('convnext_tiny', 'multiclass', 'utils',
                                        subfolder='20251117_1028/saved_predictions')
    else:
        multiclass_path = args.multiclass_path

    # Configure model comparisons
    model_configs = {
        'binary_rust': {
            'predictions_path': Path(binary_rust_path),
            'model_name': 'Binary',
            'task_type': 'binary',
            'disease_name': 'Brown Rust'
        },
        'binary_ramularia': {
            'predictions_path': Path(binary_ram_path),
            'model_name': 'Binary',
            'task_type': 'binary',
            'disease_name': 'Ramularia'
        },
        'multiclass_rust': {
            'predictions_path': Path(multiclass_path),
            'model_name': 'Multiclass',
            'task_type': 'multiclass',
            'disease_class': 1,
            'disease_name': 'Brown Rust'
        },
        'multiclass_ramularia': {
            'predictions_path': Path(multiclass_path),
            'model_name': 'Multiclass',
            'task_type': 'multiclass',
            'disease_class': 2,
            'disease_name': 'Ramularia'
        }
    }

    # Find common leaf IDs
    print("Finding common leaves across all models...")
    model_paths = [config['predictions_path'] for config in model_configs.values()]
    common_leaf_ids = find_common_leaf_ids(model_paths=model_paths)

    if len(common_leaf_ids) == 0:
        print("ERROR: No common leaves found across all models!")
        return

    # Get leaf masks path
    leaf_masks_path = get_save_path('convnext_tiny', 'multiclass', 'utils',
                                    subfolder='20251117_1028/saved_predictions')

    # Initialize evaluator
    evaluator = MultiModelEvaluator(device=args.device,
                                    min_presence_pixels=args.min_presence_pixels,
                                    leaf_masks_path=Path(leaf_masks_path))

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run F1 threshold analysis
    evaluator.run_f1_threshold_analysis(model_configs, common_leaf_ids, output_path)

    # Run main evaluation
    evaluator.run_comprehensive_comparison(model_configs, common_leaf_ids)

    # Generate and display results
    comparison_df = evaluator.generate_comparison_table()
    print("\n" + "=" * 100)
    print("MODEL COMPARISON TABLE")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("\n")

    # Save results
    evaluator.save_results(output_path)


if __name__ == "__main__":
    main()