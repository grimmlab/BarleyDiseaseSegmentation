"""
Quick start inference script for barley disease segmentation models.
Runs evaluation on test data with optional leaf-level analysis.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from barley_disease_segmentation.config import *
import argparse
from barley_disease_segmentation.pipeline_inference import FinalRetrainingPipeline
from barley_disease_segmentation.utils import get_save_path
from barley_disease_segmentation.common import set_seed
import json
from barley_disease_segmentation.evaluator import SingleModelEvaluator


def main():
    """
    Run inference with trained segmentation models.

    Loads best hyperparameters, evaluates model on test set,
    optionally runs leaf-level evaluation for detailed analysis.

    Example usage:
        python Inference_quick_start.py --encoder convnext_tiny \
          --task multiclass \
          --test_data_path Test_sample_data \
          --run_leaf_evaluation \
          --leaf_evaluation_output results
    """
    set_seed()
    parser = argparse.ArgumentParser(description='Run inference for segmentation models')
    parser.add_argument('--encoder', type=str, default='convnext_tiny',
                        choices=['resnet34', 'resnet101', 'efficientnet_b2', 'convnext_tiny'],
                        help='Encoder architecture to optimize')
    parser.add_argument('--task', type=str, required=True,
                        choices=['multiclass', 'binary_ram', 'binary_rust'],
                        help='Task to optimize for')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch', type=int,
                        help='Batch size to use')
    parser.add_argument('--mlflow_experiment_name', action='store_true',
                        help='Name of the experiment on MLFlow')
    parser.add_argument('--model_path', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data_path', type=str,
                        help='Path to test data')
    parser.add_argument('--run_leaf_evaluation', action='store_true',
                        help='Run additional leaf-level evaluation')
    parser.add_argument('--leaf_evaluation_output', type=str, default="leaf_evaluation",
                        help='Output directory for leaf-level evaluation')

    args = parser.parse_args()

    param_dir = get_save_path(args.encoder, args.task, base_dir_type="hpo_data", subfolder="HPO")
    # Load the best hyperparameters
    print(param_dir)
    with open(f"{param_dir}/best_parameters_{args.encoder}_{args.task}.json", 'r') as f:
        best_hparams = json.load(f)

    pipeline = FinalRetrainingPipeline(
        best_hparams=best_hparams,
        task_name=best_hparams['task']
    )
    print('Pipeline initialised')
    pipeline.TEST_DATA_DIR = PROJECT_ROOT / args.test_data_path

    if args.mlflow_experiment_name:
        pipeline.mlflow_run = True
        pipeline.mlflow_experiment_name = args.mlflow_experiment_name

    model_dir = get_save_path(args.encoder, args.task, base_dir_type="utils", subfolder="final_models")
    if args.model_path is not None:
        model_path = args.model_path
    else:
        if args.task == 'multiclass':
            model_path = model_dir / 'final_model_multiclass_checkpoint_55.pth'
        elif args.task == 'binary_rust':
            model_path = model_dir / 'final_model_binary_rust_checkpoint_55.pth'
        else:
            model_path = model_dir / 'final_model_binary_ram_checkpoint_175.pth'

    # Run standard evaluation
    utils_dir = pipeline.evaluate_on_test_set(model_path)

    # Run additional leaf-level evaluation if requested
    if args.run_leaf_evaluation:
        print("RUNNING ADDITIONAL LEAF-LEVEL EVALUATION")


        # Determine predictions path
        predictions_path = utils_dir/'saved_predictions'

        # Configure based on task type
        if args.task == 'multiclass':
            # Evaluate both diseases for multiclass
            for disease_name, disease_class in [("Brown Rust", 1), ("Ramularia", 2)]:
                print(f"\nEvaluating multiclass model on {disease_name}...")
                detailed_df = SingleModelEvaluator.run_leaf_level_evaluation(
                    predictions_path=predictions_path,
                    model_name=f"{args.encoder}_{args.task}",
                    task_type="multiclass",
                    disease_name=disease_name,
                    disease_class=disease_class,
                    output_dir=Path(args.leaf_evaluation_output) / disease_name.replace(" ", "_")
                )
        else:
            # Binary task
            disease_name = "Brown Rust" if args.task == 'binary_rust' else "Ramularia"
            print(f"\nEvaluating binary model on {disease_name}...")
            detailed_df =SingleModelEvaluator.run_leaf_level_evaluation(
                predictions_path=predictions_path,
                model_name=f"{args.encoder}_{args.task}",
                task_type="binary",
                disease_name=disease_name,
                disease_class=None,
                output_dir=Path(args.leaf_evaluation_output)
            )


if __name__ == "__main__":
    main()

'''
python Inference_quick_start.py --encoder convnext_tiny \
  --task multiclass \
  --test_data_path Test_sample_data \
  --run_leaf_evaluation \
  --leaf_evaluation_output quick_start_results
'''