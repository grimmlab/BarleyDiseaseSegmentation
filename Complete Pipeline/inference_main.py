"""
Main script for running inference with final retrained models.
"""

import argparse
from pathlib import Path
from barley_disease_segmentation.pipeline_inference import FinalRetrainingPipeline
from barley_disease_segmentation.utils import set_mlflow_connection, get_save_path
from barley_disease_segmentation.common import set_seed
import os
import json
from barley_disease_segmentation.evaluator import SingleModelEvaluator


def main():
    """Main entry point for inference script."""
    set_seed()
    # set_mlflow_connection()
    parser = argparse.ArgumentParser(description='Run inference for segmentation models')
    parser.add_argument('--encoder', type=str, required=True,
                        choices=['resnet34', 'resnet101', 'efficientnet_b2', 'convnext_tiny'],
                        help='Encoder architecture to optimize')
    parser.add_argument('--task', type=str, required=True,
                        choices=['multiclass', 'binary_ram', 'binary_rust'],
                        help='Task to optimize for')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch', type=int,
                        help='Batch size to use')
    parser.add_argument('--experiment_name', type=str,  # CHANGED: was action='store_true'
                        help='Name of the experiment on MLFlow')

    args = parser.parse_args()

    param_dir = get_save_path(args.encoder, args.task, base_dir_type="hpo_data", subfolder="HPO")
    # Load the best hyperparameters
    with open(f"{param_dir}/best_parameters_{args.encoder}_{args.task}.json", 'r') as f:
        best_hparams = json.load(f)

    print(f"Optimized hyperparameters: {best_hparams}")

    print("STARTING FINAL RETRAINING ")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Write permissions in current dir: {os.access('.', os.W_OK)}")

    pipeline = FinalRetrainingPipeline(
        best_hparams=best_hparams,
        task_name=best_hparams['task']
    )

    if args.experiment_name:
        pipeline.mlflow_run = True
        pipeline.mlflow_experiment_name = args.experiment_name

    try:
        print("\nSTEP 1: RETRAINING")
        model_path = pipeline.retrain_final_model(epochs=args.epochs)

        # Run standard evaluation
        utils_dir = pipeline.evaluate_on_test_set(model_path)

        print("INFERENCE AND EVALUATION COMPLETED SUCCESSFULLY")

        # If you want to see what was generated:
        if utils_dir and utils_dir.exists():
            print(f"\nGenerated outputs in: {utils_dir}")
            # List generated files
            for item in utils_dir.iterdir():
                if item.is_dir():
                    print(f" {item.name}/")
                else:
                    print(f"  {item.name}")

    except Exception as e:
        print(f" Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()