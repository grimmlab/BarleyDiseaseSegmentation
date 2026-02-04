#!/usr/bin/env python3
"""
Main script for running Hyperparameter Optimization
"""

import argparse
import sys
import os

# Add package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from barley_disease_segmentation.utils import run_single_encoder_single_task, save_study_results, set_mlflow_connection
from barley_disease_segmentation.common import set_seed


def main():
    """Main entry point for HPO script."""
    set_seed()

    parser = argparse.ArgumentParser(description='Run HPO for segmentation models')
    parser.add_argument('--encoder', type=str,
                        choices=['resnet34', 'resnet101', 'efficientnet_b2', 'convnext_tiny'],
                        help='Encoder architecture to optimize')
    parser.add_argument('--task', type=str, required=True,
                        choices=['multiclass', 'binary_ram', 'binary_rust'],
                        help='Task to optimize for')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of HPO trials to run')
    parser.add_argument('--test-setup', action='store_true',
                        help='Test the setup before running HPO')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of the experiment on MLFlow')
    parser.add_argument('--HPO_refined', action='store_true', help='HPO with refined  search space if added')

    args = parser.parse_args()

    if args.experiment_name:
        set_mlflow_connection(experiment_name=args.experiment_name)
    else:
        set_mlflow_connection()

    print(f"Starting HPO for {args.encoder} - {args.task}")
    print(f"Number of trials: {args.trials}")

    if args.encoder is not None:
    # Run HPO
        study = run_single_encoder_single_task(
            encoder=args.encoder,
            task=args.task,
            n_trials=args.trials,
            HPO_refined=args.HPO_refined
        )
        # Save results
        save_study_results(study=study, encoder=args.encoder, task=args.task, mode='append', HPO_refined=args.HPO_refined)

        print(f"\n HPO completed for {args.encoder} - {args.task}!")
        print(f"Best Dice: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
    else:
        for encoder in ['resnet34', 'efficientnet_b2', 'convnext_tiny']:
            study = run_single_encoder_single_task(
                encoder=encoder,
                task=args.task,
                n_trials=args.trials,
                HPO_refined=args.HPO_refined
            )
            # Save results
            save_study_results(study=study, encoder=encoder, task=args.task, mode='append',
                               HPO_refined=args.HPO_refined)

            print(f"\n HPO completed for {encoder} - {args.task}!")
            print(f"Best Dice: {study.best_value:.4f}")
            print(f"Best parameters: {study.best_params}")


if __name__ == "__main__":
    main()

#python3 main_hpo_script.py --task binary_rust --trials 60
#python3 main_hpo_script.py --task binary_rust --trials 30 --HPO_refined