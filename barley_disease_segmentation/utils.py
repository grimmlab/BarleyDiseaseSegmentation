"""
Hyperparameter optimization utilities for barley disease segmentation.
"""

import optuna
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from .trainer import objective_single
import urllib3
import json
import random
import albumentations as A
import mlflow
from barley_disease_segmentation.config import *
from barley_disease_segmentation.common import get_batch_size_config

__all__ = [
    'set_mlflow_connection',
    'find_best_hyperparameters',
    'get_save_path',
    'run_single_hpo',
    'run_single_encoder_single_task',
    'save_study_results'
]

def set_mlflow_connection(experiment_name: str = "BarleyDiseaseSegmentation"):
    """
    Configure MLflow connection. EDIT THIS FUNCTION WITH YOUR CREDENTIALS.
    """
    try:
        # EDIT THESE VALUES
        MLFLOW_SERVER = "https://your-mlflow-server.com:5000"  # ← CHANGE ME
        USERNAME = "your-username"  # ← CHANGE ME
        PASSWORD = "your-password"  # ← CHANGE ME
        AWS_KEY = "your-aws-access-key"  # ← CHANGE ME (optional)
        AWS_SECRET = "your-aws-secret-key"  # ← CHANGE ME (optional)


        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_SERVER)

        # Add authentication if provided
        if USERNAME and PASSWORD:
            auth_uri = MLFLOW_SERVER.replace("://", f"://{USERNAME}:{PASSWORD}@")
            mlflow.set_tracking_uri(auth_uri)

        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Set AWS for S3
        if AWS_KEY and AWS_SECRET:
            os.environ["AWS_ACCESS_KEY_ID"] = AWS_KEY
            os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET

        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Test connection
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        print(f"MLflow connected to {MLFLOW_SERVER}")
        print(f"Experiment: {experiment_name}")

        return client

    except Exception as e:
        print(f"MLflow setup failed: {e}")
        raise


def find_best_hyperparameters(preliminary_csv, optimized_csv, encoder, task, output_json):

    """
    Find the best hyperparameters from either preliminary or optimized HPO runs
    and save them to a JSON file in the specified format.
    """

    # Read both CSV files
    df_preliminary = pd.read_csv(preliminary_csv)
    df_optimized = pd.read_csv(optimized_csv)

    # Filter completed trials
    df_preliminary_complete = df_preliminary[df_preliminary['state'] == 'COMPLETE']
    df_optimized_complete = df_optimized[df_optimized['state'] == 'COMPLETE']

    print(f"Preliminary trials: {len(df_preliminary_complete)} completed")
    print(f"Optimized trials: {len(df_optimized_complete)} completed")

    # Find best run from each
    best_preliminary = df_preliminary_complete.nlargest(1, 'value').iloc[0] if len(
        df_preliminary_complete) > 0 else None
    best_optimized = df_optimized_complete.nlargest(1, 'value').iloc[0] if len(df_optimized_complete) > 0 else None

    # Compare and select the overall best
    if best_preliminary is not None and best_optimized is not None:
        if best_preliminary['value'] >= best_optimized['value']:
            best_overall = best_preliminary
            best_source = "preliminary"
            best_value = best_preliminary['value']
        else:
            best_overall = best_optimized
            best_source = "optimized"
            best_value = best_optimized['value']
    elif best_preliminary is not None:
        best_overall = best_preliminary
        best_source = "preliminary"
        best_value = best_preliminary['value']
    else:
        best_overall = best_optimized
        best_source = "optimized"
        best_value = best_optimized['value']

    print(f"\nOverall best run from: {best_source}")
    print(f"Best Dice score: {best_value:.6f}")

    # Extract parameters based on which run was best
    if best_source == "preliminary":
        # Get all parameters from preliminary run
        lr = best_overall['params_lr']
        weight_decay = best_overall['params_weight_decay']
        decoder_dropout = best_overall['params_decoder_dropout']
        dice_weight = best_overall['params_dice_weight']
        focal_alpha = best_overall['params_focal_alpha']
        focal_gamma = best_overall['params_focal_gamma']
        bottleneck_dropout = best_overall['params_bottleneck_dropout']
    else:
        # Get optimized parameters + fixed parameters
        lr = best_overall['params_lr']
        weight_decay = best_overall['params_weight_decay']
        decoder_dropout = best_overall['params_decoder_dropout']
        dice_weight = 0.721
        focal_alpha = 0.737
        focal_gamma = 1.847
        bottleneck_dropout = 0.385

    # Handle batch size configuration
    batch_config = get_batch_size_config(encoder)
    batch_choice = batch_config['recommended']

    # Create the output structure
    best_params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'decoder_dropout': decoder_dropout,
        'dice_weight': dice_weight,
        'focal_alpha': focal_alpha,
        'focal_gamma': focal_gamma,
        'bottleneck_dropout': bottleneck_dropout,
        'batch_size': batch_choice,
        'best_dice_score': float(best_value),
        'source': best_source,
        'encoder_name': encoder,
        'task': task
    }

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest hyperparameters saved to: {output_json}")

    # Print summary
    print("\nBEST HYPERPARAMETERS ")
    print(f"Learning Rate: {lr:.2e}")
    print(f"Weight Decay: {weight_decay:.2e}")
    print(f"Decoder Dropout: {decoder_dropout:.3f}")
    print(f"Dice Weight: {dice_weight:.3f}")
    print(f"Focal Alpha: {focal_alpha:.3f}")
    print(f"Focal Gamma: {focal_gamma:.3f}")
    print(f"Bottleneck Dropout: {bottleneck_dropout:.3f}")

    return best_params, best_overall


def get_save_path(encoder, task, base_dir_type="results", subfolder="final_models"):
    """Get the appropriate save path based on task and encoder"""
    try:
        task_paths = {
            "multiclass": MULTICLASS_PATHS,
            "binary_ram": BINARY_RAM_PATHS,
            "binary_rust": BINARY_RUST_PATHS
        }.get(task.lower())

        if not task_paths:
            raise ValueError(f"Unknown task name: {task}")

        encoder_mapping = {
            'resnet34': 'Resnet',
            'resnet50': 'Resnet',
            'resnet101': 'Resnet',
            'convnext_tiny': 'Convnext',
            'convnext_small': 'Convnext',
            'convnext_base': 'Convnext',
            'efficientnet_b0': 'Efficientnet',
            'efficientnet_b1': 'Efficientnet',
            'efficientnet_b2': 'Efficientnet',
            'efficientnet_b3': 'Efficientnet',
            'efficientnet_b4': 'Efficientnet'
        }

        folder_name = encoder_mapping.get(encoder)
        base_dir = task_paths[base_dir_type][folder_name]
        full_path = Path(base_dir) / subfolder
        full_path.mkdir(parents=True, exist_ok=True)

        return full_path

    except Exception as e:
        print(f"ERROR in _get_save_path: {e}")
        fallback_path = Path("debug_results") / subfolder
        fallback_path.mkdir(parents=True, exist_ok=True)
        print(f"Using fallback path: {fallback_path}")
        return fallback_path



def run_single_hpo(encoder, task, n_trials=30, HPO_refined=False):
    """
    Run HPO for a single encoder-task combination
    """
    from .initialiser import create_pruning_study
    # Create study first
    study = create_pruning_study()

    def save_callback(study, trial):
        """Callback to save results after each trial completes"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Auto-saving results after trial {trial.number}...")
            save_study_results(study, encoder, task, mode='append')

    try:
        #mlflow.set_experiment(f"hpo_{encoder}_{task}")
        with mlflow.start_run(run_name=f"hpo_{encoder}_{task}") as main_run:
            # Only log CONSTANT parameters here
            mlflow.log_param("encoder", encoder)
            mlflow.log_param("task", task)
            mlflow.log_param("total_trials", n_trials)

            start_time = time.time()
            study.optimize(
                lambda trial: objective_single(trial, encoder, task, mlflow_run=main_run, HPO_refined=HPO_refined),
                n_trials=n_trials,
                callbacks=[save_callback]
            )
            end_time = time.time()

            print("HPO COMPLETED!")
            print(f"Total time: {(end_time - start_time) / 60:.1f} minutes")
            print(f"Total trials: {len(study.trials)}")
            print(f"Best trial value (Dice): {study.best_value:.4f}")
            print(f"Best parameters: {study.best_params}")

            # Log results
            _log_single_study_results(study, start_time, end_time, encoder, task)

    except Exception as e:
        print(f"MLflow failed: {e}, running without...")
        study = create_pruning_study()
        study.optimize(
            lambda trial: objective_single(trial, encoder, task, mlflow_run=None, HPO_refined=HPO_refined),
            n_trials=n_trials,
            callbacks=[save_callback]
        )

    # Final save to ensure everything is captured
    save_study_results(study, encoder, task, mode='append')
    print("HPO completed and final results saved!")
    return study



def _log_single_study_results(study, start_time, end_time, encoder, task):
    """
    Log results from a single HPO study to MLflow
    """
    total_time = (end_time - start_time) / 60  # Convert to minutes
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    # Log overall results
    mlflow.log_metric("best_dice", study.best_value)
    mlflow.log_metric("total_time_minutes", total_time)
    mlflow.log_metric("completed_trials", completed_trials)
    mlflow.log_metric("pruned_trials", pruned_trials)
    mlflow.log_metric("total_trials", len(study.trials))

    # Log best parameters
    for key, value in study.best_params.items():
        mlflow.log_param(f"best_{key}", value)

    # Log efficiency metrics
    mlflow.log_metric("trials_per_hour", completed_trials / (total_time / 60) if total_time > 0 else 0)
    mlflow.log_metric("pruning_rate", pruned_trials / len(study.trials) if len(study.trials) > 0 else 0)

    # Log study statistics
    if completed_trials > 0:
        all_scores = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metric("mean_dice", np.mean(all_scores))
        mlflow.log_metric("std_dice", np.std(all_scores))
        mlflow.log_metric("median_dice", np.median(all_scores))

    print(f"HPO completed for {encoder}-{task}")
    print(f"Best Dice: {study.best_value:.4f}")
    print(f"Time: {total_time:.1f} minutes")
    print(f"Trials: {completed_trials} completed, {pruned_trials} pruned")



def run_single_encoder_single_task(encoder, task, n_trials=30, HPO_refined=False):
    """
    Run HPO for one specific encoder-task combination
    Perfect for running overnight or on specific GPUs
    """
    print(f"Starting HPO: {encoder} - {task}")
    print(f"Trials: {n_trials}, Epochs per trial: 15")
    print(f"Estimated time: ~{(n_trials * 15 * 2) / 60:.1f} hours")  # Rough estimate

    study = run_single_hpo(encoder, task, n_trials, HPO_refined)

    print(f"\nCOMPLETED: {encoder} - {task}")
    print(f"Best Dice: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study


def save_study_results(study, encoder, task, mode='append', HPO_refined=False):
    """Save study results to CSV with trial metrics from user attributes"""
    # Get the basic study results
    results_df = study.trials_dataframe()

    # Extract additional metrics
    additional_metrics = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            metrics = {
                'final_train_loss': trial.user_attrs.get('final_train_loss', None),
                'final_learning_rate': trial.user_attrs.get('final_learning_rate', None),
                'epochs_completed': trial.user_attrs.get('epochs_completed', None)
            }
            additional_metrics.append(metrics)
        else:
            # For pruned/failed trials, add None values
            additional_metrics.append({
                'final_train_loss': None,
                'final_learning_rate': None,
                'epochs_completed': None
            })

    # Add additional metrics to dataframe
    metrics_df = pd.DataFrame(additional_metrics)
    results_df = pd.concat([results_df, metrics_df], axis=1)

    # Add encoder and task info
    results_df['encoder'] = encoder
    results_df['task'] = task

    if HPO_refined:
        filename = f"hpo_optimized_{encoder}_{task}.csv"
    else:
        filename = f"hpo_results_{encoder}_{task}.csv"

    if mode == 'append' and os.path.exists(filename):
        # Append to existing file
        try:
            existing_df = pd.read_csv(filename)

            # Create a mapping for duplicate trial numbers
            existing_trials = set(existing_df['number'])
            new_trials_renamed = []

            for _, trial_row in results_df.iterrows():
                trial_number = trial_row['number']
                if trial_number in existing_trials:
                    # Check if there are already suffixed versions
                    suffix = 2
                    new_number = f"{trial_number}_{suffix}"

                    # Increment suffix if {number}_2 already exists, etc.
                    while new_number in existing_trials or any(
                            f"{trial_number}_{s}" in existing_trials for s in range(2, suffix + 1)):
                        suffix += 1
                        new_number = f"{trial_number}_{suffix}"

                    # Create a copy of the trial row with modified number
                    modified_trial = trial_row.copy()
                    modified_trial['number'] = new_number
                    new_trials_renamed.append(modified_trial)
                else:
                    new_trials_renamed.append(trial_row)

            # Convert the list of trials back to a DataFrame
            results_df_renamed = pd.DataFrame(new_trials_renamed)

            # Append all renamed trials
            final_df = pd.concat([existing_df, results_df_renamed], ignore_index=True)
            final_df.to_csv(filename, index=False)

            # Count how many trials were renamed
            renamed_count = len([t for t in new_trials_renamed if '_' in str(t['number'])])
            print(
                f"Appended {len(results_df_renamed)} trials to {filename} ({renamed_count} renamed, total: {len(final_df)} trials)")

        except Exception as e:
            print(f"Failed to append to {filename}, creating new file: {e}")
            results_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
    else:
        # Create new file or overwrite
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    if HPO_refined:
        best_hparams_filename = f"best_hparams_optimized_{encoder}_{task}.json"
    else:
        best_hparams_filename = f"best_hparams_{encoder}_{task}.json"

    with open(best_hparams_filename, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best hyperparameters saved to {best_hparams_filename}")

    return filename