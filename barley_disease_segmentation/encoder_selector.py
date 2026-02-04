"""
Encoder selection and comparison for barley disease segmentation.

Analyzes hyperparameter optimization results to select the best encoder
architecture for each segmentation task (binary rust, binary ramularia, multiclass).
"""

import os
from pathlib import Path
import pandas as pd

from barley_disease_segmentation.config import PIPELINE_DIR
from barley_disease_segmentation.utils import find_best_hyperparameters, get_save_path

__all__ = [
    'find_best_trial_across_files',
    'analyze_encoder_performance',
    'compare_encoders_for_task',
    'create_encoder_selection_summary'
]

def find_best_trial_across_files(preliminary_csv, optimized_csv, encoder, task):
    """
    Find best trial from preliminary and optimized HPO CSV files.

    Args:
        preliminary_csv: Path to preliminary hyperparameter search results
        optimized_csv: Path to refined hyperparameter search results
        encoder: Encoder architecture name
        task: Task name ('Binary_rust', 'Binary_ram', 'Multiclass')

    Returns:
        tuple: (best_trial_row, best_value, source_file_name)
    """
    best_trial = None
    best_value = -float('inf')
    source_file = None

    files_to_check = []
    if os.path.exists(preliminary_csv):
        files_to_check.append((preliminary_csv, 'hpo_results'))
    if os.path.exists(optimized_csv):
        files_to_check.append((optimized_csv, 'hpo_optimized'))

    if not files_to_check:
        print(f"  No HPO files found for {encoder} on {task}")
        return None, None, None

    for file_path, source in files_to_check:
        try:
            df = pd.read_csv(file_path)
            completed_trials = df[df['state'] == 'COMPLETE']

            if len(completed_trials) > 0:
                file_best_trial = completed_trials.nlargest(1, 'value').iloc[0]
                file_best_value = file_best_trial['value']

                if file_best_value > best_value:
                    best_value = file_best_value
                    best_trial = file_best_trial
                    source_file = source
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
            continue

    return best_trial, best_value, source_file


def analyze_encoder_performance(best_trial, encoder_name, task, source):
    """
    Extract performance metrics from best trial data.

    Args:
        best_trial: DataFrame row of best trial
        encoder_name: Name of encoder architecture
        task: Segmentation task
        source: Source file containing the trial

    Returns:
        dict: Performance metrics dictionary
    """
    if best_trial is None:
        return None

    metrics = {
        'encoder_name': encoder_name,
        'task': task,
        'best_dice': best_trial['value'],
        'source_file': source,
        'final_train_loss': best_trial.get('final_train_loss', None),
        'final_learning_rate': best_trial.get('final_learning_rate', None),
        'epochs_completed': best_trial.get('epochs_completed', None),
        'duration': best_trial.get('duration', None),
    }

    return metrics


def compare_encoders_for_task(task, encoders):
    """
    Compare all encoders for a specific segmentation task.

    Args:
        task: Task name ('Binary_rust', 'Binary_ram', 'Multiclass')
        encoders: List of encoder architectures to compare

    Returns:
        tuple: (best_encoder_row, comparison_df, json_files_created)
    """
    print(f"\nCOMPARING ENCODERS FOR TASK: {task}")

    all_metrics = []
    json_files_created = []

    for encoder in encoders:
        try:
            path_dir = get_save_path(encoder, task, base_dir_type="hpo_data", subfolder="HPO")
            preliminary_csv = path_dir / f"hpo_results_{encoder}_{task}.csv"
            optimized_csv = path_dir / f"hpo_optimized_{encoder}_{task}.csv"

            print(f"\nProcessing {encoder} on {task}:")
            print(f"  Looking for files:")
            print(f"  - {preliminary_csv}")
            print(f"  - {optimized_csv}")

            best_trial, best_value, source = find_best_trial_across_files(
                preliminary_csv, optimized_csv, encoder, task
            )

            if best_trial is None:
                print(f"  No valid trials found for {encoder} on {task}")
                continue

            files_for_hpo = []
            if os.path.exists(preliminary_csv):
                files_for_hpo.append(preliminary_csv)
            if os.path.exists(optimized_csv):
                files_for_hpo.append(optimized_csv)

            if not files_for_hpo:
                continue

            json_output_path = path_dir / f"best_parameters_{encoder}_{task}.json"
            best_params, best_overall = find_best_hyperparameters(
                files_for_hpo[0],
                files_for_hpo[1] if len(files_for_hpo) > 1 else None,
                encoder,
                task,
                json_output_path
            )

            json_files_created.append(json_output_path)
            metrics = analyze_encoder_performance(best_trial, encoder, task, source)

            if metrics:
                metrics['best_lr'] = best_params.get('lr', None)
                metrics['best_weight_decay'] = best_params.get('weight_decay', None)
                metrics['best_decoder_dropout'] = best_params.get('decoder_dropout', None)
                metrics['source'] = best_params.get('source', source)
                all_metrics.append(metrics)

                print(f"  Found best trial in {source} file")
                print(f"    - Best Dice: {metrics['best_dice']:.4f}")
                if metrics['duration'] and pd.notna(metrics['duration']):
                    print(f"    - Duration: {metrics['duration']:.1f}")
                if pd.notna(metrics['final_train_loss']):
                    print(f"    - Train Loss: {metrics['final_train_loss']:.4f}")
                print(f"    - Parameters saved to JSON")

        except Exception as e:
            print(f"  Error processing {encoder} on {task}: {e}")
            continue

    if not all_metrics:
        print("No valid encoder data found for this task")
        return None

    df_comparison = pd.DataFrame(all_metrics)

    print(f"\nCOMPARISON RESULTS FOR {task}")
    print(f"{'Encoder':<15} {'Best Dice':<10} {'Source File':<12} {'Duration':<10} {'Train Loss':<12}")

    for _, row in df_comparison.iterrows():
        if row['duration'] and pd.notna(row['duration']):
            try:
                duration_float = float(row['duration'])
                duration_str = f"{duration_float:.1f}"
            except (ValueError, TypeError):
                duration_str = str(row['duration'])
        else:
            duration_str = "N/A"
        train_loss_str = f"{row['final_train_loss']:.4f}" if pd.notna(row['final_train_loss']) else "N/A"

        print(f"{row['encoder_name']:<15} {row['best_dice']:<10.4f} {row['source_file']:<12} "
              f"{duration_str:<10} {train_loss_str:<12}")

    best_encoder = df_comparison.loc[df_comparison['best_dice'].idxmax()]

    print(f"\n BEST ENCODER: {best_encoder['encoder_name']}")
    print(f"   Dice Score: {best_encoder['best_dice']:.4f}")
    print(f"  Source File: {best_encoder['source_file']}")
    if best_encoder['duration']:
        print(f"   Duration: {best_encoder['duration']:}")
    if pd.notna(best_encoder['final_train_loss']):
        print(f"   Train Loss: {best_encoder['final_train_loss']:.4f}")

    return best_encoder, df_comparison, json_files_created


def create_encoder_selection_summary(tasks, encoders,
                                     best_encoders_csv="best_encoders_summary.csv",
                                     all_trials_csv="all_encoders_trials_summary.csv"):
    """
    Create summary CSV files comparing encoder performance across all tasks.

    Args:
        tasks: List of task names to analyze
        encoders: List of encoder architectures to compare
        best_encoders_csv: Filename for best encoder summary
        all_trials_csv: Filename for all trials summary

    Returns:
        tuple: (df_best_encoders, df_all_trials)
    """
    best_encoder_results = []
    all_trials_results = []
    all_json_files = []

    summary_dir = Path(PIPELINE_DIR / "encoder_selection_results")
    summary_dir.mkdir(exist_ok=True)

    for task in tasks:
        print(f"PROCESSING TASK: {task}")

        best_encoder_data, all_encoders_data, json_files = compare_encoders_for_task(task, encoders)

        if best_encoder_data is not None:
            best_encoder_result = {
                'task': task,
                'best_encoder': best_encoder_data['encoder_name'],
                'best_dice_score': best_encoder_data['best_dice'],
                'source_file': best_encoder_data['source_file'],
                'duration': best_encoder_data.get('duration', None),
                'final_train_loss': best_encoder_data.get('final_train_loss', None),
                'best_lr': best_encoder_data.get('best_lr', None),
                'best_weight_decay': best_encoder_data.get('best_weight_decay', None),
                'best_decoder_dropout': best_encoder_data.get('best_decoder_dropout', None)
            }
            best_encoder_results.append(best_encoder_result)

            if all_encoders_data is not None:
                for _, row in all_encoders_data.iterrows():
                    trial_result = {
                        'task': task,
                        'encoder': row['encoder_name'],
                        'best_dice': row['best_dice'],
                        'source_file': row['source_file'],
                        'duration': row.get('duration', None),
                        'final_train_loss': row.get('final_train_loss', None),
                        'final_learning_rate': row.get('final_learning_rate', None),
                        'epochs_completed': row.get('epochs_completed', None),
                        'best_lr': row.get('best_lr', None),
                        'best_weight_decay': row.get('best_weight_decay', None),
                        'best_decoder_dropout': row.get('best_decoder_dropout', None),
                        'is_best_for_task': (row['encoder_name'] == best_encoder_data['encoder_name'])
                    }
                    all_trials_results.append(trial_result)

            all_json_files.extend(json_files)

    df_best_encoders = pd.DataFrame(best_encoder_results)
    df_all_trials = pd.DataFrame(all_trials_results)

    if not df_best_encoders.empty:
        best_encoders_path = summary_dir / best_encoders_csv
        df_best_encoders.to_csv(best_encoders_path, index=False)
        print(f"\nBest encoders summary saved to: {best_encoders_path}")

        print(f"\nFINAL BEST ENCODER SELECTION SUMMARY")
        print(f"{'Task':<15} {'Best Encoder':<15} {'Dice Score':<10} {'Source File':<12} {'Duration':<10}")

        for _, row in df_best_encoders.iterrows():
            duration_str = f"{row['duration']}" if pd.notna(row['duration']) else "N/A"
            print(f"{row['task']:<15} {row['best_encoder']:<15} {row['best_dice_score']:<10.4f} "
                  f"{row['source_file']:<12} {duration_str:<10}")

    if not df_all_trials.empty:
        all_trials_path = summary_dir / all_trials_csv
        df_all_trials.to_csv(all_trials_path, index=False)
        print(f"\nAll encoders trials summary saved to: {all_trials_path}")

        print(f"\nALL ENCODERS TRIALS SUMMARY")
        print(f"{'Task':<15} {'Encoder':<15} {'Dice':<10} {'Best for Task':<12} {'Source':<12}")

        df_sorted = df_all_trials.sort_values(['task', 'best_dice'], ascending=[True, False])
        for _, row in df_sorted.iterrows():
            best_marker = "✓" if row['is_best_for_task'] else ""
            print(f"{row['task']:<15} {row['encoder']:<15} {row['best_dice']:<10.4f} "
                  f"{best_marker:<12} {row['source_file']:<12}")

    return df_best_encoders, df_all_trials