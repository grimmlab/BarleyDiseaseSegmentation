#!/usr/bin/env python3
"""
Complete ML Pipeline for lesion segmentation tasks.
Simplified version focusing on encoder extraction and conditional HPO.
"""

import argparse
import subprocess
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from barley_disease_segmentation.config import TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, PIPELINE_DIR


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')

    # Required arguments
    parser.add_argument('--task', required=True,
                        choices=['binary_ram', 'binary_rust', 'multiclass'],
                        help='Task to run')

    # HPO options
    parser.add_argument('--skip-hpo', action='store_true',
                        help='Skip HPO and use existing files')
    parser.add_argument('--trials', type=int, default=60,
                        help='Number of HPO trials (default: 60 for broad, 30 for refined)')

    # MLflow options
    parser.add_argument('--experiment_name', type=str,
                        help='MLflow experiment name')

    # Other options
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs for retraining (default: 300)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')

    return parser.parse_args()


def check_data_requirements():
    """Simple data check"""
    print("=" * 60)
    print("CHECKING BASIC DATA REQUIREMENTS")
    print("=" * 60)


    try:
        dirs_to_check = [
            ("Training data", TRAIN_DATA_DIR),
            ("Validation data", VAL_DATA_DIR),
            ("Test data", TEST_DATA_DIR)
        ]

        all_ok = True
        for name, dir_path in dirs_to_check:
            if dir_path and isinstance(dir_path, Path) and dir_path.exists():
                file_count = len(list(dir_path.rglob("*.tif"))) + len(list(dir_path.rglob("*.png"))) + len(
                    list(dir_path.rglob("*.jpg")))
                print(f"{name}: {dir_path} ({file_count} image files)")
            else:
                print(f"{name}: Missing or invalid path: {dir_path}")
                all_ok = False

        if all_ok:
            print("\nBasic data requirements satisfied!")
        else:
            print("\n Missing data directories!")
            return False

        return True

    except ImportError as e:
        print(f"Could not import config: {e}")
        print("Assuming data directories exist...")
        return True
    except Exception as e:
        print(f" Error checking data: {e}")
        return True


def run_hpo_phase(args, phase):
    """Run HPO for a specific phase (broad or refined)."""
    if phase == "broad":
        trials = args.trials  # Use the provided trials (default 60)
        refined_flag = ""
        phase_name = "BROAD HPO"
    else:  # refined
        trials = args.trials // 2  # Half the trials for refined (default 30)
        refined_flag = "--HPO_refined"
        phase_name = "REFINED HPO"

    print(f"RUNNING {phase_name} ({trials} trials)")


    # Build the command
    cmd_parts = [
        "python3", "main_hpo_script.py",
        "--task", args.task,
        "--trials", str(trials)
    ]

    if refined_flag:
        cmd_parts.append(refined_flag)

    if args.experiment_name:
        cmd_parts.extend(["--experiment_name", args.experiment_name])

    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")

    if args.dry_run:
        print("(Dry run - would execute)")
        return True

    # Execute the command
    try:
        result = subprocess.run(cmd_parts, capture_output=True, text=True)

        if result.returncode == 0:
            print(f" {phase_name} completed successfully")
            if result.stdout and len(result.stdout) > 0:
                # Print last few informative lines
                lines = [l for l in result.stdout.strip().split('\n') if l]
                if lines:
                    print("\nLast output:")
                    for line in lines[-5:]:
                        print(f"  {line}")
            return True
        else:
            print(f" {phase_name} failed with exit code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])  # First 500 chars
            return False

    except Exception as e:
        print(f" Exception during {phase_name}: {str(e)}")
        return False


def extract_best_encoder(args):
    """
    Extract the best encoder from the results file.
    Simple version knowing the exact file structure.
    """
    print("EXTRACTING BEST ENCODER")

    results_file = Path(f"{PIPELINE_DIR}/encoder_selection_results/best_encoders_summary_{args.task}.csv")

    if not results_file.exists():
        print(f" Results file not found: {results_file}")
        return None

    try:
        # Read the CSV with known structure
        df = pd.read_csv(results_file)

        if df.empty:
            print("Results file is empty!")
            return None

        # Get the row for our task (there might be multiple tasks in the file)
        task_row = df[df['task'] == args.task]

        if task_row.empty:
            print(f" No results found for task: {args.task}")
            print("Available tasks:", df['task'].unique())
            return None

        # Extract best encoder and score
        best_encoder = task_row.iloc[0]['best_encoder']
        best_score = task_row.iloc[0]['best_dice_score']
        source = task_row.iloc[0]['source_file']

        print(f"Results from: {results_file}")
        print(f"Best encoder: {best_encoder}")
        print(f"Dice score: {best_score:.4f}")
        print(f"Source: {source}")

        # Print all parameters for reference
        print("\nHyperparameters:")
        print(f"   Learning rate: {task_row.iloc[0]['best_lr']:.6f}")
        print(f"   Weight decay: {task_row.iloc[0]['best_weight_decay']:.6f}")
        print(f"   Decoder dropout: {task_row.iloc[0]['best_decoder_dropout']:.4f}")

        return best_encoder

    except Exception as e:
        print(f" Error reading results file: {str(e)}")
        # Show file preview for debugging
        try:
            with open(results_file, 'r') as f:
                content = f.read()
                print(f"File preview (first 200 chars):\n{content[:200]}")
        except:
            pass
        return None


def run_encoder_selection(args):
    """Run encoder selection script."""
    print("RUNNING ENCODER SELECTION")


    cmd_parts = ["python", "Encoder_selection.py", "--task", args.task]

    if args.experiment_name:
        cmd_parts.extend(["--experiment-name", args.experiment_name])

    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")

    if args.dry_run:
        print("(Dry run - would execute)")
        return True

    result = subprocess.run(cmd_parts, capture_output=True, text=True)

    if result.returncode == 0:
        print("Encoder selection completed")
        if result.stdout:
            # Show any relevant output
            for line in result.stdout.strip().split('\n'):
                if any(keyword in line.lower() for keyword in ['encoder', 'dice', 'best', 'saved']):
                    print(f"  {line}")
        return True
    else:
        print(f" Encoder selection failed: {result.stderr}")
        return False


def run_inference(args, best_encoder):
    """Run inference with the best encoder."""

    print(f"RUNNING INFERENCE WITH: {best_encoder}")


    cmd_parts = [
        "python3", "inference_main.py",
        "--task", args.task,
        "--encoder", best_encoder,
        "--epochs", str(args.epochs)
    ]

    if args.experiment_name:
        cmd_parts.extend(["--experiment_name", args.experiment_name])

    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")

    if args.dry_run:
        print("(Dry run - would execute)")
        return True

    # Run with real-time output
    process = subprocess.Popen(
        cmd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    print("\n" + "-" * 40)
    print("Inference output:")
    print("-" * 40)

    # Stream output
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode == 0:
        print("\n" + "-" * 40)
        print(" Inference completed")
        return True
    else:
        print(f"\n Inference failed with code {process.returncode}")
        return False


def main():
    """Main pipeline function."""
    args = parse_arguments()

    start_time = time.time()

    print(f"STARTING PIPELINE FOR: {args.task}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    # Step 1: Quick data check
    check_data_requirements()

    # Step 2: HPO (if not skipped)
    if not args.skip_hpo:
        print(f"\n RUNNING HPO ({args.trials} broad + {max(30, args.trials // 2)} refined trials)")

        # Run broad HPO
        if not run_hpo_phase(args, "broad"):
            print(" Stopping - HPO failed")
            sys.exit(1)

        # Run refined HPO
        if not run_hpo_phase(args, "refined"):
            print(" Stopping - Refined HPO failed")
            sys.exit(1)
    else:
        print(f"\n SKIPPING HPO (using existing files)")

    # Step 3: Encoder selection
    if not run_encoder_selection(args):
        print(" Stopping - Encoder selection failed")
        sys.exit(1)

    # Step 4: Extract best encoder
    best_encoder = extract_best_encoder(args)
    if not best_encoder:
        print(" Stopping - Could not extract best encoder")
        sys.exit(1)

    # Step 5: Run inference
    inference_success = run_inference(args, best_encoder)

    # Final summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(" PIPELINE SUMMARY")
    print(f"Task: {args.task}")
    print(f"Best encoder: {best_encoder}")
    print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print(f"HPO skipped: {args.skip_hpo}")
    print(f"Dry run: {args.dry_run}")
    print(f"Final status: {' SUCCESS' if inference_success else ' FAILED'}")

    sys.exit(0 if inference_success else 1)


if __name__ == "__main__":
    main()