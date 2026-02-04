#!/usr/bin/env python3
"""
Master script to run the complete figure and table reproduction pipeline.
Checks all dependencies and runs scripts in the correct order.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from barley_disease_segmentation.config import *
from barley_disease_segmentation.config import UTILS_DIR, FIGURE_DIR, EXPERIMENTS_DIR

# Required data directories
DATA_REQUIREMENTS = {
    # For Table_1.py, Figure_S2.py, Count_lesions.py
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "VAL_DATA_DIR": VAL_DATA_DIR,
    "TEST_DATA_DIR": TEST_DATA_DIR,

    # For Table5_Figure_5.py and Table_3_Table_4.py
    "PREDICTIONS_DIR": UTILS_DIR,

    # For Table5_Figure5.py
    "CHECKPOINTS_DIR": UTILS_DIR,

    # For Table_2_Table_S6.py
    "HPO_FILES_DIR": EXPERIMENTS_DIR
}

# Script execution order and their dependencies
SCRIPTS_TO_RUN = [
    {
        "name": "Table_1",
        "path": FIGURE_DIR / "Table_1" / "Table_1.py",
        "dependencies": ["TRAIN_DATA_DIR", "VAL_DATA_DIR", "TEST_DATA_DIR"],
        "description": "Generates Table 1 with dataset statistics"
    },
    {
        "name": "Figure_S2",
        "path": FIGURE_DIR / "Figure_S2" / "Figure_S2.py",
        "dependencies": ["TRAIN_DATA_DIR", "VAL_DATA_DIR", "TEST_DATA_DIR"],
        "description": "Generates Figure S2"
    },
    {
        "name": "Count_lesions",
        "path": FIGURE_DIR / "Table_S2_Figure_S3" / "Count_lesions.py",
        "dependencies": ["TRAIN_DATA_DIR", "VAL_DATA_DIR", "TEST_DATA_DIR"],
        "description": "Counts lesions for Table S2 and Figure S3"
    },
    {
        "name": "Table_S2_Figure_S3",
        "path": FIGURE_DIR / "Table_S2_Figure_S3" / "Table_S2_Figure_S3.py",
        "dependencies": [],  # Depends on Count_lesions output
        "description": "Generates Table S2 and Figure S3"
    },
    {
        "name": "Table_2_Table_S6",
        "path": FIGURE_DIR / "Table_2_Table_S6" / "Table_2_Table_S6.py",
        "dependencies": ["HPO_FILES_DIR"],
        "description": "Generates Table 2 and Table S6 from HPO results"
    },
    {
        "name": "Table_3_Table_4",
        "path": FIGURE_DIR / "Table_3_Table_4" / "Table_3_Table_4.py",
        "dependencies": ["PREDICTIONS_DIR"],
        "description": "Generates Table 3 and Table 4 from predictions"
    },
    {
        "name": "Table_5_Figure_5",
        "path": FIGURE_DIR / "Table_5_Figure_5" / "Table_5_Figure_5.py",
        "dependencies": ["PREDICTIONS_DIR", "CHECKPOINTS_DIR"],
        "description": "Generates Table 5 and Figure 5"
    }
]


def print_header(title: str):
    """Print a header."""
    print(title)



def check_data_dependencies() -> Tuple[bool, Dict]:
    """
    Check if all required data directories and files exist.

    Returns:
        Tuple of (all_dependencies_met, missing_items)
    """
    print_header("CHECKING DATA DEPENDENCIES")

    all_ok = True
    missing = {}

    for dep_name, dep_path in DATA_REQUIREMENTS.items():
        if isinstance(dep_path, Path):
            if dep_path.exists():
                # Check if directory has files (if it's a directory)
                if dep_path.is_dir():
                    files = list(dep_path.glob("*"))
                    if files:
                        print(f" {dep_name}: Found at {dep_path}")
                    else:
                        print(f" {dep_name}: Directory exists but is empty at {dep_path}")
                else:
                    print(f"{dep_name}: Found at {dep_path}")
            else:
                print(f"{dep_name}: Missing at {dep_path}")
                missing[dep_name] = str(dep_path)
                all_ok = False
        else:
            print(f" {dep_name}: Path is not a Path object: {dep_path}")

    print("=" * 60)

    if all_ok:
        print(" All data dependencies are satisfied!")
    else:
        print(f" Missing {len(missing)} data dependencies:")
        for dep, path in missing.items():
            print(f"   - {dep}: {path}")

    return all_ok, missing


def check_script_dependencies() -> Tuple[bool, Dict]:
    """
    Check if all scripts exist.

    Returns:
        Tuple of (all_scripts_exist, missing_scripts)
    """
    print_header("CHECKING SCRIPT DEPENDENCIES")

    all_ok = True
    missing = {}

    for script_info in SCRIPTS_TO_RUN:
        script_path = script_info["path"]
        if script_path.exists():
            print(f" {script_info['name']}: Found at {script_path}")
        else:
            print(f" {script_info['name']}: Missing at {script_path}")
            missing[script_info['name']] = str(script_path)
            all_ok = False



    if all_ok:
        print(" All scripts are available!")
    else:
        print(f" Missing {len(missing)} scripts")

    return all_ok, missing


def run_script(script_path: Path, script_name: str) -> bool:
    """
    Run a single script.

    Returns:
        True if script executed successfully, False otherwise
    """
    print(f"RUNNING: {script_name}")
    print(f"PATH: {script_path}")
    print(f"TIME: {datetime.now().strftime('%H:%M:%S')}")


    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,  # Run in script's directory
            capture_output=True,
            text=True,
            check=False
        )

        execution_time = time.time() - start_time

        # Print output if there is any
        if result.stdout:
            print("Output:", result.stdout.strip())

        if result.stderr:
            print("Errors/Warnings:", result.stderr.strip())

        # Check result
        if result.returncode == 0:
            print(f"{script_name} completed in {execution_time:.1f} seconds")
            return True
        else:
            print(f" {script_name} failed with exit code {result.returncode}")
            print(f"Execution time: {execution_time:.1f} seconds")
            return False

    except Exception as e:
        print(f" Exception while running {script_name}: {str(e)}")
        return False


def main():
    """Main pipeline execution function."""
    print_header("STARTING ANALYSIS PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")

    # Check dependencies
    data_ok, missing_data = check_data_dependencies()
    scripts_ok, missing_scripts = check_script_dependencies()

    if not data_ok or not scripts_ok:
        print("\n PIPELINE CANNOT START: Missing dependencies")
        if missing_data:
            print("Missing data:")
            for item, path in missing_data.items():
                print(f"  - {item}: {path}")
        if missing_scripts:
            print("Missing scripts:")
            for item, path in missing_scripts.items():
                print(f"  - {item}: {path}")
        sys.exit(1)

    # Run scripts in order
    print_header("STARTING SCRIPT EXECUTION")

    successful_scripts = []
    failed_scripts = []

    overall_start_time = time.time()

    for script_info in SCRIPTS_TO_RUN:
        success = run_script(script_info["path"], script_info["name"])

        if success:
            successful_scripts.append(script_info["name"])
        else:
            failed_scripts.append(script_info["name"])

    # Summary
    overall_time = time.time() - overall_start_time

    print_header("PIPELINE EXECUTION SUMMARY")
    print(f"Total execution time: {overall_time:.1f} seconds")
    print(f"Successful scripts: {len(successful_scripts)}/{len(SCRIPTS_TO_RUN)}")
    print(f"Failed scripts: {len(failed_scripts)}/{len(SCRIPTS_TO_RUN)}")

    if failed_scripts:
        print("\n Failed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")

    if not failed_scripts:
        print("\n ALL SCRIPTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"\n {len(failed_scripts)} script(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()