"""
Encoder selection summary generation for barley disease segmentation.
"""

import pandas as pd
import json
import os
from pathlib import Path
from barley_disease_segmentation.config import *
from barley_disease_segmentation.encoder_selector import *


if __name__ == "__main__":
    """Main execution for encoder selection summary generation."""
    # Configuration
    TASKS = ["binary_ram", "binary_rust", "multiclass"]
    ENCODERS = ["resnet34", "convnext_tiny", "efficientnet_b2"]

    # Run the analysis
    df_best, df_all = create_encoder_selection_summary(
        tasks=TASKS,
        encoders=ENCODERS,
        best_encoders_csv="best_encoders_summary_.csv",
        all_trials_csv="all_encoders_trials_summary.csv"
    )

    if df_best is not None and df_all is not None:
        print(f"Analysis complete! Check 'encoder_selection_results' folder for:")
        print(f"  1. best_encoders_summary.csv - Best encoder for each task")
        print(f"  2. all_encoders_trials_summary.csv - All best trials for each encoder-task combination")
        print(f"  3. Individual JSON files with best parameters for each encoder-task combination")

        # Print summary statistics

        print(f"SUMMARY STATISTICS")

        print(f"Total tasks analyzed: {len(TASKS)}")
        print(f"Total encoders analyzed: {len(ENCODERS)}")
        print(f"Total encoder-task combinations: {len(TASKS) * len(ENCODERS)}")
        print(f"Valid trials found: {len(df_all)}")

        # Count best encoders by model
        if len(df_best) > 0:
            best_counts = df_best['best_encoder'].value_counts()
            print(f"\nBest encoders by frequency:")
            for encoder, count in best_counts.items():
                print(f"  {encoder}: {count} tasks")