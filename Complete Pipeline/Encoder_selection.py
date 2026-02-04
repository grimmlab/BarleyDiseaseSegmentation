"""
Script for encoder selection analysis for barley disease segmentation.
"""

from barley_disease_segmentation.encoder_selector import *
import argparse


if __name__ == "__main__":
    """Main entry point for encoder selection script."""
    parser = argparse.ArgumentParser(description='Run Encoder Selection for one task')
    parser.add_argument('--task', type=str, required=True,
                        choices=['multiclass', 'binary_ram', 'binary_rust'],
                        help='Task to optimize for')

    args = parser.parse_args()

    TASKS = [args.task]  # Single task in a list
    ENCODERS = ["resnet34", "convnext_tiny", "efficientnet_b2"]

    # Run the analysis
    df_best, df_all = create_encoder_selection_summary(
        tasks=TASKS,
        encoders=ENCODERS,
        best_encoders_csv=f"best_encoders_summary_{args.task}.csv",
        all_trials_csv=f"all_encoders_trials_summary_{args.task}.csv"
    )

    if df_best is not None and df_all is not None:
        print(f"\n{'=' * 80}")
        print(f"Analysis complete! Check 'encoder_selection_results' folder for:")
        print(f"  1. best_encoders_summary_{args.task}.csv - Best encoder for {args.task}")
        print(f"  2. all_encoders_trials_summary_{args.task}.csv - All encoder trials for {args.task}")
        print(f"  3. Individual JSON files with best parameters for each encoder-{args.task} combination")

        # Print summary statistics
        print(f"SUMMARY STATISTICS")

        print(f"Task analyzed: {args.task}")
        print(f"Total encoders analyzed: {len(ENCODERS)}")
        print(f"Valid trials found: {len(df_all)}")

        # Display best encoder result
        if not df_best.empty:
            best_row = df_best.iloc[0]
            print(f"\nBEST ENCODER FOR {args.task}: {best_row['best_encoder']}")
            print(f"  Dice Score: {best_row['best_dice_score']:.4f}")
            print(f"  Source File: {best_row['source_file']}")


#python Encoder_selection.py --task binary_ram