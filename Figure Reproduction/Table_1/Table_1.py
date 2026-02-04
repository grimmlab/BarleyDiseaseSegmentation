"""
Leaf pixel counting and class distribution analysis for barley disease segmentation.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from barley_disease_segmentation.dataset import BarleyLeafDataset
from barley_disease_segmentation.config import *

# CONFIG
SAVE_DIR = (".")
os.makedirs(SAVE_DIR, exist_ok=True)


def count_leaf_pixels_only(dataset, dataset_name):
    """
    Count pixels only on leaf areas (excluding background)
    Uses total leaf pixels as denominator for all percentages
    """
    print(f"Counting leaf pixels for {dataset_name}...")

    # Initialize counters
    total_leaf_pixels = 0
    class_leaf_pixels = {0: 0, 1: 0, 2: 0}  # Counts on leaf only

    for img, mask, metadata, background_mask in tqdm(dataset, desc=dataset_name, leave=False):
        mask_np = mask.numpy()
        bg_mask_np = background_mask.numpy()

        # Leaf pixels (non-background)
        leaf_mask = ~bg_mask_np
        leaf_pixels_in_image = np.sum(leaf_mask)
        total_leaf_pixels += leaf_pixels_in_image

        # Count classes on leaf only
        for cls in [0, 1, 2]:
            class_mask = (mask_np == cls)
            leaf_class_pixels = np.sum(class_mask & leaf_mask)
            class_leaf_pixels[cls] += leaf_class_pixels

    # Calculate percentages
    if total_leaf_pixels > 0:
        healthy_pct = (class_leaf_pixels[0] / total_leaf_pixels) * 100
        brown_rust_pct = (class_leaf_pixels[1] / total_leaf_pixels) * 100
        ramularia_pct = (class_leaf_pixels[2] / total_leaf_pixels) * 100
    else:
        healthy_pct = brown_rust_pct = ramularia_pct = 0

    # Calculate total pixels and background for reference
    total_pixels = 0
    background_pixels = 0
    for img, mask, metadata, background_mask in dataset:
        mask_np = mask.numpy()
        bg_mask_np = background_mask.numpy()
        total_pixels += mask_np.size
        background_pixels += np.sum(bg_mask_np)

    background_pct = (background_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    result = {
        'dataset': dataset_name,
        'total_leaf_pixels': total_leaf_pixels,
        'total_pixels': total_pixels,
        'background_pixels': background_pixels,
        'background_pct': round(background_pct, 2),
        'healthy_pixels': class_leaf_pixels[0],
        'brown_rust_pixels': class_leaf_pixels[1],
        'ramularia_pixels': class_leaf_pixels[2],
        'healthy_pct': round(healthy_pct, 4),
        'brown_rust_pct': round(brown_rust_pct, 4),
        'ramularia_pct': round(ramularia_pct, 4)
    }

    # Print summary
    print(f"  {dataset_name}:")
    print(f"    Total leaf pixels: {total_leaf_pixels:,}")
    print(f"    Healthy leaf: {healthy_pct:.4f}%")
    print(f"    Brown rust: {brown_rust_pct:.4f}%")
    print(f"    Ramularia: {ramularia_pct:.4f}%")
    print(f"    Sum check: {(healthy_pct + brown_rust_pct + ramularia_pct):.2f}%")
    print(f"    Background: {background_pct:.1f}% of total pixels")

    return result


if __name__ == "__main__":
    """Main execution for leaf pixel counting."""
    print("Starting leaf-only pixel counting...")
    print(" Using total leaf pixels as denominator for all percentages")

    # Define datasets to analyze
    datasets = {
        "train_reflect": BarleyLeafDataset(TRAIN_DATA_DIR, TRAIN_GENOTYPES, task="multiclass"),
        "val_constant": BarleyLeafDataset(VAL_DATA_DIR, VAL_GENOTYPES, task="multiclass"),
        "test_constant": BarleyLeafDataset(TEST_DATA_DIR, TEST_GENOTYPES, task="multiclass"),
    }

    # Count leaf pixels onlydocker
    results = []
    for name, dataset in datasets.items():
        result = count_leaf_pixels_only(dataset, name)
        results.append(result)

    # Create DataFrame
    df_leaf = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'dataset',
        'total_pixels',
        'background_pixels',
        'background_pct',
        'total_leaf_pixels',
        'healthy_pixels',
        'healthy_pct',
        'brown_rust_pixels',
        'brown_rust_pct',
        'ramularia_pixels',
        'ramularia_pct'
    ]

    # Ensure all columns exist
    existing_columns = [col for col in column_order if col in df_leaf.columns]
    df_leaf = df_leaf[existing_columns]

    # Save
    csv_path = os.path.join(SAVE_DIR, "leaf_only_pixel_counts.csv")
    df_leaf.to_csv(csv_path, index=False)
    print(f" Leaf-only counts saved to: {csv_path}")

    # Print summary table
    print("\nPercentage of leaf pixels per class:")
    print(df_leaf[['dataset', 'healthy_pct', 'brown_rust_pct', 'ramularia_pct']].to_string(index=False))

    print(f"\n Analysis complete, CSV saved to: {csv_path}")