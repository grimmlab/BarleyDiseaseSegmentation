"""
Dataset statistics computation for barley disease segmentation.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from barley_disease_segmentation.config import *
from barley_disease_segmentation.dataset import BarleyLeafDataset


SAVE_DIR = (".")
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_class_stats(dataset, max_classes=3):
    """
    Compute per-class pixel counts and lesion counts.
    """
    class_counts_leaf_only = np.zeros(max_classes, dtype=np.int64)
    lesion_counts = {c: 0 for c in range(max_classes)}

    for img, mask, metadata, background_mask in tqdm(dataset, desc="Computing stats", leave=False):
        mask_np = mask.numpy()
        bg_mask_np = background_mask.numpy()

        # Leaf-only mask
        leaf_mask = ~bg_mask_np

        # Count pixels for each class in leaf-only areas
        for c in range(max_classes):
            class_mask = (mask_np == c) & leaf_mask
            class_counts_leaf_only[c] += np.sum(class_mask)

            # Count lesions for class 1 and 2
            if c in [1, 2] and np.any(class_mask):
                # Count connected components (lesions)
                labeled_array, num_features = ndimage.label(class_mask)
                lesion_counts[c] += num_features

    stats = {
        "class_counts_leaf_only": class_counts_leaf_only,
        "lesion_counts": lesion_counts,
        "total_leaf_pixels": np.sum(class_counts_leaf_only)
    }
    return stats


def summarize_stats(dataset_name, stats):
    """
    Return summary row as DataFrame with only requested columns.
    """
    n_classes = len(stats["class_counts_leaf_only"])
    total_leaf_pixels = stats["total_leaf_pixels"]

    row = {
        "dataset": dataset_name,
    }

    # Compute class ratios
    for i in range(n_classes):
        row[f"class_{i}_ratio_leaf_only"] = stats["class_counts_leaf_only"][i] / total_leaf_pixels

    # Add lesion counts for classes 1 and 2
    if "lesion_counts" in stats:
        row["class_1_lesion_count"] = stats["lesion_counts"].get(1, 0)
        row["class_2_lesion_count"] = stats["lesion_counts"].get(2, 0)

    return pd.DataFrame([row])


if __name__ == "__main__":
    """Main execution for dataset statistics computation."""
    print(" Starting dataset composition analysis...")

    dataset_variants = {
        "train_reflect": BarleyLeafDataset(TRAIN_DATA_DIR, TRAIN_GENOTYPES, task="multiclass"),
        "val_constant": BarleyLeafDataset(VAL_DATA_DIR, VAL_GENOTYPES, task="multiclass"),
        "test_constant": BarleyLeafDataset(TEST_DATA_DIR, TEST_GENOTYPES, task="multiclass"),
    }

    # Compute statistics
    summaries = []
    for name, dataset in dataset_variants.items():
        print(f"\n[INFO] Processing {name} ({len(dataset)} patches)...")
        stats = compute_class_stats(dataset, max_classes=dataset.num_classes)
        summary = summarize_stats(name, stats)
        summaries.append(summary)

    df_summary = pd.concat(summaries, ignore_index=True)

    columns = [
        "dataset",
        "class_0_ratio_leaf_only",
        "class_1_ratio_leaf_only",
        "class_1_lesion_count",
        "class_2_ratio_leaf_only",
        "class_2_lesion_count"
    ]

    available_columns = [col for col in columns if col in df_summary.columns]
    df_summary = df_summary[available_columns]

    # Save to CSV
    csv_path = os.path.join(SAVE_DIR, "dataset_stats.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f" Saved dataset summary {csv_path}")
    print(" Analysis complete.")