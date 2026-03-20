import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from barley_disease_segmentation.config import PROJECT_ROOT

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
TASKS = {
    "multiclass": {
        "pred": Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/predictions"),
        "data": Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/data"),
        "type": "multiclass",
    },
    "binary_rust": {
        "pred": Path(PROJECT_ROOT / "inference_data/Binary_rust/Convnext/20251117_1014/saved_predictions/predictions"),
        "data": Path(PROJECT_ROOT / "inference_data/Binary_rust/Convnext/20251117_1014/saved_predictions/data"),
        "type": "binary_rust",
    },
    "binary_ram": {
        "pred": Path(PROJECT_ROOT / "inference_data/Binary_ram/Convnext/20251117_1002/saved_predictions/predictions"),
        "data": Path(PROJECT_ROOT / "inference_data/Binary_ram/Convnext/20251117_1002/saved_predictions/data"),
        "type": "binary_ram",
    },
}

WINTER = ["9635"]  # only winter genotype


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------
def load_img(path):
    return np.array(Image.open(path))


def get_leaf_mask(img):
    return ~((img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255))


def parse_filename(fn):
    g, leaf = fn.replace(".png", "").split("_")
    return g, leaf


def map_score(pct):
    score_ranges = {
        1: (0.0, 0.0),
        2: (0.0, 2.0),
        3: (2.0, 5.0),
        4: (5.0, 8.0),
        5: (8.0, 14.0),
        6: (14.0, 22.0),
        7: (22.0, 37.0),
        8: (37.0, 61.0),
        9: (61.0, 100.0)
    }
    for score in sorted(score_ranges.keys()):
        low, high = score_ranges[score]
        if score == max(score_ranges.keys()):
            if low <= pct <= high:
                return score
        else:
            if low <= pct < high:
                return score
    return np.nan


# ---------------------------------------------------------
# Process all tasks to get predictions
# ---------------------------------------------------------
all_records = []

for task_name, cfg in TASKS.items():
    pred_dir = cfg["pred"]
    data_dir = cfg["data"]
    ttype = cfg["type"]

    for fn in os.listdir(pred_dir):
        if not fn.endswith(".png"):
            continue

        genotype, leafID = parse_filename(fn)

        pred = load_img(pred_dir / fn)
        if pred.ndim == 3:
            pred = pred[:, :, 0]

        img = load_img(data_dir / fn)
        leaf_mask = get_leaf_mask(img)
        leaf_area = leaf_mask.sum()

        rust_area = 0
        ram_area = 0

        if ttype == "multiclass":
            rust_area = ((pred == 1) & leaf_mask).sum()
            ram_area = ((pred == 2) & leaf_mask).sum()
        elif ttype == "binary_rust":
            rust_area = ((pred == 1) & leaf_mask).sum()
        elif ttype == "binary_ram":
            ram_area = ((pred == 1) & leaf_mask).sum()

        all_records.append({
            "task": task_name,
            "genotype": genotype,
            "rust_pct": rust_area / leaf_area * 100,
            "ram_pct": ram_area / leaf_area * 100,
        })

df = pd.DataFrame(all_records)

# ---------------------------------------------------------
# Get ground truth labels
# ---------------------------------------------------------
labels_dir = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/labels")
gt_records = []

for fn in os.listdir(labels_dir):
    if not fn.endswith(".png"):
        continue
    genotype, leafID = parse_filename(fn)
    label = load_img(labels_dir / fn)
    if label.ndim == 3:
        label = label[:, :, 0]

    img = load_img(TASKS["multiclass"]["data"] / fn)
    leaf_mask = get_leaf_mask(img)
    leaf_area = leaf_mask.sum()

    rust_area = ((label == 1) & leaf_mask).sum()
    ram_area = ((label == 2) & leaf_mask).sum()

    gt_records.append({
        "genotype": genotype,
        "rust_pct_gt": rust_area / leaf_area * 100,
        "ram_pct_gt": ram_area / leaf_area * 100,
    })

df_gt = pd.DataFrame(gt_records)
geno_gt_summary = df_gt.groupby("genotype").agg(
    rust_pct_avg_gt=("rust_pct_gt", "mean"),
    ram_pct_avg_gt=("ram_pct_gt", "mean"),
).reset_index()

# Add scores to ground truth
geno_gt_summary["rust_score_gt"] = geno_gt_summary["rust_pct_avg_gt"].apply(map_score)
geno_gt_summary["ram_score_gt"] = geno_gt_summary["ram_pct_avg_gt"].apply(map_score)

# ---------------------------------------------------------
# Create comprehensive genotype-level table with all models
# ---------------------------------------------------------
all_model_records = []

for task_name in ["multiclass", "binary_rust", "binary_ram"]:
    df_task = df[df["task"] == task_name].copy()

    if task_name == "binary_rust":
        rust_col = "rust_pct"
        ram_col = None
    elif task_name == "binary_ram":
        rust_col = None
        ram_col = "ram_pct"
    else:
        rust_col = "rust_pct"
        ram_col = "ram_pct"

    genotypes = df_task["genotype"].unique()
    task_records = []

    for genotype in genotypes:
        genotype_data = df_task[df_task["genotype"] == genotype]
        record = {"genotype": genotype}

        if rust_col:
            record[f"rust_pct_pred_{task_name}"] = genotype_data[rust_col].mean()
            record[f"rust_score_pred_{task_name}"] = map_score(record[f"rust_pct_pred_{task_name}"])
        else:
            record[f"rust_pct_pred_{task_name}"] = np.nan
            record[f"rust_score_pred_{task_name}"] = np.nan

        if ram_col:
            record[f"ram_pct_pred_{task_name}"] = genotype_data[ram_col].mean()
            record[f"ram_score_pred_{task_name}"] = map_score(record[f"ram_pct_pred_{task_name}"])
        else:
            record[f"ram_pct_pred_{task_name}"] = np.nan
            record[f"ram_score_pred_{task_name}"] = np.nan

        task_records.append(record)

    task_df = pd.DataFrame(task_records)

    if len(all_model_records) == 0:
        all_model_records = task_df
    else:
        all_model_records = pd.merge(all_model_records, task_df, on="genotype", how="outer")

# Merge with ground truth
final_table = pd.merge(
    all_model_records,
    geno_gt_summary[["genotype", "rust_pct_avg_gt", "ram_pct_avg_gt", "rust_score_gt", "ram_score_gt"]],
    on="genotype",
    how="left"
)

# Create compact version
compact_table = final_table[[
    "genotype",
    "rust_pct_pred_multiclass", "rust_pct_pred_binary_rust", "rust_pct_avg_gt",
    "rust_score_pred_multiclass", "rust_score_pred_binary_rust", "rust_score_gt",
    "ram_pct_pred_multiclass", "ram_pct_pred_binary_ram", "ram_pct_avg_gt",
    "ram_score_pred_multiclass", "ram_score_pred_binary_ram", "ram_score_gt"
]].copy()

# Round percentages to 2 decimal places
pct_cols = [col for col in compact_table.columns if "pct" in col]
for col in pct_cols:
    compact_table[col] = compact_table[col].round(2)

# Save to CSV
compact_csv = Path("genotype_summary_all_models_compact.csv")
compact_table.to_csv(compact_csv, index=False)

print(f"Compact CSV file saved to: {compact_csv}")
print("\nPreview of the compact table:")
print(compact_table.to_string())