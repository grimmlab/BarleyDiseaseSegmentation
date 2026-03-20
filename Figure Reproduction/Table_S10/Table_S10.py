import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from barley_disease_segmentation.config import PROJECT_ROOT

# -------------------------------------------------------------------
# 1. SET PATHS (Multiclass model ONLY)
# -------------------------------------------------------------------
PRED_DIR = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/predictions")
DATA_DIR = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/data")
LABELS_DIR = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/labels")

# -------------------------------------------------------------------
# 2. BASIC HELPERS
# -------------------------------------------------------------------
def load_img(path):
    return np.array(Image.open(path))

def get_leaf_mask(img):
    """Mask = True for leaf pixels, False for white background."""
    return ~((img[:,:,0] == 255) & (img[:,:,1] == 255) & (img[:,:,2] == 255))

def parse_filename(fn):
    g, leaf = fn.replace(".png", "").split("_")
    return g, leaf

# -------------------------------------------------------------------
# 3. LOAD ALL MULTICLASS PREDICTIONS + GROUND TRUTH
# -------------------------------------------------------------------
records = []

for fn in os.listdir(PRED_DIR):
    if not fn.endswith(".png"):
        continue

    genotype, leafID = parse_filename(fn)

    # prediction
    pred = load_img(PRED_DIR / fn)
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    # ground truth
    gt = load_img(LABELS_DIR / fn)
    if gt.ndim == 3:
        gt = gt[:, :, 0]

    # leaf mask
    img = load_img(DATA_DIR / fn)
    leaf_mask = get_leaf_mask(img)
    leaf_area = leaf_mask.sum()

    # predicted areas
    rust_pred = ((pred == 1) & leaf_mask).sum() / leaf_area * 100
    ram_pred = ((pred == 2) & leaf_mask).sum() / leaf_area * 100

    # ground truth areas
    rust_gt = ((gt == 1) & leaf_mask).sum() / leaf_area * 100
    ram_gt = ((gt == 2) & leaf_mask).sum() / leaf_area * 100

    records.append({
        "genotype": genotype,
        "rust_pred": rust_pred,
        "ram_pred": ram_pred,
        "rust_gt": rust_gt,
        "ram_gt": ram_gt
    })

df = pd.DataFrame(records)

# -------------------------------------------------------------------
# 4. GENOTYPE-LEVEL AVERAGES
# -------------------------------------------------------------------
geno_summary = df.groupby("genotype").mean().reset_index()

# -------------------------------------------------------------------
# 5. MAE AND RMSE
# -------------------------------------------------------------------
rust_abs_err = (geno_summary["rust_pred"] - geno_summary["rust_gt"]).abs()
ram_abs_err  = (geno_summary["ram_pred"] - geno_summary["ram_gt"]).abs()

mae_rust = rust_abs_err.mean()
mae_ram  = ram_abs_err.mean()

rmse_rust = np.sqrt(((geno_summary["rust_pred"] - geno_summary["rust_gt"]) ** 2).mean())
rmse_ram  = np.sqrt(((geno_summary["ram_pred"] - geno_summary["ram_gt"]) ** 2).mean())

# -------------------------------------------------------------------
# 6. CREATE FINAL CSV (ONLY MAE + RMSE)
# -------------------------------------------------------------------
error_summary = pd.DataFrame({
    "disease": ["brown_rust", "ramularia"],
    "MAE (%)": [mae_rust, mae_ram],
    "RMSE (%)": [rmse_rust, rmse_ram]
})

output_csv = Path("multiclass_error_metrics.csv")
output_csv.parent.mkdir(exist_ok=True, parents=True)
error_summary.to_csv(output_csv, index=False)

print(f"\nSaved MAE/RMSE table to: {output_csv}\n")
print(error_summary)