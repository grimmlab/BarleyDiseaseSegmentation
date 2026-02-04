# BarleyDiseaseSegmentation

*A fully reproducible deep-learning pipeline for rust, ramularia, and multiclass segmentation.*

This repository provides a unified, end-to-end system for automated segmentation and phenotyping of barley leaf diseases. It supports:

- A **quick demo** using sample data  
- Fully automated **figure reproduction** from the associated manuscript  
- A **complete training pipeline** with two-phase HPO, encoder selection, retraining, and final inference  

---

# 1. Environment Setup

## 1.1 Clone Repository
```bash
git clone https://github.com/grimmlab/barley-disease-segmentation.git
cd barley-disease-segmentation
````

## 1.2 Choose Installation Method

### Docker (Recommended)

```bash
docker build -t barley_disease_segmentation_image .
docker run -it \
    --gpus '"device=0"' \
    --volume $(pwd):/workspace \
    barley_disease_segmentation_image
```

### Local Installation

```bash
pip install -e .
# Optional: edit config.py if you want to customize paths
```

---

# 2. Dataset Setup

## 2.1 Sample Data (for Quick Start)

Included in the repository:

```
Test_sample_data/
```

## 2.2 Full Dataset (required for Figures & Complete Pipeline)

Download via script:

```bash
still don't know the link 
```

The script automatically creates the correct folder structure for training, validation, and testing.

---

# 3. Usage Options

You can use this repository in **three different ways**, depending on your goal.

---

# 3A. Quick Start (Sample Data Only)

A minimal demonstration: no full dataset required.

### Run Inference

```bash
python Inference_quick_start.py \
    --encoder convnext_tiny \
    --task multiclass \
    --test_data_path Test_sample_data \
    --run_leaf_evaluation \
    --leaf_evaluation_output quick_start_results
```

**Output:**
`quick_start_results/` containing predictions and evaluation metrics.

---

# 3B. Figure Reproduction (Full Dataset Required)

Reproduce all figures from the manuscript.

```bash
cd Figure_Reproduction
python reproduce_figures.py
```
---

# 3C. Complete Pipeline

*A full workflow: 2-phase HPO → encoder selection → retraining → inference.*

Supports three tasks:

* `binary_ram`
* `binary_rust`
* `multiclass`

### Run Pipeline

```bash
cd Complete_Pipeline

python Complete_pipeline.py \
    --task multiclass \
    --skip-hpo
```

If `--skip-hpo` is removed, the pipeline performs a **broad + refined Optuna HPO** stage (~2 days on a single GPU).

---

## Pipeline Arguments (Overview Table)

| Argument                                       | Description                                        | Default |
| ---------------------------------------------- | -------------------------------------------------- | ------- |
| `--task {binary_ram, binary_rust, multiclass}` | Segmentation task (required)                       | —       |
| `--skip-hpo`                                   | Use existing HPO results instead of running Optuna | False   |
| `--trials N`                                   | Number of HPO trials (if HPO active)               | 60      |
| `--epochs N`                                   | Epochs for final retraining                        | 300     |
| `--experiment-name`                            | MLflow experiment name                             | None    |
| `--dry-run`                                    | Print commands without executing                   | False   |

The pipeline automatically executes:

* Broad-phase HPO
* Refined HPO
* Encoder benchmark
* Final retraining with the best hyperparameters
* Leaf-level inference and evaluation

---

# 4. Citing This Work

If you use this repository in scientific work, please cite:


