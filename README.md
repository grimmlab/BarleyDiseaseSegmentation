
# BarleyDiseaseSegmentation

*A fully reproducible deep-learning pipeline for brown rust, ramularia, and multiclass segmentation.*

This repository provides a unified, end-to-end system for automated segmentation and phenotyping of barley leaf diseases. It supports:

- A **quick demo** inference  
- Fully automated **figure reproduction** from the associated manuscript  
- A **complete training pipeline** with two-phase HPO, encoder selection, retraining, and inference  

**Reference Paper:**

## 1. Environment Setup

### 1.1 Clone Repository
```bash
git clone https://github.com/grimmlab/barley-disease-segmentation.git
cd barley-disease-segmentation
````

### 1.2 Install Environment

#### Docker Container

```bash
docker build -t barley_disease_segmentation_image .
docker run -it \
    --gpus '"device=0"' \
    --volume $(pwd):/workspace \
    barley_disease_segmentation_image
```

#### Local Installation

```bash
pip install -e .
```
Optional: edit barley_disease_segmentation>config.py to customise paths
## 2. Dataset Setup 

Download the following three folders from Mendeley Data, DOI: [doi.org/10.17632/4ny92p2r8f.1](doi.org/10.17632/4ny92p2r8f.1):

* `hpo_data/`
* `inference_data/`
* `data_patches/`

Place them in the repository root.


## 3. Usage Options

You can use the repository in **three main ways**, depending on your goal.

### 3A. Quick Start Inference

Run a minimal demo using default settings.

```bash
python Inference_quick_start.py \
    --encoder convnext_tiny \
    --task multiclass \
    --test_data_path Sample_test_data \
    --run_leaf_evaluation \
    --leaf_evaluation_output quick_start_results
```

**Output:**
`quick_start_results/` containing predictions, leaf-level visualisations, and evaluation metrics.


### 3B. Figure Reproduction

Reproduce all figures from the manuscript.

```bash
cd Figure\ Reproduction/
python Reproduce_figures.py
```
The figures are in the respective folders in Figure Reproduction

### 3C. Complete Pipeline

*A full workflow: 2-phase HPO → encoder selection → retraining → inference.*

Supported segmentation tasks:

* `binary_ram`
* `binary_rust`
* `multiclass`

#### Run Pipeline

```bash
cd Complete\ Pipeline/

python Complete_pipeline.py \
    --task multiclass \
    --skip-hpo
```

If `--skip-hpo` is omitted, the pipeline performs **broad + refined Optuna HPO** (~6 days on a single GPU).

---

#### Pipeline Arguments (Reference Table)

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
* Encoder benchmarking
* Final retraining with best hyperparameters
* Leaf-level inference and evaluation

---

## 4. Citing This Work

This work is currently under review:  
```
Deep Learning–Based Identification of Visually Similar Foliar Diseases in Field-Grown Barley  
Sofia Martello, Nikita Genze & Dominik G. Grimm
```
