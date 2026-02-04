"""
Configuration settings for barley leaf disease segmentation project.

Defines project structure, paths, dataset splits, and hyperparameters for
brown rust and ramularia leaf spot segmentation experiments.
"""

from pathlib import Path
import torch

__all__ = [
    'PROJECT_ROOT',
    'DEVICE',
    'NUM_GPUS',
    'DATA_DIR',
    'TRAIN_DATA_DIR',
    'VAL_DATA_DIR',
    'TEST_DATA_DIR',
    'TASKS',
    'MODELS',
    'BINARY_RUST_PATHS',
    'BINARY_RAM_PATHS',
    'MULTICLASS_PATHS',
    'get_model_paths',
    'TRAIN_GENOTYPES',
    'VAL_GENOTYPES',
    'TEST_GENOTYPES',
    'EDGE_CASES',
    'CLASS_WEIGHTS',
    'NUM_CLASSES'
]

#  PROJECT STRUCTURE
PROJECT_ROOT = Path(__file__).parent.parent
print(f"Config loaded: Project root is {PROJECT_ROOT}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

# MAIN DIRECTORIES
DATA_DIR = PROJECT_ROOT / "data_patches"
EXPERIMENTS_DIR = PROJECT_ROOT / "hpo_data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "inference_data"
FIGURE_DIR = PROJECT_ROOT / "Figure Reproduction"
PIPELINE_DIR = PROJECT_ROOT / "Complete Pipeline"
PACKAGE_DIR = PROJECT_ROOT / "barley_disease_segmentation"
CONFIG_FILE = PACKAGE_DIR / "config.py"

# DATA PATHS
RAW_DATA_DIR = DATA_DIR / "Raw_data"
TRAIN_DATA_DIR = DATA_DIR / "Train_data"
VAL_DATA_DIR = DATA_DIR / "Validation_data"
TEST_DATA_DIR = DATA_DIR / "Test_data"

# TASK TYPES
TASKS = ["Dataset_characterization", "Binary_rust", "Binary_ram", "Multiclass"]
MODELS = ["Convnext", "Resnet", "Efficientnet"]

#  EXPERIMENT PATHS

# Binary Rust Task
BINARY_RUST_EXP = EXPERIMENTS_DIR / "Binary_rust"
BINARY_RUST_RES = RESULTS_DIR / "Binary_rust"

# Binary Ramularia Task
BINARY_RAM_EXP = EXPERIMENTS_DIR / "Binary_ram"
BINARY_RAM_RES = RESULTS_DIR / "Binary_ram"

# Multiclass Task
MULTICLASS_EXP = EXPERIMENTS_DIR / "Multiclass"
MULTICLASS_RES = RESULTS_DIR / "Multiclass"


def get_model_paths(task_name: str):
    """
    Get experiment and results paths for all models in a task.

    Args:
        task_name: One of ["Binary_rust", "Binary_ram", "Multiclass"]

    Returns:
        dict: Paths organized by model type and purpose.
    """
    task_exp = EXPERIMENTS_DIR / task_name
    task_res = RESULTS_DIR / task_name
    task_utils = UTILS_DIR / task_name

    return {
        'hpo_data': {
            'Convnext': task_exp / "Convnext",
            'Resnet': task_exp / "Resnet",
            'Efficientnet': task_exp / "Efficientnet"
        },
        'results': {
            'Convnext': task_res / "Convnext",
            'Resnet': task_res / "Resnet",
            'Efficientnet': task_res / "Efficientnet"
        },
        'utils': {
            'Convnext': task_utils / "Convnext",
            'Resnet': task_utils / "Resnet",
            'Efficientnet': task_utils / "Efficientnet"
        }
    }


# Specific model paths for easy access
BINARY_RUST_PATHS = get_model_paths("Binary_rust")
BINARY_RAM_PATHS = get_model_paths("Binary_ram")
MULTICLASS_PATHS = get_model_paths("Multiclass")

# DATASET CONFIGURATION
# TRAINING SET (43 genotypes)
TRAIN_GENOTYPES = [
    # Set1 (9)
    '9610', '9633', '9641', '9648', '9650', '9669', '9799', '9815', '9842',
    # Set2 (4)
    '41422', '41530', '41542', '41611',
    # Set3 (30)
    '694001', '694002', '694003', '694004', '694005', '694006', '694007', '694008', '694009', '694010',
    '694011', '694012', '694013', '694014', '694015', '694016', '694017', '694018', '694019', '694020',
    '694021', '694022', '694023', '694024', '694025', '694026', '694027', '694028', '694029', '694030'
]

# VALIDATION SET (9 genotypes)
VAL_GENOTYPES = [
    # Set1 (2)
    '9783', '9830',
    # Set2 (1)
    '41424',
    # Set3 (6)
    '694031', '694032', '694033', '694034', '694035', '694036'
]

# TEST SET (10 genotypes)
TEST_GENOTYPES = [
    # Set1 (1)
    '9635',
    # Set2 (1)
    '41561',
    # Set3 (8)
    '694037', '694038', '694039', '694040', '694041', '694042', '694043', '694044'
]

# Edge cases dictionary
EDGE_CASES = {
    '9610': ('c', 'Set1'), '9633': ('d', 'Set1'), '9635': ('f', 'Set1'),
    '9783': ('e', 'Set1'), '9830': ('c', 'Set1'), '9842': ('c', 'Set1'),
    '41422': ('b', 'Set2'), '41424': ('e', 'Set2'), '41530': ('g', 'Set2'),
    '41542': ('g', 'Set2'), '41561': ('e', 'Set2'), '41611': ('b', 'Set2'),
    '694010': ('d', 'Set3'), '694014': ('b', 'Set3'), '694016': ('a', 'Set3'),
    '694021': ('a', 'Set3'), '694022': ('a', 'Set3'), '694034': ('d', 'Set3')
}

CLASS_WEIGHTS = torch.tensor([0.0032737066503614187, 0.5850458145141602, 0.4116804301738739], dtype=torch.float32)
NUM_CLASSES = 2