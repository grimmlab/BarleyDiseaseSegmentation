"""
Final retraining pipeline for barley disease segmentation models.
"""

from barley_disease_segmentation.common import get_augmentations
from barley_disease_segmentation.config import *
import mlflow
import torch
from barley_disease_segmentation.training_inference import TrainingModule
from .evaluation_inference import EvaluationModule
from .model_architecture import FlexibleUNet
from .visualization_inference import VisualizationModule
from .dataset import BarleyLeafDataset

__all__ = ['FinalRetrainingPipeline']

class FinalRetrainingPipeline:
    """Final retraining pipeline using train+val for training and test for evaluation."""

    def __init__(self, best_hparams, task_name, mlflow_experiment=None):
        """
        Final retraining pipeline - uses train+val for training, test only for evaluation
        """
        self.best_hparams = best_hparams
        self.task_name = task_name
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_run = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_class = FlexibleUNet
        self.dataset_class = BarleyLeafDataset
        self.TRAIN_GENOTYPES = TRAIN_GENOTYPES
        self.VAL_GENOTYPES = VAL_GENOTYPES
        self.TEST_GENOTYPES = TEST_GENOTYPES
        self.TRAIN_DATA_DIR = TRAIN_DATA_DIR
        self.VAL_DATA_DIR = VAL_DATA_DIR
        self.TEST_DATA_DIR = TEST_DATA_DIR

        # Initialize modules
        self.training = TrainingModule(self)
        self.evaluation = EvaluationModule(self)
        self.visualization = VisualizationModule(self)

        print(f" Final Retraining Pipeline for {task_name}")
        print(f" Training on: {len(TRAIN_GENOTYPES)} train + {len(VAL_GENOTYPES)} val genotypes")
        print(f" Testing on: {len(TEST_GENOTYPES)} test genotypes")

    def create_combined_dataset(self):
        """Create combined train+val dataset for final training"""
        task_mapping = {
            "multiclass": "multiclass",
            "binary_rust": "brownrust",
            "binary_ram": "ramularia"
        }
        dataset_task = task_mapping.get(self.best_hparams['task'])

        print("Creating combined training set (train + val genotypes)...")
        combined_genotypes = self.TRAIN_GENOTYPES + self.VAL_GENOTYPES
        print(f"   Combined genotypes: {len(combined_genotypes)} total")

        combined_dataset = self.dataset_class(
            all_genotypes_dir=self.TRAIN_DATA_DIR,
            genotypes_list=combined_genotypes,
            task=dataset_task,
            augmentations=get_augmentations(),
            standardize=True,
            exclude_invalid=True,
            calculate_weights=True
        )

        print(f"  Total patches: {len(combined_dataset.patches)}")
        print("Recomputing class weights on combined dataset...")
        combined_dataset.class_weights = combined_dataset.calculate_class_weights()

        return combined_dataset

    def retrain_final_model(self, epochs=100, save_suffix="final"):
        """Final retraining on train+val data without validation during training"""
        return self.training.retrain_final_model(epochs, save_suffix)

    def evaluate_on_test_set(self, model_path=None):
        """Comprehensive evaluation on the test set with CSV production"""
        return self.evaluation.evaluate_on_test_set(model_path)

    def _create_model(self):
        """Create model with best HPO parameters"""
        return self.training._create_model()

    def _get_save_path(self, base_dir_type="results", subfolder="final_models"):
        """Get the appropriate save path based on task and encoder"""
        return self.training._get_save_path(base_dir_type, subfolder)