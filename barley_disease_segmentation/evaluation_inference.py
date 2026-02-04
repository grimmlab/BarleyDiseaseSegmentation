"""
Evaluation module for barley disease segmentation models.

Provides comprehensive evaluation metrics and analysis for:
- Binary rust segmentation
- Binary ramularia segmentation  
- Multiclass segmentation
"""

import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mlflow

from barley_disease_segmentation.config import *
from barley_disease_segmentation.common import extract_sample_metadata

__all__ = ['EvaluationModule', 'SegmentationMetrics']

class EvaluationModule:
    """Handles model evaluation on test sets with MLflow integration."""

    def __init__(self, pipeline):
        """
        Initialize evaluation module.

        Args:
            pipeline: Main training pipeline object containing model and config
        """
        self.pipeline = pipeline

        if self.pipeline.mlflow_run:
            print("MLflow logging enabled")
        else:
            print("Running in non-MLflow mode")

    def evaluate_on_test_set(self, model_path):
        """
        Comprehensive evaluation on test set.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Path to evaluation results directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        eval_log_dir = f"test_evaluation_{self.pipeline.task_name}_{self.pipeline.best_hparams['encoder_name']}_{timestamp}"
        print(f"Evaluation directory: {eval_log_dir}")

        if self.pipeline.mlflow_run:
            utils_dir = self._run_evaluation_with_mlflow(eval_log_dir, model_path)
        else:
            utils_dir = self._run_evaluation_without_mlflow(eval_log_dir, model_path)

        return utils_dir

    def _run_evaluation_with_mlflow(self, eval_log_dir, model_path):
        """Run evaluation with MLflow logging."""
        with mlflow.start_run(nested=True, run_name=eval_log_dir) as trial_run:
            if model_path:
                checkpoint = torch.load(model_path)
                self.pipeline.model = self.pipeline.training._create_model()
                self.pipeline.model.load_state_dict(checkpoint['model_state_dict'])
                print(f" Loaded model from: {model_path}")
                mlflow.log_param("model_path", str(model_path))

            utils_dir = self._run_evaluation_core(mlflow=True)
            return utils_dir

    def _run_evaluation_without_mlflow(self, eval_log_dir, model_path):
        """Run evaluation without MLflow logging."""
        print("Running evaluation without MLflow logging...")

        if model_path:
            checkpoint = torch.load(model_path)
            self.pipeline.model = self.pipeline.training._create_model()
            self.pipeline.model.load_state_dict(checkpoint['model_state_dict'])
            print(f" Loaded model from: {model_path}")

        utils_dir = self._run_evaluation_core(mlflow=None)
        return utils_dir

    def _run_evaluation_core(self, mlflow=None):
        """Core evaluation logic."""
        print(" Running comprehensive evaluation on TEST set...")

        # Map task names to dataset format
        task_mapping = {
            "multiclass": "multiclass",
            "binary_rust": "brownrust",
            "binary_ram": "ramularia"
        }
        dataset_task = task_mapping.get(self.pipeline.best_hparams['task'])

        # Create test dataset
        test_dataset = self.pipeline.dataset_class(
            all_genotypes_dir=self.pipeline.TEST_DATA_DIR,
            genotypes_list=self.pipeline.TEST_GENOTYPES,
            task=dataset_task,
            augmentations=None,
            standardize=True,
            exclude_invalid=True,
            calculate_weights=False
        )

        # Create dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=2,
            pin_memory=True, persistent_workers=False
        )

        # Compute normalization stats
        test_dataset.mean, test_dataset.std = test_dataset.compute_mean_and_std(test_loader, 'imagenet')

        # Log to MLflow if enabled
        if mlflow:
            mlflow.log_param("test_genotypes_count", len(self.pipeline.TEST_GENOTYPES))
            mlflow.log_param("test_patches_count", len(test_dataset.patches))
            mlflow.log_param("test_batch_size", 32)

        # Run patch-level inference
        patch_metrics_df = self._run_patch_level_inference(test_loader, test_dataset)

        # Create output directories
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
        eval_dir = self.pipeline.training._get_save_path("results", f"evaluations/{time_stamp}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        utils_dir = self.pipeline.training._get_save_path("utils", f"{time_stamp}")
        utils_dir.mkdir(parents=True, exist_ok=True)

        if mlflow:
            mlflow.log_param("evaluation_dir", str(eval_dir))

        # Generate visualizations and analyses
        print("\nSaving stitched whole-leaf predictions...")
        self.pipeline.visualization.save_stitched_predictions(test_dataset, utils_dir)

        # Identify and save edge cases
        edge_cases = self._identify_edge_cases(patch_metrics_df)
        edge_cases_path = eval_dir / "edge_cases.json"
        with open(edge_cases_path, 'w') as f:
            json.dump(edge_cases, f, indent=2)

        # Generate misclassification plots
        self.pipeline.visualization._generate_misclassification_plots(test_dataset, edge_cases, eval_dir)

        return utils_dir

    def _run_patch_level_inference(self, test_loader, test_dataset):
        """Run inference and compute patch-level metrics."""
        patch_metrics = []
        self.pipeline.model.eval()

        with torch.no_grad():
            for batch_idx, (images, masks, metadata, bg_masks) in enumerate(test_loader):
                images = images.to(self.pipeline.device)
                masks = masks.to(self.pipeline.device)
                bg_masks = bg_masks.to(self.pipeline.device)

                outputs = self.pipeline.model(images)
                preds = torch.argmax(outputs, dim=1)

                for i in range(images.shape[0]):
                    sample_metadata = extract_sample_metadata(metadata, i)
                    patch_metric = self._compute_patch_metrics(
                        preds[i], masks[i], outputs[i], bg_masks[i], sample_metadata
                    )
                    patch_metrics.append(patch_metric)

                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx}/{len(test_loader)} batches")

        return pd.DataFrame(patch_metrics)

    def _compute_patch_metrics(self, pred, target, logits, bg_mask, metadata):
        """Compute comprehensive metrics for a single patch."""
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        bg_mask_np = bg_mask.cpu().numpy() if bg_mask is not None else None

        # Create valid mask excluding background
        if bg_mask_np is not None:
            valid_mask = ~bg_mask_np
        else:
            valid_mask = np.ones_like(pred_np, dtype=bool)

        pred_masked = pred_np[valid_mask]
        target_masked = target_np[valid_mask]
        total_pixels = pred_np.size
        total_valid_pixels = np.sum(valid_mask)

        # Initialize patch metrics
        patch_metrics = {
            'patch_id': metadata['img_name'],
            'patch_total_pixels': total_pixels,
            'total_valid_pixels': total_valid_pixels,
        }

        # Binary task metrics
        task = self.pipeline.best_hparams['task']
        if task in ['binary_rust', 'binary_ram']:
            pred_binary = (pred_masked > 0).astype(int)
            target_binary = (target_masked > 0).astype(int)

            tp = np.sum((pred_binary == 1) & (target_binary == 1))
            tn = np.sum((pred_binary == 0) & (target_binary == 0))
            fp = np.sum((pred_binary == 1) & (target_binary == 0))
            fn = np.sum((pred_binary == 0) & (target_binary == 1))

            patch_metrics.update({
                'true_positive_pixels': tp,
                'true_negative_pixels': tn,
                'false_positive_pixels': fp,
                'false_negative_pixels': fn,
            })

        # Multiclass task metrics
        elif task == 'multiclass':
            class_names = ['background', 'rust', 'ramularia']
            pred_lesion = (pred_masked > 0).astype(int)
            target_lesion = (target_masked > 0).astype(int)

            tp_lesion = np.sum((pred_lesion == 1) & (target_lesion == 1))
            tn_lesion = np.sum((pred_lesion == 0) & (target_lesion == 0))
            fp_lesion = np.sum((pred_lesion == 1) & (target_lesion == 0))
            fn_lesion = np.sum((pred_lesion == 0) & (target_lesion == 1))

            patch_metrics.update({
                'true_positive_pixels': tp_lesion,
                'true_negative_pixels': tn_lesion,
                'false_positive_pixels': fp_lesion,
                'false_negative_pixels': fn_lesion,
            })

            # Confusion between disease classes
            for true_cls in [1, 2]:
                for pred_cls in [1, 2]:
                    if true_cls != pred_cls:
                        confusion_count = np.sum((target_masked == true_cls) & (pred_masked == pred_cls))
                        col_name = f"{class_names[true_cls]}_as_{class_names[pred_cls]}_pixels"
                        patch_metrics[col_name] = confusion_count

            background_correct = np.sum((target_masked == 0) & (pred_masked == 0))
            patch_metrics['background_correct_pixels'] = background_correct

        # Area-based metrics
        gt_lesion_area = np.sum(target_masked > 0)
        pred_lesion_area = np.sum(pred_masked > 0)

        patch_metrics.update({
            'gt_lesion_area_pixels': gt_lesion_area,
            'pred_lesion_area_pixels': pred_lesion_area,
            'lesion_coverage_ratio': gt_lesion_area / total_valid_pixels if total_valid_pixels > 0 else 0,
        })

        # Standard segmentation metrics
        num_classes = 3 if task == "multiclass" else 2
        metrics_obj = SegmentationMetrics(
            num_classes=num_classes,
            task='multiclass' if task == "multiclass" else 'binary'
        )

        pred_expanded = pred.unsqueeze(0) if pred.dim() == 2 else pred
        target_expanded = target.unsqueeze(0) if target.dim() == 2 else target
        bg_expanded = bg_mask.unsqueeze(0) if bg_mask is not None and bg_mask.dim() == 2 else bg_mask

        all_metrics = metrics_obj.get_all_metrics(pred_expanded, target_expanded, bg_expanded)

        patch_metrics.update({
            'pixel_accuracy': all_metrics['accuracy'],
            'mean_iou': all_metrics['mean_iou'],
            'dice_score': all_metrics['mean_dice'],
        })

        # Add precision/recall based on task
        if task in ['binary_rust', 'binary_ram']:
            if 'foreground_precision' in all_metrics:
                patch_metrics.update({
                    'precision': all_metrics['foreground_precision'],
                    'recall': all_metrics['foreground_recall'],
                })
        else:
            if len(all_metrics['precision_per_class']) > 1:
                patch_metrics['precision'] = all_metrics['precision_per_class'][1]
                patch_metrics['recall'] = all_metrics['recall_per_class'][1]

        # Calculate specificity
        if 'true_negative_pixels' in patch_metrics and 'false_positive_pixels' in patch_metrics:
            tn = patch_metrics['true_negative_pixels']
            fp = patch_metrics['false_positive_pixels']
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            patch_metrics['specificity'] = specificity

        # Categorize patch type and error severity
        lesion_coverage = patch_metrics['lesion_coverage_ratio']
        patch_metrics['patch_type'] = self._categorize_patch_type(lesion_coverage)

        total_error = patch_metrics.get('false_positive_pixels', 0) + patch_metrics.get('false_negative_pixels', 0)
        error_ratio = total_error / total_valid_pixels if total_valid_pixels > 0 else 0
        patch_metrics['error_severity'] = self._categorize_error_severity(error_ratio)

        # Add metadata
        patch_metrics.update({
            'genotype': metadata['orig_image_id'].split('_')[0],
            'leaf_id': metadata['orig_image_id'],
            'x_offset': metadata['x_offset'],
            'y_offset': metadata['y_offset'],
        })

        return patch_metrics

    def _categorize_patch_type(self, coverage):
        """Categorize patch based on lesion coverage."""
        if coverage < 0.1:
            return "mostly_healthy"
        elif coverage > 0.7:
            return "mostly_lesion"
        else:
            return "mixed"

    def _categorize_error_severity(self, error_ratio):
        """Categorize error severity."""
        if error_ratio < 0.05:
            return "low"
        elif error_ratio < 0.15:
            return "medium"
        else:
            return "high"

    def _identify_edge_cases(self, df, n_cases=5):
        """Identify edge cases for analysis."""
        area_ratio = df['pred_lesion_area_pixels'] / df['gt_lesion_area_pixels']
        area_ratio = area_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        segmentation_quality = []
        for ratio in area_ratio:
            if ratio > 1.5:
                segmentation_quality.append("over_segmented")
            elif ratio < 0.5:
                segmentation_quality.append("under_segmented")
            else:
                segmentation_quality.append("balanced")

        df_with_quality = df.copy()
        df_with_quality['segmentation_quality'] = segmentation_quality

        return {
            'high_error': df.nlargest(n_cases, 'false_positive_pixels')['patch_id'].tolist(),
            'over_segmented': df_with_quality[df_with_quality['segmentation_quality'] == 'over_segmented']
            .head(n_cases)['patch_id'].tolist(),
            'under_segmented': df_with_quality[df_with_quality['segmentation_quality'] == 'under_segmented']
            .head(n_cases)['patch_id'].tolist(),
            'high_confidence_errors': df[(df['error_severity'] == 'high') & (df['dice_score'] < 0.3)]
            .head(n_cases)['patch_id'].tolist(),
            'best_performers': df.nlargest(n_cases, 'dice_score')['patch_id'].tolist()
        }


class SegmentationMetrics:
    """Computes segmentation metrics for binary and multiclass tasks."""

    def __init__(self, num_classes, task="multiclass", ignore_index=None):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes (including background)
            task: "binary" or "multiclass"
            ignore_index: Class index to ignore (e.g., background)
        """
        self.num_classes = num_classes
        self.task = task
        self.ignore_index = ignore_index

        if task not in ["binary", "multiclass"]:
            raise ValueError("task must be 'binary' or 'multiclass'")

    def _prepare_predictions_targets(self, predictions, targets, background_mask):
        """Prepare predictions and targets for metric computation."""
        # Get class predictions from logits
        if predictions.dim() == 4:  # [N, C, H, W] - raw logits
            pred_classes = torch.argmax(predictions, dim=1)  # [N, H, W]
        else:  # [N, H, W]
            pred_classes = predictions

        # Ensure targets are 3D
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # Apply background masking
        if background_mask is not None:
            if background_mask.dim() == 4:
                background_mask = background_mask.squeeze(1)

            if background_mask.shape != pred_classes.shape:
                try:
                    background_mask = F.interpolate(
                        background_mask.unsqueeze(1).float(),
                        size=pred_classes.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).bool()
                except:
                    # Skip background masking if resize fails
                    return pred_classes, targets

            targets = targets.clone()
            if self.ignore_index is not None:
                targets[background_mask] = self.ignore_index
            else:
                pred_classes = pred_classes.clone()
                targets = targets.clone()
                pred_classes[background_mask] = -1
                targets[background_mask] = -1

        return pred_classes, targets

    def _get_valid_mask(self, targets):
        """Get mask of valid pixels."""
        if self.ignore_index is not None:
            return targets != self.ignore_index
        else:
            return targets != -1

    def iou_per_class(self, predictions, targets, background_mask=None):
        """Calculate IoU per class."""
        pred_classes, targets = self._prepare_predictions_targets(predictions, targets, background_mask)
        valid_mask = self._get_valid_mask(targets)

        iou_per_class = []
        classes = [0, 1] if self.task == "binary" else range(self.num_classes)

        for class_id in classes:
            pred_mask = (pred_classes == class_id) & valid_mask
            target_mask = (targets == class_id) & valid_mask

            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                iou_per_class.append(float('nan'))
                continue

            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()

            if union == 0:
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append((intersection / union).item())

        return iou_per_class

    def mean_iou(self, predictions, targets, background_mask=None):
        """Calculate mean IoU across all classes."""
        iou_per_class = self.iou_per_class(predictions, targets, background_mask)
        valid_ious = [iou for iou in iou_per_class if not torch.isnan(torch.tensor(iou))]

        if len(valid_ious) == 0:
            return float('nan')
        return sum(valid_ious) / len(valid_ious)

    def dice_per_class(self, predictions, targets, background_mask=None):
        """Calculate Dice coefficient per class."""
        pred_classes, targets = self._prepare_predictions_targets(predictions, targets, background_mask)
        valid_mask = self._get_valid_mask(targets)

        dice_per_class = []
        classes = [0, 1] if self.task == "binary" else range(self.num_classes)

        for class_id in classes:
            pred_mask = (pred_classes == class_id) & valid_mask
            target_mask = (targets == class_id) & valid_mask

            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                dice_per_class.append(float('nan'))
                continue

            intersection = (pred_mask & target_mask).float().sum()
            total = pred_mask.float().sum() + target_mask.float().sum()

            if total == 0:
                dice_per_class.append(float('nan'))
            else:
                dice_per_class.append((2. * intersection / total).item())

        return dice_per_class

    def mean_dice(self, predictions, targets, background_mask=None):
        """Calculate mean Dice coefficient across all classes."""
        dice_per_class = self.dice_per_class(predictions, targets, background_mask)
        valid_dice = [dice for dice in dice_per_class if not torch.isnan(torch.tensor(dice))]

        if len(valid_dice) == 0:
            return float('nan')
        return sum(valid_dice) / len(valid_dice)

    def precision_recall_f1_per_class(self, predictions, targets, background_mask=None):
        """Calculate Precision, Recall, and F1-score per class."""
        pred_classes, targets = self._prepare_predictions_targets(predictions, targets, background_mask)
        valid_mask = self._get_valid_mask(targets)

        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        classes = [0, 1] if self.task == "binary" else range(self.num_classes)

        for class_id in classes:
            pred_mask = (pred_classes == class_id) & valid_mask
            target_mask = (targets == class_id) & valid_mask

            tp = (pred_mask & target_mask).float().sum()
            fp = (pred_mask & ~target_mask).float().sum()
            fn = (~pred_mask & target_mask).float().sum()

            # Precision
            if (tp + fp) == 0:
                precision = float('nan')
            else:
                precision = (tp / (tp + fp)).item()

            # Recall
            if (tp + fn) == 0:
                recall = float('nan')
            else:
                recall = (tp / (tp + fn)).item()

            # F1-score
            if torch.isnan(torch.tensor(precision)) or torch.isnan(torch.tensor(recall)) or (precision + recall) == 0:
                f1 = float('nan')
            else:
                f1 = float(2 * precision * recall / (precision + recall))

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        return precision_per_class, recall_per_class, f1_per_class

    def accuracy(self, predictions, targets, background_mask=None):
        """Calculate overall accuracy."""
        pred_classes, targets = self._prepare_predictions_targets(predictions, targets, background_mask)
        valid_mask = self._get_valid_mask(targets)

        correct = (pred_classes[valid_mask] == targets[valid_mask]).float().sum()
        total = valid_mask.float().sum()

        if total == 0:
            return float('nan')
        return (correct / total).item()

    def get_all_metrics(self, predictions, targets, background_mask=None):
        """Compute all metrics at once."""
        metrics = {}

        # Per-class metrics
        metrics['iou_per_class'] = self.iou_per_class(predictions, targets, background_mask)
        metrics['dice_per_class'] = self.dice_per_class(predictions, targets, background_mask)
        metrics['precision_per_class'], metrics['recall_per_class'], metrics['f1_per_class'] = \
            self.precision_recall_f1_per_class(predictions, targets, background_mask)

        # Mean metrics
        metrics['mean_iou'] = self.mean_iou(predictions, targets, background_mask)
        metrics['mean_dice'] = self.mean_dice(predictions, targets, background_mask)
        metrics['accuracy'] = self.accuracy(predictions, targets, background_mask)

        # For binary task, add foreground metrics
        if self.task == "binary" and len(metrics['iou_per_class']) > 1:
            metrics['foreground_iou'] = metrics['iou_per_class'][1]
            metrics['foreground_dice'] = metrics['dice_per_class'][1]
            metrics['foreground_precision'] = metrics['precision_per_class'][1]
            metrics['foreground_recall'] = metrics['recall_per_class'][1]
            metrics['foreground_f1'] = metrics['f1_per_class'][1]

        return metrics