"""
Evaluation framework for barley disease segmentation models.

Provides comprehensive evaluation metrics including:
- Pixel-level segmentation (Dice, IoU)
- Instance-level detection (precision, recall, F1)
- Multi-model comparison
- F1 vs IoU threshold analysis
"""

import json
from collections import defaultdict
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage import measure

from barley_disease_segmentation.config import *
__all__ = ['BaseEvaluator',
           'SingleModelEvaluator',
           'MultiModelEvaluator']

class BaseEvaluator:
    """
    Base evaluator with shared functionality for barley disease evaluation.

    Handles mask loading, basic metrics calculation, and leaf mask processing.
    """

    def __init__(self, device="cuda", min_presence_pixels=10,
                 leaf_masks_path=None):
        """
        Initialize base evaluator.

        Args:
            device: Computing device ('cuda' or 'cpu')
            min_presence_pixels: Minimum lesion pixels to consider leaf valid
            leaf_masks_path: Path to leaf segmentation masks
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_presence_pixels = int(min_presence_pixels)
        self.leaf_masks_path = leaf_masks_path
        print(f"Using device: {self.device}")
        print(f"Lesion presence min pixels: {self.min_presence_pixels}")

    def _load_leaf_mask(self, leaf_id):
        """
        Load leaf segmentation mask from PNG file.

        Args:
            leaf_id: Leaf identifier string

        Returns:
            Binary mask where 1=leaf tissue, 0=background
        """
        if self.leaf_masks_path is None:
            return None
        leaf_mask_path = self.leaf_masks_path / "data" / f"{leaf_id}.png"
        if not leaf_mask_path.exists():
            return None
        leaf_rgb = cv2.imread(str(leaf_mask_path))
        if leaf_rgb is None:
            return None
        leaf_rgb = cv2.cvtColor(leaf_rgb, cv2.COLOR_BGR2RGB)
        leaf_mask = ~np.all(leaf_rgb == [255, 255, 255], axis=2)
        return leaf_mask.astype(np.uint8)

    def load_masks(self, predictions_path, leaf_id, task_type,
                   disease_class=None):
        """
        Load ground truth, prediction, and leaf masks.

        Args:
            predictions_path: Directory containing predictions
            leaf_id: Leaf identifier
            task_type: 'binary' or 'multiclass'
            disease_class: For multiclass, which disease to evaluate (1=rust, 2=ramularia)

        Returns:
            tuple: (gt_binary_mask, pred_binary_mask, leaf_mask)
        """
        labels_path = predictions_path / "labels" / f"{leaf_id}.png"
        pred_path = predictions_path / "predictions" / f"{leaf_id}.png"

        gt_mask = self._load_gray_mask(labels_path)
        pred_mask = self._load_gray_mask(pred_path)
        leaf_mask = self._load_leaf_mask(leaf_id)

        if gt_mask is None or pred_mask is None:
            return None, None, None

        # Resize if needed
        if gt_mask.shape != pred_mask.shape:
            try:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception:
                return None, None, None

        # Apply leaf mask
        if leaf_mask is not None and leaf_mask.shape != gt_mask.shape:
            leaf_mask = cv2.resize(leaf_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            gt_mask = gt_mask * leaf_mask
            pred_mask = pred_mask * leaf_mask

        # Convert to binary for specific disease if multiclass
        if task_type == "multiclass" and disease_class is not None:
            gt_bin = (gt_mask == disease_class).astype(np.uint8)
            pred_bin = (pred_mask == disease_class).astype(np.uint8)
        else:
            gt_bin = (gt_mask > 0).astype(np.uint8)
            pred_bin = (pred_mask > 0).astype(np.uint8)

        return gt_bin, pred_bin, leaf_mask

    def _load_gray_mask(self, path):
        """Load grayscale mask from path."""
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return mask

    def calculate_dice_boolean(self, gt_mask, pred_mask, leaf_mask):
        """
        Calculate Dice coefficient using only leaf tissue areas.

        Args:
            gt_mask: Ground truth binary mask
            pred_mask: Prediction binary mask
            leaf_mask: Leaf tissue mask

        Returns:
            Dice coefficient (float)
        """
        if leaf_mask is None:
            gt_tensor = torch.as_tensor(gt_mask.astype(np.float32), device=self.device)
            pred_tensor = torch.as_tensor(pred_mask.astype(np.float32), device=self.device)
        else:
            leaf_pixels = leaf_mask.astype(bool)
            gt_leaf_only = gt_mask[leaf_pixels]
            pred_leaf_only = pred_mask[leaf_pixels]

            if len(gt_leaf_only) == 0:
                return 0.0

            gt_tensor = torch.as_tensor(gt_leaf_only.astype(np.float32), device=self.device)
            pred_tensor = torch.as_tensor(pred_leaf_only.astype(np.float32), device=self.device)

        intersection = (gt_tensor * pred_tensor).sum()
        union = gt_tensor.sum() + pred_tensor.sum()
        dice = (2.0 * intersection) / (union + 1e-8)
        return float(dice.item())

    def calculate_iou(self, gt_mask, pred_mask, leaf_mask=None):
        """
        Calculate Intersection over Union (IoU).

        Args:
            gt_mask: Ground truth binary mask
            pred_mask: Prediction binary mask
            leaf_mask: Optional leaf tissue mask

        Returns:
            IoU score (float)
        """
        if leaf_mask is not None:
            leaf_pixels = leaf_mask.astype(bool)
            gt_leaf_only = gt_mask[leaf_pixels]
            pred_leaf_only = pred_mask[leaf_pixels]

            if len(gt_leaf_only) == 0:
                return 0.0

            intersection = (gt_leaf_only & pred_leaf_only).sum()
            union = (gt_leaf_only | pred_leaf_only).sum()
        else:
            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)

    def _iou_of_masks(self, mask_a, mask_b):
        """Calculate IoU between two boolean masks."""
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return inter / union

    @staticmethod
    def get_leaf_ids_from_predictions(predictions_path):
        """
        Extract leaf IDs from predictions folder.

        Args:
            predictions_path: Directory containing predictions

        Returns:
            List of leaf ID strings
        """
        pred_path = predictions_path / "predictions"
        if not pred_path.exists():
            print(f"Warning: No predictions found in {pred_path}")
            return []

        leaf_ids = sorted([f.stem for f in pred_path.glob("*.png")])
        print(f"Found {len(leaf_ids)} leaves in predictions")
        return leaf_ids

    def _compute_iou_matrix(self, gt_labels, pred_labels,
                            gt_regions, pred_regions):
        """
        Compute IoU matrix between ground truth and predicted regions.

        Optimized using bounding box checks before full mask comparison.
        """
        G = len(gt_regions)
        P = len(pred_regions)
        iou_mat = np.zeros((G, P), dtype=np.float32)

        for i, g in enumerate(gt_regions):
            g_slice = g.bbox
            g_minr, g_minc, g_maxr, g_maxc = g_slice
            g_mask = (gt_labels == g.label)

            for j, p in enumerate(pred_regions):
                p_slice = p.bbox
                rmin = max(g_minr, p_slice[0])
                cmin = max(g_minc, p_slice[1])
                rmax = min(g_maxr, p_slice[2])
                cmax = min(g_maxc, p_slice[3])
                if rmin >= rmax or cmin >= cmax:
                    continue

                p_mask = (pred_labels == p.label)
                iou = self._iou_of_masks(g_mask, p_mask)
                iou_mat[i, j] = iou

        return iou_mat

    def calculate_detection_metrics(self, gt_mask, pred_mask,
                                    iou_threshold=0.5):
        """
        Calculate instance-level detection metrics.

        Uses Hungarian algorithm for optimal matching between
        ground truth and predicted lesion instances.

        Args:
            gt_mask: Ground truth binary mask
            pred_mask: Prediction binary mask
            iou_threshold: Minimum IoU for considering a match

        Returns:
            dict: Precision, recall, and F1 scores
        """
        gt_labels = measure.label(gt_mask)
        pred_labels = measure.label(pred_mask)

        gt_regions = measure.regionprops(gt_labels)
        pred_regions = measure.regionprops(pred_labels)

        G = len(gt_regions)
        P = len(pred_regions)

        if G == 0 and P == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        if G == 0:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
        if P == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        iou_mat = self._compute_iou_matrix(gt_labels, pred_labels, gt_regions, pred_regions)

        n = max(iou_mat.shape[0], iou_mat.shape[1])
        cost = np.ones((n, n), dtype=np.float32)
        cost[:iou_mat.shape[0], :iou_mat.shape[1]] = 1.0 - iou_mat

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        matched_gt = set()
        matched_pred = set()
        for r, c in zip(row_ind, col_ind):
            if r < iou_mat.shape[0] and c < iou_mat.shape[1]:
                iou = iou_mat[r, c]
                if iou >= iou_threshold:
                    matches.append((r, c, iou))
                    matched_gt.add(r)
                    matched_pred.add(c)

        tp = len(matches)
        fp = P - len(matched_pred)
        fn = G - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def compute_f1_vs_iou_threshold(self, predictions_path, leaf_ids,
                                    task_type, disease_class=None):
        """
        Compute average F1 scores across 10 IoU thresholds.

        Used for robustness analysis of detection performance.

        Returns:
            tuple: (iou_thresholds, avg_f1_scores, valid_leaves_count)
        """
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        all_f1_scores = {thresh: [] for thresh in iou_thresholds}
        valid_leaves = 0

        for leaf_id in leaf_ids:
            try:
                gt_mask, pred_mask, _ = self.load_masks(predictions_path, leaf_id, task_type, disease_class)

                if gt_mask is None or pred_mask is None:
                    continue

                if gt_mask.sum() < self.min_presence_pixels:
                    continue

                for iou_thresh in iou_thresholds:
                    det_metrics = self.calculate_detection_metrics(gt_mask, pred_mask, iou_threshold=iou_thresh)
                    all_f1_scores[iou_thresh].append(det_metrics['f1'])

                valid_leaves += 1

            except Exception as e:
                continue

        avg_f1_scores = []
        for iou_thresh in iou_thresholds:
            if all_f1_scores[iou_thresh]:
                avg_f1 = np.mean(all_f1_scores[iou_thresh])
                avg_f1_scores.append(avg_f1)
            else:
                avg_f1_scores.append(0.0)

        return iou_thresholds, avg_f1_scores, valid_leaves


class SingleModelEvaluator(BaseEvaluator):
    """
    Evaluator for detailed single model evaluation.

    Provides leaf-level metrics and comprehensive statistics.
    """

    def __init__(self, device="cuda", min_presence_pixels=10,
                 leaf_masks_path=None):
        super().__init__(device, min_presence_pixels, leaf_masks_path)
        self.results = defaultdict(dict)

    def evaluate_single_leaf(self, predictions_path, leaf_id, task_type,
                             disease_class=None, disease_name="Disease"):
        """
        Evaluate a single leaf comprehensively.

        Args:
            predictions_path: Path to model predictions
            leaf_id: Leaf identifier
            task_type: 'binary' or 'multiclass'
            disease_class: Disease class index (for multiclass)
            disease_name: Name of disease for reporting

        Returns:
            dict: Comprehensive metrics for the leaf
        """
        try:
            gt_mask, pred_mask, leaf_mask = self.load_masks(predictions_path, leaf_id,
                                                            task_type, disease_class)

            if gt_mask is None or pred_mask is None:
                return None

            # Skip if no disease presence in ground truth
            if gt_mask.sum() < self.min_presence_pixels:
                return None

            # Calculate segmentation metrics
            dice = self.calculate_dice_boolean(gt_mask, pred_mask, leaf_mask)
            iou = self.calculate_iou(gt_mask, pred_mask, leaf_mask)

            # Calculate detection metrics at different thresholds
            det_lenient = self.calculate_detection_metrics(gt_mask, pred_mask, iou_threshold=0.2)
            det_strict = self.calculate_detection_metrics(gt_mask, pred_mask, iou_threshold=0.5)

            # Count lesions
            gt_labels = measure.label(gt_mask)
            pred_labels = measure.label(pred_mask)
            gt_lesions = len(measure.regionprops(gt_labels))
            pred_lesions = len(measure.regionprops(pred_labels))

            # Area statistics
            gt_area = gt_mask.sum()
            pred_area = pred_mask.sum()

            return {
                'leaf_id': leaf_id,
                'disease_name': disease_name,
                'dice_score': dice,
                'iou_score': iou,
                'detection_f1_lenient': det_lenient['f1'],
                'detection_precision_lenient': det_lenient['precision'],
                'detection_recall_lenient': det_lenient['recall'],
                'detection_f1_strict': det_strict['f1'],
                'detection_precision_strict': det_strict['precision'],
                'detection_recall_strict': det_strict['recall'],
                'gt_lesion_count': gt_lesions,
                'pred_lesion_count': pred_lesions,
                'gt_lesion_area': int(gt_area),
                'pred_lesion_area': int(pred_area),
                'has_disease': int(gt_area) > 0
            }

        except Exception as e:
            print(f"Error evaluating leaf {leaf_id}: {e}")
            return None

    def evaluate_model(self, predictions_path, model_name, leaf_ids,
                       task_type, disease_class=None,
                       disease_name="Disease"):
        """
        Evaluate a model on all leaves.

        Args:
            predictions_path: Path to model predictions
            model_name: Name of model for reporting
            leaf_ids: List of leaf IDs to evaluate
            task_type: 'binary' or 'multiclass'
            disease_class: Disease class index
            disease_name: Disease name for reporting

        Returns:
            DataFrame: Leaf-level metrics for all valid leaves
        """
        print(f"\nEvaluating: {model_name} on {disease_name}")
        print(f"Predictions path: {predictions_path}")
        print(f"Number of leaves to evaluate: {len(leaf_ids)}")

        results = []
        valid_leaves = 0

        for i, leaf_id in enumerate(leaf_ids):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/{len(leaf_ids)} leaves...")

            leaf_metrics = self.evaluate_single_leaf(predictions_path, leaf_id, task_type,
                                                     disease_class, disease_name)
            if leaf_metrics:
                results.append(leaf_metrics)
                valid_leaves += 1

        print(f"  Completed. Valid leaves evaluated: {valid_leaves}/{len(leaf_ids)}")

        if results:
            df = pd.DataFrame(results)
            return df
        else:
            print("Warning: No valid leaves found for evaluation!")
            return pd.DataFrame()

    def run_f1_threshold_analysis_single(self, predictions_path, model_name,
                                         leaf_ids, task_type,
                                         disease_class=None,
                                         disease_name="Disease",
                                         output_path=Path(".")):
        """
        Run F1 vs IoU threshold analysis for single model.

        Args:
            predictions_path: Path to model predictions
            model_name: Model name for reporting
            leaf_ids: List of leaf IDs
            task_type: 'binary' or 'multiclass'
            disease_class: Disease class index
            disease_name: Disease name for reporting
            output_path: Directory to save results

        Returns:
            dict: Analysis results with thresholds and F1 scores
        """
        print("\n" + "=" * 60)
        print(f"F1 THRESHOLD ANALYSIS: {model_name} - {disease_name}")
        print("=" * 60)

        iou_thresholds, avg_f1_scores, valid_leaves = self.compute_f1_vs_iou_threshold(
            predictions_path=predictions_path,
            leaf_ids=leaf_ids,
            task_type=task_type,
            disease_class=disease_class
        )

        result = {
            'model': model_name,
            'disease': disease_name,
            'iou_thresholds': iou_thresholds,
            'avg_f1_scores': avg_f1_scores,
            'valid_leaves': valid_leaves
        }

        # Save CSV results
        csv_data = []
        for iou, f1 in zip(iou_thresholds, avg_f1_scores):
            csv_data.append({
                'model': model_name,
                'disease': disease_name,
                'iou_threshold': iou,
                'f1_score': f1
            })

        df = pd.DataFrame(csv_data)
        csv_path = output_path / f'f1_vs_iou_threshold_{model_name}_{disease_name.replace(" ", "_")}.csv'
        df.to_csv(csv_path, index=False)
        print(f"F1 threshold results saved to: {csv_path}")

        return result

    def generate_summary_statistics(self, detailed_df):
        """
        Generate summary statistics from leaf-level DataFrame.

        Args:
            detailed_df: DataFrame with leaf-level metrics

        Returns:
            DataFrame: Summary statistics (mean, std, median, counts)
        """
        if detailed_df.empty:
            return pd.DataFrame()

        summary_data = []

        # Overall statistics
        summary_data.append({
            'statistic': 'Overall Mean',
            'dice_score': detailed_df['dice_score'].mean(),
            'iou_score': detailed_df['iou_score'].mean(),
            'detection_f1_lenient': detailed_df['detection_f1_lenient'].mean(),
            'detection_f1_strict': detailed_df['detection_f1_strict'].mean(),
            'gt_lesion_count': detailed_df['gt_lesion_count'].mean(),
            'pred_lesion_count': detailed_df['pred_lesion_count'].mean(),
            'gt_lesion_area': detailed_df['gt_lesion_area'].mean(),
            'pred_lesion_area': detailed_df['pred_lesion_area'].mean(),
        })

        # Standard deviations
        summary_data.append({
            'statistic': 'Overall Std',
            'dice_score': detailed_df['dice_score'].std(),
            'iou_score': detailed_df['iou_score'].std(),
            'detection_f1_lenient': detailed_df['detection_f1_lenient'].std(),
            'detection_f1_strict': detailed_df['detection_f1_strict'].std(),
            'gt_lesion_count': detailed_df['gt_lesion_count'].std(),
            'pred_lesion_count': detailed_df['pred_lesion_count'].std(),
            'gt_lesion_area': detailed_df['gt_lesion_area'].std(),
            'pred_lesion_area': detailed_df['pred_lesion_area'].std(),
        })

        # Medians
        summary_data.append({
            'statistic': 'Median',
            'dice_score': detailed_df['dice_score'].median(),
            'iou_score': detailed_df['iou_score'].median(),
            'detection_f1_lenient': detailed_df['detection_f1_lenient'].median(),
            'detection_f1_strict': detailed_df['detection_f1_strict'].median(),
            'gt_lesion_count': detailed_df['gt_lesion_count'].median(),
            'pred_lesion_count': detailed_df['pred_lesion_count'].median(),
            'gt_lesion_area': detailed_df['gt_lesion_area'].median(),
            'pred_lesion_area': detailed_df['pred_lesion_area'].median(),
        })

        # Counts
        summary_data.append({
            'statistic': 'Counts',
            'dice_score': len(detailed_df),
            'iou_score': len(detailed_df),
            'detection_f1_lenient': len(detailed_df),
            'detection_f1_strict': len(detailed_df),
            'gt_lesion_count': len(detailed_df),
            'pred_lesion_count': len(detailed_df),
            'gt_lesion_area': len(detailed_df),
            'pred_lesion_area': len(detailed_df),
        })

        return pd.DataFrame(summary_data)

    @staticmethod
    def run_leaf_level_evaluation(predictions_path, model_name,
                                  task_type, disease_name,
                                  disease_class=None,
                                  output_dir=None):
        """
        Convenience method to run complete leaf-level evaluation.

        Args:
            predictions_path: Path to model predictions
            model_name: Model name for reporting
            task_type: 'binary' or 'multiclass'
            disease_name: Disease name for reporting
            disease_class: Disease class index
            output_dir: Directory to save results

        Returns:
            DataFrame: Detailed leaf-level metrics
        """
        if output_dir is None:
            output_dir = Path(f"leaf_evaluation_results_{task_type}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get leaf IDs
        leaf_ids = BaseEvaluator.get_leaf_ids_from_predictions(predictions_path=predictions_path)
        if len(leaf_ids) == 0:
            print("ERROR: No leaf predictions found!")
            return pd.DataFrame()

        # Initialize evaluator
        evaluator = SingleModelEvaluator(
            device="cuda",
            min_presence_pixels=10,
            leaf_masks_path=predictions_path
        )

        # Run evaluation
        detailed_df = evaluator.evaluate_model(
            predictions_path=predictions_path,
            model_name=model_name,
            leaf_ids=leaf_ids,
            task_type=task_type,
            disease_class=disease_class,
            disease_name=disease_name
        )

        if not detailed_df.empty:
            # Save detailed results
            detailed_csv_path = output_dir / f'leaf_level_detailed_metrics_{task_type}.csv'
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"Detailed leaf-level metrics saved to: {detailed_csv_path}")

            # Save summary statistics
            summary_df = evaluator.generate_summary_statistics(detailed_df)
            summary_csv_path = output_dir / f'leaf_level_summary_{task_type}.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Leaf-level summary saved to: {summary_csv_path}")

            # Print key metrics
            print("\n" + "=" * 80)
            print(f"LEAF-LEVEL EVALUATION RESULTS for {model_name} - {disease_name}")
            print("=" * 80)
            print(f"Number of leaves evaluated: {len(detailed_df)}")
            print(f"Average Dice Score: {detailed_df['dice_score'].mean():.3f} ± {detailed_df['dice_score'].std():.3f}")
            print(f"Average IoU Score: {detailed_df['iou_score'].mean():.3f} ± {detailed_df['iou_score'].std():.3f}")
            print(f"Detection F1 @IoU=0.2: {detailed_df['detection_f1_lenient'].mean():.3f}")
            print(f"Detection F1 @IoU=0.5: {detailed_df['detection_f1_strict'].mean():.3f}")

        return detailed_df


class MultiModelEvaluator(BaseEvaluator):
    """
    Evaluator for comparing multiple models.

    Focuses on per-class metrics and generates comparison tables.
    """

    def __init__(self, device="cuda", min_presence_pixels=10,
                 leaf_masks_path=None):
        super().__init__(device, min_presence_pixels, leaf_masks_path)
        self.results = defaultdict(dict)
        self.per_leaf_results = []

    def evaluate_model_on_disease(self, model_predictions_path, model_name, leaf_ids,
                                  task_type, disease_class, disease_name,
                                  iou_lenient=0.2, iou_strict=0.5):
        """
        Evaluate a specific model on a specific disease.

        Args:
            model_predictions_path: Path to model predictions
            model_name: Model name for reporting
            leaf_ids: List of leaf IDs
            task_type: 'binary' or 'multiclass'
            disease_class: Disease class index
            disease_name: Disease name for reporting
            iou_lenient: Lenient IoU threshold (0.2)
            iou_strict: Strict IoU threshold (0.5)
        """
        print(f"Evaluating: {model_name} on {disease_name}")

        dice_scores = []
        iou_scores = []
        detection_f1_lenient = []
        detection_f1_strict = []
        detection_precision_lenient = []
        detection_recall_lenient = []
        detection_precision_strict = []
        detection_recall_strict = []

        for leaf_id in leaf_ids:
            try:
                gt_mask, pred_mask, leaf_mask = self.load_masks(model_predictions_path, leaf_id, task_type,
                                                                disease_class)

                if gt_mask is None or pred_mask is None:
                    continue

                dice = self.calculate_dice_boolean(gt_mask, pred_mask, leaf_mask)
                dice_scores.append(dice)

                iou = self.calculate_iou(gt_mask, pred_mask, leaf_mask)
                iou_scores.append(iou)

                det_len = self.calculate_detection_metrics(gt_mask, pred_mask, iou_threshold=iou_lenient)
                det_str = self.calculate_detection_metrics(gt_mask, pred_mask, iou_threshold=iou_strict)

                detection_f1_lenient.append(det_len['f1'])
                detection_f1_strict.append(det_str['f1'])
                detection_precision_lenient.append(det_len['precision'])
                detection_recall_lenient.append(det_len['recall'])
                detection_precision_strict.append(det_str['precision'])
                detection_recall_strict.append(det_str['recall'])

            except Exception as e:
                continue

        key = f"{model_name.lower()}_{disease_name.replace(' ', '_').lower()}"
        if len(dice_scores) > 0:
            self.results[key] = {
                'model': model_name,
                'disease': disease_name,
                'dice_mean': float(np.mean(dice_scores)),
                'iou_mean': float(np.mean(iou_scores)),
                'detection_f1_lenient_mean': float(np.mean(detection_f1_lenient)) if detection_f1_lenient else 0.0,
                'detection_f1_strict_mean': float(np.mean(detection_f1_strict)) if detection_f1_strict else 0.0,
                'detection_precision_lenient_mean': float(
                    np.mean(detection_precision_lenient)) if detection_precision_lenient else 0.0,
                'detection_recall_lenient_mean': float(
                    np.mean(detection_recall_lenient)) if detection_recall_lenient else 0.0,
                'detection_precision_strict_mean': float(
                    np.mean(detection_precision_strict)) if detection_precision_strict else 0.0,
                'detection_recall_strict_mean': float(
                    np.mean(detection_recall_strict)) if detection_recall_strict else 0.0,
                'num_leaves_evaluated': int(len(dice_scores))
            }

    def run_comprehensive_comparison(self, model_configs, common_leaf_ids):
        """Run comparison across all model configurations."""
        for config_name, config in model_configs.items():
            self.evaluate_model_on_disease(
                model_predictions_path=config['predictions_path'],
                model_name=config['model_name'],
                leaf_ids=common_leaf_ids,
                task_type=config['task_type'],
                disease_class=config.get('disease_class'),
                disease_name=config['disease_name']
            )

    def run_f1_threshold_analysis(self, model_configs, common_leaf_ids, output_path):
        """
        Run F1 vs IoU threshold analysis for all models.

        Generates plot and CSV of detection performance across IoU thresholds.
        """
        print("\n" + "=" * 60)
        print("F1 THRESHOLD ANALYSIS")
        print("=" * 60)

        f1_threshold_results = {}
        csv_data = []

        # Analyze each model/disease combination
        for config_name, config in model_configs.items():
            model_name = config['model_name']
            disease_name = config['disease_name']

            print(f"\nAnalysing: {model_name} - {disease_name}")

            iou_thresholds, avg_f1_scores, valid_leaves = self.compute_f1_vs_iou_threshold(
                predictions_path=config['predictions_path'],
                leaf_ids=common_leaf_ids,
                task_type=config['task_type'],
                disease_class=config.get('disease_class')
            )

            key = f"{model_name}_{disease_name}"
            f1_threshold_results[key] = {
                'iou_thresholds': iou_thresholds,
                'avg_f1_scores': avg_f1_scores,
                'model': model_name,
                'disease': disease_name,
                'valid_leaves': valid_leaves
            }

            for iou, f1 in zip(iou_thresholds, avg_f1_scores):
                csv_data.append({
                    'model': model_name,
                    'disease': disease_name,
                    'iou_threshold': iou,
                    'f1_score': f1
                })

            print(f"  Used {valid_leaves} leaves with disease presence")
            print(f"  F1 @ IoU=0.2: {avg_f1_scores[1]:.3f}")
            print(f"  F1 @ IoU=0.5: {avg_f1_scores[4]:.3f}")

        # Save CSV results
        df = pd.DataFrame(csv_data)
        csv_path = output_path / 'f1_vs_iou_threshold_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nF1 threshold results saved to: {csv_path}")

        # Create and save plot
        self._create_f1_vs_iou_plot(f1_threshold_results, output_path)

        # Print summary table
        self._print_f1_threshold_summary(f1_threshold_results)

        return f1_threshold_results

    def _create_f1_vs_iou_plot(self, results, output_path):
        """Create F1 vs IoU threshold plot comparing models."""
        plt.figure(figsize=(10, 6))

        color_map = {
            ('Binary', 'Brown Rust'): '#1f77b4',
            ('Binary', 'Ramularia'): '#ff7f0e',
            ('Multiclass', 'Brown Rust'): '#2ca02c',
            ('Multiclass', 'Ramularia'): '#d62728',
        }

        line_styles = {'Binary': '-', 'Multiclass': '--'}
        markers = {'Brown Rust': 'o', 'Ramularia': 's'}

        for key, data in results.items():
            plt.plot(
                data['iou_thresholds'],
                data['avg_f1_scores'],
                marker=markers[data['disease']],
                linestyle=line_styles[data['model']],
                linewidth=2,
                markersize=7,
                color=color_map[(data['model'], data['disease'])],
                label=f"{data['model']} - {data['disease']}",
                markeredgecolor='black',
                markeredgewidth=0.5
            )

        plt.xlabel('IoU Threshold (Detection Matching Criterion)', fontsize=12, fontweight='bold')
        plt.ylabel('Detection F1 Score', fontsize=12, fontweight='bold')
        plt.title('Detection Performance vs Matching Strictness\n(F1 Score at Different IoU Thresholds)',
                  fontsize=14, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
        plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        plt.ylim(0, 1)
        plt.gca().yaxis.grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = output_path / 'f1_vs_iou_thresholds.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path / 'f1_vs_iou_thresholds.pdf', bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"F1 vs IoU threshold plot saved to: {plot_path}")

    def _print_f1_threshold_summary(self, results):
        """Print summary table of F1 scores at different thresholds."""
        print("\n" + "=" * 60)
        print("SUMMARY: F1 SCORES AT DIFFERENT IOU THRESHOLDS")
        print("=" * 60)

        summary_data = []
        for key, data in results.items():
            for iou, f1 in zip(data['iou_thresholds'], data['avg_f1_scores']):
                summary_data.append({
                    'model': data['model'],
                    'disease': data['disease'],
                    'iou_threshold': iou,
                    'f1_score': f1
                })

        df = pd.DataFrame(summary_data)
        pivot_df = df.pivot_table(
            index='iou_threshold',
            columns=['model', 'disease'],
            values='f1_score'
        )

        print(pivot_df.round(3))

    def generate_comparison_table(self):
        """
        Generate main comparison table between binary and multiclass models.

        Returns:
            DataFrame: Formatted comparison table for paper
        """
        comparison_data = []

        diseases = set()
        for key in self.results.keys():
            disease = self.results[key]['disease']
            diseases.add(disease)

        for disease in sorted(diseases):
            disease_results = {k: v for k, v in self.results.items() if v['disease'] == disease}

            binary_key = f"binary_{disease.replace(' ', '_').lower()}"
            multiclass_key = f"multiclass_{disease.replace(' ', '_').lower()}"

            binary_metrics = disease_results.get(binary_key)
            multiclass_metrics = disease_results.get(multiclass_key)

            comparison_data.append({
                'Disease': disease,
                'Model': 'Binary',
                'Per-class Dice': f"{binary_metrics['dice_mean']:.3f}" if binary_metrics else "N/A",
                'Per-class IoU': f"{binary_metrics['iou_mean']:.3f}" if binary_metrics else "N/A",
                'Detection F1 @0.2': f"{binary_metrics['detection_f1_lenient_mean']:.3f}" if binary_metrics else "N/A",
                'Detection F1 @0.5': f"{binary_metrics['detection_f1_strict_mean']:.3f}" if binary_metrics else "N/A",
                'Precision @0.2': f"{binary_metrics['detection_precision_lenient_mean']:.3f}" if binary_metrics else "N/A",
                'Recall @0.2': f"{binary_metrics['detection_recall_lenient_mean']:.3f}" if binary_metrics else "N/A",
                'Precision @0.5': f"{binary_metrics['detection_precision_strict_mean']:.3f}" if binary_metrics else "N/A",
                'Recall @0.5': f"{binary_metrics['detection_recall_strict_mean']:.3f}" if binary_metrics else "N/A"
            })

            comparison_data.append({
                'Disease': disease,
                'Model': 'Multiclass',
                'Per-class Dice': f"{multiclass_metrics['dice_mean']:.3f}" if multiclass_metrics else "N/A",
                'Per-class IoU': f"{multiclass_metrics['iou_mean']:.3f}" if multiclass_metrics else "N/A",
                'Detection F1 @0.2': f"{multiclass_metrics['detection_f1_lenient_mean']:.3f}" if multiclass_metrics else "N/A",
                'Detection F1 @0.5': f"{multiclass_metrics['detection_f1_strict_mean']:.3f}" if multiclass_metrics else "N/A",
                'Precision @0.2': f"{multiclass_metrics['detection_precision_lenient_mean']:.3f}" if multiclass_metrics else "N/A",
                'Recall @0.2': f"{multiclass_metrics['detection_recall_lenient_mean']:.3f}" if multiclass_metrics else "N/A",
                'Precision @0.5': f"{multiclass_metrics['detection_precision_strict_mean']:.3f}" if multiclass_metrics else "N/A",
                'Recall @0.5': f"{multiclass_metrics['detection_recall_strict_mean']:.3f}" if multiclass_metrics else "N/A"
            })

        return pd.DataFrame(comparison_data)

    def save_results(self, output_path):
        """
        Save evaluation results to CSV and JSON files.

        Args:
            output_path: Directory to save results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        df = self.generate_comparison_table()
        csv_path = output_path / 'model_comparison_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved to: {csv_path}")

        # Save detailed results as JSON
        with open(output_path / 'detailed_metrics.json', 'w') as f:
            json_ready = {}
            for key, value in self.results.items():
                json_ready[key] = value
            json.dump(json_ready, f, indent=2)