"""
Visualization utilities for barley disease segmentation results.
"""

import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import cv2

__all__ = ['VisualizationModule']

class VisualizationModule:
    """Visualization module for generating plots and saving predictions."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _create_misclassification_overlay(self, gt_mask, pred_mask, background_mask=None):
        """Create misclassification overlay that excludes background areas"""
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)

        if background_mask is None:
            valid_areas = np.ones_like(gt_mask, dtype=bool)
        else:
            valid_areas = ~background_mask

        if self.pipeline.best_hparams['task'] in ['binary_rust', 'binary_ram']:
            overlay = self._create_misclassification_overlay_binary(gt_mask, pred_mask, valid_areas)
        else:
            overlay = self._create_misclassification_overlay_multiclass(gt_mask, pred_mask, valid_areas)

        return overlay

    def _create_misclassification_overlay_multiclass(self, gt_mask, pred_mask, valid_areas):
        """Multiclass overlay that only colors valid leaf areas"""
        overlay = np.ones((*gt_mask.shape, 3), dtype=np.float32)

        colors = {
            'correct_background': [0.96, 0.96, 0.96],
            'correct_rust': [0.63, 0.63, 0.63],
            'correct_ramularia': [0.38, 0.38, 0.38],
            'over_rust': [0.89, 0.45, 0.36],
            'over_ramularia': [0.49, 0.36, 0.61],
            'under_rust': [0.96, 0.66, 0.71],
            'under_ramularia': [0.78, 0.64, 0.78],
            'rust_to_ram': [0.54, 0.60, 0.36],
            'ram_to_rust': [0.83, 0.69, 0.22],
        }

        correct_bg = (gt_mask == 0) & (pred_mask == 0) & valid_areas
        correct_rust = (gt_mask == 1) & (pred_mask == 1) & valid_areas
        correct_ram = (gt_mask == 2) & (pred_mask == 2) & valid_areas
        over_rust = (gt_mask == 0) & (pred_mask == 1) & valid_areas
        over_ram = (gt_mask == 0) & (pred_mask == 2) & valid_areas
        under_rust = (gt_mask == 1) & (pred_mask == 0) & valid_areas
        under_ram = (gt_mask == 2) & (pred_mask == 0) & valid_areas
        rust_to_ram = (gt_mask == 1) & (pred_mask == 2) & valid_areas
        ram_to_rust = (gt_mask == 2) & (pred_mask == 1) & valid_areas

        overlay[correct_bg] = colors['correct_background']
        overlay[correct_rust] = colors['correct_rust']
        overlay[correct_ram] = colors['correct_ramularia']
        overlay[over_rust] = colors['over_rust']
        overlay[over_ram] = colors['over_ramularia']
        overlay[under_rust] = colors['under_rust']
        overlay[under_ram] = colors['under_ramularia']
        overlay[rust_to_ram] = colors['rust_to_ram']
        overlay[ram_to_rust] = colors['ram_to_rust']

        return overlay

    def _create_misclassification_overlay_binary(self, gt_mask, pred_mask, valid_areas):
        """Binary task misclassification overlay that only colors valid leaf areas"""
        overlay = np.ones((*gt_mask.shape, 3), dtype=np.float32)

        colors, lesion_name = self._get_binary_colors_and_labels()

        correct_bg = (gt_mask == 0) & (pred_mask == 0) & valid_areas
        correct_lesion = (gt_mask == 1) & (pred_mask == 1) & valid_areas
        over_seg = (gt_mask == 0) & (pred_mask == 1) & valid_areas
        under_seg = (gt_mask == 1) & (pred_mask == 0) & valid_areas

        overlay[correct_bg] = colors['correct_background']
        overlay[correct_lesion] = colors['correct_lesion']
        overlay[over_seg] = colors['over_segmentation']
        overlay[under_seg] = colors['under_segmentation']

        return overlay

    def _get_binary_colors_and_labels(self):
        """Get task-specific colors and lesion name for binary tasks"""
        if self.pipeline.best_hparams['task'] == 'binary_rust':
            colors = {
                'correct_background': [0.96, 0.96, 0.96],
                'correct_lesion': [0.63, 0.63, 0.63],
                'over_segmentation': [0.89, 0.45, 0.36],
                'under_segmentation': [0.96, 0.66, 0.71],
                'gt_lesion': [0.8, 0.2, 0.2],
            }
            lesion_name = "Brown Rust"
        else:
            colors = {
                'correct_background': [0.96, 0.96, 0.96],
                'correct_lesion': [0.38, 0.38, 0.38],
                'over_segmentation': [0.49, 0.36, 0.61],
                'under_segmentation': [0.78, 0.64, 0.78],
                'gt_lesion': [0.2, 0.4, 0.8],
            }
            lesion_name = "Ramularia"

        return colors, lesion_name

    def _add_organized_legend(self, fig):
        """Add legend that works for both multiclass and binary tasks"""
        if self.pipeline.best_hparams['task'] in ['binary_rust', 'binary_ram']:
            self._add_binary_legend(fig)
        else:
            self._add_multiclass_legend(fig)

    def _add_binary_legend(self, fig):
        """Binary task legend with task-specific labels"""
        colors, lesion_name = self._get_binary_colors_and_labels()

        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['correct_background'], markersize=12,
                       label='Correct: Background'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['correct_lesion'], markersize=12,
                       label=f'Correct: {lesion_name}'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['gt_lesion'], markersize=12,
                       label=f'GT: {lesion_name}'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['over_segmentation'], markersize=12,
                       label=f'Over: Bkg → {lesion_name}'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['under_segmentation'], markersize=12,
                       label=f'Under: {lesion_name} → Bkg'),
        ]

        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11,
                   frameon=True, fancybox=True, shadow=True,
                   title="Legend Categories", title_fontsize=12)

    def _add_multiclass_legend(self, fig):
        """Add organized multiclass legend with grouped categories"""
        colors = {
            'correct_background': [0.96, 0.96, 0.96],
            'correct_rust': [0.63, 0.63, 0.63],
            'correct_ramularia': [0.38, 0.38, 0.38],
            'gt_rust': [0.8, 0.2, 0.2],
            'gt_ramularia': [0.2, 0.4, 0.8],
            'over_rust': [0.89, 0.45, 0.36],
            'under_rust': [0.96, 0.66, 0.71],
            'rust_to_ram': [0.54, 0.60, 0.36],
            'over_ramularia': [0.49, 0.36, 0.61],
            'under_ramularia': [0.78, 0.64, 0.78],
            'ram_to_rust': [0.83, 0.69, 0.22],
        }

        legend_elements = []

        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['correct_background'], markersize=12,
                       label='Correct: Background'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['correct_rust'], markersize=12,
                       label='Correct: Rust'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['correct_ramularia'], markersize=12,
                       label='Correct: Ramularia'),
        ])

        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['gt_rust'], markersize=12,
                       label='GT: Rust'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['gt_ramularia'], markersize=12,
                       label='GT: Ramularia'),
        ])

        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['over_rust'], markersize=12,
                       label='Over: Bkg → Rust'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['under_rust'], markersize=12,
                       label='Under: Rust → Bkg'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['rust_to_ram'], markersize=12,
                       label='Cross: Rust → Ram'),
        ])

        legend_elements.extend([
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['over_ramularia'], markersize=12,
                       label='Over: Bkg → Ram'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['under_ramularia'], markersize=12,
                       label='Under: Ram → Bkg'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=colors['ram_to_rust'], markersize=12,
                       label='Cross: Ram → Rust'),
        ])

        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11,
                   frameon=True, fancybox=True, shadow=True,
                   title="Legend Categories", title_fontsize=12)

    def _mask_to_colored_image(self, mask, mask_type='gt'):
        """Convert mask to colored image for both multiclass and binary tasks"""
        colored = np.zeros((*mask.shape, 3), dtype=np.float32)

        if self.pipeline.best_hparams['task'] in ['binary_rust', 'binary_ram']:
            colors, lesion_name = self._get_binary_colors_and_labels()

            if mask_type == 'gt':
                colored[mask == 0] = [1.0, 1.0, 1.0]
                colored[mask == 1] = colors['gt_lesion']
            else:
                colored[mask == 0] = [1.0, 1.0, 1.0]
                colored[mask == 1] = colors['gt_lesion']
        else:
            if mask_type == 'gt':
                colored[mask == 0] = [1.0, 1.0, 1.0]
                colored[mask == 1] = [0.8, 0.2, 0.2]
                colored[mask == 2] = [0.2, 0.4, 0.8]
            else:
                colored[mask == 0] = [1.0, 1.0, 1.0]
                colored[mask == 1] = [0.8, 0.2, 0.2]
                colored[mask == 2] = [0.2, 0.4, 0.8]

        return colored

    def _generate_misclassification_plots(self, test_dataset, edge_cases, output_dir):
        """Generate misclassification plots for ALL leaves in the test dataset"""
        print("Generating misclassification plots for all leaves...")

        plots_dir = output_dir / "misclassification_plots"
        plots_dir.mkdir(exist_ok=True)

        # Get ALL patch IDs from the test dataset
        all_patch_ids = []
        leaf_to_patches = defaultdict(list)

        # Extract patch information directly from the dataset
        for i in range(len(test_dataset)):
            try:
                # Get sample to extract metadata
                sample = test_dataset[i]
                metadata = sample[2]

                img_name = metadata["img_name"]  # Full patch ID
                orig_image_id = metadata["orig_image_id"]  # This should be genotype_leafid

                all_patch_ids.append(img_name)

                # Group by original image ID (genotype_leafid)
                leaf_to_patches[orig_image_id].append(img_name)

            except Exception as e:
                print(f"Warning: Could not process sample {i}: {e}")
                continue

        print(f"Found {len(all_patch_ids)} total patches in test dataset")
        print(f"Found {len(leaf_to_patches)} unique leaves")

        # Sort leaves
        all_leaves = sorted(leaf_to_patches.items(), key=lambda x: len(x[0]), reverse=True)

        print(f"Generating plots for all {len(all_leaves)} leaves...")
        for leaf_id, patch_ids in all_leaves:
            self._plot_leaf_misclassification(test_dataset, leaf_id, patch_ids, plots_dir)

        print(f"Successfully generated misclassification plots for all leaves!")

    def _stitch_leaf_patches(self, test_dataset, leaf_patches):
        """Stitch all patches together to reconstruct the whole leaf"""
        max_x = max(p['x_offset'] for p in leaf_patches) + 512
        max_y = max(p['y_offset'] for p in leaf_patches) + 512

        leaf_image = np.ones((max_y, max_x, 3), dtype=np.float32)
        leaf_gt = np.zeros((max_y, max_x), dtype=np.uint8)
        leaf_pred = np.zeros((max_y, max_x), dtype=np.uint8)
        overlap_count = np.ones((max_y, max_x), dtype=np.int32)

        for patch_data in leaf_patches:
            try:
                patch_idx = test_dataset.patches.index(patch_data)
                img_tensor, mask_tensor, metadata, bg_mask = test_dataset[patch_idx]

                img_batch = img_tensor.unsqueeze(0).to(self.pipeline.device)
                with torch.no_grad():
                    output = self.pipeline.model(img_batch)
                    pred_tensor = torch.argmax(output, dim=1).squeeze(0)

                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                mask_np = mask_tensor.cpu().numpy()
                pred_np = pred_tensor.cpu().numpy()

                img_denorm = self._denormalize_image(img_np, test_dataset.mean, test_dataset.std)

                x_start = patch_data['x_offset']
                y_start = patch_data['y_offset']
                x_end = x_start + 512
                y_end = y_start + 512

                if bg_mask is not None:
                    bg_mask_np = bg_mask.cpu().numpy()
                    valid_mask = ~bg_mask_np
                else:
                    valid_mask = np.ones_like(mask_np, dtype=bool)

                patch_region = leaf_image[y_start:y_end, x_start:x_end]
                patch_weight = valid_mask[..., None].astype(np.float32)
                existing_weight = 1.0 - patch_weight

                leaf_image[y_start:y_end, x_start:x_end] = (
                        patch_region * existing_weight +
                        img_denorm * patch_weight
                )

                leaf_gt[y_start:y_end, x_start:x_end] = np.where(valid_mask, mask_np,
                                                                 leaf_gt[y_start:y_end, x_start:x_end])
                leaf_pred[y_start:y_end, x_start:x_end] = np.where(valid_mask, pred_np,
                                                                   leaf_pred[y_start:y_end, x_start:x_end])

                overlap_count[y_start:y_end, x_start:x_end] += valid_mask.astype(int)

            except Exception as e:
                print(f"    Error processing patch {patch_data['img_name']}: {e}")
                continue

        leaf_image = np.clip(leaf_image, 0, 1)
        background_mask = (overlap_count == 1)

        return leaf_image, leaf_gt, leaf_pred, background_mask

    def _denormalize_image(self, img_np, mean, std):
        """Denormalize image for visualization"""
        if mean is not None and std is not None:
            img_denorm = img_np.copy()
            for i in range(3):
                img_denorm[..., i] = img_denorm[..., i] * std[i] + mean[i]
            img_denorm = np.clip(img_denorm, 0, 1)
            return img_denorm
        return img_np

    def _plot_leaf_misclassification(self, test_dataset, leaf_id, patch_ids, plots_dir):
        """Plot whole-leaf misclassification with three-panel layout for publications"""
        try:
            all_leaf_patches = [p for p in test_dataset.patches if p['orig_image_id'] == leaf_id]

            if not all_leaf_patches:
                print(f"  No patches found for leaf {leaf_id}")
                return

            leaf_image, leaf_gt, leaf_pred, background_mask = self._stitch_leaf_patches(test_dataset, all_leaf_patches)

            if leaf_image is None:
                print(f"  Failed to stitch leaf {leaf_id}")
                return

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 20))

            fig.patch.set_facecolor('white')
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor('white')

            ax1.imshow(leaf_image)
            ax1.set_title(f"Original Leaf: {leaf_id}", fontsize=18, pad=20, fontweight='bold')
            ax1.axis('off')

            gt_overlay = self._mask_to_colored_image(leaf_gt, mask_type='gt')
            ax2.imshow(leaf_image, alpha=0.4)
            ax2.imshow(gt_overlay, alpha=0.8)
            ax2.set_title("Ground Truth Annotations", fontsize=18, pad=20, fontweight='bold')
            ax2.axis('off')

            misclassification_overlay = self._create_misclassification_overlay(leaf_gt, leaf_pred, background_mask)
            ax3.imshow(leaf_image, alpha=0.3)
            ax3.imshow(misclassification_overlay, alpha=0.9)
            ax3.set_title("Misclassification Analysis", fontsize=18, pad=20, fontweight='bold')
            ax3.axis('off')

            self._add_organized_legend(fig)

            plt.tight_layout()

            base_path = plots_dir / f"{leaf_id}_whole_leaf_analysis"
            plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight',
                        facecolor='white', transparent=False)
            plt.savefig(f"{base_path}.pdf", bbox_inches='tight',
                        facecolor='white', transparent=False)
            plt.close()

            print(f"    Saved whole-leaf analysis: {base_path}.png/.pdf")

        except Exception as e:
            print(f"  Failed to plot whole-leaf analysis for {leaf_id}: {e}")
            import traceback
            traceback.print_exc()

    def save_stitched_predictions(self, test_dataset, output_dir):
        """Save stitched whole-leaf predictions for the entire test set"""
        print("Saving stitched whole-leaf predictions...")

        predictions_dir = output_dir / "saved_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        data_dir = predictions_dir / "data"
        labels_dir = predictions_dir / "labels"
        predictions_dir_sub = predictions_dir / "predictions"

        data_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        predictions_dir_sub.mkdir(exist_ok=True)

        all_leaves = set(p['orig_image_id'] for p in test_dataset.patches)

        print(f"Found {len(all_leaves)} unique leaves in test set")
        print(f"Task type: {self.pipeline.best_hparams['task']}")

        for leaf_id in sorted(all_leaves):
            try:
                print(f"  Processing leaf: {leaf_id}")
                leaf_patches = [p for p in test_dataset.patches if p['orig_image_id'] == leaf_id]

                leaf_image, leaf_gt, leaf_pred, background_mask = self._stitch_leaf_patches(test_dataset, leaf_patches)

                if leaf_image is None:
                    print(f"    Failed to stitch leaf {leaf_id}")
                    continue

                self._save_individual_components(leaf_image, leaf_gt, leaf_pred, leaf_id, data_dir, labels_dir,
                                                 predictions_dir_sub, background_mask, predictions_dir)
                print(f"    Saved stitched predictions for {leaf_id}")

            except Exception as e:
                print(f"    Error processing leaf {leaf_id}: {e}")
                continue

        print(f"All stitched predictions saved to: {predictions_dir}")
        print(f"  - Original images: {data_dir}")
        print(f"  - Ground truth: {labels_dir}")
        print(f"  - Predictions: {predictions_dir_sub}")

    def _save_individual_components(self, leaf_image, leaf_gt, leaf_pred, leaf_id, data_dir, labels_dir,
                                    predictions_dir, background_mask, pred_dir):
        """Save individual stitched components as binary PNG files in organized folders"""

        # Save original image with cv2 to preserve dimensions
        # Convert from float [0,1] to uint8 [0,255] for cv2
        if leaf_image.dtype == np.float32:
            leaf_image_uint8 = (leaf_image * 255).astype(np.uint8)
        else:
            leaf_image_uint8 = leaf_image.astype(np.uint8)

        # Convert RGB to BGR if needed
        if len(leaf_image_uint8.shape) == 3 and leaf_image_uint8.shape[2] == 3:
            leaf_image_cv = cv2.cvtColor(leaf_image_uint8, cv2.COLOR_RGB2BGR)
        else:
            leaf_image_cv = leaf_image_uint8

        cv2.imwrite(str(data_dir / f"{leaf_id}.png"), leaf_image_cv)

        # Apply background mask to both ground truth and predictions
        leaf_gt_masked = leaf_gt.copy()
        leaf_pred_masked = leaf_pred.copy()
        leaf_gt_masked[background_mask] = 0
        leaf_pred_masked[background_mask] = 0

        # Handle different task types
        task_type = self.pipeline.best_hparams['task']

        if task_type == 'multiclass':
            print(f"    Saving multiclass predictions for {leaf_id}")
            cv2.imwrite(str(labels_dir / f"{leaf_id}.png"), leaf_gt_masked)
            cv2.imwrite(str(predictions_dir / f"{leaf_id}.png"), leaf_pred_masked)
        else:
            print(f"    Saving binary predictions for {leaf_id}")
            gt_binary = leaf_gt_masked.astype(np.uint8)
            pred_binary = leaf_pred_masked.astype(np.uint8)
            cv2.imwrite(str(labels_dir / f"{leaf_id}.png"), gt_binary)
            cv2.imwrite(str(predictions_dir / f"{leaf_id}.png"), pred_binary)