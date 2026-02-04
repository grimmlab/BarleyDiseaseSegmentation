#!/usr/bin/env python3
"""
Disease area correlation analysis for barley disease segmentation.
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse
import warnings
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from barley_disease_segmentation.utils import get_save_path
from barley_disease_segmentation.config import *
from scipy import stats


class AreaCorrelationPlotter:
    """Area correlation plotter for disease area analysis."""

    def __init__(self, min_presence_pixels: int = 10):
        self.min_presence_pixels = min_presence_pixels
        self.area_data = []

    def _load_gray_mask(self, path: Path) -> np.ndarray:
        """Load grayscale mask."""
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return mask

    def load_masks(self, predictions_path: Path, leaf_id: str, task_type: str, disease_class: int = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """Load ground truth and prediction masks."""
        labels_path = predictions_path / "labels" / f"{leaf_id}.png"
        pred_path = predictions_path / "predictions" / f"{leaf_id}.png"

        gt_mask = self._load_gray_mask(labels_path)
        pred_mask = self._load_gray_mask(pred_path)

        if gt_mask is None or pred_mask is None:
            return None, None

        if gt_mask.shape != pred_mask.shape:
            try:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception:
                return None, None

        if task_type == "multiclass" and disease_class is not None:
            gt_bin = (gt_mask == disease_class).astype(np.uint8)
            pred_bin = (pred_mask == disease_class).astype(np.uint8)
            return gt_bin, pred_bin

        gt_bin = (gt_mask > 0).astype(np.uint8)
        pred_bin = (pred_mask > 0).astype(np.uint8)
        return gt_bin, pred_bin

    def load_original_rgb_image(self, leaf_id: str, predictions_path: Path) -> np.ndarray:
        """Load original RGB image from saved_predictions/data directory."""
        data_folder = predictions_path / "data"

        if not data_folder.exists():
            return None

        image_path = data_folder / f"{leaf_id}.png"
        if image_path.exists():
            image = cv2.imread(str(image_path))
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for image_file in data_folder.glob("*"):
            if leaf_id in image_file.stem:
                image = cv2.imread(str(image_file))
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return None

    def calculate_leaf_area_from_rgb(self, rgb_image: np.ndarray) -> int:
        """Calculate leaf area from RGB image by detecting non-white pixels."""
        if rgb_image is None:
            return 0

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        _, leaf_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

        return np.sum(leaf_mask > 0)

    def extract_genotype_from_leaf_id(self, leaf_id: str) -> str:
        """Extract genotype from leaf_id (format: genotype_leafid.png)."""
        return leaf_id.split('_')[0]

    def collect_area_data(self, model_configs: Dict, common_leaf_ids: List[str]):
        """Collect area data for all model configurations."""
        print("Collecting area data across all models...")

        for config_name, config in model_configs.items():
            model_predictions_path = config['predictions_path']
            model_name = config['model_name']
            task_type = config['task_type']
            disease_class = config.get('disease_class')
            disease_name = config['disease_name']

            print(f"Processing {model_name} on {disease_name}...")

            for i, leaf_id in enumerate(common_leaf_ids):
                try:
                    gt_mask, pred_mask = self.load_masks(model_predictions_path, leaf_id,
                                                         task_type, disease_class)

                    if gt_mask is None or pred_mask is None:
                        continue

                    rgb_image = self.load_original_rgb_image(leaf_id, model_predictions_path)
                    if rgb_image is None:
                        continue

                    total_leaf_pixels = self.calculate_leaf_area_from_rgb(rgb_image)
                    if total_leaf_pixels == 0:
                        continue

                    gt_area_pct = (np.sum(gt_mask > 0) / total_leaf_pixels) * 100.0
                    pred_area_pct = (np.sum(pred_mask > 0) / total_leaf_pixels) * 100.0
                    genotype = self.extract_genotype_from_leaf_id(leaf_id)

                    self.area_data.append({
                        'model': model_name,
                        'disease': disease_name,
                        'task_type': task_type,
                        'leaf_id': leaf_id,
                        'genotype': genotype,
                        'gt_area_pct': gt_area_pct,
                        'pred_area_pct': pred_area_pct,
                        'total_leaf_pixels': total_leaf_pixels,
                        'predictions_path': str(model_predictions_path)
                    })

                    if (i + 1) % 50 == 0:
                        print(f"  Processed {i + 1}/{len(common_leaf_ids)} leaves")

                except Exception as e:
                    continue

            completed_count = len(
                [d for d in self.area_data if d['model'] == model_name and d['disease'] == disease_name])
            print(f"  Completed {model_name} - {disease_name}: {completed_count} leaves")

    def calculate_correlation_with_pvalue(self, x_vals, y_vals):
        """Calculate Pearson correlation with p-value."""
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)

        valid_mask = (x > 0) & (y > 0) & ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < 3:
            return 0.0, 1.0, len(x_valid)

        try:
            r, p = stats.pearsonr(x_valid, y_valid)
            return r, p, len(x_valid)
        except:
            return 0.0, 1.0, len(x_valid)

    def generate_main_figures(self, output_path: Path):
        """Generate main comparison figures for each disease with log scale."""
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'mathtext.fontset': 'dejavuserif',
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        diseases_found = set([d['disease'] for d in self.area_data])
        model_colors = {'Binary': '#2E8B57', 'Multiclass': '#8B4513'}
        model_markers = {'Binary': 'o', 'Multiclass': 's'}

        score_ranges = {
            1: (0.0, 0.0), 2: (0.0, 2.0), 3: (2.0, 5.0), 4: (5.0, 8.0),
            5: (8.0, 14.0), 6: (14.0, 22.0), 7: (22.0, 37.0), 8: (37.0, 61.0), 9: (61.0, 100.0)
        }

        score_colors = {
            1: '#FFFFFF', 2: '#E6F7FF', 3: '#BAE7FF', 4: '#91D5FF',
            5: '#69C0FF', 6: '#FFCCC7', 7: '#FFA39E', 8: '#FF7875', 9: '#FF4D4F'
        }

        for disease in diseases_found:
            all_disease_data = [d for d in self.area_data if d['disease'] == disease]

            if not all_disease_data:
                print(f"No data found for disease: {disease}, skipping...")
                continue

            all_area_values = []
            for data in all_disease_data:
                all_area_values.append(data['gt_area_pct'])
                all_area_values.append(data['pred_area_pct'])

            if not all_area_values:
                continue

            max_data = max(all_area_values)
            plot_min = 0.01
            plot_max = max(70, max_data * 1.5)

            visual_boundaries = [high for _, high in score_ranges.values() if high > 0 and high <= plot_max]
            all_boundaries = [plot_min] + visual_boundaries
            if plot_max > visual_boundaries[-1] if visual_boundaries else 0:
                all_boundaries.append(plot_max)

            fig, ax = plt.subplots(figsize=(8, 8))

            for boundary in visual_boundaries:
                if boundary <= plot_max:
                    ax.axvline(x=boundary, ymin=0, ymax=1, color='gray', alpha=0.6,
                               linestyle='--', linewidth=1, zorder=1)
                    ax.axhline(y=boundary, xmin=0, xmax=1, color='gray', alpha=0.6,
                               linestyle='--', linewidth=1, zorder=1)

            for i in range(len(all_boundaries) - 1):
                low_x = all_boundaries[i]
                high_x = all_boundaries[i + 1]
                low_y = all_boundaries[i]
                high_y = all_boundaries[i + 1]

                if high_x <= plot_max and high_y <= plot_max:
                    score = None
                    for score_num, (low_range, high_range) in score_ranges.items():
                        if i == 0:
                            score = 2
                        elif low_x >= low_range and high_x <= high_range:
                            score = score_num
                            break

                    if score and score in score_colors:
                        cell_color = score_colors[score]
                        diagonal_cell = plt.Rectangle((low_x, low_y),
                                                      high_x - low_x,
                                                      high_y - low_y,
                                                      facecolor=cell_color, alpha=0.6,
                                                      edgecolor='gray', linewidth=0.5, zorder=1)
                        ax.add_patch(diagonal_cell)

            for model_type in ['Binary', 'Multiclass']:
                plot_data = [d for d in self.area_data
                             if d['disease'] == disease and d['model'] == model_type]

                if not plot_data:
                    continue

                df = pd.DataFrame(plot_data)
                genotype_means = df.groupby('genotype').agg({
                    'gt_area_pct': 'mean',
                    'pred_area_pct': 'mean'
                }).reset_index()

                ax.scatter(genotype_means['gt_area_pct'],
                           genotype_means['pred_area_pct'],
                           c=model_colors[model_type],
                           marker=model_markers[model_type],
                           s=80, alpha=0.8,
                           edgecolors='black',
                           linewidth=0.8,
                           zorder=3,
                           label=f"{model_type} model")

            log_perfect_x = np.logspace(np.log10(plot_min), np.log10(plot_max), 100)
            log_perfect_y = log_perfect_x
            ax.plot(log_perfect_x, log_perfect_y, 'k-', linewidth=2, alpha=0.8,
                    label='Perfect correlation', zorder=2)

            ax.set_xscale('log')
            ax.set_yscale('log')

            disease_title = disease.title()
            ax.set_xlabel('Ground Truth Disease Area (% of leaf) - Log Scale', fontsize=12, fontweight='bold',
                          labelpad=15.0)
            ax.set_ylabel('Predicted Disease Area (% of leaf) - Log Scale', fontsize=12, fontweight='bold',
                          labelpad=15.0)
            ax.set_title(f'{disease_title} - Model Comparison', fontsize=14, fontweight='bold', pad=20)

            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

            ax.set_xticks([0.01, 0.1, 1, 10, 100])
            ax.set_yticks([0.01, 0.1, 1, 10, 100])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

            ax.legend(loc='upper left', frameon=True, fancybox=True,
                      shadow=True, framealpha=0.9)

            safe_disease = disease.replace(' ', '_').lower()
            plt.savefig(output_path / f"main_figure_{safe_disease}_comparison_log.png",
                        dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
            plt.savefig(output_path / f"main_figure_{safe_disease}_comparison_log.pdf",
                        bbox_inches='tight', pad_inches=0.1, facecolor='white')

            plt.close()
            print(f"  Saved LOG plot for {disease}")

        print(f"Main comparison figures saved to: {output_path}")

    def generate_correlation_table(self, output_path: Path):
        """Generate table with r_leaf and r_genotype for each model-task combination."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        table_data = []

        model_configs = [
            ('Binary', 'binary', 'Brown Rust'),
            ('Binary', 'binary', 'Ramularia'),
            ('Multiclass', 'multiclass', 'Brown Rust'),
            ('Multiclass', 'multiclass', 'Ramularia')
        ]

        for model_name, task_type, disease in model_configs:
            plot_data = [d for d in self.area_data if
                         d['model'] == model_name and
                         d['task_type'] == task_type and
                         d['disease'] == disease]

            if not plot_data:
                table_data.append({
                    'Model': model_name,
                    'Task': task_type,
                    'Disease': disease,
                    'r_leaf': 'N/A',
                    'r_genotype': 'N/A',
                    'n_leaves': 0,
                    'n_genotypes': 0
                })
                continue

            df = pd.DataFrame(plot_data)

            # Leaf-level correlation
            valid_df = df[(df['gt_area_pct'] > 0) & (df['pred_area_pct'] > 0)]
            if len(valid_df) > 2:
                leaf_correlation, leaf_p, leaf_n = self.calculate_correlation_with_pvalue(
                    valid_df['gt_area_pct'].values,
                    valid_df['pred_area_pct'].values
                )
            else:
                leaf_correlation, leaf_p, leaf_n = 0, 1.0, 0

            # Genotype-level correlation
            genotype_means = df.groupby('genotype').agg({
                'gt_area_pct': 'mean',
                'pred_area_pct': 'mean'
            }).reset_index()

            valid_genotype_means = genotype_means[
                (genotype_means['gt_area_pct'] > 0) &
                (genotype_means['pred_area_pct'] > 0)
                ]

            if len(valid_genotype_means) > 2:
                genotype_correlation, genotype_p, genotype_n = self.calculate_correlation_with_pvalue(
                    valid_genotype_means['gt_area_pct'].values,
                    valid_genotype_means['pred_area_pct'].values
                )
            else:
                genotype_correlation, genotype_p, genotype_n = 0, 1.0, 0

            table_data.append({
                'Model': model_name,
                'Task': task_type,
                'Disease': disease,
                'r_leaf': f"{leaf_correlation:.3f}",
                'r_genotype': f"{genotype_correlation:.3f}",
                'n_leaves': len(valid_df),
                'n_genotypes': len(valid_genotype_means)
            })

        correlation_table = pd.DataFrame(table_data)

        # Save as CSV
        csv_path = output_path / "correlation_table.csv"
        correlation_table.to_csv(csv_path, index=False)

        # Print table
        print("\n" + "=" * 80)
        print("CORRELATION TABLE: r_leaf and r_genotype by Model-Task Combination")
        print("=" * 80)
        print(correlation_table.to_string(index=False))
        print(f"\nTable saved to: {csv_path}")

        return correlation_table


def find_common_leaf_ids(model_paths: List[Path]) -> List[str]:
    """Find leaf IDs common across all model predictions."""
    all_leaf_sets = []

    for model_path in model_paths:
        labels_path = model_path / "labels"
        if labels_path.exists():
            leaf_ids = set(f.stem for f in labels_path.glob("*.png"))
            all_leaf_sets.append(leaf_ids)
            print(f"  {model_path.name}: {len(leaf_ids)} leaves")
        else:
            print(f"Warning: No labels found in {model_path}")

    if not all_leaf_sets:
        raise ValueError("No valid model paths found")

    common_leaf_ids = set.intersection(*all_leaf_sets)
    print(f"Common leaves across all models: {len(common_leaf_ids)}")

    return sorted(common_leaf_ids)


def main():
    """Main entry point for area correlation analysis."""
    parser = argparse.ArgumentParser(description='Generate disease area correlation plots and correlation tables')
    parser.add_argument('--min_presence_pixels', type=int, default=10,
                        help='Minimum GT pixels to count as disease presence')
    parser.add_argument('--binary_rust_path', type=str, default=None,
                        help='Path to binary rust predictions')
    parser.add_argument('--binary_ram_path', type=str, default=None,
                        help='Path to binary ramularia predictions')
    parser.add_argument('--multiclass_path', type=str, default=None,
                        help='Path to multiclass predictions')
    parser.add_argument('--output', type=str,
                        default=str("Area_Correlation"),
                        help='Output folder for plots')
    args = parser.parse_args()

    # Set up model paths
    if args.binary_rust_path is None:
        binary_rust_path = get_save_path('convnext_tiny', 'binary_rust', 'utils',
                                         subfolder='20251117_1014/saved_predictions')
    else:
        binary_rust_path = args.binary_rust_path

    if args.binary_ram_path is None:
        binary_ram_path = get_save_path('convnext_tiny', 'binary_ram', 'utils',
                                        subfolder='20251117_1002/saved_predictions')
    else:
        binary_ram_path = args.binary_ram_path

    if args.multiclass_path is None:
        multiclass_path = get_save_path('convnext_tiny', 'multiclass', 'utils',
                                        subfolder='20251117_1028/saved_predictions')
    else:
        multiclass_path = args.multiclass_path

    output = args.output
    model_configs = {
        'binary_rust': {
            'predictions_path': Path(binary_rust_path),
            'model_name': 'Binary',
            'task_type': 'binary',
            'disease_name': 'Brown Rust'
        },
        'binary_ramularia': {
            'predictions_path': Path(binary_ram_path),
            'model_name': 'Binary',
            'task_type': 'binary',
            'disease_name': 'Ramularia'
        },
        'multiclass_rust': {
            'predictions_path': Path(multiclass_path),
            'model_name': 'Multiclass',
            'task_type': 'multiclass',
            'disease_class': 1,
            'disease_name': 'Brown Rust'
        },
        'multiclass_ramularia': {
            'predictions_path': Path(multiclass_path),
            'model_name': 'Multiclass',
            'task_type': 'multiclass',
            'disease_class': 2,
            'disease_name': 'Ramularia'
        }
    }

    # Find common leaf IDs
    print("Finding common leaves across all models...")
    model_paths = [config['predictions_path'] for config in model_configs.values()]
    common_leaf_ids = find_common_leaf_ids(model_paths)

    if len(common_leaf_ids) == 0:
        print("ERROR: No common leaves found across all models!")
        return

    # Initialize plotter
    plotter = AreaCorrelationPlotter(min_presence_pixels=args.min_presence_pixels)

    # Collect data
    plotter.collect_area_data(model_configs, common_leaf_ids)

    # 1. Generate main figures

    print("GENERATING MAIN COMPARISON FIGURES")

    plotter.generate_main_figures(Path(output))

    # 2. Generate correlation table

    print("GENERATING CORRELATION TABLE")

    correlation_table = plotter.generate_correlation_table(Path(output))

    print("ANALYSIS COMPLETE!")

    print(f"Output directory: {output}")
    print("Generated files:")
    print(f"  1. main_figure_brown_rust_comparison_log.[png/pdf]")
    print(f"  2. main_figure_ramularia_comparison_log.[png/pdf]")
    print(f"  3. correlation_table.csv")


if __name__ == "__main__":
    main()