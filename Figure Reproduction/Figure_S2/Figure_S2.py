"""
Lesion analysis and visualization for barley disease segmentation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from skimage import measure
import warnings
import torch
from barley_disease_segmentation.dataset import BarleyLeafDataset
from barley_disease_segmentation.config import *

warnings.filterwarnings('ignore')

# ========= CONFIG =========
SAVE_DIR = (".")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= PROFESSIONAL STYLING =========
plt.style.use('default')
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'axes.edgecolor': '.4',
    'axes.linewidth': 1.2
})

COLORS = {
    'brown_rust': '#C44E52',  # Muted red
    'ramularia': '#8172B2',  # Muted purple
    'train': '#4C72B0',
    'val': '#DD8452',
    'test': '#55A868'
}

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


# ========= LESION ANALYSIS FUNCTIONS =========

def extract_lesion_properties(mask, image, disease_class):
    """
    Extract individual lesion properties from segmentation mask and corresponding image.

    Args:
        mask: Segmentation mask (0=background, 1=brown_rust, 2=ramularia)
        image: RGB image tensor [3, H, W] or numpy array
        disease_class: Either 'brown_rust' (value 1) or 'ramularia' (value 2)

    Returns:
        List of dictionaries with lesion properties
    """
    if hasattr(mask, 'numpy'):
        mask_np = mask.numpy()
    else:
        mask_np = mask

    if hasattr(image, 'numpy'):
        image_np = image.numpy().transpose(1, 2, 0)  # [H, W, 3]
    else:
        image_np = image.transpose(1, 2, 0)

    # Get class value
    class_value = 1 if disease_class == 'brown_rust' else 2

    # Create binary mask for the specific disease
    disease_mask = (mask_np == class_value).astype(np.uint8)

    # Find connected components
    labeled_mask, num_features = measure.label(disease_mask, return_num=True, connectivity=2)

    lesions = []

    for region in measure.regionprops(labeled_mask, intensity_image=image_np):
        # Skip very small lesions
        if region.area < 5:  # pixels
            continue

        # Extract color features from the original image within this lesion
        min_row, min_col, max_row, max_col = region.bbox
        lesion_rgb = image_np[min_row:max_row, min_col:max_col]
        lesion_mask = labeled_mask[min_row:max_row, min_col:max_col] == region.label

        # Get RGB values only within the lesion
        r_vals = lesion_rgb[lesion_mask, 0]
        g_vals = lesion_rgb[lesion_mask, 1]
        b_vals = lesion_rgb[lesion_mask, 2]

        if len(r_vals) == 0:
            continue

        lesion_data = {
            'disease': disease_class,
            'size_pixels': region.area,
            'size_mm2': region.area * 0.04,
            'r_mean': np.mean(r_vals),
            'g_mean': np.mean(g_vals),
            'b_mean': np.mean(b_vals),
            'r_std': np.std(r_vals),
            'g_std': np.std(g_vals),
            'b_std': np.std(b_vals),
            'perimeter': region.perimeter,
            'solidity': region.solidity,
            'eccentricity': region.eccentricity
        }
        lesions.append(lesion_data)

    return lesions


def analyze_test_set_lesions(dataset, max_samples=1000):
    """
    Analyze lesion properties from the test set only.

    Args:
        dataset: Your BarleyLeafDataset instance (test set)
        max_samples: Maximum number of patches to process

    Returns:
        DataFrame with lesion properties
    """
    all_lesions = []
    processed_patches = 0

    print(" Analyzing lesions in test set...")

    for idx in range(min(len(dataset), max_samples)):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{min(len(dataset), max_samples)} patches...")

        try:
            image, mask, metadata, background_mask = dataset[idx]

            # Skip patches with no disease (only background/healthy)
            unique_labels = torch.unique(mask)
            has_brown_rust = 1 in unique_labels
            has_ramularia = 2 in unique_labels

            # Extract brown rust lesions
            if has_brown_rust:
                brown_rust_lesions = extract_lesion_properties(mask, image, 'brown_rust')
                for lesion in brown_rust_lesions:
                    lesion.update({
                        'patch_id': idx,
                        'image_id': metadata['orig_image_id']
                    })
                all_lesions.extend(brown_rust_lesions)

            # Extract ramularia lesions
            if has_ramularia:
                ramularia_lesions = extract_lesion_properties(mask, image, 'ramularia')
                for lesion in ramularia_lesions:
                    lesion.update({
                        'patch_id': idx,
                        'image_id': metadata['orig_image_id']
                    })
                all_lesions.extend(ramularia_lesions)

            processed_patches += 1

        except Exception as e:
            print(f"Warning: Could not process patch {idx}: {e}")
            continue

    print(f" Analyzed {len(all_lesions)} lesions from {processed_patches} patches")

    return pd.DataFrame(all_lesions)


def create_combined_panel(lesion_df):
    """Create a combined panel with color_comparison_combined on left, and lesion_size_boxplot + color_space_2d stacked on right"""

    # Create large figure
    fig = plt.figure(figsize=(18, 10))

    # Create a grid
    # Left: 2x2 grid equivalent, Right: 2 individual plots of same height as left subplots
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                          hspace=0.5, wspace=0.5)

    # Left: color_comparison_combined (2x2 grid)
    ax1 = fig.add_subplot(gs[0, 0])  # top-left
    ax2 = fig.add_subplot(gs[0, 1])  # top-right
    ax3 = fig.add_subplot(gs[1, 0])  # bottom-left
    ax4 = fig.add_subplot(gs[1, 1])  # bottom-right

    # Right: individual plots (each takes one row of the right column)
    ax5 = fig.add_subplot(gs[0, 2])  # lesion_size_boxplot (top-right)
    ax6 = fig.add_subplot(gs[1, 2])  # color_space_2d (bottom-right)

    # LEFT SIDE
    lesion_df['brightness'] = (lesion_df['r_mean'] + lesion_df['g_mean'] + lesion_df['b_mean']) / 3
    lesion_df['red_green_ratio'] = lesion_df['r_mean'] / (lesion_df['g_mean'] + 1e-8)

    features_to_plot = [
        ('brightness', 'Brightness'),
        ('red_green_ratio', 'Red-Green Ratio'),
        ('r_mean', 'Red Channel Intensity'),
        ('g_mean', 'Green Channel Intensity')
    ]

    left_axes = [ax1, ax2, ax3, ax4]

    for idx, (feature, title) in enumerate(features_to_plot):
        sns.violinplot(data=lesion_df, x='disease', y=feature,
                       palette=[COLORS['brown_rust'], COLORS['ramularia']],
                       ax=left_axes[idx], saturation=0.8)

        brown_vals = lesion_df[lesion_df['disease'] == 'brown_rust'][feature]
        ramularia_vals = lesion_df[lesion_df['disease'] == 'ramularia'][feature]
        stat, p_value = stats.mannwhitneyu(brown_vals, ramularia_vals, alternative='two-sided')

        y_max = max(brown_vals.max(), ramularia_vals.max()) * 1.2
        left_axes[idx].plot([0, 0, 1, 1], [y_max * 0.95, y_max, y_max, y_max * 0.95], 'k-', lw=1)

        left_axes[idx].set_title(title, fontweight='bold', pad=12, fontsize=11)
        left_axes[idx].set_xlabel('Disease Type', fontsize=10)
        left_axes[idx].set_ylabel(title, fontsize=10)
        left_axes[idx].tick_params(axis='both', which='major', labelsize=9)

    # Add panel label for the entire left side
    fig.text(0.02, 0.98, 'A', fontsize=16, fontweight='bold',
             va='top', ha='left', transform=fig.transFigure)

    # RIGHT TOP
    sns.boxplot(data=lesion_df, x='disease', y='size_pixels',
                palette=[COLORS['brown_rust'], COLORS['ramularia']],
                ax=ax5, width=0.6, fliersize=2)


    ax5.set_yscale('log')
    ax5.set_ylabel('Size (pixels) - Log scale', fontsize=11)
    ax5.set_xlabel('Disease Type', fontsize=11)
    ax5.set_title('B) Lesion Size Distribution', fontweight='bold', pad=15, fontsize=12)
    ax5.tick_params(axis='both', which='major', labelsize=10)

    # RIGHT BOTTOM
    # Sample points for better visualization
    if len(lesion_df) > 1000:
        plot_df = lesion_df.groupby('disease').apply(lambda x: x.sample(n=min(500, len(x)))).reset_index(drop=True)
    else:
        plot_df = lesion_df

    scatter = sns.scatterplot(data=plot_df, x='r_mean', y='g_mean', hue='disease',
                              palette=[COLORS['brown_rust'], COLORS['ramularia']],
                              alpha=0.6, s=30, ax=ax6)

    ax6.set_xlabel('Mean Red Intensity', fontsize=11)
    ax6.set_ylabel('Mean Green Intensity', fontsize=11)
    ax6.set_title('C) 2D Color Space Analysis', fontweight='bold', pad=15, fontsize=12)
    ax6.tick_params(axis='both', which='major', labelsize=10)

    # Add ellipses for confidence intervals
    for disease, color in [('brown_rust', COLORS['brown_rust']), ('ramularia', COLORS['ramularia'])]:
        subset = lesion_df[lesion_df['disease'] == disease]
        if len(subset) > 10:  # Only plot if we have enough points
            cov = np.cov(subset[['r_mean', 'g_mean']].T)
            mean = subset[['r_mean', 'g_mean']].mean().values

            # Plot confidence ellipse
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = plt.matplotlib.patches.Ellipse(xy=mean, width=lambda_[0] * 2, height=lambda_[1] * 2,
                                                 angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                                                 edgecolor=color, facecolor='none', linewidth=2, linestyle='--')
            ax6.add_patch(ell)

    ax6.legend(title='Disease Type', frameon=True, fancybox=False, framealpha=0.9,
               edgecolor='gray', loc='lower right', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save the combined panel
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(SAVE_DIR, f"combined_analysis_panel.{ext}"),
                    dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined panel saved: combined_analysis_panel")

    plt.close()



def main():
    """Main function to generate lesion analysis plots."""
    print("[INFO] Starting lesion analysis...")

    test_dataset = BarleyLeafDataset(
        all_genotypes_dir=TEST_DATA_DIR,
        genotypes_list=TEST_GENOTYPES,
        task='multiclass',
        augmentations=None,
        standardize=True,
        exclude_invalid=True,
        calculate_weights=False
    )
    lesion_df = analyze_test_set_lesions(test_dataset, max_samples=1000)

    # Generate the combined panel
    print("\n Generating combined panel...")
    create_combined_panel(lesion_df)

    # Print summary statistics
    print("\n Lesion Statistics:")
    for disease in ['brown_rust', 'ramularia']:
        subset = lesion_df[lesion_df['disease'] == disease]
        print(f"\n{disease.replace('_', ' ').title()}:")
        print(f"  Number of lesions: {len(subset)}")
        print(f"  Size (pixels): {subset['size_pixels'].mean():.1f} ± {subset['size_pixels'].std():.1f}")
        # Calculate brightness for summary
        subset_brightness = (subset['r_mean'] + subset['g_mean'] + subset['b_mean']) / 3
        print(f"  Brightness: {subset_brightness.mean():.1f} ± {subset_brightness.std():.1f}")
        print(f"  Red intensity: {subset['r_mean'].mean():.1f} ± {subset['r_mean'].std():.1f}")
        print(f"  Green intensity: {subset['g_mean'].mean():.1f} ± {subset['g_mean'].std():.1f}")

    print(f"\n Combined panel saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()