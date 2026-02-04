
"""
Dataset analysis and visualization script.
Generates summary statistics and charts for barley disease dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from matplotlib.patches import Patch
import matplotlib as mpl
from io import StringIO
import os
import re
from barley_disease_segmentation.config import *
from barley_disease_segmentation.config import FIGURE_DIR


def aggregate_lesion_counts(df):
    """
    Aggregate lesion counts into Train/Validation/Test categories.

    Args:
        df: DataFrame with 'dataset', 'class_1_lesion_count', 'class_2_lesion_count'

    Returns:
        dict: {split: {'Brown Rust': count, 'Ramularia': count}}
    """
    # Map dataset variants to categories
    dataset_mapping = {
        'train_constant': 'Train',
        'train_reflect': 'Train',
        'val_constant': 'Validation',
        'test_constant': 'Test'
    }

    # Initialize result
    aggregated = {
        'Train': {'Brown Rust': 0, 'Ramularia': 0},
        'Validation': {'Brown Rust': 0, 'Ramularia': 0},
        'Test': {'Brown Rust': 0, 'Ramularia': 0}
    }

    # Sum lesion counts
    for _, row in df.iterrows():
        dataset_name = row['dataset']
        category = dataset_mapping.get(dataset_name)

        if category:
            aggregated[category]['Brown Rust'] += row.get('class_1_lesion_count', 0)
            aggregated[category]['Ramularia'] += row.get('class_2_lesion_count', 0)

    return aggregated

# Set font sizes
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11

# Lesion counts from my data
lesion_csv = pd.read_csv('dataset_stats.csv')
lesion_counts = aggregate_lesion_counts(lesion_csv)


# Pixel data from CSV
csv_data = pd.read_csv(FIGURE_DIR/'Table_1/leaf_only_pixel_counts.csv')


def count_leaves_per_genotype(base_path=DATA_DIR):
    """
    Count leaves per genotype from patch filenames.

    Args:
        base_path: Root dataset directory

    Returns:
        dict: {genotype: leaf_count}
    """
    leaf_counts = {}
    splits = {
        'Train': TRAIN_GENOTYPES,
        'Validation': VAL_GENOTYPES,
        'Test': TEST_GENOTYPES
    }

    for split_name, genotypes in splits.items():
        split_folder = f"{split_name}_data"
        split_path = os.path.join(base_path, split_folder)

        if not os.path.exists(split_path):
            print(f"Warning: {split_folder} not found at {split_path}")
            continue

        for genotype in genotypes:
            # Updated folder name
            cropped_folder = os.path.join(split_path, genotype, f"cropped_leaves_{genotype}_boxed_preprocessed_patches")

            if os.path.exists(cropped_folder):
                # Get all patch files
                files = [f for f in os.listdir(cropped_folder)
                         if os.path.isfile(os.path.join(cropped_folder, f)) and
                         f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))]

                if not files:
                    leaf_counts[genotype] = 0
                    print(f"Warning: No patch files found in {cropped_folder}")
                    continue

                # Extract leaf numbers from filenames
                leaf_numbers = set()
                pattern = r'cropped_leaf_(\d{2})_patch'

                for file_name in files:
                    match = re.search(pattern, file_name)
                    if match:
                        leaf_num = int(match.group(1))
                        leaf_numbers.add(leaf_num)

                # Get the highest leaf number
                if leaf_numbers:
                    max_leaf_num = max(leaf_numbers)
                    leaf_counts[genotype] = max_leaf_num
                    print(
                        f"Found {len(leaf_numbers)} unique leaves (max leaf number: {max_leaf_num}) for genotype {genotype}")
                else:
                    leaf_counts[genotype] = 0
                    print(f"Warning: Could not extract leaf numbers from files in {cropped_folder}")
            else:
                print(f"Warning: {cropped_folder} not found")
                leaf_counts[genotype] = 0

    return leaf_counts


def calculate_composition(leaf_counts):
    """
    Calculate composition by disease set and edge cases.

    Args:
        leaf_counts: dict from count_leaves_per_genotype()

    Returns:
        dict: Composition data by split and set
    """
    splits = {'Train': TRAIN_GENOTYPES, 'Validation': VAL_GENOTYPES, 'Test': TEST_GENOTYPES}
    composition = {}

    for split_name, genotypes in splits.items():
        composition[split_name] = {
            'Set1': {'total': 0, 'edge': 0, 'regular': 0},
            'Set2': {'total': 0, 'edge': 0, 'regular': 0},
            'Set3': {'total': 0, 'edge': 0, 'regular': 0}
        }

        for genotype in genotypes:
            if genotype.startswith('96'):
                set_name = 'Set1'
            elif genotype.startswith('41'):
                set_name = 'Set2'
            elif genotype.startswith('694'):
                set_name = 'Set3'
            else:
                continue

            count = leaf_counts.get(genotype, 0)
            composition[split_name][set_name]['total'] += count

            if genotype in EDGE_CASES:
                composition[split_name][set_name]['edge'] += count
            else:
                composition[split_name][set_name]['regular'] += count

    return composition


def create_summary_csv(leaf_counts, composition, save_dir="dataset_visualization"):
    """
    Create CSV with all dataset statistics.

    Args:
        leaf_counts: dict from count_leaves_per_genotype()
        composition: dict from calculate_composition()
        save_dir: Output directory

    Returns:
        str: Path to saved CSV
    """
    # Calculate totals
    total_leaves = sum(leaf_counts.values())
    total_edge_cases = sum(composition[split][s]['edge'] for split in composition for s in ['Set1', 'Set2', 'Set3'])
    total_regular_cases = total_leaves - total_edge_cases

    # Calculate lesion totals
    total_brown_rust = sum(v['Brown Rust'] for v in lesion_counts.values())
    total_ramularia = sum(v['Ramularia'] for v in lesion_counts.values())
    total_lesions = total_brown_rust + total_ramularia

    # Load pixel data
    df = csv_data
    df['split'] = df['dataset'].map({
        'train_reflect': 'Train',
        'val_constant': 'Validation',
        'test_constant': 'Test'
    })

    # Create summary dictionary
    summary_data = {
        'Category': [],
        'Value': []
    }

    # Add leaf statistics
    summary_data['Category'].extend([
        'Total Leaves',
        'Total Edge Case Leaves',
        'Total Regular Case Leaves',
        'Edge Case Percentage (%)',
        'Regular Case Percentage (%)'
    ])
    summary_data['Value'].extend([
        total_leaves,
        total_edge_cases,
        total_regular_cases,
        f"{total_edge_cases / total_leaves * 100:.2f}",
        f"{total_regular_cases / total_leaves * 100:.2f}"
    ])

    # Add split-wise leaf counts
    for split in ['Train', 'Validation', 'Test']:
        split_leaves = sum(composition[split][s]['total'] for s in ['Set1', 'Set2', 'Set3'])
        summary_data['Category'].append(f'{split} Leaves')
        summary_data['Value'].append(split_leaves)

    # Add set-wise leaf counts
    for set_name in ['Set1', 'Set2', 'Set3']:
        set_total = sum(composition[split][set_name]['total'] for split in ['Train', 'Validation', 'Test'])
        summary_data['Category'].append(f'{set_name} Total Leaves')
        summary_data['Value'].append(set_total)

    # Add lesion statistics
    summary_data['Category'].extend([
        'Total Lesions',
        'Total Brown Rust Lesions',
        'Total Ramularia Lesions',
        'Brown Rust Lesion Percentage (%)',
        'Ramularia Lesion Percentage (%)'
    ])
    summary_data['Value'].extend([
        total_lesions,
        total_brown_rust,
        total_ramularia,
        f"{total_brown_rust / total_lesions * 100:.2f}",
        f"{total_ramularia / total_lesions * 100:.2f}"
    ])

    # Add split-wise lesion counts
    for split in ['Train', 'Validation', 'Test']:
        split_total = lesion_counts[split]['Brown Rust'] + lesion_counts[split]['Ramularia']
        summary_data['Category'].append(f'{split} Total Lesions')
        summary_data['Value'].append(split_total)
        summary_data['Category'].append(f'{split} Brown Rust Lesions')
        summary_data['Value'].append(lesion_counts[split]['Brown Rust'])
        summary_data['Category'].append(f'{split} Ramularia Lesions')
        summary_data['Value'].append(lesion_counts[split]['Ramularia'])

    # Add pixel statistics
    total_brown_rust_pixels = sum(df['brown_rust_pixels'])
    total_ramularia_pixels = sum(df['ramularia_pixels'])
    total_disease_pixels = total_brown_rust_pixels + total_ramularia_pixels
    total_leaf_pixels = sum(df['total_leaf_pixels'])
    total_all_pixels = sum(df['total_pixels'])
    total_healthy_pixels = sum(df['healthy_pixels'])

    summary_data['Category'].extend([
        'Total Leaf Pixels',
        'Total All Pixels',
        'Total Healthy Pixels',
        'Total Disease Pixels',
        'Total Brown Rust Pixels',
        'Total Ramularia Pixels',
        'Healthy Pixel Percentage (%)',
        'Disease Pixel Percentage (%)',
        'Brown Rust Pixel Percentage of Disease (%)',
        'Ramularia Pixel Percentage of Disease (%)',
        'Overall Background Ratio (%)',
        'Brown Rust:Ramularia Pixel Ratio'
    ])

    summary_data['Value'].extend([
        total_leaf_pixels,
        total_all_pixels,
        total_healthy_pixels,
        total_disease_pixels,
        total_brown_rust_pixels,
        total_ramularia_pixels,
        f"{total_healthy_pixels / total_leaf_pixels * 100:.2f}",
        f"{total_disease_pixels / total_leaf_pixels * 100:.2f}",
        f"{total_brown_rust_pixels / total_disease_pixels * 100:.2f}",
        f"{total_ramularia_pixels / total_disease_pixels * 100:.2f}",
        f"{((total_all_pixels - total_leaf_pixels) / total_all_pixels) * 100:.2f}",
        f"{total_brown_rust_pixels / total_ramularia_pixels:.2f}:1" if total_ramularia_pixels > 0 else "N/A"
    ])

    # Add split-wise pixel statistics
    for split in ['Train', 'Validation', 'Test']:
        row = df[df['split'] == split].iloc[0]
        summary_data['Category'].append(f'{split} Leaf Pixels')
        summary_data['Value'].append(row['total_leaf_pixels'])
        summary_data['Category'].append(f'{split} Brown Rust Pixels')
        summary_data['Value'].append(row['brown_rust_pixels'])
        summary_data['Category'].append(f'{split} Ramularia Pixels')
        summary_data['Value'].append(row['ramularia_pixels'])
        summary_data['Category'].append(f'{split} Disease Pixels')
        summary_data['Value'].append(row['brown_rust_pixels'] + row['ramularia_pixels'])
        summary_data['Category'].append(f'{split} Healthy Pixels')
        summary_data['Value'].append(row['healthy_pixels'])

    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'dataset_summary.csv')
    summary_df.to_csv(csv_path, index=False)

    print(f"Summary CSV saved: {csv_path}")
    return csv_path


def plot_dataset_composition_donut(composition, save_dir="dataset_visualization"):
    """
    Create donut chart showing dataset composition.

    Args:
        composition: dict from calculate_composition()
        save_dir: Output directory

    Returns:
        tuple: (png_path, pdf_path)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palettes
    split_colors = {
        'Train': '#4C72B0',
        'Validation': '#DD8452',
        'Test': '#55A868'
    }

    set_colors = {
        'Set1': '#FF6B6B',
        'Set2': '#4ECDC4',
        'Set3': '#FFD166'
    }

    case_colors = {
        'edge': '#2C3E50',
        'regular': '#BDC3C7'
    }

    # Calculate data
    splits = ['Train', 'Validation', 'Test']
    sets = ['Set1', 'Set2', 'Set3']

    split_totals = [sum(composition[split][s]['total'] for s in sets) for split in splits]
    total_leaves = sum(split_totals)

    # Middle ring: Disease sets
    set_data = []
    set_colors_list = []
    for split in splits:
        for set_name in sets:
            total = composition[split][set_name]['total']
            if total > 0:
                set_data.append(total)
                set_colors_list.append(set_colors[set_name])

    # Inner ring: Edge vs Regular
    edge_regular_data = []
    edge_regular_colors = []
    for split in splits:
        for set_name in sets:
            edge = composition[split][set_name]['edge']
            regular = composition[split][set_name]['regular']

            if edge > 0:
                edge_regular_data.append(edge)
                edge_regular_colors.append(case_colors['edge'])
            if regular > 0:
                edge_regular_data.append(regular)
                edge_regular_colors.append(case_colors['regular'])

    # Draw donut
    ring_widths = [0.25, 0.25, 0.25]
    radii = [0.25, 0.5, 0.75]

    # Inner ring
    wedges1 = ax.pie(
        edge_regular_data,
        colors=edge_regular_colors,
        radius=radii[0],
        startangle=90,
        wedgeprops=dict(width=ring_widths[0], edgecolor='white', linewidth=1.5),
        labels=None
    )[0]

    for wedge, color in zip(wedges1, edge_regular_colors):
        if color == case_colors['edge']:
            wedge.set_hatch('////')

    # Middle ring
    ax.pie(
        set_data,
        colors=set_colors_list,
        radius=radii[1],
        startangle=90,
        wedgeprops=dict(width=ring_widths[1], edgecolor='white', linewidth=1.5),
        labels=None
    )

    # Outer ring
    ax.pie(
        split_totals,
        colors=[split_colors[s] for s in splits],
        radius=radii[2],
        startangle=90,
        wedgeprops=dict(width=ring_widths[2], edgecolor='white', linewidth=2),
        labels=None
    )

    # Legend
    total_edge_cases = sum(composition[split][s]['edge'] for split in splits for s in sets)
    total_regular_cases = total_leaves - total_edge_cases

    legend_elements = [
        Patch(facecolor=split_colors['Train'], label=f'Train: {split_totals[0]:,} leaves'),
        Patch(facecolor=split_colors['Validation'], label=f'Validation: {split_totals[1]:,} leaves'),
        Patch(facecolor=split_colors['Test'], label=f'Test: {split_totals[2]:,} leaves'),
        Patch(facecolor=set_colors['Set1'], label='Set1: Brown Rust dominant'),
        Patch(facecolor=set_colors['Set2'], label='Set2: Ramularia dominant'),
        Patch(facecolor=set_colors['Set3'], label='Set3: Mixed incidence'),
        Patch(facecolor=case_colors['edge'], hatch='////',
              label=f'Edge Cases: {total_edge_cases:,} leaves'),
        Patch(facecolor=case_colors['regular'],
              label=f'Regular Cases: {total_regular_cases:,} leaves'),
    ]

    ax.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.05, 0.5), fontsize=9, framealpha=0.9,
              title='Dataset Composition', title_fontsize=10)

    plt.title('Dataset Composition', fontsize=14, fontweight='bold', pad=20)
    ax.set_aspect('equal')

    os.makedirs(save_dir, exist_ok=True)
    png_path = f'{save_dir}/chart1_dataset_composition.png'
    pdf_path = f'{save_dir}/chart1_dataset_composition.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Chart 1 saved: {png_path}")
    return png_path, pdf_path


def plot_lesion_distribution_donut(save_dir="dataset_visualization"):
    """
    Create donut chart showing lesion distribution.

    Args:
        save_dir: Output directory

    Returns:
        tuple: (png_path, pdf_path)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palettes
    split_colors = {
        'Train': '#4C72B0',
        'Validation': '#DD8452',
        'Test': '#55A868'
    }

    lesion_colors = {
        'Brown Rust': '#D62728',
        'Ramularia': '#1F77B4'
    }

    # Calculate data
    splits = ['Train', 'Validation', 'Test']
    split_lesion_totals = [lesion_counts[split]['Brown Rust'] + lesion_counts[split]['Ramularia']
                           for split in splits]

    # Middle ring data
    middle_ring_data = []
    middle_ring_colors = []

    for split in splits:
        br_count = lesion_counts[split]['Brown Rust']
        ra_count = lesion_counts[split]['Ramularia']

        middle_ring_data.append(br_count)
        middle_ring_colors.append(lesion_colors['Brown Rust'])

        middle_ring_data.append(ra_count)
        middle_ring_colors.append(lesion_colors['Ramularia'])

    # Draw donut
    ring_widths = [0.25, 0.25, 0.25]
    radii = [0.25, 0.5, 0.75]

    # Inner ring (empty)
    ax.pie([1], colors=['white'], radius=radii[0],
           wedgeprops=dict(width=ring_widths[0], edgecolor='none'))

    # Middle ring
    ax.pie(
        middle_ring_data,
        colors=middle_ring_colors,
        radius=radii[1],
        startangle=90,
        wedgeprops=dict(width=ring_widths[1], edgecolor='white', linewidth=1.5),
        labels=None
    )

    # Outer ring
    ax.pie(
        split_lesion_totals,
        colors=[split_colors[s] for s in splits],
        radius=radii[2],
        startangle=90,
        wedgeprops=dict(width=ring_widths[2], edgecolor='white', linewidth=2),
        labels=None
    )

    # Legend
    total_brown_rust = sum(lesion_counts[split]['Brown Rust'] for split in splits)
    total_ramularia = sum(lesion_counts[split]['Ramularia'] for split in splits)

    legend_elements = [
        Patch(facecolor=split_colors['Train'],
              label=f'Train: {split_lesion_totals[0]:,} lesions'),
        Patch(facecolor=split_colors['Validation'],
              label=f'Validation: {split_lesion_totals[1]:,} lesions'),
        Patch(facecolor=split_colors['Test'],
              label=f'Test: {split_lesion_totals[2]:,} lesions'),
        Patch(facecolor=lesion_colors['Brown Rust'],
              label=f'Brown Rust: {total_brown_rust:,} lesions'),
        Patch(facecolor=lesion_colors['Ramularia'],
              label=f'Ramularia: {total_ramularia:,} lesions'),
    ]

    ax.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.05, 0.5), fontsize=9, framealpha=0.9,
              title='Lesion Distribution', title_fontsize=10)

    plt.title('Lesion Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_aspect('equal')

    os.makedirs(save_dir, exist_ok=True)
    png_path = f'{save_dir}/chart2_lesion_distribution.png'
    pdf_path = f'{save_dir}/chart2_lesion_distribution.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Chart 2 saved: {png_path}")
    return png_path, pdf_path


def plot_disease_pixel_distribution_donut(save_dir="dataset_visualization"):
    """
    Create donut chart showing disease pixel distribution.

    Args:
        save_dir: Output directory

    Returns:
        tuple: (png_path, pdf_path)
    """
    # Load CSV data
    df = csv_data
    df['split'] = df['dataset'].map({
        'train_reflect': 'Train',
        'val_constant': 'Validation',
        'test_constant': 'Test'
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palettes
    split_colors = {
        'Train': '#4C72B0',
        'Validation': '#DD8452',
        'Test': '#55A868'
    }

    disease_colors = {
        'Brown Rust': '#D62728',
        'Ramularia': '#1F77B4'
    }

    # Calculate data
    splits = ['Train', 'Validation', 'Test']
    split_data = {}

    for split in splits:
        row = df[df['split'] == split].iloc[0]
        total_disease_pixels = row['brown_rust_pixels'] + row['ramularia_pixels']
        split_data[split] = {
            'brown_rust_pixels': row['brown_rust_pixels'],
            'ramularia_pixels': row['ramularia_pixels'],
            'total_disease_pixels': total_disease_pixels
        }

    # Outer ring data
    split_disease_totals = [split_data[split]['total_disease_pixels'] for split in splits]

    # Middle ring data
    middle_ring_data = []
    middle_ring_colors = []

    for split in splits:
        data = split_data[split]
        middle_ring_data.append(data['brown_rust_pixels'])
        middle_ring_colors.append(disease_colors['Brown Rust'])

        middle_ring_data.append(data['ramularia_pixels'])
        middle_ring_colors.append(disease_colors['Ramularia'])

    # Draw donut
    ring_widths = [0.25, 0.25, 0.25]
    radii = [0.25, 0.5, 0.75]

    # Inner ring (empty)
    ax.pie([1], colors=['white'], radius=radii[0],
           wedgeprops=dict(width=ring_widths[0], edgecolor='none'))

    # Middle ring
    ax.pie(
        middle_ring_data,
        colors=middle_ring_colors,
        radius=radii[1],
        startangle=90,
        wedgeprops=dict(width=ring_widths[1], edgecolor='white', linewidth=1.5),
        labels=None
    )

    # Outer ring
    ax.pie(
        split_disease_totals,
        colors=[split_colors[s] for s in splits],
        radius=radii[2],
        startangle=90,
        wedgeprops=dict(width=ring_widths[2], edgecolor='white', linewidth=2),
        labels=None
    )

    # Legend
    total_brown_rust_pixels = sum(df['brown_rust_pixels'])
    total_ramularia_pixels = sum(df['ramularia_pixels'])
    total_disease_pixels = total_brown_rust_pixels + total_ramularia_pixels

    br_percent_total = total_brown_rust_pixels / total_disease_pixels * 100 if total_disease_pixels > 0 else 0
    ra_percent_total = total_ramularia_pixels / total_disease_pixels * 100 if total_disease_pixels > 0 else 0

    legend_elements = [
        Patch(facecolor=split_colors['Train'],
              label=f'Train: {split_data["Train"]["total_disease_pixels"]:,} pixels'),
        Patch(facecolor=split_colors['Validation'],
              label=f'Validation: {split_data["Validation"]["total_disease_pixels"]:,} pixels'),
        Patch(facecolor=split_colors['Test'],
              label=f'Test: {split_data["Test"]["total_disease_pixels"]:,} pixels'),
        Patch(facecolor=disease_colors['Brown Rust'],
              label=f'Brown Rust: {total_brown_rust_pixels:,} pixels ({br_percent_total:.1f}%)'),
        Patch(facecolor=disease_colors['Ramularia'],
              label=f'Ramularia: {total_ramularia_pixels:,} pixels ({ra_percent_total:.1f}%)'),
    ]

    ax.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.05, 0.5), fontsize=9, framealpha=0.9,
              title='Disease Pixel Distribution', title_fontsize=10)

    plt.title('Disease Pixel Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_aspect('equal')

    os.makedirs(save_dir, exist_ok=True)
    png_path = f'{save_dir}/chart3_disease_pixel_distribution.png'
    pdf_path = f'{save_dir}/chart3_disease_pixel_distribution.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Chart 3 saved: {png_path}")
    return png_path, pdf_path


def analyze_leaf_disease_cooccurrence(base_path=DATA_DIR):
    """
    Analyze disease co-occurrence from annotated masks.

    Args:
        base_path: Root dataset directory

    Returns:
        dict: Analysis results with counts and percentages
    """

    print("ANALYZING LEAF DISEASE CO-OCCURRENCE")


    # Initialize counters
    total_leaves = 0
    both_diseases = 0
    only_brown_rust = 0
    only_ramularia = 0
    no_disease = 0

    # For split-wise statistics
    split_stats = {
        'Train': {'total': 0, 'both': 0, 'only_br': 0, 'only_ra': 0, 'none': 0},
        'Validation': {'total': 0, 'both': 0, 'only_br': 0, 'only_ra': 0, 'none': 0},
        'Test': {'total': 0, 'both': 0, 'only_br': 0, 'only_ra': 0, 'none': 0}
    }

    # For detailed statistics
    leaf_details = []

    splits = {
        'Train': TRAIN_GENOTYPES,
        'Validation': VAL_GENOTYPES,
        'Test': TEST_GENOTYPES
    }

    for split_name, genotypes in splits.items():
        print(f"\nProcessing {split_name} split...")

        for genotype in genotypes:
            # Construct path to annotated masks folder
            mask_folder = os.path.join(
                base_path,
                f"{split_name}_data",
                genotype,
                f"annotated_masks_{genotype}_boxed_preprocessed_patches"
            )

            if not os.path.exists(mask_folder):
                print(f"  Warning: Mask folder not found for {genotype}: {mask_folder}")
                continue

            # Find all annotated mask files
            mask_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                mask_files.extend([f for f in os.listdir(mask_folder)
                                   if f.lower().startswith('annotated_mask') and f.lower().endswith(ext.lower())])

            if not mask_files:
                print(f"  No mask files found for {genotype}")
                continue

            print(f"  Found {len(mask_files)} mask files for {genotype}")

            # Group masks by leaf number (assuming leaf_XX in filename)
            leaf_masks = {}
            for mask_file in mask_files:
                # Extract leaf number from filename
                match = re.search(r'leaf[_-]?(\d+)', mask_file.lower())
                if match:
                    leaf_num = match.group(1)
                    if leaf_num not in leaf_masks:
                        leaf_masks[leaf_num] = []
                    leaf_masks[leaf_num].append(mask_file)
                else:
                    # If no leaf number found, treat each file as separate leaf
                    unique_id = mask_file
                    leaf_masks[unique_id] = [mask_file]

            # Analyze each leaf
            for leaf_id, mask_list in leaf_masks.items():
                leaf_has_brown_rust = False
                leaf_has_ramularia = False
                leaf_brown_rust_pixels = 0
                leaf_ramularia_pixels = 0

                # Check each patch for this leaf
                for mask_file in mask_list:
                    mask_path = os.path.join(mask_folder, mask_file)

                    try:
                        # Load the mask image (grayscale with values 0, 1, 2)
                        mask = plt.imread(mask_path)
                        mask_array = np.array(mask)

                        # Handle different mask formats
                        if mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                            # Normalize to 0, 1, 2 if needed
                            mask_array = np.round(mask_array * 2).astype(np.uint8)

                        # Get unique values in this patch
                        unique_vals = np.unique(mask_array)

                        # Check for disease classes (1=Brown Rust, 2=Ramularia)
                        if 1 in unique_vals:
                            leaf_has_brown_rust = True
                            leaf_brown_rust_pixels += np.sum(mask_array == 1)

                        if 2 in unique_vals:
                            leaf_has_ramularia = True
                            leaf_ramularia_pixels += np.sum(mask_array == 2)

                    except Exception as e:
                        print(f"    Error reading {mask_file}: {e}")
                        continue

                # Count based on disease presence
                total_leaves += 1
                split_stats[split_name]['total'] += 1

                # Store leaf details
                leaf_detail = {
                    'split': split_name,
                    'genotype': genotype,
                    'leaf_id': leaf_id,
                    'has_brown_rust': leaf_has_brown_rust,
                    'has_ramularia': leaf_has_ramularia,
                    'brown_rust_pixels': leaf_brown_rust_pixels,
                    'ramularia_pixels': leaf_ramularia_pixels,
                    'total_disease_pixels': leaf_brown_rust_pixels + leaf_ramularia_pixels
                }
                leaf_details.append(leaf_detail)

                # Update counters
                if leaf_has_brown_rust and leaf_has_ramularia:
                    both_diseases += 1
                    split_stats[split_name]['both'] += 1
                    leaf_detail['disease_category'] = 'Both diseases'
                elif leaf_has_brown_rust and not leaf_has_ramularia:
                    only_brown_rust += 1
                    split_stats[split_name]['only_br'] += 1
                    leaf_detail['disease_category'] = 'Only Brown Rust'
                elif not leaf_has_brown_rust and leaf_has_ramularia:
                    only_ramularia += 1
                    split_stats[split_name]['only_ra'] += 1
                    leaf_detail['disease_category'] = 'Only Ramularia'
                else:
                    no_disease += 1
                    split_stats[split_name]['none'] += 1
                    leaf_detail['disease_category'] = 'No disease'

    # Calculate percentages
    if total_leaves > 0:
        pct_both = (both_diseases / total_leaves) * 100
        pct_only_br = (only_brown_rust / total_leaves) * 100
        pct_only_ra = (only_ramularia / total_leaves) * 100
        pct_none = (no_disease / total_leaves) * 100

        leaves_with_disease = both_diseases + only_brown_rust + only_ramularia
        pct_with_disease = (leaves_with_disease / total_leaves) * 100

        if leaves_with_disease > 0:
            pct_both_among_diseased = (both_diseases / leaves_with_disease) * 100
            pct_only_br_among_diseased = (only_brown_rust / leaves_with_disease) * 100
            pct_only_ra_among_diseased = (only_ramularia / leaves_with_disease) * 100
        else:
            pct_both_among_diseased = pct_only_br_among_diseased = pct_only_ra_among_diseased = 0
    else:
        pct_both = pct_only_br = pct_only_ra = pct_none = pct_with_disease = 0
        pct_both_among_diseased = pct_only_br_among_diseased = pct_only_ra_among_diseased = 0

    # Print detailed analysis
    print("DISEASE CO-OCCURRENCE ANALYSIS RESULTS")
    print(f"\n TOTAL LEAVES ANALYZED: {total_leaves:,}")

    print(" OVERALL DISEASE DISTRIBUTION")
    print(f"Leaves with BOTH diseases:       {both_diseases:>5} ({pct_both:6.2f}%)")
    print(f"Leaves with ONLY Brown Rust:    {only_brown_rust:>5} ({pct_only_br:6.2f}%)")
    print(f"Leaves with ONLY Ramularia:     {only_ramularia:>5} ({pct_only_ra:6.2f}%)")
    print(f"Leaves with NO disease:         {no_disease:>5} ({pct_none:6.2f}%)")
    print(f"Leaves with AT LEAST ONE disease: {leaves_with_disease:>5} ({pct_with_disease:6.2f}%)")


    print("AMONG DISEASED LEAVES")
    if leaves_with_disease > 0:
        print(f"Both diseases:       {pct_both_among_diseased:6.2f}% of diseased leaves")
        print(f"Only Brown Rust:     {pct_only_br_among_diseased:6.2f}% of diseased leaves")
        print(f"Only Ramularia:      {pct_only_ra_among_diseased:6.2f}% of diseased leaves")


    print(" SPLIT-WISE STATISTICS")
    for split in ['Train', 'Validation', 'Test']:
        s = split_stats[split]
        if s['total'] > 0:
            print(f"\n{split.upper()}:")
            print(f"  Total leaves:        {s['total']:>5}")
            print(f"  Both diseases:       {s['both']:>5} ({(s['both'] / s['total']) * 100:6.2f}%)")
            print(f"  Only Brown Rust:     {s['only_br']:>5} ({(s['only_br'] / s['total']) * 100:6.2f}%)")
            print(f"  Only Ramularia:      {s['only_ra']:>5} ({(s['only_ra'] / s['total']) * 100:6.2f}%)")
            print(f"  No disease:          {s['none']:>5} ({(s['none'] / s['total']) * 100:6.2f}%)")

    # Analyze pixel counts per category
    print(" PIXEL ANALYSIS BY DISEASE CATEGORY")


    # Convert to DataFrame for easier analysis
    df_leaves = pd.DataFrame(leaf_details)

    if len(df_leaves) > 0:
        # Analyze pixels for each category
        categories = ['Both diseases', 'Only Brown Rust', 'Only Ramularia', 'No disease']
        for category in categories:
            cat_leaves = df_leaves[df_leaves['disease_category'] == category]
            if len(cat_leaves) > 0:
                print(f"\n{category.upper()}:")
                print(f"  Number of leaves: {len(cat_leaves):>5}")
                if category != 'No disease':
                    avg_br_pixels = cat_leaves[
                        'brown_rust_pixels'].mean() if 'brown_rust_pixels' in cat_leaves.columns else 0
                    avg_ra_pixels = cat_leaves[
                        'ramularia_pixels'].mean() if 'ramularia_pixels' in cat_leaves.columns else 0
                    print(f"  Avg Brown Rust pixels/leaf: {avg_br_pixels:>8.0f}")
                    print(f"  Avg Ramularia pixels/leaf:  {avg_ra_pixels:>8.0f}")
                    if category == 'Both diseases':
                        print(f"  Avg total disease pixels/leaf: {(avg_br_pixels + avg_ra_pixels):>8.0f}")
                        if avg_ra_pixels > 0:
                            br_ra_ratio = avg_br_pixels / avg_ra_pixels
                            print(f"  Brown Rust:Ramularia ratio:    {br_ra_ratio:>8.2f}:1")

    # Summary statistics

    print(" SUMMARY")
    print(f" {pct_with_disease:.1f}% of leaves have at least one disease")
    print(f" {pct_both:.1f}% of leaves have both diseases (co-infection)")
    print(f" Among diseased leaves, {pct_both_among_diseased:.1f}% are co-infected")
    print(
        f" Disease co-occurrence ratio: {pct_both_among_diseased:.1f}% co-infected vs {100 - pct_both_among_diseased:.1f}% single disease")

    # Return results
    results = {
        'total_leaves': total_leaves,
        'both_diseases': both_diseases,
        'only_brown_rust': only_brown_rust,
        'only_ramularia': only_ramularia,
        'no_disease': no_disease,
        'leaves_with_disease': leaves_with_disease,
        'percentages': {
            'pct_both': pct_both,
            'pct_only_br': pct_only_br,
            'pct_only_ra': pct_only_ra,
            'pct_none': pct_none,
            'pct_with_disease': pct_with_disease,
            'pct_both_among_diseased': pct_both_among_diseased,
            'pct_only_br_among_diseased': pct_only_br_among_diseased,
            'pct_only_ra_among_diseased': pct_only_ra_among_diseased
        },
        'split_stats': split_stats,
        'leaf_details': leaf_details,
        'dataframe': df_leaves if len(df_leaves) > 0 else None
    }

    return results


def main():
    """
    Main function to run all analysis and generate outputs.
    """
    print(f"Using DATA_DIR: {DATA_DIR}")


    # Count leaves from actual data
    print("Counting leaves per genotype...")
    leaf_counts = count_leaves_per_genotype(DATA_DIR)

    disease_analysis = analyze_leaf_disease_cooccurrence(DATA_DIR)

    # Calculate composition
    print("Calculating composition...")
    composition = calculate_composition(leaf_counts)

    # Create summary CSV

    print("CREATING SUMMARY CSV")

    csv_path = create_summary_csv(leaf_counts, composition)

    # Generate plots

    print("GENERATING CHART 1: Dataset Composition")

    chart1_png, chart1_pdf = plot_dataset_composition_donut(composition)

    print("GENERATING CHART 2: Lesion Distribution")

    chart2_png, chart2_pdf = plot_lesion_distribution_donut()

    print("GENERATING CHART 3: Disease Pixel Distribution")

    chart3_png, chart3_pdf = plot_disease_pixel_distribution_donut()

    # Print summary

    print("GENERATION COMPLETE!")

    print(f"Summary CSV: {csv_path}")
    print(f"Chart 1: {chart1_png}")
    print(f"Chart 1 (PDF): {chart1_pdf}")
    print(f"Chart 2: {chart2_png}")
    print(f"Chart 2 (PDF): {chart2_pdf}")
    print(f"Chart 3: {chart3_png}")
    print(f"Chart 3 (PDF): {chart3_pdf}")


if __name__ == "__main__":
    main()