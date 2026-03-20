import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

from barley_disease_segmentation.config import PROJECT_ROOT

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Path to your detailed CSV file
INPUT_CSV = Path(PROJECT_ROOT/"Paper_Review/Table_S8_S9/table2_unlabelled_detailed_threegroups.csv")

# Output directory for plots
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
WINTER_COLOR = '#1E88E5'  # Vibrant blue
SPRING_2024_COLOR = '#FF8C00'  # Dark orange
SPRING_2025_COLOR = '#DC143C'  # Crimson

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
# ---------------------------------------------------------
# Load and process data
# ---------------------------------------------------------

print("GENOTYPE RUST:RAM RATIO ANALYSIS - SCATTERPLOT ONLY")


# Load the data
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df)} leaf records")
print(f"Winter leaves: {len(df[df['group'] == 'winter'])}")
print(f"Spring leaves: {len(df[df['group'] == 'spring'])}")

# Calculate genotype-level averages
geno_avg = df.groupby(['genotype', 'group']).agg({
    'rust_pct_pred': 'mean',
    'ram_pct_pred': 'mean',
    'genotype': 'count'
}).rename(columns={'genotype': 'n_leaves'}).reset_index()


# Add year classification for spring genotypes
def classify_spring_genotype(genotype, group):
    if group == 'winter':
        return 'winter'
    else:
        # Check if genotype name is numeric and length <= 5 (2024 population)
        try:
            # Convert to string and check length
            genotype_str = str(genotype)
            if len(genotype_str) <= 5:
                return 'spring_2024'
            else:
                return 'spring_2025'
        except:
            # If conversion fails, default to spring_2025
            return 'spring_2025'


geno_avg['subgroup'] = geno_avg.apply(
    lambda row: classify_spring_genotype(row['genotype'], row['group']),
    axis=1
)

print(f"\nFound {len(geno_avg)} unique genotypes:")
print(f"  Winter: {len(geno_avg[geno_avg['subgroup'] == 'winter'])}")
print(f"  Spring 2024: {len(geno_avg[geno_avg['subgroup'] == 'spring_2024'])}")
print(f"  Spring 2025 (names >5 chars): {len(geno_avg[geno_avg['subgroup'] == 'spring_2025'])}")

# Calculate rust:ram ratio (add small epsilon to avoid division by zero)
epsilon = 0.01
geno_avg['rust_ram_ratio'] = (geno_avg['rust_pct_pred'] + epsilon) / (geno_avg['ram_pct_pred'] + epsilon)
geno_avg['log10_ratio'] = np.log10(geno_avg['rust_ram_ratio'])

# ---------------------------------------------------------
# SINGLE PLOT: Scatter plot - Ram % vs Rust %
# ---------------------------------------------------------

plt.figure(figsize=(12,10))

# Plot each subgroup separately
subgroups = ['winter', 'spring_2024', 'spring_2025']
subgroup_colors = [WINTER_COLOR, SPRING_2024_COLOR, SPRING_2025_COLOR]
subgroup_labels = ['Winter', 'Spring 2024', 'Spring 2025']


for subgroup, color, label in zip(subgroups, subgroup_colors, subgroup_labels):
    group_data = geno_avg[geno_avg['subgroup'] == subgroup]

    if len(group_data) > 0:
        plt.scatter(
            group_data['ram_pct_pred'].values,
            group_data['rust_pct_pred'].values,
            c=color,
            label=label,
            s=group_data['n_leaves'].values * 15,  # Size by number of leaves
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8
        )

# Add diagonal line for equal proportions
max_val = max(geno_avg['rust_pct_pred'].max(), geno_avg['ram_pct_pred'].max())
x = np.linspace(0, max_val * 1.1, 100)
plt.plot(x, x, 'k--', alpha=0.5, linewidth=1.5, label='1:1 ratio')

# Add lines for 2:1 and 1:2 ratios for reference
plt.plot(x, x * 2, 'k:', alpha=0.3, linewidth=1, label='2:1 (rust:ram)')
plt.plot(x, x / 2, 'k:', alpha=0.3, linewidth=1, label='1:2 (rust:ram)')

# Labels and title
plt.xlabel('Ramularia Leaf Area (%)', fontsize=13)
plt.ylabel('Brown Rust Leaf Area (%)', fontsize=13)


# Set axis limits with some padding
plt.xlim(0, max_val * 1.1)
plt.ylim(0, 15)

# Add legend with better positioning
plt.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Add grid with light lines
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# Make axes square to better see the ratio relationships
plt.gca().set_aspect('equal', adjustable='box')

# Add minor ticks for better readability
plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True)
plt.tick_params(which='major', length=6)
plt.tick_params(which='minor', length=3)

plt.tight_layout()

# Save in multiple formats
plt.savefig(OUTPUT_DIR / 'scatter_ram_vs_rust.pdf', format='pdf')
plt.savefig(OUTPUT_DIR / 'scatter_ram_vs_rust.png', format='png', dpi=600)


print(f"\nScatter plot saved to:")
print(f"  - {OUTPUT_DIR / 'scatter_ram_vs_rust.pdf'}")
print(f"  - {OUTPUT_DIR / 'scatter_ram_vs_rust.png'}")



# ---------------------------------------------------------
# Statistical Analysis (kept for reference)
# ---------------------------------------------------------

print("STATISTICAL ANALYSIS")


# Separate groups
winter_ratios = geno_avg[geno_avg['subgroup'] == 'winter']['rust_ram_ratio']
spring2024_ratios = geno_avg[geno_avg['subgroup'] == 'spring_2024']['rust_ram_ratio']
spring2025_ratios = geno_avg[geno_avg['subgroup'] == 'spring_2025']['rust_ram_ratio']

if len(winter_ratios) > 0:
    print("\nDescriptive Statistics:")
    print(f"\nWinter genotypes (n={len(winter_ratios)}):")
    print(f"  Mean rust:ram ratio: {winter_ratios.mean():.3f}")
    print(f"  Median rust:ram ratio: {winter_ratios.median():.3f}")
    print(f"  Std dev: {winter_ratios.std():.3f}")

if len(spring2024_ratios) > 0:
    print(f"\nSpring 2024 genotypes (n={len(spring2024_ratios)}):")
    print(f"  Mean rust:ram ratio: {spring2024_ratios.mean():.3f}")
    print(f"  Median rust:ram ratio: {spring2024_ratios.median():.3f}")
    print(f"  Std dev: {spring2024_ratios.std():.3f}")

if len(spring2025_ratios) > 0:
    print(f"\nSpring 2025 genotypes (n={len(spring2025_ratios)}):")
    print(f"  Mean rust:ram ratio: {spring2025_ratios.mean():.3f}")
    print(f"  Median rust:ram ratio: {spring2025_ratios.median():.3f}")
    print(f"  Std dev: {spring2025_ratios.std():.3f}")

# Statistical tests between groups
if len(winter_ratios) > 0 and len(spring2024_ratios) > 0:
    stat, p_value = stats.mannwhitneyu(winter_ratios, spring2024_ratios, alternative='two-sided')
    print(f"\nMann-Whitney U Test (Winter vs Spring 2024):")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")

if len(winter_ratios) > 0 and len(spring2025_ratios) > 0:
    stat, p_value = stats.mannwhitneyu(winter_ratios, spring2025_ratios, alternative='two-sided')
    print(f"\nMann-Whitney U Test (Winter vs Spring 2025):")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")

if len(spring2024_ratios) > 0 and len(spring2025_ratios) > 0:
    stat, p_value = stats.mannwhitneyu(spring2024_ratios, spring2025_ratios, alternative='two-sided')
    print(f"\nMann-Whitney U Test (Spring 2024 vs Spring 2025):")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

# Save genotype averages with ratios and subgroups
output_csv = OUTPUT_DIR / 'genotype_ratios_with_years.csv'
geno_avg.to_csv(output_csv, index=False)
print(f"\nGenotype ratio data saved to: {output_csv}")

print("ANALYSIS COMPLETE")
print(f"\nAll plots and data saved to: {OUTPUT_DIR}/")