import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, levene
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

from barley_disease_segmentation.config import PROJECT_ROOT, PREDICTIONS_UNLABELLED, BROWN_RUST_GENOTYPES, \
    RAMULARIA_GENOTYPES_2024, RAMULARIA_GENOTYPES_2025

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# FOR TABLE 1: Test set paths (with ground truth)
TEST_PRED_DIR = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/predictions")
TEST_GT_DIR = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/labels")
TEST_DATA_DIR = Path(PROJECT_ROOT / "inference_data/Multiclass/Convnext/20251117_1028/saved_predictions/data")

# FOR TABLE 2: Big unlabelled dataset (predictions only)
UNLABELLED_PRED_DIR = Path(PREDICTIONS_UNLABELLED / 'predictions')
UNLABELLED_DATA_DIR = Path(PREDICTIONS_UNLABELLED / 'data')

WINTER_GENOTYPES = BROWN_RUST_GENOTYPES
RAMULARIA_GENOTYPES_2024 = RAMULARIA_GENOTYPES_2024
RAMULARIA_GENOTYPES_2025 = RAMULARIA_GENOTYPES_2025


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def load_img(path):
    """Load image as numpy array."""
    return np.array(Image.open(path))


def get_leaf_mask(img):
    """Create leaf mask (non-white pixels)."""
    return ~((img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255))


def parse_filename(fn):
    """Parse genotype and leaf ID from filename."""
    g, leaf = fn.replace(".png", "").split("_")
    return g, leaf


def get_genotype_group_from_lists(genotype):
    """Determine if genotype is winter, spring_2024, or spring_2025 using the provided lists."""
    if genotype in WINTER_GENOTYPES:
        return "winter"
    elif genotype in RAMULARIA_GENOTYPES_2024:
        return "spring_2024"
    elif genotype in RAMULARIA_GENOTYPES_2025:
        return "spring_2025"
    else:
        return None


def get_genotype_group_test_set(genotype):
    """Grouping logic for test set (Table 1)."""
    if genotype == "9635":
        return "winter"
    elif genotype == "41561":
        return "spring_2024"
    elif genotype.startswith("69"):
        return "spring_2025"
    else:
        return "other"


def perform_statistical_tests_multi_group(data, group_col='group', value_cols=['rust_pct', 'ram_pct'],
                                          groups=['winter', 'spring_2024', 'spring_2025']):
    """
    Perform statistical tests comparing all pairs of groups.
    Now properly handles one-tailed tests based on disease type.
    """
    results = {}
    group_pairs = list(combinations(groups, 2))

    for val_col in value_cols:
        results[val_col] = {}

        for group1, group2 in group_pairs:
            pair_key = f"{group1}_vs_{group2}"

            # Extract data for each group
            group1_data = data[data[group_col] == group1][val_col].dropna()
            group2_data = data[data[group_col] == group2][val_col].dropna()

            if len(group1_data) == 0 or len(group2_data) == 0:
                results[val_col][pair_key] = {"error": "Insufficient data"}
                continue

            # Determine the correct alternative hypothesis based on the disease and groups
            if 'rust' in val_col.lower():
                # For brown rust: winter > both spring groups
                if group1 == "winter" and group2.startswith("spring"):
                    test_alternative = 'greater'  # winter > spring
                    hypothesis_direction = f"{group1} > {group2}"
                elif group2 == "winter" and group1.startswith("spring"):
                    test_alternative = 'less'  # winter > spring means spring < winter
                    hypothesis_direction = f"{group1} < {group2}"
                else:
                    # Comparing spring_2024 vs spring_2025 - no expected direction
                    test_alternative = 'two-sided'
                    hypothesis_direction = "two-sided difference"

            elif 'ram' in val_col.lower():
                # For ramularia: winter < both spring groups
                if group1 == "winter" and group2.startswith("spring"):
                    test_alternative = 'less'  # winter < spring
                    hypothesis_direction = f"{group1} < {group2}"
                elif group2 == "winter" and group1.startswith("spring"):
                    test_alternative = 'greater'  # winter < spring means spring > winter
                    hypothesis_direction = f"{group1} > {group2}"
                else:
                    # Comparing spring_2024 vs spring_2025 - no expected direction
                    test_alternative = 'two-sided'
                    hypothesis_direction = "two-sided difference"
            else:
                test_alternative = 'two-sided'
                hypothesis_direction = "two-sided difference"

            # Test for normality (Shapiro-Wilk) - only for reasonable sample sizes
            if len(group1_data) < 5000 and len(group2_data) < 5000:
                _, p_norm1 = stats.shapiro(group1_data)
                _, p_norm2 = stats.shapiro(group2_data)
            else:
                p_norm1 = p_norm2 = 1.0  # Assume non-normal for large samples

            # Test for homogeneity of variances (Levene's test)
            _, p_levene = stats.levene(group1_data, group2_data)

            # Choose appropriate test based on assumptions
            if p_norm1 > 0.05 and p_norm2 > 0.05 and p_levene > 0.05 and len(group1_data) < 5000:
                # Parametric test: t-test
                stat, p_value = stats.ttest_ind(group1_data, group2_data, alternative=test_alternative)
                test_used = f"t-test (parametric, {test_alternative})"

                # Cohen's d effect size
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() +
                                      (len(group2_data) - 1) * group2_data.var()) /
                                     (len(group1_data) + len(group2_data) - 2))
                effect_size = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std != 0 else 0
                effect_size_interpretation = "Cohen's d"
            else:
                # Non-parametric test: Mann-Whitney U
                stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative=test_alternative)
                test_used = f"Mann-Whitney U (non-parametric, {test_alternative})"

                # Calculate effect size for Mann-Whitney: r = Z / sqrt(N)
                from scipy.stats import norm
                # Approximate Z-score from p-value
                if test_alternative == 'two-sided':
                    z = norm.ppf(1 - p_value / 2)
                else:  # one-tailed
                    z = norm.ppf(1 - p_value)
                effect_size = z / np.sqrt(len(group1_data) + len(group2_data))
                effect_size_interpretation = "r (rank-biserial correlation)"

            # Calculate fold change
            mean1 = group1_data.mean()
            mean2 = group2_data.mean()
            fold_change = mean1 / mean2 if mean2 != 0 else np.inf

            # Store results
            results[val_col][pair_key] = {
                "test_used": test_used,
                "hypothesis_tested": hypothesis_direction,
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "p_value_formatted": f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}",
                "mean_group1": mean1,
                "mean_group2": mean2,
                "group1": group1,
                "group2": group2,
                "fold_change": fold_change,
                "fold_change_description": f"{fold_change:.2f}x higher in {group1}" if fold_change > 1 else f"{1/fold_change:.2f}x higher in {group2}",
                "effect_size": effect_size,
                "effect_size_interpretation": effect_size_interpretation,
                "n_group1": len(group1_data),
                "n_group2": len(group2_data)
            }

            # Interpret effect size
            if effect_size_interpretation == "Cohen's d":
                if abs(effect_size) < 0.2:
                    results[val_col][pair_key]["effect_magnitude"] = "negligible"
                elif abs(effect_size) < 0.5:
                    results[val_col][pair_key]["effect_magnitude"] = "small"
                elif abs(effect_size) < 0.8:
                    results[val_col][pair_key]["effect_magnitude"] = "medium"
                else:
                    results[val_col][pair_key]["effect_magnitude"] = "large"
            else:  # r for Mann-Whitney
                if abs(effect_size) < 0.1:
                    results[val_col][pair_key]["effect_magnitude"] = "negligible"
                elif abs(effect_size) < 0.3:
                    results[val_col][pair_key]["effect_magnitude"] = "small"
                elif abs(effect_size) < 0.5:
                    results[val_col][pair_key]["effect_magnitude"] = "medium"
                else:
                    results[val_col][pair_key]["effect_magnitude"] = "large"

    return results


def print_statistical_summary_multi_group(results, dataset_name):
    """Print formatted statistical results for multiple group comparisons."""
    print(f"STATISTICAL ANALYSIS - {dataset_name}")

    for disease, comparisons in results.items():
        disease_name = "Brown Rust" if "rust" in disease else "Ramularia"
        print(f"\n{disease_name}:")

        for comparison, stats_dict in comparisons.items():
            if "error" in stats_dict:
                print(f"  {comparison}: {stats_dict['error']}")
                continue

            print(f"\n  {comparison.replace('_', ' ')}:")
            print(f"    {stats_dict['test_used']}")
            print(f"    Testing: {stats_dict['hypothesis_tested']}")
            print(f"    {stats_dict['group1']} (n={stats_dict['n_group1']}): {stats_dict['mean_group1']:.3f}%")
            print(f"    {stats_dict['group2']} (n={stats_dict['n_group2']}): {stats_dict['mean_group2']:.3f}%")
            print(f"    {stats_dict['fold_change_description']}")
            print(f"    p-value: {stats_dict['p_value_formatted']}")
            print(f"    Statistically significant: {'YES' if stats_dict['significant'] else 'NO'}")
            print(f"    Effect size ({stats_dict['effect_size_interpretation']}): {stats_dict['effect_size']:.3f}")
            print(f"    Effect magnitude: {stats_dict['effect_magnitude']}")


# ---------------------------------------------------------
# TABLE 1: Test Set Analysis (with ground truth)
# ---------------------------------------------------------

print("TABLE 1: TEST SET ANALYSIS (MULTICLASS - THREE GROUPS)")


test_records = []
group_counts = {"winter": 0, "spring_2024": 0, "spring_2025": 0, "other": 0}

for fn in os.listdir(TEST_PRED_DIR):
    if not fn.endswith(".png"):
        continue

    # Parse filename
    genotype, leafID = parse_filename(fn)

    # Determine group using test set logic
    group = get_genotype_group_test_set(genotype)
    group_counts[group] = group_counts.get(group, 0) + 1

    # Load prediction
    pred = load_img(TEST_PRED_DIR / fn)
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    # Load original image for leaf mask
    img = load_img(TEST_DATA_DIR / fn)
    leaf_mask = get_leaf_mask(img)
    leaf_area = leaf_mask.sum()

    # Calculate prediction disease areas
    rust_area_pred = ((pred == 1) & leaf_mask).sum()
    ram_area_pred = ((pred == 2) & leaf_mask).sum()

    rust_pct_pred = rust_area_pred / leaf_area * 100
    ram_pct_pred = ram_area_pred / leaf_area * 100

    # Load ground truth
    gt = load_img(TEST_GT_DIR / fn)
    if gt.ndim == 3:
        gt = gt[:, :, 0]

    # Calculate ground truth disease areas
    rust_area_gt = ((gt == 1) & leaf_mask).sum()
    ram_area_gt = ((gt == 2) & leaf_mask).sum()

    rust_pct_gt = rust_area_gt / leaf_area * 100
    ram_pct_gt = ram_area_gt / leaf_area * 100

    test_records.append({
        "genotype": genotype,
        "group": group,
        "rust_pct_pred": rust_pct_pred,
        "ram_pct_pred": ram_pct_pred,
        "rust_pct_gt": rust_pct_gt,
        "ram_pct_gt": ram_pct_gt,
    })

# Create DataFrame
df_test = pd.DataFrame(test_records)

print(f"\nSample counts per group:")
for group, count in group_counts.items():
    if count > 0:
        print(f"  {group}: {count} leaves")

# Perform statistical tests on test set predictions
print("STATISTICAL TESTS - TEST SET (Predictions)")
test_stats_pred = perform_statistical_tests_multi_group(
    df_test,
    value_cols=['rust_pct_pred', 'ram_pct_pred'],
    groups=['winter', 'spring_2024', 'spring_2025']
)
print_statistical_summary_multi_group(test_stats_pred, "TEST SET - PREDICTIONS")

# Perform statistical tests on test set ground truth
print("STATISTICAL TESTS - TEST SET (Ground Truth)")
test_stats_gt = perform_statistical_tests_multi_group(
    df_test,
    value_cols=['rust_pct_gt', 'ram_pct_gt'],
    groups=['winter', 'spring_2024', 'spring_2025']
)
print_statistical_summary_multi_group(test_stats_gt, "TEST SET - GROUND TRUTH")

# Compute group-level means for test set
test_pred_summary = df_test.groupby("group")[["rust_pct_pred", "ram_pct_pred"]].mean()
test_gt_summary = df_test.groupby("group")[["rust_pct_gt", "ram_pct_gt"]].mean()

# Create formatted table for test set (now with three groups)
test_table = pd.DataFrame({
    'Group': ['Winter (9635)', 'Spring 2024 (41561)', 'Spring 2025 (69xx)'],
    'Rust % (Pred)': [
        f"{test_pred_summary.loc['winter', 'rust_pct_pred']:.3f}%" if 'winter' in test_pred_summary.index else 'N/A',
        f"{test_pred_summary.loc['spring_2024', 'rust_pct_pred']:.3f}%" if 'spring_2024' in test_pred_summary.index else 'N/A',
        f"{test_pred_summary.loc['spring_2025', 'rust_pct_pred']:.3f}%" if 'spring_2025' in test_pred_summary.index else 'N/A'],
    'Ram % (Pred)': [
        f"{test_pred_summary.loc['winter', 'ram_pct_pred']:.3f}%" if 'winter' in test_pred_summary.index else 'N/A',
        f"{test_pred_summary.loc['spring_2024', 'ram_pct_pred']:.3f}%" if 'spring_2024' in test_pred_summary.index else 'N/A',
        f"{test_pred_summary.loc['spring_2025', 'ram_pct_pred']:.3f}%" if 'spring_2025' in test_pred_summary.index else 'N/A'],
    'Rust % (GT)': [
        f"{test_gt_summary.loc['winter', 'rust_pct_gt']:.3f}%" if 'winter' in test_gt_summary.index else 'N/A',
        f"{test_gt_summary.loc['spring_2024', 'rust_pct_gt']:.3f}%" if 'spring_2024' in test_gt_summary.index else 'N/A',
        f"{test_gt_summary.loc['spring_2025', 'rust_pct_gt']:.3f}%" if 'spring_2025' in test_gt_summary.index else 'N/A'],
    'Ram % (GT)': [
        f"{test_gt_summary.loc['winter', 'ram_pct_gt']:.3f}%" if 'winter' in test_gt_summary.index else 'N/A',
        f"{test_gt_summary.loc['spring_2024', 'ram_pct_gt']:.3f}%" if 'spring_2024' in test_gt_summary.index else 'N/A',
        f"{test_gt_summary.loc['spring_2025', 'ram_pct_gt']:.3f}%" if 'spring_2025' in test_gt_summary.index else 'N/A']
})

print("TABLE 1: TEST SET - MULTICLASS (THREE GROUPS)")
print(test_table.to_string(index=False))

# Save test set results
test_table.to_csv(Path("table1_test_set_results_threegroups.csv"), index=False)
df_test.to_csv(Path("table1_test_set_detailed_threegroups.csv"), index=False)


# ---------------------------------------------------------
# TABLE 2: Big Unlabelled Dataset Analysis (predictions only)
# ---------------------------------------------------------

print("TABLE 2: BIG UNLABELLED DATASET ANALYSIS (MULTICLASS - THREE GROUPS)")


# Process unlabelled dataset
unlabelled_records = []
found_genotypes = {"winter": set(), "spring_2024": set(), "spring_2025": set(), "unknown": set()}
group_counts_unlabelled = {"winter": 0, "spring_2024": 0, "spring_2025": 0}

for fn in os.listdir(UNLABELLED_PRED_DIR):
    if not fn.endswith(".png"):
        continue

    # Parse filename
    try:
        genotype, leafID = parse_filename(fn)
    except:
        print(f"Warning: Could not parse {fn}, skipping")
        continue

    # Determine group using the lists
    group = get_genotype_group_from_lists(genotype)

    if group is None:
        found_genotypes["unknown"].add(genotype)
        continue

    found_genotypes[group].add(genotype)
    group_counts_unlabelled[group] = group_counts_unlabelled.get(group, 0) + 1

    # Load prediction
    pred = load_img(UNLABELLED_PRED_DIR / fn)
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    # Load original image for leaf mask
    img_path = UNLABELLED_DATA_DIR / fn
    if not img_path.exists():
        print(f"Warning: Image not found for {fn}, skipping")
        continue

    img = load_img(img_path)
    leaf_mask = get_leaf_mask(img)
    leaf_area = leaf_mask.sum()

    if leaf_area == 0:
        print(f"Warning: Empty leaf mask for {fn}, skipping")
        continue

    # Calculate prediction disease areas
    rust_area_pred = ((pred == 1) & leaf_mask).sum()
    ram_area_pred = ((pred == 2) & leaf_mask).sum()

    rust_pct_pred = rust_area_pred / leaf_area * 100
    ram_pct_pred = ram_area_pred / leaf_area * 100

    unlabelled_records.append({
        "genotype": genotype,
        "group": group,
        "rust_pct_pred": rust_pct_pred,
        "ram_pct_pred": ram_pct_pred,
    })

# Create DataFrame
df_unlabelled = pd.DataFrame(unlabelled_records)

print(f"\nGenotypes found per group:")
for group, genotypes in found_genotypes.items():
    if genotypes:
        print(f"  {group}: {sorted(genotypes)}")
print(f"\nSample counts per group:")
for group, count in group_counts_unlabelled.items():
    print(f"  {group}: {count} leaves")

# Perform statistical tests on unlabelled dataset
print("STATISTICAL TESTS - UNLABELLED DATASET")
unlabelled_stats = perform_statistical_tests_multi_group(
    df_unlabelled,
    value_cols=['rust_pct_pred', 'ram_pct_pred'],
    groups=['winter', 'spring_2024', 'spring_2025']
)
print_statistical_summary_multi_group(unlabelled_stats, "UNLABELLED DATASET")

# Compute group-level means for unlabelled dataset
unlabelled_summary = df_unlabelled.groupby("group")[["rust_pct_pred", "ram_pct_pred"]].mean()

# Create formatted table for unlabelled dataset
unlabelled_table = pd.DataFrame({
    'Group': ['Winter', 'Spring 2024', 'Spring 2025'],
    'Rust % (Pred)': [
        f"{unlabelled_summary.loc['winter', 'rust_pct_pred']:.3f}%" if 'winter' in unlabelled_summary.index else 'N/A',
        f"{unlabelled_summary.loc['spring_2024', 'rust_pct_pred']:.3f}%" if 'spring_2024' in unlabelled_summary.index else 'N/A',
        f"{unlabelled_summary.loc['spring_2025', 'rust_pct_pred']:.3f}%" if 'spring_2025' in unlabelled_summary.index else 'N/A'],
    'Ram % (Pred)': [
        f"{unlabelled_summary.loc['winter', 'ram_pct_pred']:.3f}%" if 'winter' in unlabelled_summary.index else 'N/A',
        f"{unlabelled_summary.loc['spring_2024', 'ram_pct_pred']:.3f}%" if 'spring_2024' in unlabelled_summary.index else 'N/A',
        f"{unlabelled_summary.loc['spring_2025', 'ram_pct_pred']:.3f}%" if 'spring_2025' in unlabelled_summary.index else 'N/A']
})


print("TABLE 2: BIG UNLABELLED DATASET - MULTICLASS (THREE GROUPS)")
print(unlabelled_table.to_string(index=False))

# Save unlabelled dataset results
unlabelled_table.to_csv(Path("table2_unlabelled_results_threegroups.csv"), index=False)
df_unlabelled.to_csv(Path("table2_unlabelled_detailed_threegroups.csv"), index=False)



print("FILES SAVED:")

print("  - table1_test_set_results_threegroups.csv")
print("  - table1_test_set_detailed_threegroups.csv")
print("  - table2_unlabelled_results_threegroups.csv")
print("  - table2_unlabelled_detailed_threegroups.csv")
print("  - statistical_summary_for_paper_threegroups.md")
