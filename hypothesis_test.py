"""
translation_hypothesis_tests.py

Ready-to-run Python script that implements the hypothesis testing pipeline we discussed.

Features:
- Loads the experiment summary table (embedded CSV string) into a pandas DataFrame.
- Permutation-based trend test to assess whether k (3,4,5) influences a metric.
- Paired permutation tests to compare pivot vs non-pivot and full vs hermes (matched by language and k).
- Bootstrap confidence intervals for mean differences.
- Produces a concise text report to the console for BLEU, chrF, COMET (customizable).

Usage: run with Python 3.8+. Requires: pandas, numpy, scipy.
Example: python translation_hypothesis_tests.py

Notes:
- The tests operate on the aggregated experiment summaries you provided. They are valid permutation/ bootstrap tests for summary-level comparisons.
- If you have per-sentence scores, replace the input DataFrame with those and use paired Wilcoxon / t-tests where appropriate.
"""

import io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -----------------------------
# Embedded CSV (your data)
# -----------------------------
csv_data = """
Experiment,Language,Variant,k_value,num_examples,BLEU,chrF,CHRF++,TER,COMET
arabic_full_k3,arabic,full,3,97,5.401,30.0248,25.9442,107.926,0.7526
arabic_full_k4,arabic,full,4,94,5.4789,29.7556,25.5957,105.629,0.7535
arabic_full_k5,arabic,full,5,96,5.6471,29.651,25.5596,98.8017,0.758
arabic_hermes_k3,arabic,hermes,3,95,4.6553,29.5545,25.3224,92.126,0.7236
arabic_hermes_k4,arabic,hermes,4,91,5.1115,30.2251,25.9337,92.7381,0.7219
arabic_hermes_k5,arabic,hermes,5,87,3.2174,29.8429,25.5154,91.3548,0.722
arabic_hermes_no_pivot_k3,arabic,hermes_no_pivot,3,100,0.0459,5.6882,4.8915,1788.68,0.3118
arabic_hermes_no_pivot_k4,arabic,hermes_no_pivot,4,92,0.0314,4.8152,4.1233,2218.07,0.3109
arabic_hermes_no_pivot_k5,arabic,hermes_no_pivot,5,84,0.0206,4.2309,3.6245,2620.85,0.3088
arabic_no_pivot_k3,arabic,no_pivot,3,100,0.0206,5.1641,4.3463,1840.71,0.3114
arabic_no_pivot_k4,arabic,no_pivot,4,100,0.0173,4.4701,3.8048,2270.51,0.3113
arabic_no_pivot_k5,arabic,no_pivot,5,100,0.0145,3.9463,3.3527,2715.28,0.3085

konkani_full_k3,konkani,full,3,205,5.7715,25.7897,21.8988,108.09,0.5315
konkani_full_k4,konkani,full,4,204,5.7644,24.3296,20.6452,108.429,0.5197
konkani_full_k5,konkani,full,5,205,5.012,20.6846,17.6044,109.944,0.4884
konkani_hermes_k3,konkani,hermes,3,205,8.4232,30.1028,25.938,103.523,0.5211
konkani_hermes_k4,konkani,hermes,4,205,8.3438,30.013,25.7884,106.49,0.5223
konkani_hermes_k5,konkani,hermes,5,205,7.9633,27.8343,24.0337,109.365,0.503
konkani_hermes_no_pivot_k3,konkani,hermes_no_pivot,3,5,0.3367,14.611,12.6674,1038.93,0.323
konkani_hermes_no_pivot_k4,konkani,hermes_no_pivot,4,5,0.284,13.4035,11.6181,1293.89,0.3191
konkani_no_pivot_k3,konkani,no_pivot,3,205,0.2753,10.6895,9.1401,1274.85,0.3296
konkani_no_pivot_k4,konkani,no_pivot,4,205,0.2185,9.3882,7.9994,1606.33,0.3264
konkani_no_pivot_k5,konkani,no_pivot,5,205,0.1983,8.4391,7.1973,1928,0.3262
"""

# Load into DataFrame
df = pd.read_csv(io.StringIO(csv_data))

# Normalize variant labels for pivot detection
pivot_map = {
    'full': 'pivot',
    'hermes': 'pivot',
    'no_pivot': 'nonpivot',
    'hermes_no_pivot': 'nonpivot'
}

df['pivotness'] = df['Variant'].map(pivot_map)

# -----------------------------
# Utility functions
# -----------------------------

def permutation_trend_test(subdf, metric='BLEU', n_perm=10000, seed=42):
    """Test whether metric shows a trend with k using Spearman correlation and permutation.

    Works even with small sample sizes (e.g., k=3,4,5 one value each). Returns observed rho and two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    ks = subdf['k_value'].values
    vals = subdf[metric].values
    if len(vals) < 3:
        return np.nan, np.nan
    obs_rho, _ = spearmanr(ks, vals)
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm_vals = rng.permutation(vals)
        perm_rhos[i], _ = spearmanr(ks, perm_vals)
    # two-sided p-value
    pval = np.mean(np.abs(perm_rhos) >= abs(obs_rho))
    return obs_rho, pval


def paired_permutation_test(series_a, series_b, n_perm=10000, seed=42):
    """Paired permutation test for mean difference between two matched arrays.

    series_a and series_b must be same length and matched (e.g., same k values).
    Returns observed mean difference (a-b) and two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(series_a)
    b = np.asarray(series_b)
    if a.shape != b.shape or a.size == 0:
        return np.nan, np.nan
    diffs = a - b
    obs_mean = diffs.mean()
    # For each permutation, flip sign of each paired difference at random (equivalent to permuting labels)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([1, -1], size=diffs.size)
        perm_stats[i] = (diffs * signs).mean()
    pval = np.mean(np.abs(perm_stats) >= abs(obs_mean))
    return obs_mean, pval


def bootstrap_mean_diff(series_a, series_b, n_boot=10000, seed=42, ci=0.95):
    """Bootstrap the mean difference (a - b) with percentile CI. Arrays must be same length and matched.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(series_a)
    b = np.asarray(series_b)
    n = a.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = (a[idx] - b[idx]).mean()
    lower = np.quantile(boots, (1 - ci) / 2)
    upper = np.quantile(boots, 1 - (1 - ci) / 2)
    return boots.mean(), (lower, upper)

# -----------------------------
# Higher-level tests
# -----------------------------

def test_k_effect(df, language, variant, metric='BLEU'):
    subset = df[(df['Language'] == language) & (df['Variant'] == variant)].sort_values('k_value')
    rho, p = permutation_trend_test(subset, metric=metric)
    return subset[['k_value', metric]], rho, p


def test_pivot_vs_nonpivot(df, language, model_label, metric='BLEU'):
    """Compare pivot vs nonpivot for a given language and model_label.

    model_label should be either 'full' (compare full vs no_pivot) or 'hermes' (compare hermes vs hermes_no_pivot).
    Returns paired arrays (pivot, nonpivot) matched by k, the mean diff and p-value from paired permutation test, and bootstrap CI.
    """
    if model_label == 'full':
        pivot_name = 'full'
        nonpivot_name = 'no_pivot'
    elif model_label == 'hermes':
        pivot_name = 'hermes'
        nonpivot_name = 'hermes_no_pivot'
    else:
        raise ValueError('model_label must be full or hermes')

    pivot_df = df[(df['Language'] == language) & (df['Variant'] == pivot_name)].set_index('k_value')
    nonpivot_df = df[(df['Language'] == language) & (df['Variant'] == nonpivot_name)].set_index('k_value')

    common_ks = sorted(set(pivot_df.index).intersection(nonpivot_df.index))
    if len(common_ks) == 0:
        return None
    a = pivot_df.loc[common_ks, metric].values
    b = nonpivot_df.loc[common_ks, metric].values
    mean_diff, pval = paired_permutation_test(a, b)
    boot_mean, ci = bootstrap_mean_diff(a, b)
    return common_ks, a, b, mean_diff, pval, boot_mean, ci


def test_model_vs_model(df, language, variant, metric='BLEU'):
    """Compare full vs hermes for a given language and variant (pivot/nonpivot)."""
    full_df = df[(df['Language'] == language) & (df['Variant'] == 'full') & (df['pivotness'] == df['pivotness'])]
    hermes_df = df[(df['Language'] == language) & (df['Variant'] == 'hermes')]
    # Better generic approach: find matching ks where both present
    full_df = df[(df['Language'] == language) & (df['Variant'] == 'full')].set_index('k_value')
    hermes_df = df[(df['Language'] == language) & (df['Variant'] == 'hermes')].set_index('k_value')
    common_ks = sorted(set(full_df.index).intersection(hermes_df.index))
    if len(common_ks) == 0:
        return None
    a = full_df.loc[common_ks, metric].values
    b = hermes_df.loc[common_ks, metric].values
    mean_diff, pval = paired_permutation_test(a, b)
    boot_mean, ci = bootstrap_mean_diff(a, b)
    return common_ks, a, b, mean_diff, pval, boot_mean, ci

# -----------------------------
# Report generation
# -----------------------------

def run_all_tests(df, metrics=['BLEU', 'chrF', 'COMET']):
    report_lines = []
    for language in df['Language'].unique():
        report_lines.append(f"\n=== Language: {language} ===")
        for variant in ['full', 'hermes']:
            report_lines.append(f"\n-- Variant: {variant} --")
            for metric in metrics:
                ks_table, rho, p = test_k_effect(df, language, variant, metric=metric)
                report_lines.append(f"Metric={metric}: Spearman rho (k vs {metric}) = {rho:.4f}, permutation p = {p:.4f}")
                report_lines.append(f"Values:\n{ks_table.to_string(index=False)}")

        # Pivot vs nonpivot comparisons (for both model labels)
        for model_label in ['full', 'hermes']:
            res = test_pivot_vs_nonpivot(df, language, model_label, metric='BLEU')
            if res is None:
                report_lines.append(f"No matching pivot/nonpivot runs found for {model_label} in {language}")
                continue
            common_ks, a, b, mean_diff, pval, boot_mean, ci = res
            report_lines.append(f"\nPivot vs Nonpivot (model={model_label}, metric=BLEU):")
            report_lines.append(f"k values used: {common_ks}")
            report_lines.append(f"Pivot BLEU: {a}")
            report_lines.append(f"Non-pivot BLEU: {b}")
            report_lines.append(f"Observed mean difference (pivot - nonpivot) = {mean_diff:.4f}, paired permutation p = {pval:.4f}")
            report_lines.append(f"Bootstrap mean diff = {boot_mean:.4f}, 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")

        # Model comparisons (full vs hermes)
        res = test_model_vs_model(df, language, variant='pivot', metric='BLEU')
        if res is not None:
            common_ks, a, b, mean_diff, pval, boot_mean, ci = res
            report_lines.append(f"\nFull vs Hermes (metric=BLEU): k={common_ks}")
            report_lines.append(f"Full BLEU: {a}")
            report_lines.append(f"Hermes BLEU: {b}")
            report_lines.append(f"Observed mean difference (full - hermes) = {mean_diff:.4f}, paired permutation p = {pval:.4f}")
            report_lines.append(f"Bootstrap mean diff = {boot_mean:.4f}, 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")

    return '\n'.join(report_lines)

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print('Running hypothesis tests on the provided summary table...')
    txt_report = run_all_tests(df, metrics=['BLEU', 'chrF', 'COMET'])
    print(txt_report)

# End of file

# === Additional Statistical Functions ===
from itertools import product

# Exact paired permutation p-value

def exact_paired_permutation_pvalue(x, y):
    diffs = np.array(x) - np.array(y)
    obs = abs(diffs.mean())
    n = len(diffs)
    total = 2**n
    extreme = 0
    for signs in product([1, -1], repeat=n):
        sdiffs = diffs * np.array(signs)
        if abs(sdiffs.mean()) >= obs:
            extreme += 1
    return extreme / total

# Cohen's d

def cohens_d(x, y):
    x = np.array(x); y = np.array(y)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled

# Cliff's delta

def cliffs_delta(x, y):
    x = np.array(x); y = np.array(y)
    n = len(x); m = len(y)
    greater = 0; less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    return (greater - less) / (n*m)

# === Report Export ===

def export_report_md(report, filename="report.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Translation Hypothesis Testing Report")
        f.write(report)


def export_report_csv(rows, filename="report.csv"):
    import csv
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Group", "Metric", "Observed", "p_value", "CI_low", "CI_high", "Cohens_d", "Cliffs_delta"])
        for r in rows:
            writer.writerow(r)



def generate_reports(rows, md_path="report.md", csv_path="report.csv"):
    """
    Generate Markdown and CSV reports from the collected statistical test results.
    """

    # -------------------------
    # Write Markdown Report
    # -------------------------
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Translation Hypothesis Testing Report\n\n")
        f.write("This report summarizes statistical comparisons across languages, models, pivot settings, and k-values.\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Language | Model | Pivot | Metric | k_pair | Test | Value |\n")
        f.write("|----------|--------|--------|--------|--------|--------|--------|\n")

        for row in rows:
            f.write(
                f"| {row['language']} | {row['model']} | {row['pivot']} | "
                f"{row['metric']} | {row['k_pair']} | {row['test']} | {row['value']} |\n"
            )

        f.write("\n\n---\n")
        f.write("Generated automatically by hypothesis_test.py\n")

    # -------------------------
    # Write CSV Report
    # -------------------------
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "model", "pivot", "metric", "k_pair", "test", "value"])

        for row in rows:
            writer.writerow([
                row["language"],
                row["model"],
                row["pivot"],
                row["metric"],
                row["k_pair"],
                row["test"],
                row["value"]
            ])

    print(f"Reports generated:\n - {md_path}\n - {csv_path}")