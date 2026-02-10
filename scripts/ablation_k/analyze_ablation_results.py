#!/usr/bin/env python3
"""
Analyze and visualize existing ablation study results.

This script can be used to:
1. Re-generate visualizations from existing results
2. Perform statistical analysis on multiple runs
3. Create publication-ready tables and figures
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_results(results_dir):
    """Load all k-value results from a directory."""
    results_dir = Path(results_dir)
    results = []
    
    # Find all k_* subdirectories
    k_dirs = sorted(results_dir.glob("k_*"))
    
    for k_dir in k_dirs:
        k_value = int(k_dir.name.split("_")[1])
        scores_file = k_dir / f"scores_k{k_value}.json"
        
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                scores = json.load(f)
            results.append({
                'k': k_value,
                'BLEU': scores.get('BLEU Score', 0),
                'chrF': scores.get('chrF Score', 0),
                'chrF++': scores.get('CHRF++ Score', 0),
            })
    
    return pd.DataFrame(results).sort_values('k')

def analyze_improvement(df):
    """Analyze improvement over baseline (k=0)."""
    if df.empty or 0 not in df['k'].values:
        return None
    
    baseline = df[df['k'] == 0].iloc[0]
    improvements = []
    
    for _, row in df[df['k'] > 0].iterrows():
        improvements.append({
            'k': row['k'],
            'BLEU_improvement': row['BLEU'] - baseline['BLEU'],
            'BLEU_improvement_pct': ((row['BLEU'] - baseline['BLEU']) / baseline['BLEU'] * 100) if baseline['BLEU'] > 0 else 0,
            'chrF_improvement': row['chrF'] - baseline['chrF'],
            'chrF_improvement_pct': ((row['chrF'] - baseline['chrF']) / baseline['chrF'] * 100) if baseline['chrF'] > 0 else 0,
            'chrFpp_improvement': row['chrF++'] - baseline['chrF++'],
            'chrFpp_improvement_pct': ((row['chrF++'] - baseline['chrF++']) / baseline['chrF++'] * 100) if baseline['chrF++'] > 0 else 0,
        })
    
    return pd.DataFrame(improvements)

def find_optimal_k(df, metric='BLEU'):
    """Find the optimal k value for a given metric."""
    if df.empty:
        return None
    
    optimal_idx = df[metric].idxmax()
    optimal_row = df.loc[optimal_idx]
    
    return {
        'optimal_k': optimal_row['k'],
        'optimal_score': optimal_row[metric],
        'metric': metric
    }

def test_significance(df, metric='BLEU'):
    """
    Test if differences between consecutive k values are significant.
    Note: This assumes multiple runs. For single runs, it's just descriptive.
    """
    if len(df) < 2:
        return None
    
    comparisons = []
    k_values = sorted(df['k'].unique())
    
    for i in range(len(k_values) - 1):
        k1, k2 = k_values[i], k_values[i+1]
        score1 = df[df['k'] == k1][metric].values[0]
        score2 = df[df['k'] == k2][metric].values[0]
        
        comparisons.append({
            'comparison': f'k={k1} vs k={k2}',
            'difference': score2 - score1,
            'relative_change_pct': ((score2 - score1) / score1 * 100) if score1 > 0 else 0
        })
    
    return pd.DataFrame(comparisons)

def create_publication_table(df):
    """Create a publication-ready LaTeX table."""
    if df.empty:
        return ""
    
    # Calculate improvements over k=0
    improvements = analyze_improvement(df)
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Impact of Number of Few-Shot Examples (k) on Translation Quality}\n"
    latex += "\\label{tab:ablation_k}\n"
    latex += "\\begin{tabular}{c|ccc|ccc}\n"
    latex += "\\hline\n"
    latex += "& \\multicolumn{3}{c|}{Absolute Scores} & \\multicolumn{3}{c}{Improvement over k=0} \\\\\n"
    latex += "k & BLEU & chrF & chrF++ & $\\Delta$BLEU & $\\Delta$chrF & $\\Delta$chrF++ \\\\\n"
    latex += "\\hline\n"
    
    for _, row in df.iterrows():
        k = int(row['k'])
        if k == 0:
            latex += f"{k} & {row['BLEU']:.2f} & {row['chrF']:.2f} & {row['chrF++']:.2f} & - & - & - \\\\\n"
        else:
            imp_row = improvements[improvements['k'] == k].iloc[0]
            latex += f"{k} & {row['BLEU']:.2f} & {row['chrF']:.2f} & {row['chrF++']:.2f} & "
            latex += f"+{imp_row['BLEU_improvement']:.2f} & +{imp_row['chrF_improvement']:.2f} & +{imp_row['chrFpp_improvement']:.2f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def create_summary_report(results_dirs, output_file):
    """Create a comprehensive summary report across multiple experiments."""
    report = []
    report.append("="*80)
    report.append("ABLATION STUDY SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    all_results = {}
    
    for name, path in results_dirs.items():
        report.append(f"\n{name}")
        report.append("-"*80)
        
        df = load_results(path)
        if df.empty:
            report.append(f"No results found in {path}")
            continue
        
        all_results[name] = df
        
        # Basic statistics
        report.append(f"\nResults for k values: {df['k'].tolist()}")
        report.append(f"\nScore ranges:")
        report.append(f"  BLEU:  {df['BLEU'].min():.2f} - {df['BLEU'].max():.2f}")
        report.append(f"  chrF:  {df['chrF'].min():.2f} - {df['chrF'].max():.2f}")
        report.append(f"  chrF++: {df['chrF++'].min():.2f} - {df['chrF++'].max():.2f}")
        
        # Optimal k
        for metric in ['BLEU', 'chrF', 'chrF++']:
            optimal = find_optimal_k(df, metric)
            if optimal:
                report.append(f"\nOptimal k for {metric}: k={optimal['optimal_k']} (score={optimal['optimal_score']:.2f})")
        
        # Improvements
        improvements = analyze_improvement(df)
        if improvements is not None and not improvements.empty:
            report.append(f"\nImprovements over k=0 baseline:")
            report.append(improvements.to_string(index=False))
        
        # Statistical comparisons
        report.append(f"\nConsecutive k comparisons (BLEU):")
        comparisons = test_significance(df, 'BLEU')
        if comparisons is not None:
            report.append(comparisons.to_string(index=False))
    
    # Cross-language comparison
    if len(all_results) > 1:
        report.append("\n" + "="*80)
        report.append("CROSS-LANGUAGE COMPARISON")
        report.append("="*80)
        
        for metric in ['BLEU', 'chrF', 'chrF++']:
            report.append(f"\n{metric} Scores:")
            report.append("-"*80)
            
            comparison_data = {}
            for name, df in all_results.items():
                comparison_data[name] = df.set_index('k')[metric]
            
            comparison_df = pd.DataFrame(comparison_data)
            report.append(comparison_df.to_string())
    
    # Write report
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    print(report_text)
    return report_text

def main():
    parser = argparse.ArgumentParser(
        description="Analyze existing ablation study results"
    )
    
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing ablation results (with k_* subdirectories)")
    parser.add_argument("--output-dir", 
                       help="Directory to save analysis outputs (default: same as results-dir)")
    parser.add_argument("--language-name", default="",
                       help="Language pair name for labeling (e.g., 'Konkani', 'Arabic')")
    parser.add_argument("--create-latex", action="store_true",
                       help="Generate LaTeX table")
    parser.add_argument("--compare-with", nargs='+',
                       help="Additional result directories to compare with")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing results from: {results_dir}")
    print("="*80)
    
    # Load results
    df = load_results(results_dir)
    
    if df.empty:
        print(f"ERROR: No results found in {results_dir}")
        print("Expected structure: results_dir/k_*/scores_k*.json")
        return
    
    print(f"Found results for k values: {df['k'].tolist()}")
    print("\nSummary Statistics:")
    print(df.to_string(index=False))
    
    # Analyze improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    improvements = analyze_improvement(df)
    if improvements is not None:
        print(improvements.to_string(index=False))
    
    # Find optimal k
    print("\n" + "="*80)
    print("OPTIMAL k VALUES")
    print("="*80)
    for metric in ['BLEU', 'chrF', 'chrF++']:
        optimal = find_optimal_k(df, metric)
        if optimal:
            print(f"{metric}: k={optimal['optimal_k']} (score={optimal['optimal_score']:.2f})")
    
    # Significance tests
    print("\n" + "="*80)
    print("CONSECUTIVE k COMPARISONS")
    print("="*80)
    for metric in ['BLEU', 'chrF', 'chrF++']:
        print(f"\n{metric}:")
        comparisons = test_significance(df, metric)
        if comparisons is not None:
            print(comparisons.to_string(index=False))
    
    # Generate LaTeX table
    if args.create_latex:
        latex_table = create_publication_table(df)
        latex_file = output_dir / "ablation_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {latex_file}")
    
    # Save detailed analysis
    analysis_file = output_dir / "detailed_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ABLATION STUDY ANALYSIS: {args.language_name}\n")
        f.write("="*80 + "\n\n")
        f.write("Summary Statistics:\n")
        f.write(df.to_string(index=False) + "\n\n")
        
        if improvements is not None:
            f.write("Improvements over k=0:\n")
            f.write(improvements.to_string(index=False) + "\n\n")
        
        f.write("Optimal k values:\n")
        for metric in ['BLEU', 'chrF', 'chrF++']:
            optimal = find_optimal_k(df, metric)
            if optimal:
                f.write(f"  {metric}: k={optimal['optimal_k']} (score={optimal['optimal_score']:.2f})\n")
    
    print(f"\nDetailed analysis saved to: {analysis_file}")
    
    # Cross-experiment comparison
    if args.compare_with:
        print("\n" + "="*80)
        print("CROSS-EXPERIMENT COMPARISON")
        print("="*80)
        
        results_dirs = {args.language_name or "Main": results_dir}
        for i, other_dir in enumerate(args.compare_with, 1):
            results_dirs[f"Experiment_{i}"] = Path(other_dir)
        
        report_file = output_dir / "comparison_report.txt"
        create_summary_report(results_dirs, report_file)

if __name__ == "__main__":
    main()


