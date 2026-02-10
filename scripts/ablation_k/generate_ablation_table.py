#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class AblationResultsGenerator:
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.results = {}
    
    def load_results(self, pattern: str = "scores_*.json") -> None:
        
        score_files = list(self.results_dir.glob(pattern))
        
        if not score_files:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {self.results_dir}")
        
        for score_file in score_files:
            config_name = score_file.stem.replace("scores_", "").replace("_", " ").title()
            with open(score_file) as f:
                self.results[config_name] = json.load(f)
            print(f"Loaded: {config_name}")
    
    def create_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.results:
            raise ValueError("No results loaded. Call load_results() first.")
        
        # Determine which metrics to include
        all_metrics = set()
        for scores in self.results.values():
            all_metrics.update(scores.keys())
        
        if metrics is None:
            metrics = sorted(list(all_metrics))
        else:
            metrics = [m for m in metrics if m in all_metrics]
        
        # Build DataFrame
        data = {}
        for config_name, scores in self.results.items():
            data[config_name] = {metric: scores.get(metric, None) for metric in metrics}
        
        df = pd.DataFrame(data).T
        df.index.name = "Configuration"
        
        return df[metrics]  # Ensure consistent column order
    
    def save_table(self, df: pd.DataFrame, output_file: str, format: str = "csv") -> None:
        output_path = self.results_dir / output_file
        
        if format == "csv":
            df.to_csv(output_path)
        elif format == "markdown":
            df.to_markdown(output_path)
        elif format == "latex":
            df.to_latex(output_path)
        elif format == "html":
            df.to_html(output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Saved table to: {output_path}")
    
    def print_table(self, df: pd.DataFrame, precision: int = 2) -> None:
        print("\n" + "="*80)
        print("ABLATION RESULTS")
        print("="*80)
        print(df.round(precision).to_string())
        print("="*80 + "\n")
    
    def compare_ablations(self, df: pd.DataFrame, baseline_config: str, 
                         metric: str = "COMET Score") -> pd.DataFrame:
        if baseline_config not in df.index:
            raise ValueError(f"Baseline '{baseline_config}' not found in results")
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        baseline_score = df.loc[baseline_config, metric]
        comparison = df.copy()
        comparison[f"{metric} Δ"] = comparison[metric] - baseline_score
        comparison[f"{metric} % Δ"] = (comparison[f"{metric} Δ"] / baseline_score * 100).round(2)
        
        return comparison
    
    def get_best_config(self, df: pd.DataFrame, metric: str) -> tuple:
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        best_idx = df[metric].idxmax()
        return best_idx, df.loc[best_idx, metric]
    
    def summary_stats(self, df: pd.DataFrame) -> Dict:
        summary = {}
        for col in df.columns:
            summary[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "best_config": df[col].idxmax()
            }
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation-style results tables"
    )
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Directory containing scores JSON files"
    )
    parser.add_argument(
        "--pattern",
        default="scores_*.json",
        help="Glob pattern for score files"
    )
    parser.add_argument(
        "--output",
        default="ablation_results.csv",
        help="Output table filename"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown", "latex", "html"],
        default="csv",
        help="Output format"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to include (default: all)"
    )
    parser.add_argument(
        "--baseline",
        help="Baseline configuration for comparison"
    )
    parser.add_argument(
        "--comparison-metric",
        default="COMET Score",
        help="Metric to use for baseline comparison"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal precision for display"
    )
    
    args = parser.parse_args()
    
    # Generate results
    generator = AblationResultsGenerator(args.results_dir)
    generator.load_results(args.pattern)
    
    # Create table
    df = generator.create_table(args.metrics)
    
    # Print summary
    generator.print_table(df, args.precision)
    
    # Save table
    generator.save_table(df, args.output, args.format)
    
    # Baseline comparison if specified
    if args.baseline:
        comparison_df = generator.compare_ablations(df, args.baseline, args.comparison_metric)
        comparison_file = args.output.replace(f".{args.format}", f"_comparison.{args.format}")
        generator.save_table(comparison_df, comparison_file, args.format)
        print(f"\nBaseline comparison saved to: {comparison_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    stats = generator.summary_stats(df)
    for metric, stat in stats.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stat['mean']:.4f}")
        print(f"  Std: {stat['std']:.4f}")
        print(f"  Min: {stat['min']:.4f}")
        print(f"  Max: {stat['max']:.4f}")
        print(f"  Best: {stat['best_config']}")


if __name__ == "__main__":
    main()
