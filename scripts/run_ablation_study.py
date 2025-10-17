#!/usr/bin/env python3
"""
Ablation Study: Impact of Number of Few-Shot Examples (k) on Translation Quality

This script runs experiments with varying numbers of few-shot examples (k=0,1,3,5,7,10)
to evaluate the impact on translation performance.

Addresses Reviewer 1 comment about lack of ablation on the core hyperparameter k.
"""

import argparse
import json
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import time
import sys

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WARNING: wandb not installed. Install with: pip install wandb")
    print("    Experiments will run without wandb logging.\n")

def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

def run_inference_with_k(k, dataset, model, pivot, source, target, db, base_output_dir, k_idx, total_k):
    """
    Run inference with a specific number of few-shot examples.
    
    Args:
        k: Number of few-shot examples
        dataset: Dataset name
        model: Model name
        pivot: Pivot language
        source: Source language
        target: Target language
        db: Database name
        base_output_dir: Base directory for outputs
        k_idx: Current k index (for progress)
        total_k: Total number of k values
    
    Returns:
        dict: Results including scores and file paths
    """
    start_time = time.time()
    
    # Create output directory for this k value
    output_dir = Path(base_output_dir) / f"k_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output files
    output_csv = output_dir / f"results_k{k}.csv"
    scores_json = output_dir / f"scores_k{k}.json"
    
    log(f"\n{'='*80}", "INFO")
    log(f"🔬 EXPERIMENT {k_idx}/{total_k}: k={k} few-shot examples", "INFO")
    log(f"{'='*80}", "INFO")
    log(f"Output directory: {output_dir}", "INFO")
    
    if k == 0:
        log("Running ZERO-SHOT baseline (no few-shot examples)", "INFO")
    else:
        log(f"Using {k} semantically similar examples for few-shot learning", "INFO")
    
    # Build command
    cmd = [
        "python", "scripts/run_inference.py",
        "--dataset", dataset,
        "--model", model,
        "--pivot", pivot,
        "--source", source,
        "--target", target,
        "--db", db,
        "--output", str(output_csv),
        "--scores", str(scores_json),
        "--num-examples", str(k)
    ]
    
    log(f"Command: {' '.join(cmd)}", "DEBUG")
    
    # Note: We don't pass --wandb flags to individual inference runs because
    # this ablation script handles W&B logging at a higher level. Individual
    # runs within an ablation study are tracked as part of the overall experiment.
    
    if k_idx == 1:
        log("⏳ First run: Model will be downloaded if not cached (~14 GB, ~10-15 min)", "INFO")
        log("   Subsequent runs will load from cache (much faster!)", "INFO")
    
    log(f"▶️  Starting inference for k={k}...", "INFO")
    
    # Run inference
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print relevant output
        for line in result.stdout.split('\n'):
            if any(keyword in line.lower() for keyword in ['downloading', 'loading', 'bleu', 'chrf', 'score', 'generated']):
                print(f"    {line}")
        
        # Load scores
        with open(scores_json, 'r') as f:
            scores = json.load(f)
        
        elapsed = time.time() - start_time
        log(f"✅ COMPLETED k={k} in {elapsed/60:.1f} minutes", "SUCCESS")
        log(f"   BLEU: {scores.get('BLEU Score', 0):.2f} | chrF: {scores.get('chrF Score', 0):.2f} | chrF++: {scores.get('CHRF++ Score', 0):.2f}", "SUCCESS")
        
        return {
            'k': k,
            'output_csv': str(output_csv),
            'scores_json': str(scores_json),
            'scores': scores,
            'status': 'success',
            'elapsed_time': elapsed
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        log(f"❌ FAILED k={k} after {elapsed/60:.1f} minutes", "ERROR")
        log(f"Error: {e}", "ERROR")
        log(f"STDERR: {e.stderr}", "ERROR")
        return {
            'k': k,
            'status': 'failed',
            'error': str(e),
            'elapsed_time': elapsed
        }

def compile_results(results, output_dir, use_wandb=False):
    """
    Compile results from all k values into summary tables and plots.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save compiled results
        use_wandb: Whether to log to wandb
    """
    output_dir = Path(output_dir)
    
    log("📊 Compiling results from all experiments...", "INFO")
    
    # Extract successful results
    successful_results = [r for r in results if r['status'] == 'success']
    failed_count = len([r for r in results if r['status'] == 'failed'])
    
    if not successful_results:
        log("❌ No successful results to compile!", "ERROR")
        return
    
    log(f"✅ Successfully completed: {len(successful_results)}/{len(results)} experiments", "INFO")
    if failed_count > 0:
        log(f"⚠️  Failed experiments: {failed_count}", "WARNING")
    
    # Create summary dataframe
    summary_data = []
    for result in successful_results:
        k = result['k']
        scores = result['scores']
        summary_data.append({
            'k': k,
            'BLEU': scores.get('BLEU Score', 0),
            'chrF': scores.get('chrF Score', 0),
            'chrF++': scores.get('CHRF++ Score', 0),
            'Normalized_BLEU': scores.get('Normalized BLEU Score', 0),
            'time_minutes': result.get('elapsed_time', 0) / 60
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('k')
    
    # Save summary table
    summary_csv = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    log(f"💾 Summary table saved to: {summary_csv}", "INFO")
    
    log("\n" + "="*80, "INFO")
    log("📈 ABLATION STUDY RESULTS: Impact of k (Number of Few-Shot Examples)", "INFO")
    log("="*80, "INFO")
    print(summary_df.to_string(index=False))
    log("="*80, "INFO")
    
    # Calculate improvements
    if len(summary_df) > 1:
        baseline = summary_df[summary_df['k'] == 0]
        if not baseline.empty:
            log("\n📊 Improvement over Zero-Shot (k=0):", "INFO")
            log("-" * 80, "INFO")
            for _, row in summary_df[summary_df['k'] > 0].iterrows():
                k = row['k']
                bleu_imp = row['BLEU'] - baseline['BLEU'].values[0]
                chrf_imp = row['chrF'] - baseline['chrF'].values[0]
                chrfpp_imp = row['chrF++'] - baseline['chrF++'].values[0]
                log(f"k={k:2d}: BLEU: {bleu_imp:+.2f}  |  chrF: {chrf_imp:+.2f}  |  chrF++: {chrfpp_imp:+.2f}", "INFO")
            log("-" * 80, "INFO")
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                for _, row in summary_df[summary_df['k'] > 0].iterrows():
                    k = row['k']
                    bleu_imp = row['BLEU'] - baseline['BLEU'].values[0]
                    chrf_imp = row['chrF'] - baseline['chrF'].values[0]
                    chrfpp_imp = row['chrF++'] - baseline['chrF++'].values[0]
                    wandb.log({
                        f'improvement/k{k}_bleu': bleu_imp,
                        f'improvement/k{k}_chrf': chrf_imp,
                        f'improvement/k{k}_chrfpp': chrfpp_imp
                    })
    
    # Create visualizations
    log("📉 Generating visualizations...", "INFO")
    plot_results(summary_df, output_dir, use_wandb=use_wandb)
    
    # Save detailed JSON
    detailed_json = output_dir / "ablation_detailed_results.json"
    with open(detailed_json, 'w') as f:
        json.dump({
            'experiment_date': datetime.now().isoformat(),
            'summary': summary_data,
            'all_results': results
        }, f, indent=2)
    log(f"💾 Detailed results saved to: {detailed_json}", "INFO")
    
    # Log summary table to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary_table": wandb.Table(dataframe=summary_df)})

def plot_results(summary_df, output_dir, use_wandb=False):
    """
    Create visualization plots for the ablation study.
    
    Args:
        summary_df: DataFrame with summary results
        output_dir: Directory to save plots
        use_wandb: Whether to log to wandb
    """
    output_dir = Path(output_dir)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study: Impact of Number of Few-Shot Examples (k)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: BLEU score vs k
    axes[0, 0].plot(summary_df['k'], summary_df['BLEU'], 
                    marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
    axes[0, 0].set_ylabel('BLEU Score', fontsize=12)
    axes[0, 0].set_title('BLEU Score vs k', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    for i, row in summary_df.iterrows():
        axes[0, 0].annotate(f'{row["BLEU"]:.2f}', 
                           (row['k'], row['BLEU']), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
    
    # Plot 2: chrF score vs k
    axes[0, 1].plot(summary_df['k'], summary_df['chrF'], 
                    marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
    axes[0, 1].set_ylabel('chrF Score', fontsize=12)
    axes[0, 1].set_title('chrF Score vs k', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    for i, row in summary_df.iterrows():
        axes[0, 1].annotate(f'{row["chrF"]:.2f}', 
                           (row['k'], row['chrF']), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
    
    # Plot 3: chrF++ score vs k
    axes[1, 0].plot(summary_df['k'], summary_df['chrF++'], 
                    marker='^', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
    axes[1, 0].set_ylabel('chrF++ Score', fontsize=12)
    axes[1, 0].set_title('chrF++ Score vs k', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    for i, row in summary_df.iterrows():
        axes[1, 0].annotate(f'{row["chrF++"]:.2f}', 
                           (row['k'], row['chrF++']), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9)
    
    # Plot 4: All metrics together (normalized)
    # Normalize scores to 0-100 scale for comparison
    axes[1, 1].plot(summary_df['k'], summary_df['BLEU'], 
                    marker='o', linewidth=2, markersize=8, label='BLEU', color='#2E86AB')
    axes[1, 1].plot(summary_df['k'], summary_df['chrF'], 
                    marker='s', linewidth=2, markersize=8, label='chrF', color='#A23B72')
    axes[1, 1].plot(summary_df['k'], summary_df['chrF++'], 
                    marker='^', linewidth=2, markersize=8, label='chrF++', color='#F18F01')
    axes[1, 1].set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "ablation_study_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    log(f"💾 Plots saved to: {plot_file}", "INFO")
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"ablation_plots": wandb.Image(str(plot_file))})
    
    plt.close(fig)
    
    # Create a separate bar chart for better comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    x = range(len(summary_df))
    width = 0.25
    
    ax.bar([i - width for i in x], summary_df['BLEU'], width, label='BLEU', color='#2E86AB')
    ax.bar([i for i in x], summary_df['chrF'], width, label='chrF', color='#A23B72')
    ax.bar([i + width for i in x], summary_df['chrF++'], width, label='chrF++', color='#F18F01')
    
    ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study: Translation Quality Metrics by k', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['k'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    bar_plot_file = output_dir / "ablation_study_bar_chart.png"
    plt.savefig(bar_plot_file, dpi=300, bbox_inches='tight')
    log(f"💾 Bar chart saved to: {bar_plot_file}", "INFO")
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"bar_chart": wandb.Image(str(bar_plot_file))})
    
    plt.close(fig2)

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on number of few-shot examples (k)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Konkani experiment
  python scripts/run_ablation_study.py \\
      --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \\
      --model "Unbabel/TowerInstruct-7B-v0.1" \\
      --pivot "hin" \\
      --source "mar" \\
      --target "gom" \\
      --db "konkani_translations" \\
      --output-dir "ablation_results/konkani"
  
  # Tunisian Arabic experiment
  python scripts/run_ablation_study.py \\
      --dataset "predictionguard/arabic_acl_corpus" \\
      --model "Unbabel/TowerInstruct-7B-v0.1" \\
      --pivot "msa" \\
      --source "eng" \\
      --target "tun" \\
      --db "arabic_translations" \\
      --output-dir "ablation_results/arabic"
        """
    )
    
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--pivot", required=True, help="Pivot language column")
    parser.add_argument("--source", required=True, help="Source language column")
    parser.add_argument("--target", required=True, help="Target language column")
    parser.add_argument("--db", required=True, help="Database name")
    parser.add_argument("--output-dir", default="ablation_results", 
                       help="Base output directory for results")
    parser.add_argument("--k-values", nargs='+', type=int, 
                       default=[0, 1, 3, 5, 7, 10],
                       help="Values of k to test (default: 0 1 3 5 7 10)")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="low-resource-translation",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None,
                       help="W&B run name (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    log("="*80, "INFO")
    log("🚀 ABLATION STUDY: Number of Few-Shot Examples (k)", "INFO")
    log("="*80, "INFO")
    log(f"Dataset: {args.dataset}", "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Languages: {args.pivot} (pivot) -> {args.source} (source) -> {args.target} (target)", "INFO")
    log(f"Database: {args.db}", "INFO")
    log(f"Testing k values: {args.k_values}", "INFO")
    log(f"Output directory: {output_dir}", "INFO")
    
    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        log("⚠️  WARNING: --wandb flag set but wandb not installed!", "WARNING")
        log("   Install with: pip install wandb", "WARNING")
        use_wandb = False
    
    if use_wandb:
        run_name = args.wandb_run_name or f"ablation_{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log(f"📊 Initializing Weights & Biases logging...", "INFO")
        log(f"   Project: {args.wandb_project}", "INFO")
        log(f"   Run name: {run_name}", "INFO")
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset": args.dataset,
                "model": args.model,
                "pivot": args.pivot,
                "source": args.source,
                "target": args.target,
                "k_values": args.k_values,
                "num_experiments": len(args.k_values)
            },
            tags=["ablation", args.target, f"k_{min(args.k_values)}-{max(args.k_values)}"]
        )
        log("✅ W&B initialized successfully", "SUCCESS")
    
    log("="*80, "INFO")
    
    # Run experiments for each k value
    results = []
    total_k = len(args.k_values)
    
    log(f"\n🔬 Running {total_k} experiments with k values: {args.k_values}", "INFO")
    log(f"⏱️  Estimated time: {total_k * 25}min - {total_k * 40}min", "INFO")
    log("", "INFO")
    
    for idx, k in enumerate(args.k_values, 1):
        result = run_inference_with_k(
            k=k,
            dataset=args.dataset,
            model=args.model,
            pivot=args.pivot,
            source=args.source,
            target=args.target,
            db=args.db,
            base_output_dir=output_dir,
            k_idx=idx,
            total_k=total_k
        )
        results.append(result)
        
        # Log to wandb after each experiment
        if use_wandb and result['status'] == 'success':
            scores = result['scores']
            wandb.log({
                f'k_{k}/bleu': scores.get('BLEU Score', 0),
                f'k_{k}/chrf': scores.get('chrF Score', 0),
                f'k_{k}/chrfpp': scores.get('CHRF++ Score', 0),
                f'k_{k}/time_minutes': result.get('elapsed_time', 0) / 60,
                'progress': idx / total_k * 100
            })
    
    # Compile and analyze results
    log("\n" + "="*80, "INFO")
    log("📊 COMPILING RESULTS", "INFO")
    log("="*80, "INFO")
    compile_results(results, output_dir, use_wandb=use_wandb)
    
    total_time = time.time() - start_time
    
    log("\n" + "="*80, "INFO")
    log("🎉 ABLATION STUDY COMPLETE!", "SUCCESS")
    log("="*80, "INFO")
    log(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)", "INFO")
    log(f"💾 All results saved to: {output_dir}", "INFO")
    log("\n📁 Files generated:", "INFO")
    log(f"  ✓ ablation_summary.csv: Summary table of all results", "INFO")
    log(f"  ✓ ablation_study_plots.png: Visualization of results", "INFO")
    log(f"  ✓ ablation_study_bar_chart.png: Bar chart comparison", "INFO")
    log(f"  ✓ ablation_detailed_results.json: Detailed results in JSON format", "INFO")
    log(f"  ✓ k_*/: Individual results for each k value", "INFO")
    
    if use_wandb:
        log(f"\n🌐 View results on W&B: {wandb.run.url}", "INFO")
        wandb.finish()

if __name__ == "__main__":
    main()

