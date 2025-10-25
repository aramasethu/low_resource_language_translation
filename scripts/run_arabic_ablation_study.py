#!/usr/bin/env python3
"""
Arabic Ablation Study Script - Mirrors run_ablation_study.py

Runs ablation study for Arabic translation with comprehensive W&B logging
"""

import os
import sys
import time
import json
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")

def log(message, level="INFO"):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️ "
    }.get(level, "  ")
    print(f"[{timestamp}] {prefix} {message}")

def run_inference_with_k(k, dataset, model, db, base_output_dir, k_idx, total_k, batch_size=8):
    """
    Run inference for a specific k value using run_inference_arabic.py
    """
    output_dir = Path(base_output_dir) / f"k_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"results_k{k}.csv"
    scores_file = output_dir / f"scores_k{k}.json"
    
    log("", "INFO")
    log("="*80, "INFO")
    log(f"EXPERIMENT {k_idx}/{total_k}: k={k}", "INFO")
    log("="*80, "INFO")
    log(f"Output: {output_file}", "INFO")
    log(f"Scores: {scores_file}", "INFO")
    
    # Build command
    cmd = [
        "python", "scripts/run_inference_arabic.py",
        "--dataset", dataset,
        "--model", model,
        "--output", str(output_file),
        "--scores", str(scores_file),
        "--db", db,
        "--num-examples", str(k),
        "--batch-size", str(batch_size)
    ]
    
    log(f"Running: {' '.join(cmd)}", "INFO")
    
    start_time = time.time()
    
    try:
        # Run without capturing output so we see real-time logs
        subprocess.run(cmd, check=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        # Load scores
        score_file = output_dir / f"scores_k{k}.json"
        if score_file.exists():
            with open(score_file) as f:
                scores = json.load(f)
            
            log(f"✅ k={k} complete in {elapsed_time/60:.1f} min", "SUCCESS")
            log(f"   BLEU: {scores.get('BLEU Score', 0):.4f}", "INFO")
            log(f"   chrF: {scores.get('chrF Score', 0):.4f}", "INFO")
            log(f"   chrF++: {scores.get('CHRF++ Score', 0):.4f}", "INFO")
            
            return {
                'k': k,
                'status': 'success',
                'scores': scores,
                'elapsed_time': elapsed_time,
                'output_dir': str(output_dir)
            }
        else:
            log(f"⚠️  Scores file not found: {score_file}", "WARNING")
            return {
                'k': k,
                'status': 'no_scores',
                'elapsed_time': elapsed_time,
                'output_dir': str(output_dir)
            }
            
    except subprocess.CalledProcessError as e:
        log(f"❌ k={k} failed: {e}", "ERROR")
        return {
            'k': k,
            'status': 'failed',
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }

def compile_results(results, output_dir, use_wandb=False):
    """
    Compile results from all k values into summary
    """
    log("", "INFO")
    log("Compiling results across all k values...", "INFO")
    
    # Extract successful results
    data = []
    for result in results:
        if result['status'] == 'success' and 'scores' in result:
            k = int(result['k'])
            scores = result['scores']
            data.append({
                'k': k,
                'BLEU': scores.get('BLEU Score', 0),
                'chrF': scores.get('chrF Score', 0),
                'chrF++': scores.get('CHRF++ Score', 0),
                'Time (min)': result['elapsed_time'] / 60
            })
    
    if not data:
        log("❌ No successful results to compile", "ERROR")
        return None
    
    summary_df = pd.DataFrame(data).sort_values('k')
    
    # Save summary
    summary_file = Path(output_dir) / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    log(f"✅ Summary saved: {summary_file}", "SUCCESS")
    
    # Calculate improvements over baseline (k=0)
    baseline = summary_df[summary_df['k'] == 0]
    if len(baseline) > 0:
        log("", "INFO")
        log("Improvements vs Baseline (k=0):", "INFO")
        log("-" * 80, "INFO")
        
        for _, row in summary_df.iterrows():
            k = int(row['k'])
            if k == 0:
                log(f"k={k:2d}: BASELINE", "INFO")
            else:
                bleu_imp = row['BLEU'] - baseline['BLEU'].values[0]
                chrf_imp = row['chrF'] - baseline['chrF'].values[0]
                chrfpp_imp = row['chrF++'] - baseline['chrF++'].values[0]
                
                log(f"k={k:2d}: BLEU: {bleu_imp:+.2f} | chrF: {chrf_imp:+.2f} | chrF++: {chrfpp_imp:+.2f}", "INFO")
                
                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        f'improvement/k{k}_bleu': bleu_imp,
                        f'improvement/k{k}_chrf': chrf_imp,
                        f'improvement/k{k}_chrfpp': chrfpp_imp,
                    })
    
    # Find best k
    best_bleu_idx = summary_df['BLEU'].idxmax()
    best_bleu_k = summary_df.loc[best_bleu_idx, 'k']
    best_bleu_score = summary_df.loc[best_bleu_idx, 'BLEU']
    
    log("", "INFO")
    log("="*80, "INFO")
    log(f"🏆 BEST k: {int(best_bleu_k)} (BLEU: {best_bleu_score:.4f})", "INFO")
    log("="*80, "INFO")
    
    # Log summary table to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"summary_table": wandb.Table(dataframe=summary_df)})
    
    return summary_df

def plot_results(summary_df, output_dir, use_wandb=False):
    """
    Create plots for ablation results
    """
    log("", "INFO")
    log("Creating visualization plots...", "INFO")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: BLEU vs k
    axes[0].plot(summary_df['k'], summary_df['BLEU'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
    axes[0].set_ylabel('BLEU Score', fontsize=12)
    axes[0].set_title('BLEU Score vs k', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: chrF vs k
    axes[1].plot(summary_df['k'], summary_df['chrF'], marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
    axes[1].set_ylabel('chrF Score', fontsize=12)
    axes[1].set_title('chrF Score vs k', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: chrF++ vs k
    axes[2].plot(summary_df['k'], summary_df['chrF++'], marker='^', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
    axes[2].set_ylabel('chrF++ Score', fontsize=12)
    axes[2].set_title('chrF++ Score vs k', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(output_dir) / "ablation_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    log(f"✅ Line plots saved: {plot_file}", "SUCCESS")
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"ablation_plots": wandb.Image(str(plot_file))})
    
    plt.close(fig)
    
    # Create bar chart comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    x = summary_df['k'].astype(str)
    width = 0.25
    x_pos = range(len(x))
    
    ax.bar([p - width for p in x_pos], summary_df['BLEU'], width, label='BLEU', alpha=0.8)
    ax.bar(x_pos, summary_df['chrF'], width, label='chrF', alpha=0.8)
    ax.bar([p + width for p in x_pos], summary_df['chrF++'], width, label='chrF++', alpha=0.8)
    
    ax.set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Arabic Translation: All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save bar chart
    bar_plot_file = Path(output_dir) / "bar_chart.png"
    plt.savefig(bar_plot_file, dpi=300, bbox_inches='tight')
    log(f"✅ Bar chart saved: {bar_plot_file}", "SUCCESS")
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"bar_chart": wandb.Image(str(bar_plot_file))})
    
    plt.close(fig2)

def main():
    parser = argparse.ArgumentParser(
        description="Arabic Ablation Study - Test different numbers of few-shot examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python scripts/run_arabic_ablation_study.py \\
        --dataset "predictionguard/arabic_acl_corpus" \\
        --model "Unbabel/TowerInstruct-7B-v0.1" \\
        --db "arabic_translations" \\
        --output-dir "ablation_results/arabic_600tokens" \\
        --k-values 0 1 2 3 4 5 6 7 8 9 10 \\
        --batch-size 4 \\
        --wandb \\
        --wandb-project "low-resource-translation-ablation" \\
        --wandb-run-name "arabic-ablation-600tokens"
        """
    )
    
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--db", required=True, help="Database name")
    parser.add_argument("--output-dir", default="ablation_results/arabic", 
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
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference (default: 8)")

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        log("", "INFO")
        log("Initializing Weights & Biases...", "INFO")
        
        run_name = args.wandb_run_name or f"arabic_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log(f"   Project: {args.wandb_project}", "INFO")
        log(f"   Run name: {run_name}", "INFO")
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset": args.dataset,
                "model": args.model,
                "k_values": args.k_values,
                "batch_size": args.batch_size,
                "language_pair": "MSA→Tunisian_Arabic",
                "pivot": "English"
            }
        )
    
    # Print experiment configuration
    log("", "INFO")
    log("="*80, "INFO")
    log("ARABIC ABLATION STUDY STARTING", "INFO")
    log("="*80, "INFO")
    log(f"Dataset: {args.dataset}", "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Database: {args.db}", "INFO")
    log(f"Output directory: {args.output_dir}", "INFO")
    log(f"k values: {args.k_values}", "INFO")
    log(f"Batch size: {args.batch_size}", "INFO")
    log(f"W&B enabled: {use_wandb}", "INFO")
    log("="*80, "INFO")
    
    start_time = time.time()
    
    # Run experiments for each k value
    results = []
    total_k = len(args.k_values)
    
    for idx, k in enumerate(args.k_values, 1):
        result = run_inference_with_k(
            k, 
            args.dataset,
            args.model,
            args.db,
            args.output_dir,
            idx,
            total_k,
            args.batch_size
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
    summary_df = compile_results(results, args.output_dir, use_wandb=use_wandb)
    
    if summary_df is not None:
        # Create plots
        plot_results(summary_df, args.output_dir, use_wandb=use_wandb)
    
    total_time = time.time() - start_time
    
    log("\n" + "="*80, "INFO")
    log("🎉 ARABIC ABLATION STUDY COMPLETE!", "INFO")
    log("="*80, "INFO")
    log(f"Total time: {total_time/60:.1f} minutes", "INFO")
    log(f"Results saved to: {args.output_dir}", "INFO")
    
    if use_wandb:
        log(f"W&B run: {wandb.run.url}", "INFO")
        wandb.finish()
    
    log("="*80, "INFO")

if __name__ == "__main__":
    main()

