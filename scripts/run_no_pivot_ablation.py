#!/usr/bin/env python3
"""
Run NO-PIVOT ablation study for k=3,4,5.

This tests whether few-shot examples help when there is NO pivot language.

Comparison:
- WITH pivot: English→Marathi→Konkani (with k examples)
- NO pivot: English→Konkani (with k examples) ← THIS SCRIPT

The goal is to isolate the contribution of few-shot examples independent of the pivot.
"""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

def run_experiment(k, language, dataset, model, source, target, db, base_output_dir, use_wandb=False, wandb_project=None):
    """
    Run a single no-pivot experiment.
    
    Args:
        k: Number of few-shot examples
        language: Language name (for logging)
        dataset: Dataset name
        model: Model name
        source: Source language
        target: Target language
        db: Database path
        base_output_dir: Base output directory
        use_wandb: Whether to enable W&B logging
        wandb_project: W&B project name
    
    Returns:
        dict: Results including scores
    """
    log("="*80, "INFO")
    log(f"🔬 EXPERIMENT: {language} - k={k} (NO PIVOT)", "INFO")
    log("="*80, "INFO")
    
    # Create output directory
    output_dir = Path(base_output_dir) / f"k_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_dir / f"results_k{k}_no_pivot.csv"
    scores_json = output_dir / f"scores_k{k}_no_pivot.json"
    
    log(f"Output directory: {output_dir}", "INFO")
    log(f"Translation: {source.upper()} → {target.upper()} (NO PIVOT)", "INFO")
    log(f"Few-shot examples: k={k}", "INFO")
    
    # Build command
    cmd = [
        "python", "scripts/run_inference_no_pivot.py",
        "--dataset", dataset,
        "--model", model,
        "--source", source,
        "--target", target,
        "--db", db,
        "--output", str(output_csv),
        "--scores", str(scores_json),
        "--num-examples", str(k),
        "--batch-size", "8"
    ]
    
    # Add W&B flags if enabled
    if use_wandb:
        cmd.extend(["--wandb"])
        if wandb_project:
            cmd.extend(["--wandb-project", wandb_project])
        run_name = f"no-pivot-{language}-k{k}"
        cmd.extend(["--wandb-run-name", run_name])
    
    log(f"Command: {' '.join(cmd)}", "DEBUG")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        log(f"✅ Experiment completed in {elapsed:.2f}s", "SUCCESS")
        
        # Load scores
        with open(scores_json, 'r') as f:
            scores = json.load(f)
        
        scores['k'] = k
        scores['elapsed_time'] = elapsed
        scores['language'] = language
        scores['pivot'] = 'NONE'
        
        return scores
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        log(f"❌ Experiment failed after {elapsed:.2f}s: {e}", "ERROR")
        return {
            'k': k,
            'language': language,
            'pivot': 'NONE',
            'status': 'FAILED',
            'error': str(e),
            'elapsed_time': elapsed
        }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run no-pivot ablation study for k=3,4,5",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="low-resource-translation-no-pivot", 
                       help="W&B project name")
    parser.add_argument("--k-values", nargs='+', type=int, default=[3, 4, 5],
                       help="K values to test (default: 3 4 5)")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name="no-pivot-ablation-study",
            config={
                "k_values": args.k_values,
                "experiment_type": "no-pivot-ablation"
            }
        )
        log("✅ W&B logging enabled", "SUCCESS")
    elif args.wandb and not WANDB_AVAILABLE:
        log("⚠️  W&B requested but not available - continuing without logging", "WARNING")
    
    log("="*80, "INFO")
    log("🚀 NO-PIVOT ABLATION STUDY", "INFO")
    log("Testing k=3,4,5 WITHOUT pivot language", "INFO")
    log("="*80, "INFO")
    
    # Configuration
    K_VALUES = args.k_values
    
    EXPERIMENTS = [
        {
            'language': 'Konkani',
            'dataset': 'predictionguard/english-hindi-marathi-konkani-corpus',
            'model': 'Unbabel/TowerInstruct-7B-v0.2',
            'source': 'eng',
            'target': 'gom',
            'db': 'konkani_no_pivot_db',
            'output_dir': 'ablation_no_pivot/konkani'
        },
        {
            'language': 'Tunisian_Arabic',
            'dataset': 'predictionguard/arabic_acl_corpus',
            'model': 'Unbabel/TowerInstruct-7B-v0.2',
            'source': 'eng',
            'target': 'tun',
            'db': 'arabic_no_pivot_db',
            'output_dir': 'ablation_no_pivot/arabic'
        }
    ]
    
    log(f"📊 Configuration:", "INFO")
    log(f"   K values: {K_VALUES}", "INFO")
    log(f"   Languages: {[exp['language'] for exp in EXPERIMENTS]}", "INFO")
    log(f"   Model: {EXPERIMENTS[0]['model']}", "INFO")
    log("="*80, "INFO")
    
    all_results = []
    total_start = time.time()
    
    # Run all experiments
    for exp in EXPERIMENTS:
        log(f"\n{'='*80}", "INFO")
        log(f"📍 STARTING {exp['language']} EXPERIMENTS", "INFO")
        log(f"{'='*80}", "INFO")
        
        for k in K_VALUES:
            result = run_experiment(
                k=k,
                language=exp['language'],
                dataset=exp['dataset'],
                model=exp['model'],
                source=exp['source'],
                target=exp['target'],
                db=exp['db'],
                base_output_dir=exp['output_dir'],
                use_wandb=use_wandb,
                wandb_project=args.wandb_project
            )
            all_results.append(result)
            
            log(f"\n📊 Results for {exp['language']} k={k} (NO PIVOT):", "INFO")
            if 'bleu' in result:
                log(f"   BLEU:   {result['bleu']:.2f}", "INFO")
                log(f"   chrF:   {result['chrf']:.2f}", "INFO")
                log(f"   chrF++: {result['chrf++']:.2f}", "INFO")
            else:
                log(f"   Status: {result.get('status', 'UNKNOWN')}", "WARNING")
            log("", "INFO")
    
    total_elapsed = time.time() - total_start
    
    # Save summary
    log("="*80, "INFO")
    log("💾 Saving experiment summary...", "INFO")
    
    summary_dir = Path("ablation_no_pivot/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_experiments': len(all_results),
            'total_time': total_elapsed,
            'k_values': K_VALUES,
            'results': all_results
        }, f, indent=2)
    
    log(f"   Saved to: {summary_file}", "INFO")
    
    # Create comparison table
    log("\n📊 SUMMARY TABLE (NO-PIVOT RESULTS):", "INFO")
    log("="*80, "INFO")
    
    df_results = pd.DataFrame(all_results)
    
    if 'bleu' in df_results.columns:
        for language in df_results['language'].unique():
            lang_data = df_results[df_results['language'] == language]
            log(f"\n{language}:", "INFO")
            log(f"{'k':<5} {'BLEU':<8} {'chrF':<8} {'chrF++':<8} {'Time(s)':<10}", "INFO")
            log("-" * 50, "INFO")
            for _, row in lang_data.iterrows():
                log(f"{row['k']:<5} {row['bleu']:<8.2f} {row['chrf']:<8.2f} {row['chrf++']:<8.2f} {row['elapsed_time']:<10.2f}", "INFO")
    
    log("\n" + "="*80, "INFO")
    log("✅ NO-PIVOT ABLATION STUDY COMPLETE!", "SUCCESS")
    log(f"   Total time: {total_elapsed/60:.2f} minutes", "INFO")
    log(f"   Results saved to: ablation_no_pivot/", "INFO")
    log("="*80, "INFO")
    
    # Log summary to W&B
    if use_wandb:
        # Log overall summary
        wandb.log({
            "total_experiments": len(all_results),
            "total_time_minutes": total_elapsed/60,
            "k_values": K_VALUES
        })
        
        # Log per-language summaries
        for language in df_results['language'].unique():
            lang_data = df_results[df_results['language'] == language]
            for _, row in lang_data.iterrows():
                if 'bleu' in row:
                    wandb.log({
                        f"{language}/k{int(row['k'])}/bleu": row['bleu'],
                        f"{language}/k{int(row['k'])}/chrf": row['chrf'],
                        f"{language}/k{int(row['k'])}/chrf++": row['chrf++'],
                        f"{language}/k{int(row['k'])}/time": row['elapsed_time']
                    })
        
        wandb.finish()
        log("✅ W&B logging complete", "SUCCESS")

if __name__ == "__main__":
    main()

