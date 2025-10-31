#!/usr/bin/env python3
"""
Orchestrator script to run sampling ablation study across:
- Multiple k values (0-10)
- Both retrieval strategies (semantic, random)
- Both models (Tower, Hermes)

GPU Configuration:
- Automatically uses available GPUs
- Set CUDA_VISIBLE_DEVICES environment variable to control which GPUs to use
- Example: CUDA_VISIBLE_DEVICES=0,1 python scripts/run_sampling_ablation_all.py
"""

import subprocess
import os
import sys
import time
from datetime import datetime
import json
import torch

# Configuration
MODELS = {
    "tower": "Unbabel/TowerInstruct-7B-v0.2",
    "hermes": "NousResearch/Hermes-2-Pro-Llama-3-8B"
}

DATASET = "predictionguard/english-hindi-marathi-konkani-corpus"
VECTOR_DB_PATH = "data/translations_db"
SOURCE_LANG = "eng"
PIVOT_LANG = "mar"
TARGET_LANG = "gom"

K_VALUES = list(range(0, 11))  # 0 to 10
RETRIEVAL_STRATEGIES = ["semantic", "random"]

OUTPUT_DIR = "outputs/sampling_ablation"
WANDB_PROJECT = "sampling-ablation-study"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

def run_inference(model_name, model_short, k, strategy):
    """
    Run inference for a single configuration.
    
    Returns:
        dict: Result with status and metrics
    """
    # Determine prompt type
    prompt_type = "few_shot" if k > 0 else "zero_shot"
    
    # Create output filename
    output_file = f"{OUTPUT_DIR}/{model_short}_{strategy}_k{k}.json"
    
    # Build command
    cmd = [
        "python", "scripts/run_inference_with_random_sampling.py",
        "--model_name", model_name,
        "--dataset_name", DATASET,
        "--output_path", output_file,
        "--prompt_type", prompt_type,
        "--num_fs_examples", str(k),
        "--retrieval_strategy", strategy,
        "--source_lang", SOURCE_LANG,
        "--pivot_lang", PIVOT_LANG,
        "--target_lang", TARGET_LANG,
        "--wandb",
        "--wandb-project", WANDB_PROJECT
    ]
    
    # Add vector DB path only for semantic strategy and k > 0
    if strategy == "semantic" and k > 0:
        cmd.extend(["--vector_db_path", VECTOR_DB_PATH])
    
    print("\n" + "=" * 80)
    print(f"🚀 RUNNING: {model_short} | {strategy} | k={k}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        # Don't capture output - let it stream to terminal in real-time
        result = subprocess.run(
            cmd,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ EXPERIMENT SUCCESS in {elapsed/60:.1f} minutes")
            print("=" * 80)
            sys.stdout.flush()
            
            # Try to load metrics from output file
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    return {
                        "status": "success",
                        "time_minutes": elapsed / 60,
                        "metrics": metrics
                    }
            except Exception as e:
                print(f"⚠️  Warning: Could not load metrics from {output_file}: {e}")
                return {
                    "status": "success",
                    "time_minutes": elapsed / 60,
                    "metrics": {}
                }
        else:
            print(f"\n❌ EXPERIMENT FAILED with exit code {result.returncode}")
            print("=" * 80)
            sys.stdout.flush()
            return {
                "status": "failed",
                "time_minutes": elapsed / 60,
                "error": f"Exit code: {result.returncode}"
            }
    
    except subprocess.TimeoutExpired:
        print(f"\n⏰ TIMEOUT after 1 hour")
        sys.stdout.flush()
        return {
            "status": "timeout",
            "time_minutes": 60.0
        }
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            "status": "error",
            "time_minutes": 0,
            "error": str(e)
        }

def main():
    """Run all experiment combinations."""
    print("\n" + "=" * 80)
    print("🔬 SAMPLING ABLATION STUDY - ALL COMBINATIONS")
    print("=" * 80)
    
    # Check GPU availability
    print("\n🔧 GPU CONFIGURATION")
    print("-" * 80)
    if torch.cuda.is_available():
        print(f"✅ CUDA available: YES")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all GPUs')
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    else:
        print("❌ CUDA available: NO")
        print("⚠️  WARNING: Running on CPU will be extremely slow!")
        print("   Press Ctrl+C to abort and configure GPU access")
    print("-" * 80)
    
    print(f"\n📋 EXPERIMENT CONFIGURATION")
    print("-" * 80)
    print(f"Models: {list(MODELS.keys())}")
    print(f"K values: {K_VALUES}")
    print(f"Strategies: {RETRIEVAL_STRATEGIES}")
    print(f"Total experiments: {len(MODELS) * len(K_VALUES) * len(RETRIEVAL_STRATEGIES)}")
    print("-" * 80)
    print("=" * 80)
    sys.stdout.flush()
    
    ensure_output_dir()
    
    # Track all results
    all_results = []
    start_time = time.time()
    
    experiment_num = 0
    total_experiments = len(MODELS) * len(K_VALUES) * len(RETRIEVAL_STRATEGIES)
    
    # Run all combinations
    for model_short, model_name in MODELS.items():
        for strategy in RETRIEVAL_STRATEGIES:
            for k in K_VALUES:
                experiment_num += 1
                
                print(f"\n📊 Progress: {experiment_num}/{total_experiments}")
                sys.stdout.flush()
                
                result = run_inference(model_name, model_short, k, strategy)
                
                # Store result
                all_results.append({
                    "model": model_short,
                    "model_full": model_name,
                    "strategy": strategy,
                    "k": k,
                    "result": result
                })
                
                # Save intermediate results
                summary_file = f"{OUTPUT_DIR}/experiment_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump({
                        "completed": experiment_num,
                        "total": total_experiments,
                        "elapsed_time_minutes": (time.time() - start_time) / 60,
                        "results": all_results
                    }, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    # Count successes and failures
    successes = sum(1 for r in all_results if r['result']['status'] == 'success')
    failures = sum(1 for r in all_results if r['result']['status'] != 'success')
    
    print(f"✅ Successes: {successes}")
    print(f"❌ Failures: {failures}")
    
    # Generate results table
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    for model_short in MODELS.keys():
        print(f"\n{model_short.upper()} Model:")
        print("-" * 80)
        print(f"{'Strategy':<12} {'k':<4} {'BLEU':<8} {'chrF':<8} {'chrF++':<8} {'Time(min)':<10} {'Status'}")
        print("-" * 80)
        
        for strategy in RETRIEVAL_STRATEGIES:
            for k in K_VALUES:
                # Find result
                result_entry = next(
                    (r for r in all_results 
                     if r['model'] == model_short and r['strategy'] == strategy and r['k'] == k),
                    None
                )
                
                if result_entry:
                    result = result_entry['result']
                    status = result['status']
                    
                    if status == 'success':
                        metrics = result.get('metrics', {})
                        bleu = metrics.get('bleu', 0)
                        chrf = metrics.get('chrf', 0)
                        chrf_pp = metrics.get('chrf_pp', 0)
                        time_min = result.get('time_minutes', 0)
                        
                        print(f"{strategy:<12} {k:<4} {bleu:<8.2f} {chrf:<8.2f} {chrf_pp:<8.2f} {time_min:<10.1f} ✅")
                    else:
                        print(f"{strategy:<12} {k:<4} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} ❌ {status}")
    
    print("\n" + "=" * 80)
    print(f"📁 Full results saved to: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

