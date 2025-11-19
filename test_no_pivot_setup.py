#!/usr/bin/env python3
"""
Test script for no-pivot ablation setup.
Validates all components before running full experiments.

NOTE: Run this in the lrlt_exp conda environment:
    conda activate lrlt_exp
    python test_no_pivot_setup.py
"""
import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import os

def log(message, level="INFO"):
    """Print colored log message."""
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m"
    }
    color = colors.get(level, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{level}] {message}{reset}")
    sys.stdout.flush()

def test_vector_db_creation():
    """Test vector database creation for Konkani."""
    log("="*80, "INFO")
    log("TEST 1: Vector Database Creation (Konkani)", "INFO")
    log("="*80, "INFO")
    
    test_db = "test_konkani_no_pivot_db"
    
    cmd = [
        "python", "scripts/create_vector_db_no_pivot.py",
        "--dataset", "predictionguard/english-hindi-marathi-konkani-corpus",
        "--source", "eng",
        "--target", "gom",
        "--db", test_db
    ]
    
    log(f"Running: {' '.join(cmd)}", "INFO")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log("✅ Vector database created successfully!", "SUCCESS")
        log(result.stdout, "INFO")
        
        # Check if database was created
        db_path = Path(test_db)
        if db_path.exists():
            log(f"✅ Database directory exists: {db_path}", "SUCCESS")
            return True
        else:
            log(f"❌ Database directory not found: {db_path}", "ERROR")
            return False
            
    except subprocess.CalledProcessError as e:
        log(f"❌ Vector database creation failed!", "ERROR")
        log(f"Error: {e.stderr}", "ERROR")
        return False

def test_inference_script():
    """Test inference script with 3 samples."""
    log("\n" + "="*80, "INFO")
    log("TEST 2: Inference Script (3 samples, k=3)", "INFO")
    log("="*80, "INFO")
    
    test_db = "test_konkani_no_pivot_db"
    output_csv = "test_no_pivot_output.csv"
    scores_json = "test_no_pivot_scores.json"
    
    cmd = [
        "python", "scripts/run_inference_no_pivot.py",
        "--dataset", "predictionguard/english-hindi-marathi-konkani-corpus",
        "--model", "Unbabel/TowerInstruct-7B-v0.2",
        "--source", "eng",
        "--target", "gom",
        "--db", test_db,
        "--output", output_csv,
        "--scores", scores_json,
        "--num-examples", "3",
        "--batch-size", "2",
        "--test-limit", "3"  # Only test 3 samples
        # NOTE: --wandb is NOT passed for tests
    ]
    
    log(f"Running: {' '.join(cmd)}", "INFO")
    log("This will load the model and run inference on 3 samples...", "INFO")
    log("W&B logging is DISABLED for tests", "INFO")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log("✅ Inference completed successfully!", "SUCCESS")
        
        # Check outputs
        if Path(output_csv).exists():
            log(f"✅ Output CSV created: {output_csv}", "SUCCESS")
            df = pd.read_csv(output_csv)
            log(f"   Samples processed: {len(df)}", "INFO")
            log(f"   Columns: {list(df.columns)}", "INFO")
        else:
            log(f"❌ Output CSV not found: {output_csv}", "ERROR")
            return False
        
        if Path(scores_json).exists():
            log(f"✅ Scores JSON created: {scores_json}", "SUCCESS")
            with open(scores_json, 'r') as f:
                scores = json.load(f)
            log(f"   BLEU: {scores['bleu']:.2f}", "INFO")
            log(f"   chrF: {scores['chrf']:.2f}", "INFO")
            log(f"   chrF++: {scores['chrf++']:.2f}", "INFO")
            log(f"   Config: {scores['config']}", "INFO")
            
            # Validate config
            if scores['config']['pivot'] != 'NONE (direct translation)':
                log("❌ Pivot should be 'NONE' for no-pivot experiments!", "ERROR")
                return False
            if scores['config']['source'] != 'eng':
                log("❌ Source should be 'eng'!", "ERROR")
                return False
            if scores['config']['target'] != 'gom':
                log("❌ Target should be 'gom'!", "ERROR")
                return False
                
            log("✅ Config validation passed!", "SUCCESS")
        else:
            log(f"❌ Scores JSON not found: {scores_json}", "ERROR")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        log(f"❌ Inference failed!", "ERROR")
        log(f"Error output: {e.stderr}", "ERROR")
        return False

def test_prompt_format():
    """Verify prompt doesn't use pivot language."""
    log("\n" + "="*80, "INFO")
    log("TEST 3: Prompt Format Validation", "INFO")
    log("="*80, "INFO")
    
    output_csv = "test_no_pivot_output.csv"
    
    if not Path(output_csv).exists():
        log("❌ Cannot test prompt format - output CSV not found", "ERROR")
        return False
    
    df = pd.read_csv(output_csv)
    
    if 'prompt' not in df.columns:
        log("❌ No 'prompt' column in output CSV", "ERROR")
        return False
    
    # Check first prompt
    first_prompt = df['prompt'].iloc[0]
    log("Sample prompt (first 500 chars):", "INFO")
    log(first_prompt[:500], "INFO")
    log("...", "INFO")
    
    # Validate prompt doesn't mention pivot languages
    # Check for actual language name words, not substrings
    import re
    prompt_lower = first_prompt.lower()
    
    # Check for pivot language mentions (word boundaries to avoid false positives)
    forbidden_patterns = [
        r'\bmarathi\b',
        r'\bhindi\b', 
        r'\bstandard arabic\b',
        r'\bmsa\b',
        r'\bmar\b',  # MAR as language code
        r'\bhin\b'   # HIN as language code
    ]
    
    found_forbidden = []
    for pattern in forbidden_patterns:
        if re.search(pattern, prompt_lower):
            found_forbidden.append(pattern.strip('\\b'))
    
    if found_forbidden:
        log(f"❌ Prompt contains pivot language references: {found_forbidden}", "ERROR")
        log("   This suggests the script is using pivot language!", "ERROR")
        return False
    
    # Check for correct direct translation language pair
    # Should say "from ENG to GOM" or "English to Konkani"
    if not (('eng' in prompt_lower and 'gom' in prompt_lower) or 
            ('english' in prompt_lower and 'konkani' in prompt_lower)):
        log("❌ Prompt should mention English/ENG and Konkani/GOM", "ERROR")
        return False
    
    log("✅ Prompt format is correct (no pivot language detected)", "SUCCESS")
    return True

def cleanup_test_files():
    """Clean up test files."""
    log("\n" + "="*80, "INFO")
    log("CLEANUP: Removing test files", "INFO")
    log("="*80, "INFO")
    
    test_files = [
        "test_konkani_no_pivot_db",
        "test_no_pivot_output.csv",
        "test_no_pivot_scores.json"
    ]
    
    for file_path in test_files:
        path = Path(file_path)
        try:
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                log(f"✅ Removed directory: {file_path}", "SUCCESS")
            elif path.is_file():
                path.unlink()
                log(f"✅ Removed file: {file_path}", "SUCCESS")
        except Exception as e:
            log(f"⚠️  Could not remove {file_path}: {e}", "WARNING")

def main():
    log("="*80, "INFO")
    log("🧪 NO-PIVOT ABLATION SETUP TESTS", "INFO")
    log("="*80, "INFO")
    log("\nThis will test all components before running full experiments.", "INFO")
    log("Tests include:", "INFO")
    log("  1. Vector database creation", "INFO")
    log("  2. Inference script (3 samples)", "INFO")
    log("  3. Prompt format validation", "INFO")
    log("\nEstimated time: 5-10 minutes", "INFO")
    log("="*80 + "\n", "INFO")
    
    all_passed = True
    
    # Test 1: Vector DB creation
    if not test_vector_db_creation():
        log("\n❌ TEST 1 FAILED: Vector database creation", "ERROR")
        all_passed = False
        cleanup_test_files()
        return
    
    # Test 2: Inference script
    if not test_inference_script():
        log("\n❌ TEST 2 FAILED: Inference script", "ERROR")
        all_passed = False
        cleanup_test_files()
        return
    
    # Test 3: Prompt format
    if not test_prompt_format():
        log("\n❌ TEST 3 FAILED: Prompt format validation", "ERROR")
        all_passed = False
        cleanup_test_files()
        return
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    log("\n" + "="*80, "INFO")
    if all_passed:
        log("✅ ALL TESTS PASSED!", "SUCCESS")
        log("="*80, "INFO")
        log("\nYou can now run the full experiments with:", "INFO")
        log("  ./setup_no_pivot_experiments.sh", "INFO")
        log("\nOr manually:", "INFO")
        log("  python scripts/run_no_pivot_ablation.py", "INFO")
    else:
        log("❌ SOME TESTS FAILED", "ERROR")
        log("="*80, "INFO")
        log("\nPlease fix the issues before running full experiments.", "ERROR")
    log("="*80 + "\n", "INFO")

if __name__ == "__main__":
    main()

