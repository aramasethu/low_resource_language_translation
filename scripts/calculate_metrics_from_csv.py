#!/usr/bin/env python3
"""
Script to calculate translation metrics for CSV files.
Processes CSV files from ablation_results directory with various k values.

Metrics calculated:
- COMET: Neural metric using Unbabel/wmt22-comet-da
- BLEU: Bilingual Evaluation Understudy
- chrF: Character n-gram F-score (good for low-resource languages)
- CHRF++: chrF with word unigrams and bigrams
- TER: Translation Edit Rate (lower is better)
"""

import pandas as pd
import json
import argparse
import os
import glob
from pathlib import Path
from datetime import datetime
import traceback
import sacrebleu

# MetricX-24 from Google Research (https://github.com/google-research/metricx)
# requires transformers==4.30.2 which conflicts with other dependencies.
# Setting METRICX_AVAILABLE = False to skip MetricX calculations.
# To enable MetricX, create a separate environment with transformers==4.30.2
# and install: git+https://github.com/google-research/mt-metrics-eval
METRICX_AVAILABLE = False
print("Note: MetricX-24 requires transformers==4.30.2 (incompatible). Skipping MetricX.")

from comet import load_from_checkpoint


def calculate_bleu(references, hypotheses):
    """Calculate BLEU score using sacrebleu."""
    try:
        # Format references as list of lists (sacrebleu format)
        formatted_refs = [[str(ref) for ref in references]]
        formatted_hyps = [str(hyp) for hyp in hypotheses]
        
        bleu = sacrebleu.corpus_bleu(formatted_hyps, formatted_refs)
        return float(bleu.score)
    except Exception as e:
        print(f"  Error calculating BLEU: {e}")
        return None


def calculate_chrf(references, hypotheses):
    """Calculate chrF score (character n-gram F-score).
    
    Good for low-resource languages as it:
    - Doesn't require tokenization
    - Works at character level
    - Better correlates with human judgment for morphologically rich languages
    """
    try:
        formatted_refs = [[str(ref) for ref in references]]
        formatted_hyps = [str(hyp) for hyp in hypotheses]
        
        chrf = sacrebleu.corpus_chrf(formatted_hyps, formatted_refs)
        return float(chrf.score)
    except Exception as e:
        print(f"  Error calculating chrF: {e}")
        return None


def calculate_chrf_plus_plus(references, hypotheses):
    """Calculate CHRF++ score (chrF with word unigrams and bigrams).
    
    Extends chrF with word-level information for better performance.
    """
    try:
        formatted_refs = [[str(ref) for ref in references]]
        formatted_hyps = [str(hyp) for hyp in hypotheses]
        
        # CHRF++ uses word_order=2 (word unigrams and bigrams)
        chrf_pp = sacrebleu.corpus_chrf(formatted_hyps, formatted_refs, word_order=2)
        return float(chrf_pp.score)
    except Exception as e:
        print(f"  Error calculating CHRF++: {e}")
        return None


def calculate_ter(references, hypotheses):
    """Calculate TER (Translation Edit Rate).
    
    Measures the number of edits needed to change the hypothesis to match the reference.
    Lower is better. Good for understanding how much post-editing is needed.
    """
    try:
        formatted_refs = [[str(ref) for ref in references]]
        formatted_hyps = [str(hyp) for hyp in hypotheses]
        
        ter = sacrebleu.corpus_ter(formatted_hyps, formatted_refs)
        return float(ter.score)
    except Exception as e:
        print(f"  Error calculating TER: {e}")
        return None

def calculate_comet(references, hypotheses, sources, model_path="Unbabel/wmt22-comet-da"):
    """Calculate COMET score given references, hypotheses, and sources."""
    try:
        print(f"  Loading COMET model from {model_path}...")
        # Try to load model, download if needed
        try:
            model = load_from_checkpoint(model_path)
        except Exception as e:
            # If loading fails, try downloading first
            if "Invalid checkpoint path" in str(e) or "not found" in str(e).lower():
                print(f"  Model not found locally, attempting to download...")
                try:
                    from comet import download_model
                    downloaded_path = download_model(model_path)
                    model = load_from_checkpoint(downloaded_path)
                except Exception as download_error:
                    print(f"  Download failed: {download_error}")
                    raise e
            else:
                raise e
        
        # Prepare data in COMET format (source, hypothesis, reference)
        data = []
        for src, hyp, ref in zip(sources, hypotheses, references):
            data.append({
                "src": str(src) if pd.notna(src) else "",
                "mt": str(hyp) if pd.notna(hyp) else "",
                "ref": str(ref) if pd.notna(ref) else ""
            })
        
        # Filter out empty entries
        data = [d for d in data if d["src"].strip() and d["mt"].strip() and d["ref"].strip()]
        
        if not data:
            print("  Warning: No valid data for COMET calculation")
            return None
        
        print(f"  Calculating COMET for {len(data)} examples...")
        # Calculate scores
        result = model.predict(data, batch_size=32, gpus=1)
        # Result is a Prediction object with system_score attribute
        if hasattr(result, 'system_score'):
            comet_score = result.system_score
        elif isinstance(result, tuple):
            # Fallback for older API
            scores, comet_score = result
        else:
            # Try to get score from result
            comet_score = result
        return float(comet_score)
    except Exception as e:
        print(f"  Error calculating COMET score: {e}")
        traceback.print_exc()
        return None

def calculate_metricx(references, hypotheses, model_variant="GLOBAL"):
    """Calculate MetricX score given references and hypotheses."""
    if not METRICX_AVAILABLE:
        print("  MetricX not available, skipping...")
        return None
    
    try:
        print(f"  Initializing MetricX with variant {model_variant}...")
        metricx = MetricX(variant=model_variant)
        
        # Prepare data - filter out empty entries
        valid_pairs = [(str(ref), str(hyp)) for ref, hyp in zip(references, hypotheses) 
                      if pd.notna(ref) and pd.notna(hyp) and str(ref).strip() and str(hyp).strip()]
        
        if not valid_pairs:
            print("  Warning: No valid data for MetricX calculation")
            return None
        
        print(f"  Calculating MetricX for {len(valid_pairs)} examples...")
        # Calculate scores
        scores = []
        for ref, hyp in valid_pairs:
            try:
                score = metricx.score(reference=ref, translation=hyp)
                scores.append(score)
            except Exception as e:
                print(f"  Warning: Error scoring one example: {e}")
                continue
        
        if scores:
            avg_score = sum(scores) / len(scores)
            return float(avg_score)
        return None
    except Exception as e:
        print(f"  Error calculating MetricX score: {e}")
        traceback.print_exc()
        return None

def parse_file_path(file_path):
    """Parse file path to extract metadata: language, variant, k value."""
    # Example: ablation_results/arabic_600tokens/k_0/results_k0.csv
    # Example: ablation_results/konkani_full/k_5/results_k5.csv
    
    parts = Path(file_path).parts
    if len(parts) < 3:
        return None
    
    # Find the directory name (e.g., "arabic_600tokens" or "konkani_full")
    # It should be the parent of the k_X directory
    dir_name = None
    k_value = None
    
    # Look for k_X directory
    for i, part in enumerate(parts):
        if part.startswith('k_'):
            try:
                k_value = int(part.replace('k_', ''))
                # The directory name should be the previous part
                if i > 0:
                    dir_name = parts[i-1]
                break
            except:
                pass
    
    # If we didn't find k_X, try to find language directory directly
    if dir_name is None:
        for part in parts:
            if part.startswith('arabic') or part.startswith('konkani'):
                dir_name = part
                break
    
    if dir_name is None:
        return None
    
    # Extract language and variant
    if dir_name.startswith('arabic'):
        language = 'arabic'
        variant = dir_name.replace('arabic_', '')
    elif dir_name.startswith('konkani'):
        language = 'konkani'
        variant = dir_name.replace('konkani_', '')
    else:
        return None
    
    # If k_value still not found, try from filename
    if k_value is None:
        filename = parts[-1]
        if 'k' in filename.lower():
            try:
                # Extract number after k
                import re
                match = re.search(r'k(\d+)', filename.lower())
                if match:
                    k_value = int(match.group(1))
            except:
                pass
    
    return {
        'language': language,
        'variant': variant,
        'k_value': k_value
    }

def get_column_mapping(language):
    """Get column names for target, source, and reference based on language."""
    if language == 'arabic':
        return {
            'target': 'tn',  # Tunisian Arabic
            'source': 'msa',  # Modern Standard Arabic (for COMET)
            'hypothesis': 'response'
        }
    elif language == 'konkani':
        return {
            'target': 'gom',  # Konkani
            'source': 'mar',  # Marathi (for COMET)
            'hypothesis': 'response'
        }
    return None

def process_csv_file(csv_path, output_dir=None, error_log=None):
    """
    Process a single CSV file and calculate metrics.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save results
        error_log: File handle for error logging
    """
    print(f"\n{'='*60}")
    print(f"Processing: {csv_path}")
    print(f"{'='*60}")
    
    # Parse metadata from file path
    metadata = parse_file_path(csv_path)
    if not metadata:
        error_msg = f"Could not parse metadata from path: {csv_path}"
        print(f"ERROR: {error_msg}")
        if error_log:
            error_log.write(f"{error_msg}\n")
        return None
    
    language = metadata['language']
    variant = metadata['variant']
    k_value = metadata['k_value']
    
    print(f"Language: {language}, Variant: {variant}, k: {k_value}")
    
    # Get column mapping
    col_map = get_column_mapping(language)
    if not col_map:
        error_msg = f"Unknown language: {language}"
        print(f"ERROR: {error_msg}")
        if error_log:
            error_log.write(f"{error_msg}\n")
        return None
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        error_msg = f"Error reading CSV: {e}"
        print(f"ERROR: {error_msg}")
        if error_log:
            error_log.write(f"{error_msg}\n")
        return None
    
    # Check required columns
    required_cols = [col_map['target'], col_map['hypothesis']]
    if col_map['source']:
        required_cols.append(col_map['source'])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}"
        print(f"ERROR: {error_msg}")
        if error_log:
            error_log.write(f"{error_msg}\n")
        return None
    
    # Extract data
    references = df[col_map['target']].tolist()  # Ground truth
    hypotheses = df[col_map['hypothesis']].tolist()  # Model predictions
    sources = df[col_map['source']].tolist() if col_map['source'] in df.columns else None
    
    # Filter out empty values
    valid_indices = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        if pd.notna(ref) and pd.notna(hyp) and str(ref).strip() and str(hyp).strip():
            valid_indices.append(i)
    
    if not valid_indices:
        error_msg = "No valid reference-hypothesis pairs found"
        print(f"ERROR: {error_msg}")
        if error_log:
            error_log.write(f"{error_msg}\n")
        return None
    
    references = [references[i] for i in valid_indices]
    hypotheses = [hypotheses[i] for i in valid_indices]
    if sources:
        sources = [sources[i] for i in valid_indices]
    
    print(f"Valid examples: {len(references)}")
    
    # Calculate all metrics
    print("\nCalculating metrics...")
    
    # BLEU
    print("  Calculating BLEU...")
    bleu_score = calculate_bleu(references, hypotheses)
    
    # chrF (character-level, good for low-resource)
    print("  Calculating chrF...")
    chrf_score = calculate_chrf(references, hypotheses)
    
    # CHRF++ (chrF with word order)
    print("  Calculating CHRF++...")
    chrf_pp_score = calculate_chrf_plus_plus(references, hypotheses)
    
    # TER (Translation Edit Rate)
    print("  Calculating TER...")
    ter_score = calculate_ter(references, hypotheses)
    
    # MetricX (skipped due to version incompatibility)
    print("  Calculating MetricX...")
    metricx_score = calculate_metricx(references, hypotheses)
    
    # Calculate COMET (requires sources)
    comet_score = None
    if sources:
        print("  Calculating COMET...")
        comet_score = calculate_comet(references, hypotheses, sources)
    else:
        print("  Skipping COMET (no source column available)")
    
    # Prepare results
    results = {
        "source_file": csv_path,
        "language": language,
        "variant": variant,
        "k_value": k_value,
        "num_examples": len(references),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "BLEU": bleu_score,
            "chrF": chrf_score,
            "CHRF++": chrf_pp_score,
            "TER": ter_score,
            "COMET": comet_score,
            "MetricX": metricx_score
        }
    }
    
    # Save results
    if output_dir is None:
        output_dir = "metrics_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on metadata
    output_filename = f"{language}_{variant}_k{k_value}_metrics.json"
    output_file = os.path.join(output_dir, output_filename)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"  BLEU: {bleu_score}")
    print(f"  chrF: {chrf_score}")
    print(f"  CHRF++: {chrf_pp_score}")
    print(f"  TER: {ter_score}")
    print(f"  COMET: {comet_score}")
    print(f"  MetricX: {metricx_score}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Calculate translation metrics (BLEU, chrF, CHRF++, TER, COMET) for CSV files"
    )
    parser.add_argument(
        "--ablation-dir",
        default="ablation_results",
        help="Directory containing ablation_results (default: ablation_results)"
    )
    parser.add_argument(
        "--output-dir",
        default="metrics_results",
        help="Directory to save metric JSON files (default: metrics_results)"
    )
    parser.add_argument(
        "--metricx-variant",
        default="GLOBAL",
        choices=["GLOBAL", "REGIONAL"],
        help="MetricX model variant (default: GLOBAL)"
    )
    parser.add_argument(
        "--comet-model",
        default="Unbabel/wmt22-comet-da",
        help="COMET model path (default: Unbabel/wmt22-comet-da)"
    )
    parser.add_argument(
        "--pattern",
        default="**/results_k*.csv",
        help="Glob pattern for CSV files (default: **/results_k*.csv)"
    )
    
    args = parser.parse_args()
    
    # Find all CSV files
    pattern = os.path.join(args.ablation_dir, args.pattern)
    csv_files = glob.glob(pattern, recursive=True)
    
    # Filter out summary files
    csv_files = [f for f in csv_files if 'summary' not in f.lower() and 'ablation_summary' not in f.lower()]
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Open error log
    error_log_path = os.path.join(args.output_dir, "calculation_errors.log")
    error_log = open(error_log_path, "w")
    
    # Process each CSV file
    all_results = {}
    successful = 0
    failed = 0
    
    for csv_file in sorted(csv_files):
        try:
            results = process_csv_file(csv_file, args.output_dir, error_log)
            if results:
                all_results[os.path.basename(csv_file)] = results
                successful += 1
            else:
                failed += 1
        except Exception as e:
            error_msg = f"Unexpected error processing {csv_file}: {e}"
            print(f"ERROR: {error_msg}")
            error_log.write(f"{error_msg}\n")
            traceback.print_exc(file=error_log)
            failed += 1
    
    error_log.close()
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "all_metrics_summary.json")
    summary_data = {
        "total_files": len(csv_files),
        "successful": successful,
        "failed": failed,
        "timestamp": datetime.now().isoformat(),
        "results": all_results
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Summary saved to: {summary_file}")
    print(f"Errors logged to: {error_log_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

