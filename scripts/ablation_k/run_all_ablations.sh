#!/bin/bash

# Run ablation studies for all language pairs
# This script addresses Reviewer 1's comment about lack of ablation on k hyperparameter

set -e  # Exit on error

echo "=========================================="
echo "Running Complete Ablation Study"
echo "Varying k (number of few-shot examples)"
echo "=========================================="
echo ""

# Configuration
K_VALUES="0 1 3 5 7 10"
BASE_OUTPUT_DIR="ablation_results"

# Create output directory
mkdir -p ${BASE_OUTPUT_DIR}

# Save experiment configuration
cat > ${BASE_OUTPUT_DIR}/experiment_config.txt << EOF
Ablation Study Configuration
========================================
Date: $(date)
K values tested: ${K_VALUES}
Purpose: Address Reviewer 1 comment on lack of k ablation

Language Pairs:
1. Konkani: Hindi (pivot) -> Marathi (source) -> Konkani (target)
2. Tunisian Arabic: MSA (pivot) -> English (source) -> Tunisian (target)
========================================
EOF

echo "Experiment 1: Konkani Translation"
echo "----------------------------------------"
python scripts/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations" \
    --output-dir "${BASE_OUTPUT_DIR}/konkani" \
    --k-values ${K_VALUES}

echo ""
echo "=========================================="
echo ""
echo "Experiment 2: Tunisian Arabic Translation"
echo "----------------------------------------"
python scripts/run_ablation_study.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "msa" \
    --source "eng" \
    --target "tun" \
    --db "arabic_translations" \
    --output-dir "${BASE_OUTPUT_DIR}/arabic" \
    --k-values ${K_VALUES}

echo ""
echo "=========================================="
echo "ALL ABLATION STUDIES COMPLETE"
echo "=========================================="
echo ""
echo "Results saved in: ${BASE_OUTPUT_DIR}/"
echo ""
echo "Summary:"
echo "  Konkani results: ${BASE_OUTPUT_DIR}/konkani/"
echo "  Arabic results:  ${BASE_OUTPUT_DIR}/arabic/"
echo ""
echo "Key files:"
echo "  - ablation_summary.csv: Quantitative results"
echo "  - ablation_study_plots.png: Visual analysis"
echo "  - ablation_study_bar_chart.png: Comparative bar chart"
echo ""

# Create a combined comparison if both experiments completed
python - << 'PYTHON_SCRIPT'
import json
import pandas as pd
from pathlib import Path

base_dir = Path("ablation_results")
konkani_file = base_dir / "konkani" / "ablation_summary.csv"
arabic_file = base_dir / "arabic" / "ablation_summary.csv"

if konkani_file.exists() and arabic_file.exists():
    print("Creating cross-language comparison...")
    
    konkani_df = pd.read_csv(konkani_file)
    konkani_df['Language'] = 'Konkani'
    
    arabic_df = pd.read_csv(arabic_file)
    arabic_df['Language'] = 'Tunisian Arabic'
    
    combined_df = pd.concat([konkani_df, arabic_df], ignore_index=True)
    
    # Save combined results
    combined_file = base_dir / "combined_ablation_results.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined results saved to: {combined_file}")
    
    # Create summary table
    print("\n" + "="*80)
    print("CROSS-LANGUAGE ABLATION COMPARISON")
    print("="*80)
    pivot_table = combined_df.pivot_table(
        index='k', 
        columns='Language', 
        values=['BLEU', 'chrF', 'chrF++'],
        aggfunc='first'
    )
    print(pivot_table)
    print("="*80)
else:
    print("Not all experiments completed. Skipping combined analysis.")
PYTHON_SCRIPT

echo ""
echo "To view results, check the following files:"
echo "  - ${BASE_OUTPUT_DIR}/combined_ablation_results.csv"
echo "  - ${BASE_OUTPUT_DIR}/konkani/ablation_study_plots.png"
echo "  - ${BASE_OUTPUT_DIR}/arabic/ablation_study_plots.png"


