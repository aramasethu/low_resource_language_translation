#!/bin/bash
################################################################################
# Rerun Ablation Study with max_new_tokens=600
# 
# This script reruns BOTH Konkani and Arabic ablation studies with the fixed
# max_new_tokens=600 to eliminate truncation artifacts.
#
# CRITICAL: Konkani results with 200 tokens had 47-75% truncation
# Arabic results with 200 tokens had only 1-3% truncation (mostly fine)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "RERUNNING ABLATION STUDY WITH max_new_tokens=600"
echo "================================================================================"
echo ""
echo "This will run:"
echo "  1. Konkani: k=0 to k=10 (CRITICAL - previous results invalid)"
echo "  2. Arabic:  k=0 to k=10 (optional - for consistency)"
echo ""
echo "Total estimated time: 7-10 hours"
echo "================================================================================"
echo ""

# Configuration
KONKANI_GPU=0
ARABIC_GPU=1
BATCH_SIZE=4
WANDB_PROJECT="low-resource-translation-ablation"

################################################################################
# PART 1: KONKANI (CRITICAL)
################################################################################

echo ""
echo "================================================================================"
echo "PART 1: KONKANI ABLATION (max_new_tokens=600)"
echo "================================================================================"
echo ""
echo "Starting Konkani ablation study..."
echo "GPU: ${KONKANI_GPU}"
echo "Batch size: ${BATCH_SIZE}"
echo "k values: 0 to 10"
echo ""

CUDA_VISIBLE_DEVICES=${KONKANI_GPU} python scripts/run_ablation_study.py \
  --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --pivot "hin" \
  --source "mar" \
  --target "gom" \
  --db "konkani_translations" \
  --output-dir "ablation_results/konkani_600tokens" \
  --k-values 0 1 2 3 4 5 6 7 8 9 10 \
  --batch-size ${BATCH_SIZE} \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "konkani-ablation-600tokens-full"

echo ""
echo "✅ Konkani ablation complete!"
echo ""

################################################################################
# PART 2: ARABIC (OPTIONAL BUT RECOMMENDED)
################################################################################

echo ""
echo "================================================================================"
echo "PART 2: ARABIC ABLATION (max_new_tokens=600)"
echo "================================================================================"
echo ""
echo "Starting Arabic ablation study..."
echo "GPU: ${ARABIC_GPU}"
echo "Batch size: ${BATCH_SIZE}"
echo "k values: 0 to 10"
echo ""

# Variables
DATASET="predictionguard/arabic_acl_corpus"
MODEL="Unbabel/TowerInstruct-7B-v0.1"
OUTPUT_BASE="ablation_results/arabic_600tokens"
DB_NAME="arabic_translations"

# Run Arabic ablation using comprehensive Python script
CUDA_VISIBLE_DEVICES=${ARABIC_GPU} python scripts/run_arabic_ablation_study.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --db "${DB_NAME}" \
    --output-dir "${OUTPUT_BASE}" \
    --k-values 0 1 2 3 4 5 6 7 8 9 10 \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "arabic-ablation-600tokens-full"

echo ""
echo "✅ Arabic ablation complete!"
echo ""

################################################################################
# SUMMARY
################################################################################

echo ""
echo "================================================================================"
echo "✅ ALL ABLATION EXPERIMENTS COMPLETE WITH max_new_tokens=600"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  • Konkani: ablation_results/konkani_600tokens/"
echo "  • Arabic:  ablation_results/arabic_600tokens/"
echo ""
echo "Next steps:"
echo "  1. Compare with old results (200 tokens) to see improvement"
echo "  2. Analyze new scores (should be higher, especially for Konkani)"
echo "  3. Update ABLATION_STUDY.md with new results"
echo "  4. Verify truncation is now <5% (was 47-75% for Konkani)"
echo ""
echo "To analyze results, run:"
echo "  conda run -n lrlt_exp python check_generated_lengths.py"
echo ""
echo "================================================================================"

