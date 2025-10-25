#!/bin/bash
################################################################################
# Rerun ARABIC ONLY with max_new_tokens=600
# 
# Arabic had only 1-3% truncation with 200 tokens (mostly fine)
# This rerun is for consistency with Konkani and to eliminate any edge cases
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "ARABIC ABLATION STUDY - RERUN WITH max_new_tokens=600"
echo "================================================================================"
echo ""
echo "ℹ️  Previous results had only 1-3% truncation (mostly valid)"
echo "   This rerun is for consistency and to eliminate edge cases"
echo ""
echo "Configuration:"
echo "  • Model: Unbabel/TowerInstruct-7B-v0.1"
echo "  • GPU: 1"
echo "  • Batch size: 4"
echo "  • k values: 0 to 10 (all)"
echo "  • max_new_tokens: 600 (was 200)"
echo ""
echo "Estimated time: 3-4 hours"
echo "================================================================================"
echo ""

# Configuration
GPU_DEVICE=1
BATCH_SIZE=4
WANDB_PROJECT="low-resource-translation-ablation"
DATASET="predictionguard/arabic_acl_corpus"
MODEL="Unbabel/TowerInstruct-7B-v0.1"
OUTPUT_BASE="ablation_results/arabic_600tokens"
DB_NAME="arabic_translations"

# Run ablation study using the comprehensive Python script
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python scripts/run_arabic_ablation_study.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --db "${DB_NAME}" \
  --output-dir "${OUTPUT_BASE}" \
  --k-values 0 1 2 3 4 5 6 7 8 9 10 \
  --batch-size ${BATCH_SIZE} \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "arabic-ablation-600tokens"

echo ""
echo "================================================================================"
echo "✅ ARABIC ABLATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: ablation_results/arabic_600tokens/"
echo ""
echo "Expected changes (minimal since only 1-3% were truncated):"
echo "  ✅ BLEU scores should be slightly higher"
echo "  ✅ Truncation should be ~0% (was 1-3%)"
echo "  ✅ Edge cases eliminated"
echo ""
echo "To verify the fix worked:"
echo "  conda run -n lrlt_exp python check_generated_lengths.py"
echo ""
echo "To compare with old results:"
echo "  • Old (200 tokens): ablation_results/arabic_full/"
echo "  • New (600 tokens): ablation_results/arabic_600tokens/"
echo ""
echo "================================================================================"

