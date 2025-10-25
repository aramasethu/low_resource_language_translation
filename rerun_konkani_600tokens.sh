#!/bin/bash
################################################################################
# Rerun KONKANI ONLY with max_new_tokens=600
# 
# This is the CRITICAL fix - Konkani had 47-75% truncation with 200 tokens
# Use this script if you only want to rerun Konkani (faster, ~4-5 hours)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "KONKANI ABLATION STUDY - RERUN WITH max_new_tokens=600"
echo "================================================================================"
echo ""
echo "⚠️  CRITICAL FIX: Previous results had 47-75% truncation"
echo "   This rerun will fix the truncation issue"
echo ""
echo "Configuration:"
echo "  • Model: Unbabel/TowerInstruct-7B-v0.1"
echo "  • GPU: 0"
echo "  • Batch size: 4"
echo "  • k values: 0 to 10 (all)"
echo "  • max_new_tokens: 600 (was 200)"
echo ""
echo "Estimated time: 4-5 hours"
echo "================================================================================"
echo ""

# Configuration
GPU_DEVICE=0
BATCH_SIZE=4
WANDB_PROJECT="low-resource-translation-ablation"

# Run ablation
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python scripts/run_ablation_study.py \
  --dataset "ai4bharat/IN22-Conv" \
  --dataset-config "kok" \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --pivot "mar" \
  --source "hin" \
  --target "gom" \
  --db "konkani_translations" \
  --output-dir "ablation_results/konkani_600tokens" \
  --k-values 0 1 2 3 4 5 6 7 8 9 10 \
  --batch-size ${BATCH_SIZE} \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "konkani-ablation-600tokens"

echo ""
echo "================================================================================"
echo "✅ KONKANI ABLATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: ablation_results/konkani_600tokens/"
echo ""
echo "Expected improvements:"
echo "  ✅ BLEU scores should be HIGHER (complete translations)"
echo "  ✅ Truncation should be <5% (was 47-75%)"
echo "  ✅ k=10 should work better (less truncation)"
echo "  ✅ Fair comparison between k values"
echo ""
echo "To verify the fix worked:"
echo "  conda run -n lrlt_exp python check_generated_lengths.py"
echo ""
echo "To compare with old results:"
echo "  • Old (200 tokens): ablation_results/konkani_full/"
echo "  • New (600 tokens): ablation_results/konkani_600tokens/"
echo ""
echo "================================================================================"

