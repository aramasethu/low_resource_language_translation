#!/bin/bash
################################################################################
# Arabic Ablation with Hermes Model - max_new_tokens=600
# 
# Model: NousResearch/Hermes-2-Pro-Llama-3-8B
# GPU: 4
# This compares Hermes vs Tower model performance
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "ARABIC ABLATION - HERMES MODEL (max_new_tokens=600)"
echo "================================================================================"
echo ""
echo "⚡ Running with Hermes-2-Pro-Llama-3-8B model"
echo "   Comparing against Tower model baseline"
echo ""
echo "Configuration:"
echo "  • Model: NousResearch/Hermes-2-Pro-Llama-3-8B"
echo "  • GPU: 4"
echo "  • Batch size: 4"
echo "  • k values: 0 to 10 (all)"
echo "  • max_new_tokens: 600"
echo ""
echo "Estimated time: 3-4 hours"
echo "================================================================================"
echo ""

# Configuration
GPU_DEVICE=4
BATCH_SIZE=4
WANDB_PROJECT="low-resource-translation-ablation"
DATASET="predictionguard/arabic_acl_corpus"
MODEL="NousResearch/Hermes-2-Pro-Llama-3-8B"
OUTPUT_BASE="ablation_results/arabic_hermes_600tokens"
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
  --wandb-run-name "arabic-hermes-600tokens"

echo ""
echo "================================================================================"
echo "✅ ARABIC ABLATION (HERMES) COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: ablation_results/arabic_hermes_600tokens/"
echo ""
echo "Compare with Tower model:"
echo "  • Tower:  ablation_results/arabic_600tokens/"
echo "  • Hermes: ablation_results/arabic_hermes_600tokens/"
echo ""
echo "Expected comparison:"
echo "  ✅ Both should have ~0% truncation (max_new_tokens=600)"
echo "  📊 Compare BLEU scores to see which model is better"
echo "  📊 Check if Hermes has different k-value sensitivity"
echo ""
echo "================================================================================"

