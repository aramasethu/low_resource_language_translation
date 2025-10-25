#!/bin/bash
################################################################################
# Konkani Ablation with Hermes Model - max_new_tokens=600
# 
# Model: NousResearch/Hermes-2-Pro-Llama-3-8B
# GPU: 3
# This compares Hermes vs Tower model performance
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "KONKANI ABLATION - HERMES MODEL (max_new_tokens=600)"
echo "================================================================================"
echo ""
echo "⚡ Running with Hermes-2-Pro-Llama-3-8B model"
echo "   Comparing against Tower model baseline"
echo ""
echo "Configuration:"
echo "  • Model: NousResearch/Hermes-2-Pro-Llama-3-8B"
echo "  • GPU: 3"
echo "  • Batch size: 4"
echo "  • k values: 0 to 10 (all)"
echo "  • max_new_tokens: 600"
echo ""
echo "Estimated time: 4-5 hours"
echo "================================================================================"
echo ""

# Configuration
GPU_DEVICE=3
BATCH_SIZE=4
WANDB_PROJECT="low-resource-translation-ablation"
MODEL="NousResearch/Hermes-2-Pro-Llama-3-8B"

# Run ablation
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python scripts/run_ablation_study.py \
  --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
  --model "${MODEL}" \
  --pivot "hin" \
  --source "mar" \
  --target "gom" \
  --db "konkani_translations" \
  --output-dir "ablation_results/konkani_hermes_600tokens" \
  --k-values 0 1 2 3 4 5 6 7 8 9 10 \
  --batch-size ${BATCH_SIZE} \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "konkani-hermes-600tokens"

echo ""
echo "================================================================================"
echo "✅ KONKANI ABLATION (HERMES) COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: ablation_results/konkani_hermes_600tokens/"
echo ""
echo "Compare with Tower model:"
echo "  • Tower:  ablation_results/konkani_600tokens/"
echo "  • Hermes: ablation_results/konkani_hermes_600tokens/"
echo ""
echo "Expected comparison:"
echo "  ✅ Both should have <5% truncation (max_new_tokens=600)"
echo "  📊 Compare BLEU scores to see which model is better"
echo "  📊 Compare k-value sensitivity between models"
echo ""
echo "================================================================================"

