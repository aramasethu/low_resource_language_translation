#!/bin/bash
################################################################################
# Re-run Ablation Study for Tunisian Arabic with Hermes Model (k=0 to 10)
# 
# Purpose: Verify the k=3 complete failure result
#
# Usage:
#   ./scripts/ablation_k/rerun_arabic_hermes_ablation.sh [GPU_ID] [BATCH_SIZE]
#
# Examples:
#   ./scripts/ablation_k/rerun_arabic_hermes_ablation.sh 0 4   # GPU 0, batch 4
#   ./scripts/ablation_k/rerun_arabic_hermes_ablation.sh       # Defaults: GPU 0, batch 4
################################################################################

set -e  # Exit on error

# Parse arguments
GPU_ID=${1:-0}          # Default GPU 0
BATCH_SIZE=${2:-4}      # Default batch size 4

# Configuration
DATASET="predictionguard/arabic_acl_corpus"
MODEL="NousResearch/Hermes-2-Pro-Llama-3-8B"
PIVOT="msa"
SOURCE="en"
TARGET="tn"
DB="arabic_translations"

# Date-stamped output directory
DATE_STAMP=$(date +%Y%m%d)
OUTPUT_DIR="ablation_results/arabic_hermes_${DATE_STAMP}"

echo "================================================================================"
echo "ARABIC HERMES ABLATION STUDY (RE-RUN) - Verifying k=3 result"
echo "================================================================================"
echo "Date:         $(date)"
echo "GPU ID:       $GPU_ID"
echo "Batch size:   $BATCH_SIZE"
echo "Dataset:      $DATASET"
echo "Model:        $MODEL"
echo "Languages:    $PIVOT (pivot) -> $SOURCE (source) -> $TARGET (target)"
echo "Output:       $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save experiment config
cat > "${OUTPUT_DIR}/experiment_config.txt" << EOF
Arabic Hermes Ablation Study - Verification Run
================================================
Date: $(date)
Purpose: Verify k=3 complete failure result (0.00 BLEU)

Configuration:
  Dataset:     $DATASET
  Model:       $MODEL
  Pivot:       $PIVOT (Modern Standard Arabic)
  Source:      $SOURCE (English)
  Target:      $TARGET (Tunisian Arabic)
  Database:    $DB
  GPU:         $GPU_ID
  Batch Size:  $BATCH_SIZE
  K Values:    0 1 2 3 4 5 6 7 8 9 10

Previous Results (from arabic_hermes_600tokens):
  k=0:  BLEU 4.37
  k=1:  BLEU 4.79
  k=2:  BLEU 5.52
  k=3:  BLEU 0.00  <-- SUSPECTED OUTLIER
  k=4:  BLEU 5.52
  k=5:  BLEU 6.27
  k=6:  BLEU 3.17
  k=7:  BLEU 5.30
  k=8:  BLEU 4.46
  k=9:  BLEU 6.74
  k=10: BLEU 5.87

Hypothesis: k=3 failure may be a random artifact, should be ~4-5 BLEU
================================================
EOF

# Start time
START_TIME=$(date +%s)
echo "🚀 Starting ablation study at $(date)"
echo ""

# Run ablation study using the main script
echo "Running ablation with all k values: 0 1 2 3 4 5 6 7 8 9 10"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/ablation_k/run_ablation_study.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --pivot "$PIVOT" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --db "$DB" \
    --output-dir "$OUTPUT_DIR" \
    --k-values 0 1 2 3 4 5 6 7 8 9 10 \
    --batch-size $BATCH_SIZE

# End time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "================================================================================"
echo "🎉 ARABIC HERMES ABLATION STUDY COMPLETE!"
echo "================================================================================"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  Summary:  ${OUTPUT_DIR}/ablation_summary.csv"
echo "  Plots:    ${OUTPUT_DIR}/ablation_study_plots.png"
echo "  Details:  ${OUTPUT_DIR}/ablation_detailed_results.json"
echo ""
echo "To compare with previous results:"
echo "  Previous: ablation_results/arabic_hermes_600tokens/ablation_summary.csv"
echo "  Current:  ${OUTPUT_DIR}/ablation_summary.csv"
echo ""
echo "Check k=3 specifically:"
echo "  cat ${OUTPUT_DIR}/k_3/scores_k3.json"
echo "================================================================================"

