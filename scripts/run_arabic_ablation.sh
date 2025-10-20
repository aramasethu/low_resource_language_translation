#!/bin/bash
################################################################################
# Run Ablation Study for Tunisian Arabic (k=0 to 10)
#
# The Arabic dataset has a nested structure that requires the specialized
# run_inference_arabic.py script.
#
# Usage:
#   ./scripts/run_arabic_ablation.sh [GPU_ID] [BATCH_SIZE]
#
# Examples:
#   ./scripts/run_arabic_ablation.sh 7 4       # Use GPU 7, batch size 4
#   ./scripts/run_arabic_ablation.sh 0 8       # Use GPU 0, batch size 8
################################################################################

set -e  # Exit on error

# Parse arguments
GPU_ID=${1:-7}          # Default GPU 7
BATCH_SIZE=${2:-4}      # Default batch size 4

# Configuration
DATASET="predictionguard/arabic_acl_corpus"
MODEL="Unbabel/TowerInstruct-7B-v0.1"
PIVOT="msa"
SOURCE="en"
TARGET="tn"
DB="arabic_translations"
OUTPUT_DIR="ablation_results/arabic_full"

echo "================================================================================"
echo "ARABIC ABLATION STUDY: k=0 to k=10"
echo "================================================================================"
echo "GPU ID:       $GPU_ID"
echo "Batch size:   $BATCH_SIZE"
echo "Dataset:      $DATASET"
echo "Model:        $MODEL"
echo "Languages:    $PIVOT (pivot) -> $SOURCE (source) -> $TARGET (target)"
echo "Output:       $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Check if vector database exists
if [ ! -d "lancedb_data/${DB}" ]; then
    echo "⚠️  WARNING: Vector database not found: lancedb_data/${DB}"
    echo "Creating vector database first..."
    python scripts/create_vector_db_arabic.py \
        --dataset "$DATASET" \
        --pivot "$PIVOT" \
        --source "$SOURCE" \
        --target "$TARGET" \
        --db "$DB"
    echo "✅ Vector database created!"
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)
echo "🚀 Starting ablation study at $(date)"
echo ""

# Run ablation for each k value
for k in 0 1 2 3 4 5 6 7 8 9 10; do
    echo "================================================================================"
    echo "🔬 EXPERIMENT: k=$k"
    echo "================================================================================"
    
    K_START=$(date +%s)
    
    # Create k-specific output directory
    mkdir -p "${OUTPUT_DIR}/k_${k}"
    
    # Run inference
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_inference_arabic.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --pivot "$PIVOT" \
        --source "$SOURCE" \
        --target "$TARGET" \
        --db "$DB" \
        --output "${OUTPUT_DIR}/k_${k}/results_k${k}.csv" \
        --scores "${OUTPUT_DIR}/k_${k}/scores_k${k}.json" \
        --num-examples $k \
        --batch-size $BATCH_SIZE
    
    K_END=$(date +%s)
    K_DURATION=$((K_END - K_START))
    K_MINUTES=$((K_DURATION / 60))
    K_SECONDS=$((K_DURATION % 60))
    
    # Read and display scores
    if [ -f "${OUTPUT_DIR}/k_${k}/scores_k${k}.json" ]; then
        BLEU=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/k_${k}/scores_k${k}.json'))['BLEU Score'])" 2>/dev/null || echo "N/A")
        CHRF=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/k_${k}/scores_k${k}.json'))['chrF Score'])" 2>/dev/null || echo "N/A")
        echo ""
        echo "✅ k=$k completed in ${K_MINUTES}m ${K_SECONDS}s"
        echo "   BLEU: $BLEU | chrF: $CHRF"
    else
        echo ""
        echo "⚠️  k=$k completed but scores file not found"
    fi
    
    echo ""
done

echo "================================================================================"
echo "📊 GENERATING SUMMARY AND PLOTS"
echo "================================================================================"

# Generate summary
python scripts/analyze_ablation_results.py \
    --results-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

# End time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "================================================================================"
echo "🎉 ARABIC ABLATION STUDY COMPLETE!"
echo "================================================================================"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View summary:"
echo "  cat ${OUTPUT_DIR}/ablation_summary.csv"
echo ""
echo "View plots:"
echo "  ls -lh ${OUTPUT_DIR}/*.png"
echo "================================================================================"

