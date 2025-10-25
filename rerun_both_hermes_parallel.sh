#!/bin/bash
################################################################################
# Run BOTH Konkani and Arabic with Hermes Model IN PARALLEL
# 
# Konkani on GPU 3 - Hermes-2-Pro-Llama-3-8B
# Arabic on GPU 4  - Hermes-2-Pro-Llama-3-8B
#
# This provides model comparison: Tower vs Hermes
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "PARALLEL ABLATION STUDY - HERMES MODEL - max_new_tokens=600"
echo "================================================================================"
echo ""
echo "⚡ Running BOTH experiments with Hermes-2-Pro-Llama-3-8B"
echo "   Model comparison: Tower vs Hermes"
echo ""
echo "This will run BOTH experiments IN PARALLEL:"
echo "  • Konkani on GPU 3"
echo "  • Arabic on GPU 4"
echo ""
echo "Total wall-clock time: ~4-5 hours (since they run in parallel)"
echo "================================================================================"
echo ""

# Make scripts executable
chmod +x rerun_konkani_hermes.sh
chmod +x rerun_arabic_hermes.sh

# Create log directory
mkdir -p logs

echo "Starting Konkani (Hermes) on GPU 3 (background)..."
./rerun_konkani_hermes.sh > logs/konkani_hermes_600tokens.log 2>&1 &
KONKANI_PID=$!
echo "  ✅ Konkani started (PID: ${KONKANI_PID})"
echo "  📝 Log: logs/konkani_hermes_600tokens.log"

echo ""
echo "Starting Arabic (Hermes) on GPU 4 (background)..."
./rerun_arabic_hermes.sh > logs/arabic_hermes_600tokens.log 2>&1 &
ARABIC_PID=$!
echo "  ✅ Arabic started (PID: ${ARABIC_PID})"
echo "  📝 Log: logs/arabic_hermes_600tokens.log"

echo ""
echo "================================================================================"
echo "BOTH HERMES EXPERIMENTS RUNNING"
echo "================================================================================"
echo ""
echo "Konkani PID: ${KONKANI_PID} (GPU 3)"
echo "Arabic PID:  ${ARABIC_PID} (GPU 4)"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/konkani_hermes_600tokens.log"
echo "  tail -f logs/arabic_hermes_600tokens.log"
echo ""
echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To check if still running:"
echo "  ps aux | grep ${KONKANI_PID}"
echo "  ps aux | grep ${ARABIC_PID}"
echo ""
echo "Waiting for both experiments to complete..."
echo "(This will take approximately 4-5 hours)"
echo "================================================================================"
echo ""

# Wait for both to complete
wait ${KONKANI_PID}
KONKANI_EXIT=$?
echo "✅ Konkani (Hermes) finished (exit code: ${KONKANI_EXIT})"

wait ${ARABIC_PID}
ARABIC_EXIT=$?
echo "✅ Arabic (Hermes) finished (exit code: ${ARABIC_EXIT})"

echo ""
echo "================================================================================"
echo "PARALLEL HERMES ABLATION COMPLETE"
echo "================================================================================"
echo ""

if [ ${KONKANI_EXIT} -eq 0 ] && [ ${ARABIC_EXIT} -eq 0 ]; then
    echo "✅ Both Hermes experiments completed successfully!"
    echo ""
    echo "Hermes Results:"
    echo "  • Konkani: ablation_results/konkani_hermes_600tokens/"
    echo "  • Arabic:  ablation_results/arabic_hermes_600tokens/"
    echo ""
    echo "Tower Results (for comparison):"
    echo "  • Konkani: ablation_results/konkani_600tokens/"
    echo "  • Arabic:  ablation_results/arabic_600tokens/"
    echo ""
    echo "Logs:"
    echo "  • Konkani: logs/konkani_hermes_600tokens.log"
    echo "  • Arabic:  logs/arabic_hermes_600tokens.log"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "MODEL COMPARISON ANALYSIS"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "You now have results for BOTH models on BOTH languages:"
    echo ""
    echo "┌─────────────────┬──────────────────────┬──────────────────────┐"
    echo "│ Language        │ Tower Model          │ Hermes Model         │"
    echo "├─────────────────┼──────────────────────┼──────────────────────┤"
    echo "│ Konkani         │ GPU 0 (done/running) │ GPU 3 (✅ done)     │"
    echo "│ Arabic          │ GPU 1 (done/running) │ GPU 4 (✅ done)     │"
    echo "└─────────────────┴──────────────────────┴──────────────────────┘"
    echo ""
    echo "Compare:"
    echo "  1. Which model achieves higher BLEU scores?"
    echo "  2. Which model benefits more from few-shot learning (k>0)?"
    echo "  3. Do models show same optimal k value?"
    echo "  4. Which model is more robust to varying k?"
    echo ""
    echo "Check W&B for side-by-side comparison:"
    echo "  • konkani-ablation-600tokens (Tower)"
    echo "  • konkani-hermes-600tokens (Hermes)"
    echo "  • arabic-ablation-600tokens (Tower)"
    echo "  • arabic-hermes-600tokens (Hermes)"
    echo ""
else
    echo "❌ One or more experiments failed!"
    echo "  Konkani exit code: ${KONKANI_EXIT}"
    echo "  Arabic exit code:  ${ARABIC_EXIT}"
    echo ""
    echo "Check logs for errors:"
    echo "  cat logs/konkani_hermes_600tokens.log | tail -50"
    echo "  cat logs/arabic_hermes_600tokens.log | tail -50"
fi

echo "================================================================================"

