#!/bin/bash
################################################################################
# Run BOTH Konkani and Arabic ablation studies IN PARALLEL
# 
# Konkani on GPU 0 (CRITICAL - 47-75% truncation)
# Arabic on GPU 1 (consistency - only 1-3% truncation)
#
# This script runs both experiments simultaneously on different GPUs
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "PARALLEL ABLATION STUDY - max_new_tokens=600"
echo "================================================================================"
echo ""
echo "This will run BOTH experiments IN PARALLEL:"
echo "  • Konkani on GPU 0 (CRITICAL fix)"
echo "  • Arabic on GPU 1 (consistency)"
echo ""
echo "Total wall-clock time: ~4-5 hours (since they run in parallel)"
echo "================================================================================"
echo ""

# Make scripts executable
chmod +x rerun_konkani_600tokens.sh
chmod +x rerun_arabic_600tokens.sh

# Create log directory
mkdir -p logs

echo "Starting Konkani on GPU 0 (background)..."
./rerun_konkani_600tokens.sh > logs/konkani_600tokens.log 2>&1 &
KONKANI_PID=$!
echo "  ✅ Konkani started (PID: ${KONKANI_PID})"
echo "  📝 Log: logs/konkani_600tokens.log"

echo ""
echo "Starting Arabic on GPU 1 (background)..."
./rerun_arabic_600tokens.sh > logs/arabic_600tokens.log 2>&1 &
ARABIC_PID=$!
echo "  ✅ Arabic started (PID: ${ARABIC_PID})"
echo "  📝 Log: logs/arabic_600tokens.log"

echo ""
echo "================================================================================"
echo "BOTH EXPERIMENTS RUNNING"
echo "================================================================================"
echo ""
echo "Konkani PID: ${KONKANI_PID} (GPU 0)"
echo "Arabic PID:  ${ARABIC_PID} (GPU 1)"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/konkani_600tokens.log"
echo "  tail -f logs/arabic_600tokens.log"
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
echo "✅ Konkani finished (exit code: ${KONKANI_EXIT})"

wait ${ARABIC_PID}
ARABIC_EXIT=$?
echo "✅ Arabic finished (exit code: ${ARABIC_EXIT})"

echo ""
echo "================================================================================"
echo "PARALLEL ABLATION COMPLETE"
echo "================================================================================"
echo ""

if [ ${KONKANI_EXIT} -eq 0 ] && [ ${ARABIC_EXIT} -eq 0 ]; then
    echo "✅ Both experiments completed successfully!"
    echo ""
    echo "Results:"
    echo "  • Konkani: ablation_results/konkani_600tokens/"
    echo "  • Arabic:  ablation_results/arabic_600tokens/"
    echo ""
    echo "Logs:"
    echo "  • Konkani: logs/konkani_600tokens.log"
    echo "  • Arabic:  logs/arabic_600tokens.log"
    echo ""
    echo "Next steps:"
    echo "  1. Verify truncation is fixed:"
    echo "     conda run -n lrlt_exp python check_generated_lengths.py"
    echo ""
    echo "  2. Compare with old results:"
    echo "     - Old Konkani (200): ablation_results/konkani_full/"
    echo "     - New Konkani (600): ablation_results/konkani_600tokens/"
    echo "     - Old Arabic (200):  ablation_results/arabic_full/"
    echo "     - New Arabic (600):  ablation_results/arabic_600tokens/"
    echo ""
    echo "  3. Update ABLATION_STUDY.md with new results"
else
    echo "❌ One or more experiments failed!"
    echo "  Konkani exit code: ${KONKANI_EXIT}"
    echo "  Arabic exit code:  ${ARABIC_EXIT}"
    echo ""
    echo "Check logs for errors:"
    echo "  cat logs/konkani_600tokens.log | tail -50"
    echo "  cat logs/arabic_600tokens.log | tail -50"
fi

echo "================================================================================"

