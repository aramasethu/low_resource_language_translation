#!/bin/bash
# Test no-pivot ablation setup in conda environment

set -e  # Exit on error

echo "========================================================================"
echo "🧪 NO-PIVOT ABLATION SETUP TESTS"
echo "========================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda not found in PATH"
    echo "   Please ensure conda is installed and initialized"
    exit 1
fi

# Get conda base directory
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
echo "🔧 Activating conda environment: lrlt_exp"
conda activate lrlt_exp

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to activate conda environment 'lrlt_exp'"
    echo "   Please create the environment first"
    exit 1
fi

echo "✅ Environment activated"
echo ""

# Verify Python
echo "📍 Python location: $(which python)"
echo "📍 Python version: $(python --version)"
echo ""

# Run tests
echo "🚀 Running tests..."
echo "========================================================================"
echo ""

python test_no_pivot_setup.py

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ ALL TESTS COMPLETED"
    echo ""
    echo "If tests passed, you can now run the full experiments:"
    echo "  ./run_no_pivot_experiments.sh"
else
    echo "❌ TESTS FAILED (exit code: $TEST_EXIT_CODE)"
    echo ""
    echo "Please fix the issues before running full experiments."
fi

echo "========================================================================"

exit $TEST_EXIT_CODE

