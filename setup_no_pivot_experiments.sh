#!/bin/bash
# Setup and run no-pivot ablation experiments

set -e  # Exit on error

echo "========================================================================"
echo "NO-PIVOT ABLATION STUDY SETUP"
echo "Testing k=3,4,5 WITHOUT pivot language"
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
echo "📍 Python location: $(which python)"
echo "📍 Python version: $(python --version)"
echo ""

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "rohin/no-pivot-ablation" ]; then
    echo "⚠️  WARNING: You're on branch '$CURRENT_BRANCH'"
    echo "   Expected: 'rohin/no-pivot-ablation'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Creating No-Pivot Vector Databases"
echo "========================================================================" 
echo ""

echo "📊 Creating Konkani database (English→Konkani, no Marathi pivot)..."
python scripts/create_vector_db_no_pivot.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --source "eng" \
    --target "gom" \
    --db "konkani_no_pivot_db"

echo ""
echo "✅ Konkani database created!"
echo ""

echo "📊 Creating Tunisian Arabic database (English→Tunisian, no MSA pivot)..."
python scripts/create_vector_db_no_pivot.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --source "eng" \
    --target "tun" \
    --db "arabic_no_pivot_db"

echo ""
echo "✅ Tunisian Arabic database created!"
echo ""

echo "========================================================================"
echo "Step 2: Running Ablation Experiments"
echo "========================================================================"
echo ""
echo "This will run 6 experiments total:"
echo "  - Konkani: k=3,4,5 (no pivot)"
echo "  - Tunisian Arabic: k=3,4,5 (no pivot)"
echo ""
echo "Estimated time: 2-3 hours"
echo ""
read -p "Continue with experiments? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting experiments..."
    echo ""
    
    python scripts/run_no_pivot_ablation.py
    
    echo ""
    echo "========================================================================"
    echo "✅ ALL EXPERIMENTS COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Results saved to:"
    echo "  - ablation_results/konkani_no_pivot/"
    echo "  - ablation_results/arabic_no_pivot/"
    echo "  - ablation_results/no_pivot_summary/summary.json"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results in summary.json"
    echo "  2. Compare with main paper results (with pivot)"
    echo "  3. Update paper with findings"
    echo ""
else
    echo ""
    echo "Experiments skipped. Run manually with:"
    echo "  python scripts/run_no_pivot_ablation.py"
    echo ""
fi

