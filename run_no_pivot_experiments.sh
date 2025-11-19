#!/bin/bash
# Run no-pivot ablation experiments in conda environment

set -e  # Exit on error

# Parse arguments
WANDB_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            WANDB_FLAG="--wandb"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wandb] [--wandb-project PROJECT_NAME]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "🚀 NO-PIVOT ABLATION EXPERIMENTS"
echo "Running k=3,4,5 for Konkani and Tunisian Arabic (NO PIVOT)"
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

echo "========================================================================"
echo "Step 1: Creating No-Pivot Vector Databases"
echo "========================================================================"
echo ""

# Check if databases already exist
if [ -d "konkani_no_pivot_db" ] && [ -d "arabic_no_pivot_db" ]; then
    echo "⚠️  Vector databases already exist:"
    echo "   - konkani_no_pivot_db/"
    echo "   - arabic_no_pivot_db/"
    echo ""
    read -p "Skip database creation and use existing? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating databases..."
        
        echo "📊 Creating Konkani database..."
        python scripts/create_vector_db_no_pivot.py \
            --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
            --source "eng" \
            --target "gom" \
            --db "konkani_no_pivot_db"
        
        echo ""
        echo "📊 Creating Tunisian Arabic database..."
        python scripts/create_vector_db_no_pivot.py \
            --dataset "predictionguard/arabic_acl_corpus" \
            --source "eng" \
            --target "tun" \
            --db "arabic_no_pivot_db"
    else
        echo "✅ Using existing databases"
    fi
else
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
fi

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
    
    # Build command with wandb flags if provided
    CMD="python scripts/run_no_pivot_ablation.py"
    if [ -n "$WANDB_FLAG" ]; then
        CMD="$CMD $WANDB_FLAG"
        echo "✅ W&B logging ENABLED"
    else
        echo "⚠️  W&B logging DISABLED (add --wandb flag to enable)"
    fi
    if [ -n "$WANDB_PROJECT" ]; then
        CMD="$CMD --wandb-project $WANDB_PROJECT"
        echo "   W&B project: $WANDB_PROJECT"
    fi
    echo ""
    echo "Running: $CMD"
    echo ""
    
    eval $CMD
    
    echo ""
    echo "========================================================================"
    echo "✅ ALL EXPERIMENTS COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Results saved to:"
    echo "  - ablation_no_pivot/konkani/"
    echo "  - ablation_no_pivot/arabic/"
    echo "  - ablation_no_pivot/summary/summary.json"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results: cat ablation_no_pivot/summary/summary.json"
    echo "  2. Compare with main paper results (with pivot)"
    echo "  3. Update paper with findings"
    echo ""
else
    echo ""
    echo "Experiments skipped. Run manually with:"
    echo "  conda activate lrlt_exp"
    echo "  python scripts/run_no_pivot_ablation.py"
    echo ""
fi

