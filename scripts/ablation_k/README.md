# Ablation Study Scripts (k-value experiments)

This directory contains all scripts related to the ablation study on the number of few-shot examples (k).

## Purpose

These scripts systematically evaluate the impact of varying k (number of few-shot examples) on translation quality, addressing the core research question about optimal few-shot learning for low-resource translation.

## Scripts

### Core Ablation Scripts

**`run_ablation_study.py`** - Main ablation study orchestrator
- Runs inference for multiple k values (e.g., k=0,1,3,5,7,10)
- Handles both Konkani and Arabic datasets
- Integrates W&B logging for experiment tracking
- Generates summary CSV and plots

```bash
python scripts/ablation_k/run_ablation_study.py \
  --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --pivot "hin" --source "mar" --target "gom" \
  --db "translations_db" \
  --k-values 0 1 3 5 7 10 \
  --wandb
```

### Analysis Scripts

**`analyze_ablation_results.py`** - Detailed result analysis
- Parses ablation results from multiple k values
- Generates detailed statistics and insights
- Creates publication-ready tables
- Produces LaTeX tables for papers

```bash
python scripts/ablation_k/analyze_ablation_results.py \
  --results-dir "ablation_results/konkani" \
  --language-name "Konkani" \
  --create-latex
```

**`generate_ablation_table.py`** - LaTeX table generator
- Converts results to publication-ready LaTeX format
- Supports multiple metrics (BLEU, chrF, chrF++)
- Customizable formatting

### Orchestration Scripts

**`run_all_ablations.sh`** - Run complete ablation suite
- Executes ablations for all configurations
- Handles multiple models and languages
- Useful for comprehensive experiments

**`run_arabic_ablation.sh`** - Arabic-specific ablation runner
- Pre-configured for Arabic experiments
- Handles Arabic dataset specifics

## Directory Structure

```
scripts/ablation_k/
├── README.md                      # This file
├── run_ablation_study.py          # Main ablation orchestrator
├── analyze_ablation_results.py    # Result analysis
├── generate_ablation_table.py     # LaTeX table generation
├── run_all_ablations.sh           # Complete ablation suite
└── run_arabic_ablation.sh         # Arabic ablation runner
```

## Related Directories

- `../` (scripts/) - Core inference and data preparation scripts
- `../../ablation_results/` - Output directory for all ablation results
- `../../ABLATION_STUDY.md` - Comprehensive analysis and findings

## Usage Examples

### Quick Test (3 k values)
```bash
python scripts/ablation_k/run_ablation_study.py \
  --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --pivot "hin" --source "mar" --target "gom" \
  --db "translations_db" \
  --output-dir "ablation_results/quick_test" \
  --k-values 0 5 10
```

### Full Ablation (11 k values)
```bash
python scripts/ablation_k/run_ablation_study.py \
  --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --pivot "hin" --source "mar" --target "gom" \
  --db "translations_db" \
  --output-dir "ablation_results/konkani_full" \
  --k-values 0 1 2 3 4 5 6 7 8 9 10 \
  --wandb
```

### Model Comparison
```bash
# Tower model
python scripts/ablation_k/run_ablation_study.py \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --output-dir "ablation_results/konkani_tower" \
  [... other args ...]

# Hermes model
python scripts/ablation_k/run_ablation_study.py \
  --model "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --output-dir "ablation_results/konkani_hermes" \
  [... other args ...]
```

## Output

All scripts output results to `../../ablation_results/` with the following structure:

```
ablation_results/
└── <experiment_name>/
    ├── ablation_summary.csv           # Main results table
    ├── ablation_study_plots.png       # Line plots
    ├── ablation_study_bar_chart.png   # Bar chart
    ├── ablation_detailed_results.json # JSON with all data
    ├── k_0/
    │   ├── results_k0.csv
    │   └── scores_k0.json
    ├── k_1/
    │   ├── results_k1.csv
    │   └── scores_k1.json
    └── ... (for each k value)
```

## Key Research Question

> **To what extent does the number of few-shot examples (k) impact translation quality in low-resource scenarios?**

These scripts provide empirical evidence for:
1. Optimal k value selection
2. Performance trends across k values
3. Model comparison at different k values
4. Zero-shot vs few-shot effectiveness

## Documentation

For detailed findings and analysis, see:
- `../../ABLATION_STUDY.md` - Comprehensive analysis and results
- `../../QUICK_START.md` - Quick start guide for running ablations
- `../../README.md` - Main project documentation

## Dependencies

- Python 3.10+
- PyTorch
- Transformers
- See `../../requirements.txt` for full list

## Notes

- All scripts support W&B integration with `--wandb` flag
- GPU recommended for faster experiments
- Batch size can be adjusted with `--batch-size` flag
- Results are automatically saved and plotted

