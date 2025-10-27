# Quick Start: Ablation Study

This guide will get you running the ablation study in 10 minutes.

## Step 1: Setup Environment

```bash
# Activate your conda environment
conda activate lrlt_exp

# Install required packages (if not already done)
pip install matplotlib seaborn scipy wandb
```

**Optional: Setup Weights & Biases (recommended for experiment tracking)**
```bash
pip install wandb
wandb login  # Enter your API key
```

## Step 2: Create Vector Databases

```bash
# Konkani database
python scripts/create_vector_db.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations"

# Tunisian Arabic database
python scripts/create_vector_db.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --pivot "msa" \
    --source "eng" \
    --target "tun" \
    --db "arabic_translations"
```

**Time**: ~5 minutes

## Step 3: Run Ablation Study

### Option A: Konkani Translation

**Full ablation (k=0 to 10):**
```bash
python scripts/ablation_k/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" --source "mar" --target "gom" \
    --db "translations_db" \
    --output-dir "ablation_results/konkani_tower" \
    --k-values 0 1 2 3 4 5 6 7 8 9 10 \
    --batch-size 4 \
    --wandb \
    --wandb-project "low-resource-translation-ablation" \
    --wandb-run-name "konkani-tower-ablation"
```

**Time**: ~3-4 hours | **GPU**: Any available

**Quick test (k=0, 5, 10):**
```bash
python scripts/ablation_k/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" --source "mar" --target "gom" \
    --db "translations_db" \
    --output-dir "ablation_results/konkani_quick" \
    --k-values 0 5 10 \
    --batch-size 4
```

**Time**: ~1 hour

### Option B: Arabic Translation

**Full ablation (k=0 to 10):**
```bash
python scripts/ablation_k/run_ablation_study.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "msa" --source "en" --target "tn" \
    --db "arabic_translations" \
    --output-dir "ablation_results/arabic_tower" \
    --k-values 0 1 2 3 4 5 6 7 8 9 10 \
    --batch-size 4 \
    --wandb \
    --wandb-project "low-resource-translation-ablation" \
    --wandb-run-name "arabic-tower-ablation"
```

**Time**: ~3-4 hours | **GPU**: Any available

**Note**: The unified `run_ablation_study.py` script works for both languages. Just specify the language columns with `--pivot`, `--source`, and `--target`.

### Option C: Compare Models (Tower vs Hermes)

**Konkani with different models:**
```bash
# Tower model
CUDA_VISIBLE_DEVICES=0 python scripts/ablation_k/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" --source "mar" --target "gom" \
    --db "translations_db" \
    --output-dir "ablation_results/konkani_tower" \
    --k-values 0 1 3 5 7 10 \
    --wandb

# Hermes model (run in parallel on different GPU)
CUDA_VISIBLE_DEVICES=1 python scripts/ablation_k/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "NousResearch/Hermes-2-Pro-Llama-3-8B" \
    --pivot "hin" --source "mar" --target "gom" \
    --db "translations_db" \
    --output-dir "ablation_results/konkani_hermes" \
    --k-values 0 1 3 5 7 10 \
    --wandb
```

**Same works for Arabic** - just change the language arguments:
```bash
--pivot "msa" --source "en" --target "tn"
```

**What you'll see:**
- 🔬 Progress indicators: "EXPERIMENT 1/11: k=0"
- ⏱️ Time estimates and elapsed time per experiment
- ✅ Success messages with BLEU/chrF/chrF++ scores
- 📊 W&B dashboard link (if --wandb enabled)
- 📈 Automatic plots and summary tables

## Step 4: View Results

```bash
# View summary
cat ablation_results/konkani/ablation_summary.csv

# View plots
xdg-open ablation_results/konkani/ablation_study_plots.png  # Linux
# or
open ablation_results/konkani/ablation_study_plots.png       # Mac
```

## Step 5: Analyze Results

```bash
# Generate detailed analysis and LaTeX table
python scripts/ablation_k/analyze_ablation_results.py \
    --results-dir "ablation_results/konkani" \
    --language-name "Konkani" \
    --create-latex

# LaTeX table will be in: ablation_results/konkani/ablation_table.tex
```

## Expected Output Structure

```
ablation_results/
├── konkani/
│   ├── ablation_summary.csv              # Main results table
│   ├── ablation_study_plots.png          # Line plots
│   ├── ablation_study_bar_chart.png      # Bar chart
│   ├── ablation_detailed_results.json    # JSON with all data
│   ├── detailed_analysis.txt             # Text report
│   ├── ablation_table.tex                # LaTeX table
│   ├── k_0/
│   │   ├── results_k0.csv
│   │   └── scores_k0.json
│   ├── k_1/
│   ├── k_3/
│   ├── k_5/
│   ├── k_7/
│   └── k_10/
└── arabic/
    └── ... (same structure)
```

## Troubleshooting

### Out of Memory?
```bash
# Reduce batch size or use smaller k values
python scripts/ablation_k/run_ablation_study.py ... --k-values 0 3 5
```

### GPU Not Available?
```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If False, check your PyTorch installation
```

### Vector DB Not Found?
```bash
# Recreate it
python scripts/create_vector_db.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations"
```

## What to Report in Your Paper

After running the ablation study, you'll have everything you need to address Reviewer 1:

1. **Table**: Use `ablation_table.tex` or data from `ablation_summary.csv`
2. **Figure**: Use `ablation_study_plots.png` or `ablation_study_bar_chart.png`
3. **Key findings**: From `detailed_analysis.txt`

### Example Response to Reviewer

> **Response to Reviewer 1**: We thank the reviewer for this critical observation. We have now conducted a comprehensive ablation study on k, the number of few-shot examples, testing k ∈ {0,1,3,5,7,10} on both language pairs. 
>
> Our results (Table X, Figure X) show:
> 1. Zero-shot (k=0) baseline achieves X.X BLEU on Konkani and Y.Y BLEU on Tunisian Arabic
> 2. Performance improves monotonically up to k=5, with +Z.Z BLEU improvement
> 3. Beyond k=5, we observe [diminishing returns/performance plateau/slight degradation]
> 4. The optimal k is consistent across both language pairs, validating the generalizability of k=5
>
> This provides empirical justification for our choice and demonstrates the robustness of the approach. Full results are in Section X.X.

## Next Steps

1. ✅ Run ablation study
2. ✅ Analyze results
3. ⬜ Add figure to paper
4. ⬜ Add table to paper
5. ⬜ Update methodology section with k justification
6. ⬜ Respond to Reviewer 1

For more details, see [ABLATION_STUDY.md](ABLATION_STUDY.md).


