# No-Pivot Ablation Study

## Purpose

This ablation study tests whether **few-shot examples help without a pivot language**.

### Research Question

We already know:
1. **Pivot alone helps**: k=0 with pivot (5.38 BLEU) > k=5 without pivot (3.49 BLEU)
2. **Pivot + examples is best**: k=5 with pivot (12.14 BLEU)

**New question**: Do examples help when there's NO pivot?
- k=3,4,5 WITHOUT pivot vs k=0 WITHOUT pivot

This isolates the contribution of few-shot examples independent of the pivot language.

## Experimental Design

### Configurations Tested

| Configuration | Translation Path | Few-shot Examples |
|---------------|------------------|-------------------|
| **Existing (main paper)** | Eng→Mar→Konkani | k=0,1,3,5,7,10 |
| **New (this branch)** | Eng→Konkani (direct) | k=3,4,5 |

### Key Differences from Main Experiments

1. **No Pivot Language**
   - Prompt: "Translate from English to Konkani" (not "Marathi to Konkani")
   - Examples: English→Konkani pairs (not Marathi→Konkani)
   - Vector DB: Embeds English source text (not Marathi pivot text)

2. **Limited k Values**
   - Testing k=3,4,5 only (the optimal range from main study)
   - This is sufficient to show if examples help without pivot

3. **Direct Comparison**
   - Can compare to existing k=5 no-pivot results from main paper
   - Shows whether examples help when pivot is absent

## Setup

### Prerequisites

Activate the conda environment:
```bash
conda activate lrlt_exp
```

All commands below should be run in this environment.

### Step 1: Run Tests

Before running full experiments, validate the setup:
```bash
./run_no_pivot_tests.sh
```

This will:
- Test vector database creation
- Run inference on 3 samples with k=3
- Validate prompt format (no pivot language)

**Estimated time**: 5-10 minutes

### Step 2: Create No-Pivot Vector Databases

**Konkani (English→Konkani, no Marathi pivot):**
```bash
python scripts/create_vector_db_no_pivot.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --source "eng" \
    --target "gom" \
    --db "konkani_no_pivot_db"
```

**Tunisian Arabic (English→Tunisian, no MSA pivot):**
```bash
python scripts/create_vector_db_no_pivot.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --source "eng" \
    --target "tun" \
    --db "arabic_no_pivot_db"
```

### Step 3: Run Ablation Experiments

**RECOMMENDED: Use the wrapper script (handles conda environment):**
```bash
./run_no_pivot_experiments.sh
```

This will:
1. Activate conda environment
2. Create vector databases (if needed)
3. Run all experiments (k=3,4,5 for both languages)

**OR run manually in conda environment:**
```bash
conda activate lrlt_exp
python scripts/run_no_pivot_ablation.py
```

**Enable W&B logging (optional):**
```bash
# Option 1: Using shell script
./run_no_pivot_experiments.sh --wandb --wandb-project "low-resource-translation-no-pivot"

# Option 2: Direct Python call
conda activate lrlt_exp
python scripts/run_no_pivot_ablation.py --wandb --wandb-project "low-resource-translation-no-pivot"
```

**Note**: W&B logging is **disabled by default** for all test runs and regular experiments. You must explicitly add `--wandb` to enable it.

**Estimated time**: ~2-3 hours

**Output structure**:
```
ablation_no_pivot/
├── konkani/
│   ├── k_3/
│   │   ├── results_k3_no_pivot.csv
│   │   └── scores_k3_no_pivot.json
│   ├── k_4/
│   └── k_5/
├── arabic/
│   ├── k_3/
│   ├── k_4/
│   └── k_5/
└── summary/
    └── summary.json
```

### Step 3: Run Single Experiment (Optional)

**Test with Konkani k=3:**
```bash
python scripts/run_inference_no_pivot.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.2" \
    --source "eng" \
    --target "gom" \
    --db "konkani_no_pivot_db" \
    --output "test_no_pivot.csv" \
    --scores "test_no_pivot_scores.json" \
    --num-examples 3
```

## Expected Results

### Hypothesis

**Scenario A: Examples help even without pivot**
- k=0 no-pivot: ~3.5 BLEU (existing result)
- k=3-5 no-pivot: ~5-6 BLEU (improvement)
- **Interpretation**: Few-shot learning works independently

**Scenario B: Examples don't help without pivot**
- k=0 no-pivot: ~3.5 BLEU (existing result)
- k=3-5 no-pivot: ~3.5 BLEU (no improvement)
- **Interpretation**: Pivot is necessary for few-shot to work

**Scenario C: Examples hurt without pivot**
- k=0 no-pivot: ~3.5 BLEU (existing result)
- k=3-5 no-pivot: ~2.5 BLEU (degradation)
- **Interpretation**: Examples confuse model without pivot

## Comparison to Existing Results

### Konkani (Expected)

| Configuration | Pivot | k | BLEU | Interpretation |
|--------------|-------|---|------|----------------|
| Main paper | ✅ Yes | 0 | 5.38 | Pivot alone helps |
| Main paper | ✅ Yes | 5 | 12.14 | Pivot + examples is best |
| Main paper | ❌ No | 5 | 3.49 | No pivot, with examples |
| **This study** | ❌ No | 0 | ??? | No pivot, no examples (baseline) |
| **This study** | ❌ No | 3 | ??? | No pivot, 3 examples |
| **This study** | ❌ No | 4 | ??? | No pivot, 4 examples |
| **This study** | ❌ No | 5 | ??? | No pivot, 5 examples |

### Key Insights This Will Reveal

1. **Do examples help without pivot?**
   - Compare: k=3-5 no-pivot vs k=0 no-pivot

2. **Is pivot more important than examples?**
   - Compare: k=0 with pivot vs k=5 without pivot
   - We already know: 5.38 > 3.49 ✅ **Pivot wins**

3. **Optimal strategy?**
   - k=0 with pivot: 5.38 BLEU
   - k=5 without pivot: 3.49 BLEU
   - k=5 with pivot: 12.14 BLEU
   - **Conclusion**: Both together is best

## Files Created

### New Scripts
1. `scripts/create_vector_db_no_pivot.py` - Create vector DB for no-pivot experiments
2. `scripts/run_inference_no_pivot.py` - Run inference without pivot language
3. `scripts/run_no_pivot_ablation.py` - Orchestrate all no-pivot experiments

### New Data
1. `konkani_no_pivot_db/` - Vector DB with English embeddings
2. `arabic_no_pivot_db/` - Vector DB with English embeddings
3. `ablation_results/konkani_no_pivot/` - Results for Konkani
4. `ablation_results/arabic_no_pivot/` - Results for Arabic
5. `ablation_results/no_pivot_summary/` - Summary statistics

## Git Branch

This work is on branch: `rohin/no-pivot-ablation`

To merge back into main after experiments complete:
```bash
git checkout main
git merge rohin/no-pivot-ablation
```

## Paper Integration

After running experiments, update paper with:

1. **New table**: No-pivot ablation results (k=3,4,5)
2. **Comparison**: Show that pivot > examples in importance
3. **Finding**: Whether few-shot helps without pivot

**Suggested addition to ablation section**:
> "To isolate the contribution of few-shot examples independent of the pivot language, we conducted a secondary ablation study testing k∈{3,4,5} without any pivot language. Results show that [INSERT FINDING], confirming that [the pivot language is the primary driver of improvement / few-shot examples provide complementary benefits / etc.]."

## Notes

- This is a **complementary** study to the main k-ablation (which uses pivot)
- Only tests k=3,4,5 (the optimal range) to save compute
- Provides rigorous evidence for the importance of pivot vs examples
- Strengthens the paper's ablation analysis

