# Ablation Study: Impact of Few-Shot Examples (k)

## Purpose

This ablation study addresses **Reviewer 1's comment**:
> "The number of few-shot examples was fixed at k=5 with zero justification. This is a core hyperparameter of the proposed method, and the failure to perform an ablation study on its effect is a critical omission."

We systematically evaluate the impact of the number of few-shot examples (k) on translation quality across both language pairs.

## Hypothesis

We hypothesize that:
1. **k=0 (zero-shot)**: Baseline performance without few-shot examples
2. **k=1-3**: Minimal context, may show some improvement
3. **k=5**: Original choice - expected to show significant improvement
4. **k=7-10**: Diminishing returns or potential degradation due to context length

## Experimental Design

### Tested Values of k
- **k=0**: Zero-shot baseline (no few-shot examples)
- **k=1**: Single example
- **k=3**: Three examples
- **k=5**: Original paper configuration
- **k=7**: Increased context
- **k=10**: Maximum context

### Language Pairs
1. **Konkani**: Hindi (pivot) → Marathi (source) → Konkani (target)
2. **Tunisian Arabic**: Modern Standard Arabic (pivot) → English (source) → Tunisian Arabic (target)

### Evaluation Metrics
- **BLEU**: Standard machine translation metric
- **chrF**: Character n-gram F-score (better for morphologically rich languages)
- **chrF++**: Enhanced chrF with word n-grams

### Control Variables
- Model: `Unbabel/TowerInstruct-7B-v0.1` (fixed)
- Temperature: 0.1 (fixed)
- Max tokens: 200 (fixed)
- Embedding model: `all-MiniLM-L12-v2` (fixed)
- Test set: Same for all k values

## Running the Ablation Study

### Prerequisites
```bash
# Ensure vector databases are created
python scripts/create_vector_db.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations"

python scripts/create_vector_db.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --pivot "msa" \
    --source "eng" \
    --target "tun" \
    --db "arabic_translations"
```

### Option 1: Run All Experiments (Recommended)
```bash
chmod +x scripts/run_all_ablations.sh
./scripts/run_all_ablations.sh
```

This will:
- Run ablation for Konkani (k=0,1,3,5,7,10)
- Run ablation for Tunisian Arabic (k=0,1,3,5,7,10)
- Generate visualizations for each
- Create a combined comparison report

**Estimated time**: 4-6 hours (depending on GPU)

### Option 2: Run Individual Language Pairs

#### Konkani Only
```bash
python scripts/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations" \
    --output-dir "ablation_results/konkani" \
    --k-values 0 1 3 5 7 10
```

**With Weights & Biases logging:**
```bash
python scripts/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations" \
    --output-dir "ablation_results/konkani" \
    --k-values 0 1 3 5 7 10 \
    --wandb \
    --wandb-project "low-resource-translation" \
    --wandb-run-name "konkani_ablation_v1"
```

#### Tunisian Arabic Only
```bash
# Note: Use the specialized Arabic script for the nested dataset structure
for k in 0 1 3 5 7 10; do
    mkdir -p "ablation_results/arabic/k_$k"
    python scripts/run_inference_arabic.py \
        --dataset "predictionguard/arabic_acl_corpus" \
        --model "Unbabel/TowerInstruct-7B-v0.1" \
        --pivot "msa" --source "en" --target "tn" \
        --db "arabic_translations" \
        --output "ablation_results/arabic/k_$k/results_k$k.csv" \
        --scores "ablation_results/arabic/k_$k/scores_k$k.json" \
        --num-examples $k
done

# Then analyze results
python scripts/analyze_ablation_results.py \
    --results-dir "ablation_results/arabic" \
    --language-name "Arabic" \
    --create-latex
```

### Option 3: Custom k Values
```bash
# Test only k=0, 5, 10 for quick comparison
python scripts/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations" \
    --output-dir "ablation_results/konkani_quick" \
    --k-values 0 5 10
```

## New Features: Enhanced Logging & Experiment Tracking

### 🎯 Comprehensive Logging
The ablation script now includes:
- **Timestamped logs**: `[2025-10-17 20:45:00] [INFO] message`
- **Progress indicators**: `🔬 EXPERIMENT 1/6: k=0 few-shot examples`
- **Status emojis**: 🔬 (running), ✅ (success), ❌ (error), ⚠️ (warning)
- **Time tracking**: Shows elapsed time per experiment and total time
- **Model download detection**: Warns on first run about ~14 GB download
- **Clear success/failure messages** with scores

### 📊 Weights & Biases Integration

**Setup (one-time):**
```bash
# Install wandb
pip install wandb

# Login to W&B
wandb login
```

**Command Line Options:**
```bash
--wandb                    # Enable W&B logging
--wandb-project NAME       # W&B project name (default: low-resource-translation)
--wandb-run-name NAME      # Custom run name (default: auto-generated)
```

**What Gets Logged:**
- **Per experiment**: BLEU, chrF, chrF++ scores for each k value
- **Time metrics**: Execution time per k value
- **Improvements**: Improvement over k=0 baseline
- **Visualizations**: All plots automatically uploaded
- **Summary tables**: Complete results table
- **Configuration**: Dataset, model, languages, k values tested
- **Public URL**: Shareable link to view results online

**Example Output:**
```
[2025-10-17 20:45:00] [INFO] 🚀 ABLATION STUDY: Number of Few-Shot Examples (k)
[2025-10-17 20:45:00] [INFO] Testing k values: [0, 1, 3, 5, 7, 10]
[2025-10-17 20:45:00] [INFO] 📊 Initializing Weights & Biases logging...
[2025-10-17 20:45:01] [SUCCESS] ✅ W&B initialized successfully
[2025-10-17 20:45:01] [INFO] 🔬 Running 6 experiments
[2025-10-17 20:45:01] [INFO] ⏱️  Estimated time: 150min - 240min

[2025-10-17 20:45:01] [INFO] 🔬 EXPERIMENT 1/6: k=0 few-shot examples
[2025-10-17 20:45:01] [INFO] Running ZERO-SHOT baseline
[2025-10-17 20:45:01] [INFO] ⏳ First run: Model will be downloaded (~14 GB, ~10-15 min)
[2025-10-17 20:45:01] [INFO] ▶️  Starting inference for k=0...
[2025-10-17 21:10:23] [SUCCESS] ✅ COMPLETED k=0 in 25.4 minutes
[2025-10-17 21:10:23] [SUCCESS]    BLEU: 15.23 | chrF: 42.56 | chrF++: 40.12

...

[2025-10-17 23:45:00] [SUCCESS] 🎉 ABLATION STUDY COMPLETE!
[2025-10-17 23:45:00] [INFO] ⏱️  Total time: 180.5 minutes (3.01 hours)
[2025-10-17 23:45:00] [INFO] 💾 All results saved to: ablation_results/konkani
[2025-10-17 23:45:00] [INFO] 🌐 View results on W&B: https://wandb.ai/...
```

**Benefits:**
- ✅ Real-time progress tracking
- ✅ Easy comparison across runs
- ✅ Shareable results with collaborators/reviewers
- ✅ Automatic plot generation and hosting
- ✅ Complete experiment history

## Output Files

After running the ablation study, you'll find:

```
ablation_results/
├── experiment_config.txt                    # Experiment metadata
├── combined_ablation_results.csv            # Cross-language comparison
├── konkani/
│   ├── ablation_summary.csv                 # Summary table
│   ├── ablation_detailed_results.json       # Detailed JSON results
│   ├── ablation_study_plots.png            # Line plots for all metrics
│   ├── ablation_study_bar_chart.png        # Bar chart comparison
│   ├── k_0/
│   │   ├── results_k0.csv                  # Predictions for k=0
│   │   └── scores_k0.json                  # Scores for k=0
│   ├── k_1/
│   │   ├── results_k1.csv
│   │   └── scores_k1.json
│   └── ... (k_3, k_5, k_7, k_10)
└── arabic/
    └── ... (same structure as konkani)
```

## Interpreting Results

### Key Questions to Answer

1. **Does k=0 (zero-shot) provide a reasonable baseline?**
   - This validates whether the pivot language approach works without examples

2. **At what k does performance plateau?**
   - Identifies the optimal number of examples
   - Helps understand if k=5 was indeed a good choice

3. **Are there diminishing returns after a certain k?**
   - Important for computational efficiency
   - May indicate when additional context becomes noise

4. **Is the optimal k consistent across language pairs?**
   - Tests generalizability of the approach
   - May inform language-specific recommendations

5. **What is the performance gap between k=0 and k=optimal?**
   - Quantifies the value of few-shot learning
   - Justifies the added complexity of retrieval

### Expected Patterns

**Scenario A: Linear Improvement**
```
k=0 < k=1 < k=3 < k=5 < k=7 < k=10
```
→ Suggests more examples always help (within tested range)

**Scenario B: Plateau Effect**
```
k=0 < k=1 < k=3 ≈ k=5 ≈ k=7 ≈ k=10
```
→ Suggests diminishing returns after k=3

**Scenario C: Optimal Sweet Spot**
```
k=0 < k=1 < k=3 < k=5 > k=7 > k=10
```
→ Suggests k=5 is optimal; more examples hurt performance

## Statistical Significance

For publication, consider:
1. **Multiple runs**: Run each k value 3-5 times with different random seeds
2. **Confidence intervals**: Report mean ± std dev for each metric
3. **Statistical tests**: Paired t-tests between k values
4. **Effect sizes**: Cohen's d to measure practical significance

### Example: Multiple Runs
```bash
for seed in 42 123 456; do
    python scripts/run_ablation_study.py \
        --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
        --model "Unbabel/TowerInstruct-7B-v0.1" \
        --pivot "hin" --source "mar" --target "gom" \
        --db "konkani_translations" \
        --output-dir "ablation_results/konkani_seed${seed}" \
        --k-values 0 1 3 5 7 10
done
```

## Reporting for Paper

### Recommended Visualizations

1. **Line plot**: BLEU/chrF/chrF++ vs k (already generated)
2. **Bar chart**: Side-by-side comparison (already generated)
3. **Heatmap**: Language pair × k value × metric
4. **Improvement plot**: Relative improvement over k=0

### Recommended Table Format

| k | Konkani BLEU | Konkani chrF | Arabic BLEU | Arabic chrF | Avg Improvement |
|---|-------------|--------------|-------------|-------------|-----------------|
| 0 | X.XX        | X.XX         | X.XX        | X.XX        | -               |
| 1 | X.XX (+Δ)   | X.XX (+Δ)    | X.XX (+Δ)   | X.XX (+Δ)   | +Δ%            |
| 3 | X.XX (+Δ)   | X.XX (+Δ)    | X.XX (+Δ)   | X.XX (+Δ)   | +Δ%            |
| 5 | X.XX (+Δ)   | X.XX (+Δ)    | X.XX (+Δ)   | X.XX (+Δ)   | +Δ%            |
| 7 | X.XX (+Δ)   | X.XX (+Δ)    | X.XX (+Δ)   | X.XX (+Δ)   | +Δ%            |
| 10| X.XX (+Δ)   | X.XX (+Δ)    | X.XX (+Δ)   | X.XX (+Δ)   | +Δ%            |

### Addressing the Reviewer's Concern

In your response to Reviewer 1:

> **Response to Reviewer 1**: We have now conducted a comprehensive ablation study on k, the number of few-shot examples (k ∈ {0,1,3,5,7,10}). Our results show that:
> 
> 1. **Zero-shot baseline (k=0)**: Achieves X.X BLEU on Konkani and Y.Y BLEU on Tunisian Arabic
> 2. **Optimal k**: Performance peaks at k=[best_k] with Z.Z% improvement over baseline
> 3. **k=5 justification**: Our original choice of k=5 [is/is not] optimal, achieving [X]% of maximum possible improvement
> 4. **Diminishing returns**: We observe [describe pattern] after k=[threshold]
> 5. **Cross-language consistency**: The optimal k is [consistent/varies] across language pairs
> 
> These findings provide empirical justification for our choice of k=5 and demonstrate the robustness of the few-shot retrieval approach. Full results are presented in Section X.X and Figure X.

## Troubleshooting

### Issue: Out of Memory
```bash
# Reduce k values tested
python scripts/run_ablation_study.py ... --k-values 0 3 5

# Or run sequentially with memory cleanup
for k in 0 1 3 5 7 10; do
    python scripts/run_inference.py --num-examples $k ...
    sleep 10  # Allow GPU memory to clear
done
```

### Issue: Vector DB Not Found
```bash
# Recreate the vector database
python scripts/create_vector_db.py --dataset [...] --pivot [...] --source [...] --target [...] --db [...]
```

### Issue: Missing Dependencies
```bash
# Install visualization dependencies
pip install matplotlib seaborn
```

## Next Steps

After completing the ablation study:

1. ✅ **Include results in paper** (Section on Experimental Design)
2. ✅ **Update response to Reviewer 1**
3. ✅ **Add figure to paper** showing k vs performance
4. ⬜ **Consider**: Ablate other hyperparameters (temperature, retrieval metric, embedding model)
5. ⬜ **Consider**: Test with fine-tuned models at different k values

## Citation

If you use this ablation study methodology:

```bibtex
@misc{lowresource_ablation,
  title={Ablation Study on Few-Shot Examples for Low-Resource Translation},
  note={Addresses systematic evaluation of k hyperparameter in retrieval-augmented translation}
}
```


