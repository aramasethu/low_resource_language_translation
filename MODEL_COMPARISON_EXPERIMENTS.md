# Model Comparison Experiments: Tower vs Hermes

Complete guide for running ablation studies with both Tower and Hermes models

---

## Overview

You can now run ablation experiments with **TWO different models**:

| Model | Size | Type | Good For |
|-------|------|------|----------|
| **Tower** (TowerInstruct-7B-v0.1) | 7B | Translation-specialized | High translation quality |
| **Hermes** (Hermes-2-Pro-Llama-3-8B) | 8B | General-purpose instruct | General capabilities |

This enables **model comparison** to answer:
- Which model translates better?
- Which benefits more from few-shot learning?
- Do they have same optimal k?

---

## GPU Assignment

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│ Language        │ Tower Model          │ Hermes Model         │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Konkani         │ GPU 0                │ GPU 3                │
│ Arabic          │ GPU 1                │ GPU 4                │
└─────────────────┴──────────────────────┴──────────────────────┘
```

All use `max_new_tokens=600` and `batch_size=4`

---

## Quick Start: Run All 4 Experiments

### Option 1: Run Tower and Hermes Simultaneously (RECOMMENDED)

**Terminal 1** (Tower - GPUs 0 & 1):
```bash
./rerun_both_parallel.sh
```

**Terminal 2** (Hermes - GPUs 3 & 4):
```bash
./rerun_both_hermes_parallel.sh
```

**Total time**: ~4-5 hours (all running in parallel)

---

### Option 2: Run One Model at a Time

#### Tower Model (Baseline)
```bash
# Both languages on Tower
./rerun_both_parallel.sh

# Or separately:
./rerun_konkani_600tokens.sh  # GPU 0
./rerun_arabic_600tokens.sh   # GPU 1
```

#### Hermes Model (Comparison)
```bash
# Both languages on Hermes
./rerun_both_hermes_parallel.sh

# Or separately:
./rerun_konkani_hermes.sh  # GPU 3
./rerun_arabic_hermes.sh   # GPU 4
```

---

## Output Directories

```
ablation_results/
├── konkani_600tokens/          # Tower model - Konkani
├── konkani_hermes_600tokens/   # Hermes model - Konkani
├── arabic_600tokens/           # Tower model - Arabic
└── arabic_hermes_600tokens/    # Hermes model - Arabic
```

Each directory contains:
- `k_0/`, `k_1/`, ..., `k_10/` - Results for each k value
- `summary.csv` - Aggregated results
- `ablation_plots.png` - Line plots
- `bar_chart.png` - Bar chart comparison
- `scores_k*.json` - Detailed metrics

---

## W&B Runs

| Experiment | W&B Run Name | Description |
|------------|--------------|-------------|
| Konkani (Tower) | `konkani-ablation-600tokens` | Baseline Tower model |
| Konkani (Hermes) | `konkani-hermes-600tokens` | Hermes comparison |
| Arabic (Tower) | `arabic-ablation-600tokens` | Baseline Tower model |
| Arabic (Hermes) | `arabic-hermes-600tokens` | Hermes comparison |

All in project: `low-resource-translation-ablation`

---

## Comparison Analysis

After running both models, compare:

### 1. Overall Performance
```bash
# Check summary files
cat ablation_results/konkani_600tokens/summary.csv
cat ablation_results/konkani_hermes_600tokens/summary.csv

cat ablation_results/arabic_600tokens/summary.csv
cat ablation_results/arabic_hermes_600tokens/summary.csv
```

### 2. Best k Value per Model

| Language | Tower Best k | Hermes Best k | Winner |
|----------|--------------|---------------|--------|
| Konkani | ? | ? | TBD |
| Arabic | ? | ? | TBD |

### 3. Few-Shot Learning Benefit

**Question**: Which model benefits MORE from few-shot examples?

Calculate improvement: `(Best_k_score - k0_score) / k0_score * 100%`

### 4. Robustness to k

**Question**: Which model is more stable across different k values?

Calculate standard deviation of BLEU scores across k values.

---

## Expected Results

### Tower Model (Translation-Specialized)
- ✅ Likely **higher absolute BLEU scores**
- ✅ Trained specifically for translation
- ⚠️ May be more sensitive to prompt format

### Hermes Model (General-Purpose)
- ✅ More **general capabilities**
- ✅ May handle varied prompts better
- ⚠️ Possibly lower translation scores (not specialized)

---

## Monitoring Progress

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

Should show:
- GPU 0: Tower - Konkani
- GPU 1: Tower - Arabic
- GPU 3: Hermes - Konkani
- GPU 4: Hermes - Arabic

### Check Logs
```bash
# Tower
tail -f logs/konkani_600tokens.log
tail -f logs/arabic_600tokens.log

# Hermes
tail -f logs/konkani_hermes_600tokens.log
tail -f logs/arabic_hermes_600tokens.log
```

---

## Comparison Commands

### Compare BLEU Scores
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load summaries
tower_kon = pd.read_csv('ablation_results/konkani_600tokens/summary.csv')
hermes_kon = pd.read_csv('ablation_results/konkani_hermes_600tokens/summary.csv')

tower_ar = pd.read_csv('ablation_results/arabic_600tokens/summary.csv')
hermes_ar = pd.read_csv('ablation_results/arabic_hermes_600tokens/summary.csv')

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(tower_kon['k'], tower_kon['BLEU'], marker='o', label='Tower')
axes[0].plot(hermes_kon['k'], hermes_kon['BLEU'], marker='s', label='Hermes')
axes[0].set_title('Konkani: Tower vs Hermes')
axes[0].set_xlabel('k')
axes[0].set_ylabel('BLEU')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(tower_ar['k'], tower_ar['BLEU'], marker='o', label='Tower')
axes[1].plot(hermes_ar['k'], hermes_ar['BLEU'], marker='s', label='Hermes')
axes[1].set_title('Arabic: Tower vs Hermes')
axes[1].set_xlabel('k')
axes[1].set_ylabel('BLEU')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("Saved: model_comparison.png")
```

---

## Troubleshooting

### If Hermes model not found
```bash
# Check if model exists on HuggingFace
huggingface-cli repo info NousResearch/Hermes-2-Pro-Llama-3-8B

# Model will download on first run (~16 GB)
```

### Out of memory
```bash
# Hermes is slightly larger (8B vs 7B)
# If GPU 3 or 4 runs out of memory, reduce batch size:
# Edit scripts: --batch-size 2  (instead of 4)
```

### Different results than expected
- Both models use same test sets
- Both use same k values
- Both use max_new_tokens=600
- Differences are due to model architecture/training

---

## For the Paper

### Model Comparison Section

```
We evaluate two models:
1. TowerInstruct-7B-v0.1: Translation-specialized (7B parameters)
2. Hermes-2-Pro-Llama-3-8B: General-purpose instruct (8B parameters)

Results show:
- Tower achieves X% higher BLEU scores on average
- Hermes shows Y% more benefit from few-shot learning
- Optimal k differs: Tower=X, Hermes=Y
- [Add your findings]

This demonstrates that [specialization/size/training] impacts
few-shot translation effectiveness.
```

---

## Summary

**Tower Model** (Baseline):
- Translation-specialized
- GPUs 0 & 1
- Run: `./rerun_both_parallel.sh`

**Hermes Model** (Comparison):
- General-purpose instruct
- GPUs 3 & 4
- Run: `./rerun_both_hermes_parallel.sh`

**Both Together** (Recommended):
- Use 4 GPUs simultaneously
- Complete in ~4-5 hours
- Full model comparison

**Expected Outcome**:
- 4 complete ablation studies
- Direct model comparison
- Answer: Which model + which k is best?

---

## Quick Reference

| Task | Command | Time | GPUs |
|------|---------|------|------|
| **Tower experiments** | `./rerun_both_parallel.sh` | 4-5h | 0,1 |
| **Hermes experiments** | `./rerun_both_hermes_parallel.sh` | 4-5h | 3,4 |
| **All 4 experiments** | Both scripts in parallel | 4-5h | 0,1,3,4 |
| Konkani Tower | `./rerun_konkani_600tokens.sh` | 4-5h | 0 |
| Arabic Tower | `./rerun_arabic_600tokens.sh` | 3-4h | 1 |
| Konkani Hermes | `./rerun_konkani_hermes.sh` | 4-5h | 3 |
| Arabic Hermes | `./rerun_arabic_hermes.sh` | 3-4h | 4 |

