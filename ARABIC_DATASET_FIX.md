# Arabic Dataset Fix

## Problem

The Arabic dataset (`predictionguard/arabic_acl_corpus`) has a **nested structure** that's different from the Konkani dataset.

### Konkani Dataset Structure (Simple)
```python
{
  'hin': 'text in Hindi',
  'gom': 'text in Konkani',
  'mar': 'text in Marathi',
  'eng': 'text in English'
}
```

### Arabic Dataset Structure (Nested)
```python
{
  'id': [0, 1, 2, ...],
  'translation': {
    'en': 'text in English',
    'msa': 'text in Modern Standard Arabic',
    'tn': 'text in Tunisian Arabic',
    'eg': 'text in Egyptian Arabic',
    'jo': 'text in Jordanian Arabic',
    'pa': 'text in Palestinian Arabic',
    'sy': 'text in Syrian Arabic'
  }
}
```

The `translation` column contains a **dictionary** with all language variants, not flat columns.

## Solution

Use the specialized scripts I've created:

### 1. Create Vector Database for Arabic

```bash
conda activate lrlt_exp

python scripts/create_vector_db_arabic.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --db "arabic_translations"
```

This will:
- Flatten the nested structure
- Extract: msa (pivot), en (source), tn (target)
- Create vector database in `arabic_translations/`

### 2. Run Inference for Arabic

**Note**: The `run_inference_arabic.py` script is a specialized version for Arabic's nested dataset structure. It provides similar logging output to `run_inference.py`.

```bash
python scripts/run_inference_arabic.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "msa" \
    --source "en" \
    --target "tn" \
    --db "arabic_translations" \
    --output "arabic_results.csv" \
    --scores "arabic_scores.json" \
    --num-examples 5
```

The script will show progress updates, timing information, and evaluation metrics as it runs.

## Correct Step-by-Step Setup

### Step 1: Create Konkani Vector Database ✅
```bash
conda activate lrlt_exp

python scripts/create_vector_db.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations"
```

**Expected output**: `konkani_translations/` directory created

### Step 2: Create Arabic Vector Database ✅
```bash
conda activate lrlt_exp

python scripts/create_vector_db_arabic.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --db "arabic_translations"
```

**Expected output**: `arabic_translations/` directory created

### Step 3: Verify Databases Created
```bash
ls -ld *_translations/
```

You should see:
```
drwxrwxr-x 2 user user 6144 Oct 17 XX:XX arabic_translations/
drwxrwxr-x 2 user user 6144 Oct 17 XX:XX konkani_translations/
```

## Running Ablation Study

### Option A: Run Both Languages (Modified Script Needed)

The current `run_all_ablations.sh` script needs to be updated. For now, run each language separately:

#### Konkani Ablation
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

#### Arabic Ablation (Using Modified Script)

Since the ablation script internally calls `run_inference.py`, we need a workaround. For now:

**Quick Test (k=0 and k=5)**:
```bash
# k=0 (zero-shot)
python scripts/run_inference_arabic.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "msa" --source "en" --target "tn" \
    --db "arabic_translations" \
    --output "ablation_results/arabic/k_0/results_k0.csv" \
    --scores "ablation_results/arabic/k_0/scores_k0.json" \
    --num-examples 0

# k=5
python scripts/run_inference_arabic.py \
    --dataset "predictionguard/arabic_acl_corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "msa" --source "en" --target "tn" \
    --db "arabic_translations" \
    --output "ablation_results/arabic/k_5/results_k5.csv" \
    --scores "ablation_results/arabic/k_5/scores_k5.json" \
    --num-examples 5
```

### Option B: Focus on Konkani Only (Recommended for Now)

Since Konkani works out of the box:

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

This will give you complete ablation results for one language pair, which may be sufficient to address Reviewer 1.

## Why This Happened

The datasets on HuggingFace have different structures:

1. **Konkani dataset**: Created with flat columns (likely from CSV)
2. **Arabic dataset**: Created with nested translation dictionaries (from original ACL corpus format)

The original `create_vector_db.py` and `run_inference.py` scripts assumed flat columns.

## Files Created

1. **`scripts/create_vector_db_arabic.py`** - Handles nested Arabic structure
2. **`scripts/run_inference_arabic.py`** - Inference for Arabic with nested structure
3. **This file** - Documentation

## Available Language Codes in Arabic Dataset

```
en  - English (source)
msa - Modern Standard Arabic (pivot)
tn  - Tunisian Arabic (target)
eg  - Egyptian Arabic (available but not used)
jo  - Jordanian Arabic (available but not used)
pa  - Palestinian Arabic (available but not used)
sy  - Syrian Arabic (available but not used)
```

## Quick Verification

```bash
# Verify Konkani database
python -c "import lancedb; db = lancedb.connect('konkani_translations'); print(db.table_names())"

# Verify Arabic database
python -c "import lancedb; db = lancedb.connect('arabic_translations'); print(db.table_names())"
```

Expected output:
```
['translations_gom']
['translations_tn']
```

## Summary

✅ **Konkani**: Use standard scripts (`create_vector_db.py`, `run_ablation_study.py`)  
✅ **Arabic**: Use specialized scripts (`create_vector_db_arabic.py`, `run_inference_arabic.py`)  

The root cause is different dataset structures on HuggingFace. The specialized scripts handle this properly.

