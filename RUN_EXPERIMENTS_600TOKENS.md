# How to Run Experiments with max_new_tokens=600

**Critical Fix**: Increased max_new_tokens from 200 to 600 to eliminate truncation artifacts

---

## Quick Start: Run Both in Parallel (RECOMMENDED)

**Best option**: Run both Konkani (GPU 0) and Arabic (GPU 1) simultaneously

```bash
./rerun_both_parallel.sh
```

**Time**: ~4-5 hours (wall-clock time, since they run in parallel)  
**GPUs**: Uses GPU 0 and GPU 1 simultaneously  
**Logs**: Creates `logs/konkani_600tokens.log` and `logs/arabic_600tokens.log`

### Monitor Progress

```bash
# Watch Konkani progress
tail -f logs/konkani_600tokens.log

# Watch Arabic progress (in another terminal)
tail -f logs/arabic_600tokens.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

---

## Option 2: Run Separately (Manual Terminal Management)

If you prefer to run them in separate terminals with more control:

### Terminal 1: Konkani (GPU 0)

```bash
./rerun_konkani_600tokens.sh
```

- **GPU**: 0
- **Time**: ~4-5 hours
- **Priority**: CRITICAL (fixes 47-75% truncation)
- **Output**: `ablation_results/konkani_600tokens/`

### Terminal 2: Arabic (GPU 1)

```bash
./rerun_arabic_600tokens.sh
```

- **GPU**: 1  
- **Time**: ~3-4 hours
- **Priority**: Optional (only 1-3% truncation before)
- **Output**: `ablation_results/arabic_600tokens/`

---

## Option 3: Run Only Konkani (Minimal Fix)

If you only want to fix the critical Konkani truncation issue:

```bash
./rerun_konkani_600tokens.sh
```

**Time**: ~4-5 hours  
**Note**: Arabic results from 200 tokens are mostly valid (only 1-3% truncation)

---

## What Changed

### Code Changes (Already Applied)

✅ `scripts/run_inference.py` line 228:
```python
max_new_tokens=600  # Was 200
```

✅ `scripts/run_inference_arabic.py` line 248:
```python
max_new_tokens=600  # Was 200
```

### Why This Fix is Critical

| Language | Old (200 tokens) | New (600 tokens) | Impact |
|----------|------------------|------------------|--------|
| **Konkani** | 47-75% truncated | <5% truncated | 🔴 CRITICAL |
| **Arabic** | 1-3% truncated | ~0% truncated | ✅ Minor |

**Konkani Problem**:
- Reference translations: avg 159 tokens, max 535 tokens
- With max=200: 47-75% were cut off mid-sentence
- BLEU scores artificially LOW
- k=10 failure likely due to truncation

**Arabic** (less severe):
- Reference translations: avg 46 tokens, max 122 tokens  
- With max=200: only 1-3% affected
- Results mostly valid

---

## Expected Results

### Konkani (Major Improvement Expected)

| Metric | Old (200) | New (600) Expected |
|--------|-----------|-------------------|
| BLEU scores | 5-7 range | **Higher** (8-12 range?) |
| Truncation | 47-75% | <5% |
| k=10 behavior | Failed (0.0) | Should work |
| Generated length | 127-149 tok | ~159 tok (match ref) |

### Arabic (Minimal Change Expected)

| Metric | Old (200) | New (600) Expected |
|--------|-----------|-------------------|
| BLEU scores | 4.0-4.5 range | Slightly higher |
| Truncation | 1-3% | ~0% |
| Pattern | 3 clusters | Similar pattern |
| Generated length | 45-54 tok | Similar |

---

## After Running

### 1. Verify Truncation is Fixed

```bash
conda run -n lrlt_exp python check_generated_lengths.py
```

**Expected output**:
- Konkani: <5% samples at 600 tokens (vs 47-75% at 200)
- Arabic: ~0% samples at 600 tokens (vs 1-3% at 200)

### 2. Compare Results

```bash
# Old vs New comparison
ls -la ablation_results/konkani_full/      # Old (200 tokens)
ls -la ablation_results/konkani_600tokens/ # New (600 tokens)

ls -la ablation_results/arabic_full/       # Old (200 tokens)
ls -la ablation_results/arabic_600tokens/  # New (600 tokens)
```

### 3. Analyze Score Improvements

Check `scores_k*.json` files in each directory to compare BLEU scores

### 4. Update Documentation

Update `ABLATION_STUDY.md` with new results:
- New BLEU scores
- Verification that truncation is fixed
- New optimal k values (if changed)

---

## GPU Assignment

| Experiment | GPU | Justification |
|------------|-----|---------------|
| **Konkani** | 0 | CRITICAL fix, run first/primary |
| **Arabic** | 1 | Consistency, can run in parallel |

**Both GPUs have 80 GB memory** → No memory issues expected

---

## Troubleshooting

### Check if experiments are running

```bash
# Check processes
ps aux | grep run_ablation_study
ps aux | grep run_inference_arabic

# Check GPU usage
nvidia-smi
```

### Monitor specific GPU

```bash
# GPU 0 (Konkani)
watch -n 1 'nvidia-smi | grep -A 20 "GPU  0"'

# GPU 1 (Arabic)
watch -n 1 'nvidia-smi | grep -A 20 "GPU  1"'
```

### If a run fails

```bash
# Check last 50 lines of log
tail -50 logs/konkani_600tokens.log
tail -50 logs/arabic_600tokens.log

# Check for errors
grep -i error logs/konkani_600tokens.log
grep -i error logs/arabic_600tokens.log
```

### Resume from specific k value

If Konkani fails at k=7, you can manually resume:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation_study.py \
  --dataset "ai4bharat/IN22-Conv" \
  --dataset-config "kok" \
  --model "Unbabel/TowerInstruct-7B-v0.1" \
  --pivot "mar" \
  --source "hin" \
  --target "gom" \
  --db "konkani_translations" \
  --output-dir "ablation_results/konkani_600tokens" \
  --k-values 7 8 9 10 \
  --batch-size 4 \
  --wandb \
  --wandb-project "low-resource-translation-ablation" \
  --wandb-run-name "konkani-ablation-600tokens-resume"
```

---

## Summary

✅ **RECOMMENDED**: Run `./rerun_both_parallel.sh`  
⏱️ **Time**: 4-5 hours (parallel)  
🎯 **GPUs**: 0 (Konkani) and 1 (Arabic)  
📊 **Expected**: Higher BLEU scores, especially for Konkani  
🔧 **Fix**: Eliminates 47-75% truncation in Konkani

---

## Related Files

- `TRUNCATION_ISSUE.md` - Detailed analysis of the truncation problem
- `check_generated_lengths.py` - Verify truncation is fixed
- `ABLATION_STUDY.md` - Main ablation study documentation (update after rerun)

