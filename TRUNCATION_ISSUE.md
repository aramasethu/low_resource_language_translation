# Critical Finding: max_new_tokens=200 Truncation Issue

**Date**: October 22, 2025  
**Status**: 🔴 **CRITICAL - Konkani Results Invalid**  
**Severity**: High - Affects validity of ablation study results

---

## Summary

Investigation revealed that `max_new_tokens=200` is **severely truncating Konkani translations**, making the ablation study results unreliable for that language.

---

## The Issue

### What is `max_new_tokens`?

`max_new_tokens=200` limits the **OUTPUT** length (generated translation), NOT the input prompt/context.

- ✅ Input prompts are fine (k=10 prompt = ~1583 tokens, well within model limits)
- 🔴 Output translations are truncated at 200 tokens

### Location

Set in both inference scripts:

```python
# scripts/run_inference.py line 228
# scripts/run_inference_arabic.py line 248
results = pipeline_model(
    batch_prompts,
    max_new_tokens=200,  # <-- LIMITS OUTPUT LENGTH
    ...
)
```

---

## Impact Analysis

### Arabic Translation: ✅ Mostly OK

| Metric | Value |
|--------|-------|
| **Reference mean** | 46 tokens |
| **Reference max** | 122 tokens |
| **Generated mean** | 45-54 tokens |
| **Generated max** | 201 tokens |
| **Truncated samples** | 1-3 out of 100 (1-3%) |
| **Verdict** | ✅ Acceptable - only 1-3% affected |

### Konkani Translation: 🔴 SEVERELY AFFECTED

| Metric | Value |
|--------|-------|
| **Reference mean** | **159 tokens** |
| **Reference max** | **535 tokens** |
| **Generated mean** | 127-149 tokens |
| **Generated max** | 201 tokens (HARD CAP) |
| **Truncated samples** | **47-75 out of 100 (47-75%)** |
| **Gap** | Generated are 12-20 tokens shorter than references |
| **Verdict** | 🔴 **CRITICAL ISSUE** |

#### Konkani Truncation by k Value:

| k | Samples ≥190 tokens | % Truncated | Notes |
|---|---------------------|-------------|-------|
| k=0 | 58/100 | 58% | High truncation |
| k=3 | **75/100** | **75%** | Worst truncation |
| k=5 | 54/100 | 54% | High truncation |
| k=10 | 47/100 | 47% | Still significant |

---

## Why This Invalidates Konkani Results

### 1. **Incomplete Translations**
- Translations are cut off mid-sentence
- Generated text is artificially shortened
- Missing content compared to references

### 2. **Artificially Low BLEU Scores**
- BLEU heavily penalizes length mismatch
- Incomplete translations get lower n-gram matches
- Scores are compressed and unreliable

### 3. **Wrong Interpretation of k=10 Failure**
We thought k=10's poor performance (0.0 BLEU) was due to:
- Context overload ❌
- GPU memory issues ❌
- Model degradation ❌

**Actually**: It's likely due to **TRUNCATION ARTIFACTS**:
- Longer context → model tries to generate longer, more complete translations
- These get truncated MORE severely at 200 tokens
- Truncated translations have worse BLEU scores

### 4. **Comparison Between k Values is Unfair**
- k=3 has 75% truncation
- k=10 has 47% truncation  
- Different truncation rates make comparison meaningless

---

## Evidence from the Data

### Konkani Generated vs Reference Lengths

```
k=0:   Generated mean=141.5  Reference mean=159.4  (-17.9 tokens)
k=3:   Generated mean=148.6  Reference mean=159.4  (-10.8 tokens) ⚠️ Best k but highest truncation!
k=5:   Generated mean=140.5  Reference mean=159.4  (-18.9 tokens)
k=10:  Generated mean=127.4  Reference mean=159.4  (-32.0 tokens) ⚠️ Worst performance!
```

**Pattern**: All generated translations are systematically SHORTER than references!

### Distribution of Reference Lengths

Konkani references exceed 200 tokens frequently:
- 69% are 0-200 tokens (OK)
- **31% are >200 tokens (TRUNCATED)**
- Max reference: 535 tokens (2.7x the limit!)

---

## Root Cause Analysis

### Why Was This Set to 200?

Likely reasons:
1. Conservative default to prevent runaway generation
2. Worked for development testing (short examples)
3. Based on Arabic test set (which IS under 200 tokens)
4. Memory optimization for batch processing

### Why Didn't We Notice Earlier?

1. Arabic worked fine (references are short)
2. BLEU scores seemed reasonable for Konkani (5-7 range)
3. No explicit warning when truncation occurs
4. Focus was on comparing k values, not absolute quality

---

## Required Actions

### Immediate

1. 🔴 **Mark Konkani results as preliminary/invalid**
2. ✅ **Arabic results remain valid** (only 1-3% affected)
3. 📝 **Document this issue** (this file)

### For Rerunning Experiments

1. **Update max_new_tokens to 600**
   ```python
   max_new_tokens=600  # Accommodates max ref length of 535 + buffer
   ```

2. **Rerun Konkani ablation study completely**
   - All k values (0-10)
   - With GPU 7 and batch_size=4
   - Updated scripts with new max_new_tokens

3. **Optional: Rerun Arabic with 600 for consistency**
   - Though current Arabic results are valid
   - Would enable fair comparison

### Code Changes Needed

```bash
# Update run_inference.py
sed -i 's/max_new_tokens=200/max_new_tokens=600/g' scripts/run_inference.py

# Update run_inference_arabic.py
sed -i 's/max_new_tokens=200/max_new_tokens=600/g' scripts/run_inference_arabic.py
```

---

## Implications for Paper

### What This Means for the Reviewer Response

**Before (invalid interpretation)**:
> "Konkani shows optimal performance at k=3-5 with substantial improvements (+37.8%)"

**After (with truncation artifact)**:
> "Initial experiments showed artifacts due to output length limitation (max_new_tokens=200), 
> which truncated 47-75% of Konkani translations. We have rerun all experiments with 
> max_new_tokens=600 to accommodate the longer Konkani translations (max: 535 tokens vs 
> Arabic max: 122 tokens)."

### Silver Lining

This discovery:
1. ✅ Validates the importance of thorough analysis
2. ✅ Shows we're doing due diligence  
3. ✅ May explain some unexpected patterns
4. ✅ Arabic results remain valid (strengthens those findings)

---

## Verification Checklist

Before considering Konkani results valid:

- [ ] max_new_tokens set to ≥600
- [ ] Verify generated translations are comparable in length to references
- [ ] Check that <5% of translations hit the limit
- [ ] Recompute all BLEU scores
- [ ] Compare new scores to old scores (should be higher)
- [ ] Verify k=10 behavior improves

---

## Questions for User

1. **Should we rerun Konkani experiments immediately?**
   - With max_new_tokens=600
   - All k values (0-10)
   - GPU 7, batch_size=4

2. **Should we also rerun Arabic for consistency?**
   - Current Arabic results are valid
   - But using 600 for both enables fair comparison

3. **Timeline for rerunning?**
   - Full ablation study will take several hours
   - Can run overnight if needed

---

## Related Files

- `check_token_lengths.py` - Initial analysis (Arabic only)
- `check_konkani_lengths.py` - Konkani length check (auth failed)
- `check_generated_lengths.py` - Comparative analysis that found the issue
- `scripts/run_inference.py` - Line 228: max_new_tokens=200
- `scripts/run_inference_arabic.py` - Line 248: max_new_tokens=200

---

## Conclusion

🔴 **Konkani results are invalid due to severe truncation (47-75% of samples)**  
✅ **Arabic results remain valid (only 1-3% affected)**  
🔧 **Fix: Increase max_new_tokens to 600 and rerun Konkani experiments**

This is a critical finding that requires immediate action before publication.

---

**Discovered by**: Token length analysis investigating context limits  
**Impact**: High - Invalidates primary language results  
**Fix complexity**: Low - Single parameter change  
**Rerun time**: ~4-6 hours for full Konkani ablation

