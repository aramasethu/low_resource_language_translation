# NMT Baseline Translation Results


**Model**: `facebook/nllb-200-distilled-600M`  
**Translation Approach**: Direct (English → Target, no pivot)

## Results Summary

### Konkani (English → Konkani)

**Note**: Konkani (`gom_Deva`) is not directly supported in NLLB-200. Used Marathi (`mar_Deva`) as a proxy since both languages use the Devanagari script and are closely related.

| Metric | Score |
|--------|-------|
| **BLEU** | 7.51 |
| **chrF** | 33.47 |
| **CHRF++** | 26.82 |
| **TER** | 114.05 |
| **COMET** | 0.567 |


### Tunisian Arabic (English → Tunisian Arabic)

| Metric | Score |
|--------|-------|
| **BLEU** | 4.20 |
| **chrF** | 13.89 |
| **CHRF++** | 10.42 |
| **TER** | 96.15 |
| **COMET** | 0.577 |



## Usage

```bash
python scripts/nmt_baseline.py --output-dir nmt_baseline_results
```



