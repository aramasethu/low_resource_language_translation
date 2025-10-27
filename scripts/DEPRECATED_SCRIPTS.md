# Deprecated Scripts

The following scripts have been superseded by the unified inference script.

## ⚠️ DEPRECATED (Use `run_inference.py` instead)

### `run_inference_konkani_deprecated.py` (Old Konkani-only script)
**Status**: DEPRECATED  
**Replacement**: `run_inference.py` (unified script)  
**Reason**: Replaced by unified script that handles both Konkani and Arabic datasets

### `run_inference_arabic_deprecated.py` (Old Arabic-only script)  
**Status**: DEPRECATED  
**Replacement**: `run_inference.py` (unified script)  
**Reason**: Replaced by unified script that handles both Konkani and Arabic datasets

---

## Migration Guide

### Konkani Translation:
```bash
python scripts/run_inference.py \
  --dataset predictionguard/english-hindi-marathi-konkani-corpus \
  --model Unbabel/TowerInstruct-7B-v0.1 \
  --pivot hin --source mar --target gom \
  --db translations_db \
  --num-examples 3
```

### Arabic Translation:
```bash
python scripts/run_inference.py \
  --dataset pierrebarbera/tunisian_msa_arabizi \
  --model Unbabel/TowerInstruct-7B-v0.1 \
  --pivot msa --source en --target tn \
  --db arabic_translations \
  --num-examples 3
```

---

## Key Improvements in Unified Script
1. **Single codebase**: One script handles both flat (Konkani) and nested (Arabic) datasets
2. **Auto-detection**: Automatically detects and handles dataset structure
3. **Explicit arguments**: Pivot/source/target now required (better clarity)
4. **Easier maintenance**: Fix bugs once, benefits all use cases
5. **Consistent API**: Same interface for all translation tasks

### Automatic Features:
- **Dataset flattening**: Detects nested 'translation' columns and flattens automatically
- **Column validation**: Checks that required columns exist before running
- **Error handling**: Better error messages when columns are missing

---

## Removal Timeline

These scripts are kept for backward compatibility but may be removed in future versions:
- **Phase 1** (Current): Mark as deprecated, update all ablation scripts
- **Phase 2** (After validation): Remove from repository once unified script is validated
- **Phase 3** (Cleanup): Update all documentation to use unified script only

---

## Notes

- All ablation study scripts (`run_ablation_study.py`, `run_arabic_ablation_study.py`) have been updated to use the unified script
- Old scripts will remain functional but won't receive new features or bug fixes
- If you encounter issues with the unified script, please report them

