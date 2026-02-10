# Claude.md - AI Assistant Guide for Low-Resource Translation Project

## Project Overview

This is a **research codebase** for a paper on **low-resource machine translation using pivot languages and few-shot learning**. The work involves extensive ablation studies to validate the approach.

**Current Status:** Paper in revision, addressing Reviewer 1's criticism about k=5 justification through comprehensive ablation studies.

---

## Quick Context

### Core Innovation
- **Problem:** Translating to low-resource languages (Konkani, Tunisian Arabic) with <1,000 training samples
- **Solution:** Use linguistically similar pivot languages + semantically retrieved few-shot examples
- **Languages:**
  - **Konkani** (gom): English → Marathi (mar) → Konkani
  - **Tunisian Arabic** (aeb): English → Modern Standard Arabic (msa) → Tunisian Arabic

### Models
- **TowerInstruct-7B-v0.2**: Translation-specialized (Mistral-based) - fails at k≥6
- **Hermes-2-Pro-Llama-3-8B**: General-purpose (Llama-3-based) - stable, recommended

---

## Branch Organization

**This repository has MULTIPLE branches for different ablation studies:**

### 1. `rohin/experiments` (CURRENT MAIN)
- **Purpose:** Main k ablation (k=0-10) with pivot + semantic retrieval
- **Status:** Complete, ready for paper integration
- **Key Results:**
  - Konkani: 8.22 BLEU at k=7 (Hermes) - +451% improvement
  - Arabic: 6.74 BLEU at k=9 (Hermes) - +46% improvement
  - Tower fails catastrophically at k≥6 (0.00 BLEU)
- **Documentation:** `ABLATION_STUDY.md`, `paper_updates/`

### 2. `rohin/no-pivot-ablation`
- **Purpose:** Test if few-shot examples help WITHOUT pivot language
- **Status:** Complete
- **Key Finding:** Pivot language is ESSENTIAL (+3605% to +40100% improvement)
- **Documentation:** `NO_PIVOT_ABLATION.md`, `NO_PIVOT_RESULTS.md`
- **Results in:** `ablation_no_pivot/`

### 3. `harshwardha/ablation_study`
- **Purpose:** Semantic vs Random retrieval comparison
- **Status:** Complete
- **Key Finding:** Random sampling performs equally well or BETTER than semantic
- **Impact:** Can eliminate expensive vector DB infrastructure
- **Documentation:** `SAMPLING_ABLATION_CONCLUSIONS.md`, `DOCUMENTATION.md`
- **Results in:** `outputs/sampling_ablation/`

### 4. `pairwise-jaccard-similarity`
- **Purpose:** Quantitative justification for pivot language choices
- **Status:** Complete
- **Key Finding:** Marathi-Konkani 32.6% similarity, MSA-Tunisian 31.9%
- **Results:** `jaccard_results.json`

### 5. `Random-sampling`
- **Purpose:** Implementation of random sampling as alternative to semantic
- **Status:** Code complete
- **Script:** `scripts/random_samplingv2.py`

---

## Directory Structure

```
.
├── README.md                           # Main project overview
├── ABLATION_STUDY.md                   # Main k ablation documentation (detailed!)
├── QUICK_START.md                      # Get running in 10 minutes
├── TECHNICAL_SPECS.md                  # Hardware, model specs
├── NO_PIVOT_ABLATION.md               # (no-pivot branch) No-pivot study guide
├── NO_PIVOT_RESULTS.md                # (no-pivot branch) No-pivot findings
├── SAMPLING_ABLATION_CONCLUSIONS.md   # (harshwardha branch) Semantic vs random
├── DOCUMENTATION.md                    # (harshwardha branch) Full guide
│
├── paper_updates/                      # Paper revision materials
│   ├── README.md                       # Start here for paper updates
│   ├── KEY_NUMBERS_CHEATSHEET.md      # Quick reference for stats
│   ├── RESULTS_UPDATE_SUMMARY.md      # What's changed in results
│   ├── results_section_update.tex     # Complete new Results section (~5000 words)
│   ├── additional_tables.tex          # 6 publication-ready LaTeX tables
│   ├── integration_instructions.md    # How to integrate updates
│   ├── paper.tex                       # LaTeX paper file
│   └── figures/                        # Generated plots
│       ├── konkani_ablation.pdf/png
│       ├── arabic_ablation.pdf/png
│       └── pivot_comparison.pdf/png
│
├── scripts/                            # All Python scripts
│   ├── create_vector_db.py            # Build semantic search index
│   ├── run_inference.py               # Main inference script
│   ├── translation_finetuning.py      # Fine-tune with LoRA
│   ├── generate_paper_figures.py      # Create publication plots
│   ├── random_samplingv2.py           # (Random-sampling branch)
│   └── ablation_k/                    # Ablation study scripts
│       ├── run_ablation_study.py      # Systematic k ablation
│       ├── analyze_ablation_results.py # Generate analysis
│       └── generate_ablation_table.py  # LaTeX tables
│
├── ablation_results/                   # Main k ablation results
│   ├── konkani_600tokens/             # ✅ SOURCE OF TRUTH: Tower Konkani k=0-10
│   ├── konkani_hermes_600tokens/      # ✅ SOURCE OF TRUTH: Hermes Konkani k=0-10
│   ├── arabic_600tokens/              # ✅ SOURCE OF TRUTH: Tower Arabic k=0-10
│   ├── arabic_hermes_20251126/        # ✅ SOURCE OF TRUTH: Hermes Arabic k=0-10 (latest)
│   └── plots/                         # Comparison visualizations
│
│   ⚠️  NOTE: Other folders in ablation_results/ (e.g., konkani_full, arabic_full,
│       arabic_hermes_600tokens) are from older runs on different branches.
│       ONLY use the 4 folders marked above for paper verification!
│
├── ablation_no_pivot/                  # (no-pivot branch) No-pivot results
│   ├── konkani/                       # Tower no-pivot k=3,4,5
│   ├── konkani_hermes/                # Hermes no-pivot k=3,4,5
│   ├── arabic/                        # Tower no-pivot k=3,4,5
│   └── arabic_hermes/                 # Hermes no-pivot k=3,4,5
│
├── outputs/sampling_ablation/          # (harshwardha branch) Semantic vs random
│   ├── experiment_summary.json
│   ├── tower_semantic_k0-10.json
│   ├── tower_random_k0-10.json
│   ├── hermes_semantic_k0-10.json
│   └── hermes_random_k0-10.json
│
├── data/                               # Datasets and vector DBs
├── konkani_translations/              # Vector DB for Konkani (with pivot)
├── arabic_translations/               # Vector DB for Arabic (with pivot)
├── konkani_no_pivot_db/              # Vector DB for Konkani (no pivot)
├── arabic_no_pivot_db/               # Vector DB for Arabic (no pivot)
└── wandb/                             # Weights & Biases experiment logs
```

---

## Key Concepts

### Pivot Language
- **Definition:** A linguistically similar language used as a bridge
- **Why:** Models have limited exposure to target language but know pivot well
- **Examples:**
  - Konkani ← **Marathi** ← English
  - Tunisian Arabic ← **Modern Standard Arabic** ← English

### Few-Shot Learning (k)
- **Definition:** Number of example translations shown to the model in prompt
- **Tested range:** k=0 (zero-shot) to k=10
- **Optimal range:** k=3-7 (degrades beyond k=7)
- **Main research question:** What is the optimal k?

### Semantic Retrieval
- **Definition:** Use vector database to find most similar training examples
- **Process:** Embed source text → search → retrieve top-k similar examples
- **Alternative:** Random sampling (surprisingly works just as well!)

### APE (Automatic Post-Editing)
- **Task:** Given English + Pivot translation, produce Target translation
- **Format:** "Original (English): X, Translation (Pivot): Y, Post-edited (Target): ?"

---

## Major Findings (For Quick Reference)

### Finding 1: Optimal k is 3-7
- **Evidence:** 88 experiments across 2 models, 2 languages, 11 k values
- **Pattern:** Inverted U-curve (performance peaks then degrades)
- **Best Konkani:** k=7 (Hermes) → 8.22 BLEU (+451%)
- **Best Arabic:** k=9 (Hermes) → 6.74 BLEU (+46%)

### Finding 2: Pivot Language is CRITICAL
- **Evidence:** 12 no-pivot experiments
- **Impact:** +3605% to +40100% improvement over direct translation
- **Conclusion:** Pivot language is the PRIMARY driver of performance

### Finding 3: Semantic ≈ Random (Shocking!)
- **Evidence:** 44 semantic vs random experiments
- **Result:** No significant difference (±0.10-1.73 BLEU)
- **Implication:** Can eliminate expensive vector DB infrastructure

### Finding 4: Architecture > Strategy
- **Hermes:** <2% failures across all k values (stable, production-ready)
- **Tower:** 100% failures at k≥6 (catastrophic Unicode corruption)
- **Conclusion:** Model choice matters more than retrieval strategy

### Finding 5: Language-Dependent Effectiveness
- **Konkani** (weak prior): +451% improvement (massive benefit)
- **Arabic** (strong prior): +46% improvement (modest benefit)
- **Conclusion:** Few-shot helps most when model lacks prior knowledge

---

## Common Tasks

### View Paper Update Materials
```bash
cd paper_updates
cat README.md                           # Start here
cat KEY_NUMBERS_CHEATSHEET.md          # Quick stats reference
```

### Understand Ablation Results
```bash
cat ABLATION_STUDY.md                  # Main k ablation (very detailed!)
cat NO_PIVOT_RESULTS.md                # No-pivot findings
cat SAMPLING_ABLATION_CONCLUSIONS.md   # Semantic vs random findings
```

### Generate Paper Figures
```bash
python scripts/generate_paper_figures.py
# Creates: paper_updates/figures/*.pdf and *.png
```

### Run New Ablation (if needed)
```bash
# Main k ablation
python scripts/ablation_k/run_ablation_study.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "NousResearch/Hermes-2-Pro-Llama-3-8B" \
    --pivot "hin" --source "mar" --target "gom" \
    --db "konkani_translations" \
    --output-dir "ablation_results/test" \
    --k-values 0 3 5 \
    --wandb
```

### Check Git Branches
```bash
git branch -a                          # List all branches
git checkout rohin/experiments         # Main branch
git checkout rohin/no-pivot-ablation   # No-pivot study
git checkout harshwardha/ablation_study # Semantic vs random
```

---

## Key Numbers (For Paper/Presentations)

### Best Results
- **Konkani Hermes:** 8.22 BLEU at k=7 (+451% improvement)
- **Arabic Hermes:** 6.74 BLEU at k=9 (+46% improvement)
- **Pivot Impact:** +3605% to +40100% over direct translation
- **Optimal k range:** 3-7 (empirically justified)

### Model Comparison
| Metric | Tower | Hermes | Winner |
|--------|-------|--------|--------|
| Best BLEU | 12.40 (but 20% broken) | 8.39 (1.5% problems) | Hermes |
| Stability (k≤5) | 97% ok | 99% ok | Hermes |
| Stability (k≥6) | 0% ok | 99% ok | Hermes |
| Speed | 19.5 min | 10.7 min | Hermes |

### Jaccard Similarities
- **Marathi ↔ Konkani:** 32.6% (justifies pivot choice)
- **MSA ↔ Tunisian:** 31.9% (justifies pivot choice)
- **English ↔ Konkani:** 1.5% (confirms not good pivot)

---

## Production Recommendations

**Recommended Configuration:**
```
✅ Model: Hermes-2-Pro-Llama-3-8B
✅ Strategy: Random sampling (simpler, no vector DB!)
✅ k value: 5-6
✅ With pivot: ALWAYS (critical!)
✅ Expected: 8.0-8.4 BLEU for Konkani
✅ Stability: >98% success rate
```

**What to Avoid:**
```
❌ Tower at k>5 (catastrophic failures)
❌ No pivot (97-99% performance loss)
❌ k>7 (diminishing returns → degradation)
❌ Semantic retrieval (unnecessary complexity)
```

---

## Paper Status

### Current State
- **Main paper:** `paper_updates/paper.tex`
- **Results section:** Needs integration of `results_section_update.tex` (~5000 words)
- **Figures:** Already generated in `paper_updates/figures/`
- **Tables:** Available in `additional_tables.tex`

### Integration Status
- ✅ Ablation studies complete (100+ experiments)
- ✅ Results analyzed and documented
- ✅ Figures generated (PDF + PNG)
- ✅ LaTeX tables ready
- ⏳ Awaiting integration into paper.tex
- ⏳ Reviewer response to be written

### Next Steps (For User)
1. Review `paper_updates/integration_instructions.md`
2. Replace Results section (lines 187-213) in paper.tex
3. Add figures and tables
4. Update abstract with key numbers
5. Write response to Reviewer 1

---

## Important Notes

### Data Locations
- **Training data:** Loaded from HuggingFace datasets (predictionguard/*)
- **Test sets:** 205 samples (Konkani), 100 samples (Arabic)
- **Vector DBs:** LanceDB format in *_translations/ directories

### Evaluation Metrics
- **BLEU:** Primary metric (sacrebleu)
- **chrF:** Character n-gram F-score
- **chrF++:** Enhanced chrF with word n-grams
- **All scores:** Computed at corpus level

### Hardware
- **GPUs:** 8x NVIDIA H100 80GB HBM3
- **Memory per run:** ~18 GB (FP16, no quantization for inference)
- **Typical runtime:** 8-20 minutes per experiment (205 samples)

---

## Troubleshooting

### If BLEU scores are 0.00
- Check for Unicode corruption in outputs
- Verify prompt format is correct
- May indicate model failure (especially Tower at k≥6)

### If vector DB not found
```bash
python scripts/create_vector_db.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --pivot "hin" --source "mar" --target "gom" \
    --db "konkani_translations"
```

### If branch confusion
- Main experimental branch: `rohin/experiments`
- Each ablation study has its own branch
- Check `git branch -a` to see all branches

### If looking for specific results
- Main k ablation: `ablation_results/`
- No-pivot: `ablation_no_pivot/`
- Semantic vs random: `outputs/sampling_ablation/`

---

## For AI Assistants

When helping with this codebase:

1. **Always check which branch** the user is on (`git branch`)
2. **Different branches = different ablation studies** (see Branch Organization above)
3. **Key findings are documented** in markdown files (see Directory Structure)
4. **Paper materials are in** `paper_updates/` directory
5. **100+ experiments completed** - don't suggest re-running unless necessary
6. **Random sampling is recommended** over semantic (surprising but validated!)
7. **Hermes > Tower** for production (stability matters)
8. **k=3-7 is optimal range** (empirically proven)

---

## Citation

If you use this work:
```bibtex
@article{low_resource_translation_2025,
  title={Adapting LLMs for Low-Resource Machine Translation via
         Semantically Similar Few-Shot Examples and Pivot Languages},
  author={...},
  year={2025},
  note={100+ ablation experiments validating optimal k=3-7,
        pivot language necessity, and semantic vs random retrieval}
}
```

---

**Last Updated:** 2025-11-28
**Branch:** rohin/experiments
**Status:** Ready for paper integration
**Contact:** See git log for contributors
