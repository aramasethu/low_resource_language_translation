# Repository Structure

## 📚 Documentation

### Main Files
- **`README.md`** - Project overview and quick start
- **`DOCUMENTATION.md`** - Comprehensive guide covering:
  - Quick start instructions
  - Understanding the APE approach
  - Semantic vs random retrieval
  - Running experiments
  - Sampling ablation study
  - Troubleshooting

### Analysis Results
- **`outputs/sampling_ablation_analysis/analysis_report.md`** - Detailed analysis of all 44 experiments
  - Executive summary
  - Performance comparisons
  - Problematic predictions analysis
  - Recommendations

## 🔬 Scripts

### Core Scripts
- `scripts/create_vector_db_faiss.py` - Create FAISS vector database for semantic retrieval
- `scripts/run_inference_with_random_sampling.py` - Main inference script (single experiment)
- `scripts/run_sampling_ablation_all.py` - Orchestrator to run all 44 experiments
- `scripts/analyze_sampling_ablation.py` - Generate analysis reports and plots

### Helper Scripts
- `scripts/create_vector_db.py` - LanceDB version (legacy)
- `scripts/run_inference.py` - Original inference script
- `scripts/describe_datasets.py` - Dataset exploration
- `scripts/generate_ablation_table.py` - Generate tables from results
- `scripts/translation_finetuning.py` - Finetuning script

## 📊 Data

```
data/
└── translations_db/          # FAISS vector database
    ├── index.faiss
    └── index.pkl

outputs/
├── sampling_ablation/        # 44 experiment results (JSON)
│   ├── experiment_summary.json
│   ├── tower_semantic_k*.json
│   ├── tower_random_k*.json
│   ├── hermes_semantic_k*.json
│   └── hermes_random_k*.json
└── sampling_ablation_analysis/  # Analysis outputs
    ├── analysis_report.md
    ├── tower_comparison.png
    ├── tower_delta.png
    ├── hermes_comparison.png
    └── hermes_delta.png
```

## 🚀 Quick Start

1. **Read the overview:** `README.md`
2. **Follow comprehensive guide:** `DOCUMENTATION.md`
3. **View analysis results:** `outputs/sampling_ablation_analysis/analysis_report.md`

## 📝 Key Findings (Summary)

- ✅ **Hermes outperforms Tower** (stable across all k values)
- ❌ **Semantic retrieval doesn't help** (random is just as good)
- ⚠️ **Tower degenerates at k≥6** (98-100% garbage output)
- 🎯 **Best config:** Hermes with k=5-6 (either strategy)

---

**For detailed information, see `DOCUMENTATION.md`**
