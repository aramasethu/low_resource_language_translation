# Low-Resource Language Translation Documentation

Complete guide for understanding and running experiments in this repository.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Understanding the Approach](#understanding-the-approach)
4. [Running Experiments](#running-experiments)
5. [Sampling Ablation Study](#sampling-ablation-study)
6. [Analysis Results](#analysis-results)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Basic Inference
```bash
# Create vector database for semantic retrieval
python scripts/create_vector_db_faiss.py \
  --dataset_name "predictionguard/english-hindi-marathi-konkani-corpus" \
  --source_lang "eng" \
  --pivot_lang "mar" \
  --target_lang "gom" \
  --vector_db_path "data/translations_db"

# Run inference with few-shot examples
python scripts/run_inference_with_random_sampling.py \
  --model_name "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --dataset_name "predictionguard/english-hindi-marathi-konkani-corpus" \
  --output_path "outputs/results.json" \
  --prompt_type "few_shot" \
  --num_fs_examples 5 \
  --retrieval_strategy "semantic" \
  --vector_db_path "data/translations_db" \
  --source_lang "eng" \
  --pivot_lang "mar" \
  --target_lang "gom"
```

---

## Project Overview

### Goal
Translate from English to **Konkani** (a low-resource language) using **Marathi** as a pivot/helper language.

### Why Pivot Language?
- Models have limited exposure to Konkani
- Marathi is linguistically similar to Konkani
- Models are familiar with Marathi (more training data)
- Use Marathi as a bridge to guide translation

### Models
- **Unbabel/TowerInstruct-7B-v0.2** (Tower)
- **NousResearch/Hermes-2-Pro-Llama-3-8B** (Hermes)

### Dataset
- **predictionguard/english-hindi-marathi-konkani-corpus**
- Contains parallel translations: English ↔ Marathi ↔ Konkani
- Train set: Used for few-shot examples
- Test set: Used for evaluation

---

## Understanding the Approach

### 1. APE (Automatic Post-Editing)

Instead of translating from scratch, we use **Automatic Post-Editing**:

```
Input:
  Original (English): "The cat sat on the mat"
  Translation (Marathi): "मांजर चटईवर बसली"
  
Output:
  Post-edited (Konkani): "मांजर चटयेर बसलो"
```

**Why APE?** It's theoretically easier to improve/fix an existing translation than create from nothing.

### 2. Few-Shot Learning

Show the model k examples before asking it to translate:

```
Example 1: English → Marathi → Konkani
Example 2: English → Marathi → Konkani
...
Example k: English → Marathi → Konkani

Now translate this: [Your input]
```

### 3. Semantic Retrieval

**Question:** Which examples should we show?

**Option 1: Semantic Retrieval**
- Find training examples similar to current input
- Use vector embeddings to measure similarity
- Hypothesis: Similar examples should help more

**Option 2: Random Sampling**
- Randomly select examples from training set
- No similarity matching
- Simpler, no vector database needed

**What We Retrieve:**
- Complete triplets (English + Marathi + Konkani)
- Retrieval based on English text similarity
- Stored in FAISS vector database

---

## Running Experiments

### Step 1: Create Vector Database (For Semantic Retrieval)

```bash
python scripts/create_vector_db_faiss.py \
  --dataset_name "predictionguard/english-hindi-marathi-konkani-corpus" \
  --source_lang "eng" \
  --pivot_lang "mar" \
  --target_lang "gom" \
  --vector_db_path "data/translations_db"
```

This creates a FAISS vector database with:
- Embedded English text (query field)
- Metadata: Marathi and Konkani translations

### Step 2: Run Inference

#### Zero-Shot (No Examples)
```bash
python scripts/run_inference_with_random_sampling.py \
  --model_name "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --dataset_name "predictionguard/english-hindi-marathi-konkani-corpus" \
  --output_path "outputs/zero_shot.json" \
  --prompt_type "zero_shot" \
  --num_fs_examples 0 \
  --retrieval_strategy "random" \
  --source_lang "eng" \
  --pivot_lang "mar" \
  --target_lang "gom"
```

#### Few-Shot with Semantic Retrieval
```bash
python scripts/run_inference_with_random_sampling.py \
  --model_name "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --dataset_name "predictionguard/english-hindi-marathi-konkani-corpus" \
  --output_path "outputs/semantic_k5.json" \
  --prompt_type "few_shot" \
  --num_fs_examples 5 \
  --retrieval_strategy "semantic" \
  --vector_db_path "data/translations_db" \
  --source_lang "eng" \
  --pivot_lang "mar" \
  --target_lang "gom"
```

#### Few-Shot with Random Sampling
```bash
python scripts/run_inference_with_random_sampling.py \
  --model_name "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --dataset_name "predictionguard/english-hindi-marathi-konkani-corpus" \
  --output_path "outputs/random_k5.json" \
  --prompt_type "few_shot" \
  --num_fs_examples 5 \
  --retrieval_strategy "random" \
  --source_lang "eng" \
  --pivot_lang "mar" \
  --target_lang "gom"
```

### Step 3: View Results

Results are saved as JSON with:
```json
{
  "config": { "model": "...", "k": 5, ... },
  "metrics": {
    "bleu": 8.39,
    "chrf": 38.59,
    "chrf_pp": 31.95
  },
  "outputs": [
    {
      "source": "English text",
      "pivot": "Marathi text",
      "target": "Konkani reference",
      "prediction": "Konkani prediction"
    }
  ]
}
```

---

## Sampling Ablation Study

### Purpose
Compare semantic vs random retrieval across different k values (0-10) to determine:
1. Does semantic retrieval help?
2. What's the optimal number of examples (k)?
3. Which model performs better?

### Running the Full Ablation

```bash
python scripts/run_sampling_ablation_all.py
```

This runs all 44 experiments:
- 2 models (Tower, Hermes)
- 11 k values (0-10)
- 2 strategies (semantic, random)

**Time:** 10-20 hours depending on GPU

### Output Structure

```
outputs/sampling_ablation/
├── experiment_summary.json          # Overall summary
├── tower_semantic_k0.json           # Individual results
├── tower_semantic_k1.json
├── ...
├── tower_random_k0.json
├── ...
├── hermes_semantic_k0.json
└── ...
```

### Analyzing Results

```bash
# Generate analysis report and visualizations
python scripts/analyze_sampling_ablation.py
```

Creates:
```
outputs/sampling_ablation_analysis/
├── analysis_report.md              # Detailed analysis
├── tower_comparison.png            # BLEU/chrF comparison plots
├── tower_delta.png                 # Performance delta plots
├── hermes_comparison.png
└── hermes_delta.png
```

---

## Analysis Results

### Key Findings

**See detailed analysis:** `outputs/sampling_ablation_analysis/analysis_report.md`

#### Summary:

1. **Hermes significantly outperforms Tower**
   - Hermes: Stable across all k values (0.5-2.0% problematic predictions)
   - Tower: Degenerates at k≥6, producing garbled output (98-100% failure)

2. **Semantic retrieval doesn't provide clear advantages**
   - Random sampling often performs as well or better
   - Saves overhead of maintaining vector databases

3. **Optimal configurations:**
   - **Hermes:** k=5 (semantic) or k=6 (random)
     - BLEU: ~8.25-8.39
     - Highly stable
   - **Tower:** k=6 (random) or k=0 (zero-shot)
     - BLEU: 12.40 (k=6) but 20% problematic
     - BLEU: 7.41 (k=0) with only 3% problematic

4. **Production recommendation:**
   - Use **Hermes with k=5-6** for stable, reliable output
   - Avoid Tower with k≥6 (catastrophic degeneration)

---

## Troubleshooting

### FAISS Loading Error
```
ValueError: The de-serialization relies loading a pickle file...
```

**Fix:** Already handled in code with `allow_dangerous_deserialization=True`

### Out of Memory
```bash
# Reduce k values or run fewer experiments in parallel
# Memory cleanup is automatic after each experiment
```

### W&B Not Logging
```bash
pip install wandb
wandb login
```

### Empty Predictions
**Fix:** Already handled in code - predictions are extracted by removing the prompt from model output

### Module Not Found
```bash
pip install -r requirements.txt
```

Or install specific packages:
```bash
pip install faiss-cpu InstructorEmbedding wandb sacrebleu
```

### Language Column Names
Remember to use ISO codes:
- English: `eng`
- Marathi: `mar`
- Konkani: `gom`

---

## Advanced: Understanding the Code

### Key Scripts

1. **`create_vector_db_faiss.py`**
   - Creates FAISS vector database
   - Embeds English text using Instructor model
   - Stores Marathi/Konkani as metadata

2. **`run_inference_with_random_sampling.py`**
   - Main inference script
   - Supports both semantic and random retrieval
   - Calculates BLEU/chrF/chrF++ metrics
   - W&B logging integration

3. **`run_sampling_ablation_all.py`**
   - Orchestrates all 44 experiments
   - Handles timeouts and errors gracefully
   - Saves incremental results

4. **`analyze_sampling_ablation.py`**
   - Generates comparison plots
   - Identifies problematic predictions
   - Creates detailed analysis report

### Problematic Prediction Detection

The analysis identifies genuinely problematic outputs:
- ✅ Valid: Devanagari script (Marathi/Konkani)
- ❌ Problematic:
  - Unicode replacement characters (�)
  - Excessive whitespace/punctuation repetition
  - Garbled ASCII patterns
  - High word repetition rates
  - Empty outputs

---

## Citation

If you use this work, please cite:

```bibtex
@misc{low_resource_translation_2025,
  title={Few-Shot Translation for Low-Resource Languages using Pivot Languages},
  author={Your Name},
  year={2025},
  note={Comparing semantic vs random retrieval for English-Marathi-Konkani translation}
}
```

---

## FAQ

**Q: Do we use few-shot examples in the original experiment?**  
A: Yes, default k=5. Both pivot-based translation and few-shot learning are used together.

**Q: What exactly gets retrieved in semantic search?**  
A: Complete training triplets (English + Marathi + Konkani) where the English text is similar to your query.

**Q: Why is performance poor?**  
A: Multiple factors: APE task complexity, low-resource language, context overflow at higher k, model-specific limitations.

**Q: Should I use semantic or random retrieval?**  
A: Based on results, random is simpler and often just as good. Semantic doesn't provide clear benefits.

**Q: What's the optimal k value?**  
A: Model-dependent:
- Hermes: k=5-6 (stable across all k)
- Tower: k=0 (safest) or k=6 (best BLEU but 20% problematic)

**Q: Why does Tower degenerate?**  
A: Likely context window overflow, prompt formatting issues, or model architecture limitations at higher k values.

---

**For more details, see:**
- Main paper: `low_res_translation_oct25.pdf`
- Analysis results: `outputs/sampling_ablation_analysis/analysis_report.md`
- Original README: `README.md`

