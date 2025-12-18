#!/usr/bin/env python3
"""
Evaluate few-shot translation on Konkani test set.
Runs translations with num_examples = 0, 3, 6
Stores prompts, responses, examples, and calculates metrics.
"""

import pandas as pd
import json
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import lancedb
import torch
import re
from tqdm import tqdm
import sacrebleu

# Optional COMET import
try:
    from comet import load_from_checkpoint, download_model
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

# Import functions from interactive script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Model and dataset
DEFAULT_MODEL = "Unbabel/TowerInstruct-7B-v0.1"
HERMES_MODEL = "NousResearch/Hermes-2-Pro-Llama-3-8B"
DATASET_NAME = "predictionguard/english-hindi-marathi-konkani-corpus"

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Configuration - using Marathi as pivot
CONFIG = {
    "pivot": "mar",
    "source": "eng",
    "target": "gom",
    "name": "Marathi as pivot"
}


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device if device != "mps" else None,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    if device == "mps":
        model = model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded!")
    return model, tokenizer


def load_vector_db(df, db_name="translations_db", table_name="translations_konkani", force_recreate=False):
    """Load or create vector database."""
    db = lancedb.connect(db_name)
    
    if not force_recreate:
        try:
            tbl = db.open_table(table_name)
            print(f"Vector DB '{table_name}' already exists with {len(tbl)} entries.")
            embed_model = SentenceTransformer("all-MiniLM-L12-v2")
            return db, embed_model, table_name
        except:
            print(f"Vector DB '{table_name}' not found. Creating new database...")
    
    # Create vector DB
    required_cols = ['eng', 'hin', 'mar', 'gom']
    new_df = df[required_cols].copy()
    mask = (new_df.fillna("").astype(str) == "").any(axis=1)
    new_df = new_df[~mask]
    
    print(f"Creating vector DB with {len(new_df)} rows")
    embed_model = SentenceTransformer("all-MiniLM-L12-v2")
    texts = new_df['eng'].tolist()
    embeddings = embed_model.encode(texts)
    
    data = []
    for i, row in new_df.iterrows():
        entry = {
            "text": row['eng'],
            "vector": embeddings[i].tolist(),
            "eng": row['eng'],
            "hin": row['hin'],
            "mar": row['mar'],
            "gom": row['gom']
        }
        data.append(entry)
    
    tbl = db.create_table(table_name, data, mode='overwrite')
    print(f"Created table '{table_name}' with {len(tbl)} entries")
    
    return db, embed_model, table_name


def get_semantic_examples(query_text, db, embed_model, table_name, config, num_examples):
    """Retrieve semantically similar examples."""
    if num_examples == 0:
        return pd.DataFrame()
    
    table = db.open_table(table_name)
    query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
    
    all_results = table.search(embed_model.encode(query_text)).limit(100).to_pandas()
    
    all_results = all_results[all_results['text'] != query_text]
    all_results = all_results[all_results['text'] != ""]
    all_results = all_results[all_results[config['source']] != ""]
    all_results = all_results[all_results[config['target']] != ""]
    if config['pivot']:
        all_results = all_results[all_results[config['pivot']] != ""]
    all_results.dropna(inplace=True)
    
    def word_overlap_score(row):
        source_text = str(row[config['source']]).lower()
        source_words = set(re.findall(r'\b\w+\b', source_text))
        overlap = len(query_words.intersection(source_words))
        return overlap
    
    all_results['word_overlap'] = all_results.apply(word_overlap_score, axis=1)
    
    with_matches = all_results[all_results['word_overlap'] > 0].copy()
    without_matches = all_results[all_results['word_overlap'] == 0].copy()
    
    with_matches.sort_values(by=['word_overlap', '_distance'], ascending=[False, True], inplace=True)
    without_matches.sort_values(by='_distance', ascending=True, inplace=True)
    
    if len(with_matches) >= num_examples:
        results = with_matches.head(num_examples)
    else:
        results = pd.concat([with_matches, without_matches.head(num_examples - len(with_matches))])
    
    return results.head(num_examples)


def build_prompt_messages(source_text, config, examples_df=None):
    """Build prompt messages."""
    messages = []
    
    language_constraint = (
        f"CRITICAL: You are translating to {config['target'].title()} (Konkani). "
        f"Output ONLY Konkani text in Konkani script. Never output Hindi or Marathi.\n\n"
    )
    
    if examples_df is not None and len(examples_df) > 0:
        first_example = True
        for idx, (_, row) in enumerate(examples_df.iterrows(), 1):
            if config['pivot']:
                if first_example:
                    user_content = (
                        language_constraint +
                        f"Study the following examples carefully. Use them as a guide to translate the target sentence in the same style and format.\n\n"
                        f"Example {idx}:\n"
                        f"Original (English): {row[config['source']]}\n"
                        f"Translation ({config['pivot'].title()}): {row[config['pivot']]}\n"
                        f"Post-edited ({config['target'].title()}):"
                    )
                    first_example = False
                else:
                    user_content = (
                        f"Example {idx}:\n"
                        f"Original (English): {row[config['source']]}\n"
                        f"Translation ({config['pivot'].title()}): {row[config['pivot']]}\n"
                        f"Post-edited ({config['target'].title()}):"
                    )
            else:
                if first_example:
                    user_content = (
                        language_constraint +
                        f"Study the following examples carefully. Use them as a guide to translate the target sentence in the same style and format.\n\n"
                        f"Example {idx}:\n"
                        f"Source (English): {row[config['source']]}\n"
                        f"Translation ({config['target'].title()}):"
                    )
                    first_example = False
                else:
                    user_content = (
                        f"Example {idx}:\n"
                        f"Source (English): {row[config['source']]}\n"
                        f"Translation ({config['target'].title()}):"
                    )
            
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": row[config['target']]})
    
    # Current query
    if config['pivot']:
        if examples_df is None or len(examples_df) == 0:
            user_content = (
                language_constraint +
                f"Original (English): {source_text}\n"
                f"Translation ({config['pivot'].title()}):\n"
                f"Post-edited ({config['target'].title()}):"
            )
        else:
            user_content = (
                f"Now translate the following sentence. Use the examples above as a guide to match the translation style and format:\n\n"
                f"Original (English): {source_text}\n"
                f"Translation ({config['pivot'].title()}):\n"
                f"Post-edited ({config['target'].title()}):"
            )
    else:
        if examples_df is None or len(examples_df) == 0:
            user_content = (
                language_constraint +
                f"Source (English): {source_text}\n"
                f"Translation ({config['target'].title()}):"
            )
        else:
            user_content = (
                f"Now translate the following sentence. Use the examples above as a guide to match the translation style and format:\n\n"
                f"Source (English): {source_text}\n"
                f"Translation ({config['target'].title()}):"
            )
    
    messages.append({"role": "user", "content": user_content})
    return messages


def translate_with_fewshot(source_text, model, tokenizer, db, embed_model, table_name, config, num_examples):
    """Translate text using few-shot examples."""
    examples_df = get_semantic_examples(source_text, db, embed_model, table_name, config, num_examples)
    
    messages = build_prompt_messages(source_text, config, examples_df)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up stop phrases
    stop_phrases = [
        "\n\n", "\nSource:", "\nTranslation:", "\nIntermediate:",
        "\nOriginal (English):", "\nTranslation (Hindi):", "\nTranslation (Marathi):",
        "\nPost-edited (Konkani):", "again!", "again"
    ]
    for phrase in stop_phrases:
        if phrase in translation:
            translation = translation.split(phrase)[0]
    
    # Format examples for storage
    examples_list = []
    if len(examples_df) > 0:
        for _, row in examples_df.iterrows():
            example_dict = {
                "source": str(row[config['source']]),
                "target": str(row[config['target']])
            }
            if config['pivot']:
                example_dict["pivot"] = str(row[config['pivot']])
            examples_list.append(example_dict)
    
    return translation.strip(), examples_list, prompt


def calculate_bleu(references, hypotheses):
    """Calculate BLEU score."""
    formatted_references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, formatted_references)
    return bleu.score


def calculate_chrf(references, hypotheses):
    """Calculate chrF score."""
    formatted_references = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, formatted_references)
    return chrf.score


def calculate_ter(references, hypotheses):
    """Calculate TER score."""
    formatted_references = [[ref] for ref in references]
    ter = sacrebleu.corpus_ter(hypotheses, formatted_references)
    return ter.score


def calculate_comet(references, hypotheses, sources, model_path="Unbabel/wmt22-comet-da"):
    """Calculate COMET score."""
    if not COMET_AVAILABLE:
        print("  COMET not installed. Install with: pip install unbabel-comet")
        return None
    try:
        print(f"  Loading COMET model from {model_path}...")
        # Try to load model, download if needed
        try:
            model = load_from_checkpoint(model_path)
        except Exception as e:
            # If loading fails, try downloading first
            if "Invalid checkpoint path" in str(e) or "not found" in str(e).lower():
                print(f"  Model not found locally, attempting to download...")
                try:
                    downloaded_path = download_model(model_path)
                    model = load_from_checkpoint(downloaded_path)
                except Exception as download_error:
                    print(f"  Download failed: {download_error}")
                    raise e
            else:
                raise e
        
        # Prepare data in COMET format
        data = []
        for src, hyp, ref in zip(sources, hypotheses, references):
            data.append({
                "src": str(src) if pd.notna(src) else "",
                "mt": str(hyp) if pd.notna(hyp) else "",
                "ref": str(ref) if pd.notna(ref) else ""
            })
        
        # Filter out empty entries
        data = [d for d in data if d["src"].strip() and d["mt"].strip() and d["ref"].strip()]
        
        if not data:
            print("  Warning: No valid data for COMET calculation")
            return None
        
        print(f"  Calculating COMET for {len(data)} examples...")
        # Calculate scores - use gpus parameter correctly
        try:
            if device == "cuda":
                result = model.predict(data, batch_size=32, gpus=1)
            else:
                # For CPU/MPS, use gpus=0
                result = model.predict(data, batch_size=32, gpus=0)
        except Exception as predict_error:
            # Fallback: try without gpus parameter
            print(f"  Warning: COMET predict with gpus failed, trying without...")
            result = model.predict(data, batch_size=8)  # Smaller batch for CPU
        
        # Extract system score from Prediction object
        if hasattr(result, 'system_score'):
            comet_score = result.system_score
        elif isinstance(result, tuple):
            # Fallback for older API
            scores, comet_score = result
        else:
            # Try to get score from result
            comet_score = result
        
        return float(comet_score) if comet_score is not None else None
    except Exception as e:
        print(f"Error calculating COMET: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate few-shot translation on Konkani test set")
    parser.add_argument("--output-dir", default="fewshot_evaluation_results", help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--num-examples", "-k", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--config", choices=["1", "2", "3"], default="1", help="Configuration: 1=Marathi pivot, 2=Hindi pivot, 3=No pivot")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples (for testing)")
    parser.add_argument("--no-comet", action="store_true", help="Skip COMET evaluation")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    test_df = pd.DataFrame(dataset['test'])
    train_df = pd.DataFrame(dataset['train'])
    
    # Limit test set if specified (for testing)
    if args.limit:
        test_df = test_df.head(args.limit)
        print(f"LIMITED: Using only {args.limit} test examples for testing")
    
    print(f"Test set size: {len(test_df)}")
    print(f"Train set size: {len(train_df)}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Load/create vector DB
    db, embed_model, table_name = load_vector_db(train_df)
    
    # Update config based on choice
    if args.config == "1":
        config = {"pivot": "mar", "source": "eng", "target": "gom", "name": "Marathi as pivot"}
    elif args.config == "2":
        config = {"pivot": "hin", "source": "eng", "target": "gom", "name": "Hindi as pivot"}
    else:
        config = {"pivot": None, "source": "eng", "target": "gom", "name": "No pivot"}
    
    # Process with specified num_examples
    num_examples = args.num_examples
    print(f"\n{'='*80}")
    print(f"Processing with {num_examples} few-shot examples")
    print(f"{'='*80}")
    
    results = []
    
    # Process each test example
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Translating (k={num_examples})"):
        source_text = str(row['eng'])
        reference = str(row['gom'])
        
        try:
            translation, examples_used, prompt = translate_with_fewshot(
                source_text, model, tokenizer, db, embed_model, table_name, config, num_examples
            )
            
            result = {
                "source": source_text,
                "reference": reference,
                "hypothesis": translation,
                "num_examples": num_examples,
                "prompt": prompt,
                "examples_used": json.dumps(examples_used, ensure_ascii=False)  # Store as JSON string for CSV
            }
            
            if config['pivot']:
                result["pivot"] = str(row[config['pivot']])
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    model_short = args.model.split("/")[-1]  # Get short model name
    output_file = os.path.join(args.output_dir, f"konkani_test_{model_short}_k{num_examples}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Calculate metrics
    references = results_df['reference'].tolist()
    hypotheses = results_df['hypothesis'].tolist()
    sources = results_df['source'].tolist()
    
    print("\nCalculating metrics...")
    bleu = calculate_bleu(references, hypotheses)
    chrf = calculate_chrf(references, hypotheses)
    ter = calculate_ter(references, hypotheses)
    
    if args.no_comet:
        comet = None
        print("  (COMET skipped)")
    else:
        comet = calculate_comet(references, hypotheses, sources)
    
    print(f"\nMetrics for k={num_examples}:")
    print(f"  BLEU:  {bleu:.2f}")
    print(f"  chrF:  {chrf:.2f}")
    print(f"  TER:   {ter:.2f}")
    if not args.no_comet:
        print(f"  COMET: {comet:.3f}" if comet else "  COMET: Failed")
    
    # Save metrics
    metrics = {
        "model": args.model,
        "num_examples": num_examples,
        "config": config['name'],
        "num_test_examples": len(results),
        "metrics": {
            "BLEU": float(bleu),
            "chrF": float(chrf),
            "TER": float(ter),
            "COMET": float(comet) if comet else None
        }
    }
    
    metrics_file = os.path.join(args.output_dir, f"konkani_test_{model_short}_k{num_examples}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

