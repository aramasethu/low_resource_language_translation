#!/usr/bin/env python3
"""
Interactive few-shot translation script for Konkani and Marathi.
Supports three configurations:
1. Few-shot with Marathi as pivot (eng -> mar -> gom)
2. Few-shot with Hindi as pivot (eng -> hin -> gom)
3. Few-shot with no pivot (eng -> gom)

Uses semantic similarity to find examples and Tower's chat template.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import lancedb
import pandas as pd
import torch
import sys
import os
import re

# Model
TOWER_MODEL = "Unbabel/TowerInstruct-7B-v0.1"

# Dataset
DATASET_NAME = "predictionguard/english-hindi-marathi-konkani-corpus"

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Metal Performance Shaders) for Apple Silicon acceleration")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
else:
    device = "cpu"
    print("Using CPU")

# Configuration mappings
CONFIGURATIONS = {
    "1": {
        "name": "Marathi as pivot",
        "pivot": "mar",
        "source": "eng",
        "target": "gom",
        "description": "English -> Marathi (pivot) -> Konkani"
    },
    "2": {
        "name": "Hindi as pivot",
        "pivot": "hin",
        "source": "eng",
        "target": "gom",
        "description": "English -> Hindi (pivot) -> Konkani"
    },
    "3": {
        "name": "No pivot",
        "pivot": None,
        "source": "eng",
        "target": "gom",
        "description": "English -> Konkani (direct)"
    }
}

def load_model_and_tokenizer():
    """Load Tower model and tokenizer."""
    print(f"\nLoading Tower model ({TOWER_MODEL})...")
    print("This may take a minute - model will stay loaded for fast queries...")
    
    tokenizer = AutoTokenizer.from_pretrained(TOWER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TOWER_MODEL,
        device_map=device if device != "mps" else None,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    if device == "mps":
        model = model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded!\n")
    return model, tokenizer

def load_dataset_data():
    """Load the dataset."""
    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)
    df = pd.DataFrame(dataset['train'])
    print(f"Dataset loaded with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df

def create_vector_db(df, db_name="translations_db", table_name="translations_konkani", force_recreate=False):
    """Create a single comprehensive vector database with all language pairs."""
    # Check if DB and table already exist
    db = lancedb.connect(db_name)
    
    if not force_recreate:
        try:
            tbl = db.open_table(table_name)
            print(f"\nVector DB '{table_name}' already exists with {len(tbl)} entries.")
            print("Using existing database. Set force_recreate=True to rebuild.")
            embed_model = SentenceTransformer("all-MiniLM-L12-v2")
            return db, embed_model, table_name
        except:
            print(f"\nVector DB '{table_name}' not found. Creating new database...")
    
    # Filter out rows with missing required columns
    required_cols = ['eng', 'hin', 'mar', 'gom']
    new_df = df[required_cols].copy()
    
    # Remove rows where any required column is empty
    mask = (new_df.fillna("").astype(str) == "").any(axis=1)
    new_df = new_df[~mask]
    
    print(f"\nCreating comprehensive vector DB with all language pairs")
    print(f"Columns: {required_cols}")
    print(f"Rows: {len(new_df)}")
    
    # Create embeddings based on English text (source language)
    embed_model = SentenceTransformer("all-MiniLM-L12-v2")
    texts = new_df['eng'].tolist()
    embeddings = embed_model.encode(texts)
    
    # Prepare data for LanceDB with all language columns
    data = []
    for i, row in new_df.iterrows():
        entry = {
            "text": row['eng'],  # English text for semantic search
            "vector": embeddings[i].tolist(),
            "eng": row['eng'],
            "hin": row['hin'],
            "mar": row['mar'],
            "gom": row['gom']
        }
        data.append(entry)
    
    # Create or overwrite table
    tbl = db.create_table(table_name, data, mode='overwrite')
    print(f"Created table '{table_name}' with {len(tbl)} entries")
    
    return db, embed_model, table_name

def get_semantic_examples(query_text, db, embed_model, table_name, config, num_examples):
    """Retrieve examples that contain query words, then fall back to semantic similarity."""
    table = db.open_table(table_name)
    
    # Normalize query words (lowercase, remove punctuation)
    query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
    
    # Get more results for filtering
    all_results = table.search(embed_model.encode(query_text)).limit(100).to_pandas()
    
    # Filter out empty and duplicate results
    all_results = all_results[all_results['text'] != query_text]
    all_results = all_results[all_results['text'] != ""]
    all_results = all_results[all_results[config['source']] != ""]
    all_results = all_results[all_results[config['target']] != ""]
    if config['pivot']:
        all_results = all_results[all_results[config['pivot']] != ""]
    all_results.dropna(inplace=True)
    
    # Score examples by word overlap
    def word_overlap_score(row):
        source_text = str(row[config['source']]).lower()
        source_words = set(re.findall(r'\b\w+\b', source_text))
        # Count how many query words appear in the source text
        overlap = len(query_words.intersection(source_words))
        return overlap
    
    # Calculate word overlap scores
    all_results['word_overlap'] = all_results.apply(word_overlap_score, axis=1)
    
    # Separate examples with word matches from those without
    with_matches = all_results[all_results['word_overlap'] > 0].copy()
    without_matches = all_results[all_results['word_overlap'] == 0].copy()
    
    # Sort each group
    with_matches.sort_values(by=['word_overlap', '_distance'], ascending=[False, True], inplace=True)
    without_matches.sort_values(by='_distance', ascending=True, inplace=True)
    
    # Prioritize word matches, then fill with semantic similarity if needed
    if len(with_matches) >= num_examples:
        results = with_matches.head(num_examples)
        print(f"Found {len(results)} examples with word matches (overlap scores: {results['word_overlap'].tolist()})")
    else:
        # Take all word matches, then fill with semantic similarity
        results = pd.concat([with_matches, without_matches.head(num_examples - len(with_matches))])
        if len(with_matches) > 0:
            print(f"Found {len(with_matches)} examples with word matches, {num_examples - len(with_matches)} from semantic similarity")
        else:
            print("No word matches found, using semantic similarity only")
    
    return results.head(num_examples)

def build_prompt_messages(source_text, config, examples_df=None):
    """Build prompt messages using Tower's chat template format with improved structure."""
    messages = []
    
    # Language constraint instruction (included in first message since some templates ignore system messages)
    language_constraint = (
        f"CRITICAL: You are translating to {config['target'].title()} (Konkani). "
        f"Output ONLY Konkani text in Konkani script. Never output Hindi or Marathi.\n\n"
    )
    
    # Add few-shot examples if provided
    if examples_df is not None and len(examples_df) > 0:
        # Add instruction to first example with explicit guidance to use examples
        first_example = True
        for idx, (_, row) in enumerate(examples_df.iterrows(), 1):
            if config['pivot']:
                # With pivot: show Original (eng) -> Translation (pivot) -> Post-edited (Konkani)
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
                # No pivot: direct translation
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
    else:
        # No examples, add instruction to the query itself
        language_constraint = (
            f"You are a professional translator. {language_constraint}"
        )
    
    # Add current query with explicit instruction to use examples
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
    # Get semantic examples
    examples_df = get_semantic_examples(source_text, db, embed_model, table_name, config, num_examples)
    
    # Build prompt
    messages = build_prompt_messages(source_text, config, examples_df)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode only new tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up - stop at common stop phrases and prevent language mixing
    stop_phrases = [
        "\n\n", 
        "\nSource:", 
        "\nTranslation:", 
        "\nIntermediate:",
        "\nOriginal (English):",
        "\nTranslation (Hindi):",
        "\nTranslation (Marathi):",
        "\nPost-edited (Konkani):",
        "again!", 
        "again",
        "Translation (Hindi)",
        "Translation (Marathi)",
        "Translation (English)"
    ]
    for phrase in stop_phrases:
        if phrase in translation:
            translation = translation.split(phrase)[0]
    
    return translation.strip(), examples_df

def main():
    """Main interactive loop."""
    print("="*70)
    print("Interactive Few-Shot Translation for Konkani")
    print("="*70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    df = load_dataset_data()
    
    # Select configuration
    print("\n" + "="*70)
    print("Select Configuration:")
    print("="*70)
    for key, config in CONFIGURATIONS.items():
        print(f"{key}. {config['name']}: {config['description']}")
    
    while True:
        choice = input("\nEnter configuration (1-3): ").strip()
        if choice in CONFIGURATIONS:
            config = CONFIGURATIONS[choice]
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\nSelected: {config['name']}")
    print(f"Description: {config['description']}")
    
    # Create or load vector DB (single DB for all configurations)
    db, embed_model, table_name = create_vector_db(df, force_recreate=False)
    
    # Interactive translation loop
    print("\n" + "="*70)
    print("Interactive Translation")
    print("="*70)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'config' to change configuration")
    print("Type 'examples' to see current number of examples")
    print("="*70)
    
    num_examples = 3  # Default
    
    while True:
        try:
            # Get number of examples
            num_input = input(f"\nNumber of few-shot examples (current: {num_examples}, press Enter to keep): ").strip()
            if num_input:
                try:
                    num_examples = int(num_input)
                    if num_examples < 0:
                        print("Number of examples must be >= 0")
                        num_examples = 0
                    elif num_examples > 10:
                        print("Limiting to 10 examples for performance")
                        num_examples = 10
                except ValueError:
                    print("Invalid number, using previous value")
            
            # Get source text
            source_text = input("\nEnter English text to translate: ").strip()
            
            if source_text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if source_text.lower() == 'config':
                print("\nAvailable configurations:")
                for key, cfg in CONFIGURATIONS.items():
                    print(f"{key}. {cfg['name']}: {cfg['description']}")
                choice = input("Enter new configuration (1-3): ").strip()
                if choice in CONFIGURATIONS:
                    config = CONFIGURATIONS[choice]
                    print(f"Changed to: {config['name']}")
                    # Vector DB is already created and shared across all configurations
                    print("Using existing vector DB (shared across all configurations)")
                continue
            
            if source_text.lower() == 'examples':
                print(f"Current number of examples: {num_examples}")
                continue
            
            if not source_text:
                continue
            
            # Translate
            print(f"\nTranslating with {num_examples} few-shot examples...")
            translation, examples_df = translate_with_fewshot(
                source_text, model, tokenizer, db, embed_model, table_name, config, num_examples
            )
            
            # Display results
            print("\n" + "-"*70)
            print("RESULT")
            print("-"*70)
            print(f"Source (English): {source_text}")
            if config['pivot']:
                print(f"Configuration: {config['description']}")
            print(f"Few-shot examples: {num_examples}")
            print(f"\nTranslation ({config['target'].title()}): {translation}")
            
            # Show examples used (optional)
            if len(examples_df) > 0:
                show_examples = input("\nShow few-shot examples used? (y/n): ").strip().lower()
                if show_examples == 'y':
                    print("\nFew-shot examples:")
                    for idx, (_, row) in enumerate(examples_df.iterrows(), 1):
                        print(f"\nExample {idx}:")
                        print(f"  Source: {row[config['source']]}")
                        if config['pivot']:
                            print(f"  {config['pivot'].title()}: {row[config['pivot']]}")
                        print(f"  Target: {row[config['target']]}")
            
            print("-"*70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

