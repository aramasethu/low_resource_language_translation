#!/usr/bin/env python3
"""
Token Analysis Script for Low-Resource Language Translation

Analyzes tokenization behavior of Tower (Mistral-based) and Hermes (Llama-based) 
models across all languages in the parallel corpora.

Metrics:
- Token splitting examples
- Token fertility (tokens per word)
- Character efficiency (tokens per character)
- Vocabulary coverage
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from collections import defaultdict
import json

# Model configurations
MODELS = {
    "Tower": "Unbabel/TowerInstruct-7B-v0.1",
    "Hermes": "NousResearch/Hermes-2-Pro-Llama-3-8B"
}

# Dataset configurations
DATASETS = {
    "konkani": {
        "name": "predictionguard/english-hindi-marathi-konkani-corpus",
        "languages": {
            "eng": "English",
            "hin": "Hindi", 
            "mar": "Marathi",
            "gom": "Konkani"
        }
    },
    "arabic": {
        "name": "predictionguard/arabic_acl_corpus",
        "languages": {
            "en": "English",
            "msa": "Modern Standard Arabic",
            "tn": "Tunisian Arabic",
            "eg": "Egyptian Arabic"
        }
    }
}


def load_tokenizers():
    """Load tokenizers for both models."""
    tokenizers = {}
    for model_name, model_path in MODELS.items():
        print(f"Loading tokenizer for {model_name} ({model_path})...")
        try:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            print(f"  Vocab size: {tokenizers[model_name].vocab_size}")
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
    return tokenizers


def get_sample_texts(dataset_config, num_samples=100):
    """Load sample texts from dataset for each language."""
    print(f"\nLoading dataset: {dataset_config['name']}")
    dataset = load_dataset(dataset_config['name'])
    
    samples = {}
    train_data = dataset['train']
    
    # Get column names
    if 'translation' in train_data.column_names:
        # Nested format
        for i, row in enumerate(train_data):
            if i >= num_samples:
                break
            for lang_code in dataset_config['languages'].keys():
                if lang_code not in samples:
                    samples[lang_code] = []
                if lang_code in row.get('translation', {}):
                    text = row['translation'][lang_code]
                    if text and text.strip():
                        samples[lang_code].append(text)
    else:
        # Flat format
        for lang_code in dataset_config['languages'].keys():
            if lang_code in train_data.column_names:
                samples[lang_code] = [
                    str(text) for text in train_data[lang_code][:num_samples] 
                    if text and str(text).strip()
                ]
    
    for lang, texts in samples.items():
        print(f"  {lang}: {len(texts)} samples")
    
    return samples


def analyze_tokenization(text, tokenizer):
    """Analyze tokenization of a single text."""
    # Tokenize
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Word count (simple whitespace split)
    words = text.split()
    word_count = len(words) if words else 1
    
    # Character count (excluding spaces)
    char_count = len(text.replace(" ", ""))
    
    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "num_tokens": len(tokens),
        "num_words": word_count,
        "num_chars": char_count,
        "tokens_per_word": len(tokens) / word_count if word_count > 0 else 0,
        "tokens_per_char": len(tokens) / char_count if char_count > 0 else 0
    }


def analyze_language(texts, tokenizer, lang_name):
    """Analyze tokenization for a language."""
    results = []
    
    for text in texts:
        if not text or not text.strip():
            continue
        analysis = analyze_tokenization(text, tokenizer)
        results.append(analysis)
    
    if not results:
        return None
    
    # Aggregate statistics
    avg_tokens_per_word = sum(r["tokens_per_word"] for r in results) / len(results)
    avg_tokens_per_char = sum(r["tokens_per_char"] for r in results) / len(results)
    avg_tokens = sum(r["num_tokens"] for r in results) / len(results)
    avg_words = sum(r["num_words"] for r in results) / len(results)
    
    return {
        "language": lang_name,
        "num_samples": len(results),
        "avg_tokens_per_word": avg_tokens_per_word,
        "avg_tokens_per_char": avg_tokens_per_char,
        "avg_tokens": avg_tokens,
        "avg_words": avg_words,
        "sample_results": results[:5]  # Keep first 5 for examples
    }


def show_tokenization_examples(samples, tokenizers, dataset_config, num_examples=3):
    """Show how each tokenizer splits example sentences."""
    print("\n" + "="*80)
    print("TOKENIZATION EXAMPLES")
    print("="*80)
    
    for lang_code, lang_name in dataset_config['languages'].items():
        if lang_code not in samples or not samples[lang_code]:
            continue
            
        print(f"\n### {lang_name} ({lang_code})")
        print("-" * 40)
        
        for i, text in enumerate(samples[lang_code][:num_examples]):
            print(f"\nExample {i+1}: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
            
            for model_name, tokenizer in tokenizers.items():
                tokens = tokenizer.tokenize(text)
                # Show first 20 tokens
                display_tokens = tokens[:20]
                if len(tokens) > 20:
                    display_tokens.append(f"... (+{len(tokens)-20} more)")
                print(f"  {model_name}: {display_tokens}")


def generate_report(all_results, output_file="token_analysis_results.md"):
    """Generate markdown report with analysis results."""
    
    lines = [
        "# Token Analysis Report",
        "",
        "## Overview",
        "Analysis of tokenization behavior for Tower (Mistral-based) and Hermes (Llama-based) models",
        "across languages in the parallel corpora.",
        "",
        "## Token Fertility (Tokens per Word)",
        "",
        "Lower values indicate more efficient tokenization for that language.",
        "",
        "| Dataset | Language | Tower | Hermes |",
        "|---------|----------|-------|--------|"
    ]
    
    # Organize by dataset
    by_dataset = defaultdict(lambda: defaultdict(dict))
    for result in all_results:
        by_dataset[result['dataset']][result['language']][result['model']] = result
    
    for dataset, languages in by_dataset.items():
        for lang, models in languages.items():
            tower = models.get('Tower', {}).get('avg_tokens_per_word', '-')
            hermes = models.get('Hermes', {}).get('avg_tokens_per_word', '-')
            tower_str = f"{tower:.2f}" if isinstance(tower, float) else tower
            hermes_str = f"{hermes:.2f}" if isinstance(hermes, float) else hermes
            lines.append(f"| {dataset.title()} | {lang} | {tower_str} | {hermes_str} |")
    
    lines.extend([
        "",
        "## Token Efficiency (Tokens per Character)",
        "",
        "Lower values indicate better character-level efficiency.",
        "",
        "| Dataset | Language | Tower | Hermes |",
        "|---------|----------|-------|--------|"
    ])
    
    for dataset, languages in by_dataset.items():
        for lang, models in languages.items():
            tower = models.get('Tower', {}).get('avg_tokens_per_char', '-')
            hermes = models.get('Hermes', {}).get('avg_tokens_per_char', '-')
            tower_str = f"{tower:.3f}" if isinstance(tower, float) else tower
            hermes_str = f"{hermes:.3f}" if isinstance(hermes, float) else hermes
            lines.append(f"| {dataset.title()} | {lang} | {tower_str} | {hermes_str} |")
    
    lines.extend([
        "",
        "## Interpretation",
        "",
        "### Token Fertility (Tokens per Word)",
        "- **English**: Typically ~1.3-1.5 tokens/word (well-represented in training)",
        "- **Low-resource languages**: Higher values (2-4+) indicate the tokenizer breaks words into more subword units",
        "- **High fertility = potential translation challenges**: More tokens needed to represent the same content",
        "",
        "### Token Premium",
        "The 'token premium' for a language is the ratio of its fertility compared to English:",
        "- Premium = (tokens/word for language) / (tokens/word for English)",
        "- Higher premium means the model 'pays more' to represent that language",
        ""
    ])
    
    # Calculate token premium
    lines.extend([
        "## Token Premium (vs English)",
        "",
        "| Dataset | Language | Tower Premium | Hermes Premium |",
        "|---------|----------|---------------|----------------|"
    ])
    
    for dataset, languages in by_dataset.items():
        # Find English baseline
        eng_key = 'English'
        eng_tower = None
        eng_hermes = None
        for lang, models in languages.items():
            if 'English' in lang:
                eng_tower = models.get('Tower', {}).get('avg_tokens_per_word')
                eng_hermes = models.get('Hermes', {}).get('avg_tokens_per_word')
                break
        
        for lang, models in languages.items():
            if 'English' in lang:
                lines.append(f"| {dataset.title()} | {lang} | 1.00x (baseline) | 1.00x (baseline) |")
            else:
                tower = models.get('Tower', {}).get('avg_tokens_per_word')
                hermes = models.get('Hermes', {}).get('avg_tokens_per_word')
                
                tower_premium = f"{tower/eng_tower:.2f}x" if tower and eng_tower else "-"
                hermes_premium = f"{hermes/eng_hermes:.2f}x" if hermes and eng_hermes else "-"
                lines.append(f"| {dataset.title()} | {lang} | {tower_premium} | {hermes_premium} |")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved to: {output_file}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Token analysis for translation models")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples per language")
    parser.add_argument("--output", default="token_analysis_results.md", help="Output file")
    args = parser.parse_args()
    
    # Load tokenizers
    print("Loading tokenizers...")
    tokenizers = load_tokenizers()
    
    if not tokenizers:
        print("Error: No tokenizers loaded")
        return
    
    all_results = []
    
    # Process each dataset
    for dataset_key, dataset_config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_key.upper()} dataset")
        print(f"{'='*60}")
        
        # Load samples
        samples = get_sample_texts(dataset_config, args.num_samples)
        
        # Show tokenization examples
        show_tokenization_examples(samples, tokenizers, dataset_config)
        
        # Analyze each language with each tokenizer
        print("\n" + "="*80)
        print("TOKEN FERTILITY ANALYSIS")
        print("="*80)
        
        for lang_code, lang_name in dataset_config['languages'].items():
            if lang_code not in samples or not samples[lang_code]:
                print(f"\n{lang_name}: No samples available")
                continue
                
            print(f"\n### {lang_name}")
            
            for model_name, tokenizer in tokenizers.items():
                result = analyze_language(samples[lang_code], tokenizer, lang_name)
                if result:
                    result['model'] = model_name
                    result['dataset'] = dataset_key
                    all_results.append(result)
                    
                    print(f"  {model_name}:")
                    print(f"    Avg tokens/word: {result['avg_tokens_per_word']:.2f}")
                    print(f"    Avg tokens/char: {result['avg_tokens_per_char']:.3f}")
                    print(f"    Avg tokens/sentence: {result['avg_tokens']:.1f}")
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    report = generate_report(all_results, args.output)
    
    # Print summary
    print("\n" + report)


if __name__ == "__main__":
    main()

