#!/usr/bin/env python3
"""
Calculate Jaccard similarity between each pair of languages in the corpora.
"""

import pandas as pd
from datasets import load_dataset
import numpy as np
from collections import Counter
import argparse
import json

def jaccard_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two texts based on character n-grams.
    
    Args:
        text1, text2: Input texts
    
    Returns:
        float: Jaccard similarity score
    """
    def get_ngrams(text, n=2):
        """Extract character n-grams from text."""
        text = str(text).lower().strip()
        if len(text) < n:
            return set()
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    # Get character bigrams
    ngrams1 = get_ngrams(text1, n=2)
    ngrams2 = get_ngrams(text2, n=2)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union if union > 0 else 0.0

def calculate_language_similarity(df, language_columns, dataset_name):
    """
    Calculate Jaccard similarity between all pairs of languages.
    
    Args:
        df: DataFrame with language columns
        language_columns: List of language column names
        dataset_name: Name of the dataset
    
    Returns:
        dict: Similarity matrix
    """
    print(f"\nCalculating similarities for {dataset_name}")
    print(f"Language columns: {language_columns}")
    print(f"Dataset shape: {df.shape}")
    
    similarities = {}
    
    # Calculate pairwise similarities
    for i, lang1 in enumerate(language_columns):
        for j, lang2 in enumerate(language_columns):
            if i < j:  # Only calculate upper triangle
                print(f"Calculating {lang1} vs {lang2}...")
                
                # Get non-empty pairs
                valid_pairs = df[[lang1, lang2]].dropna()
                valid_pairs = valid_pairs[
                    (valid_pairs[lang1].astype(str).str.strip() != '') & 
                    (valid_pairs[lang2].astype(str).str.strip() != '')
                ]
                
                if len(valid_pairs) == 0:
                    print(f"  No valid pairs found for {lang1} vs {lang2}")
                    continue
                
                # Calculate similarities
                similarities_list = []
                for _, row in valid_pairs.iterrows():
                    sim = jaccard_similarity(row[lang1], row[lang2])
                    similarities_list.append(sim)
                
                mean_sim = np.mean(similarities_list)
                std_sim = np.std(similarities_list)
                
                similarities[f"{lang1}_vs_{lang2}"] = {
                    "mean": float(mean_sim),
                    "std": float(std_sim),
                    "count": len(similarities_list),
                    "min": float(np.min(similarities_list)),
                    "max": float(np.max(similarities_list))
                }
                
                print(f"  {lang1} vs {lang2}: {mean_sim:.4f} ± {std_sim:.4f} (n={len(similarities_list)})")
    
    return similarities

def main():
    parser = argparse.ArgumentParser(description="Calculate Jaccard similarity between languages")
    parser.add_argument("--output", default="jaccard_similarities.json", help="Output JSON file")
    parser.add_argument("--datasets", nargs="+", 
                       default=["predictionguard/english-hindi-marathi-konkani-corpus", 
                               "predictionguard/arabic_acl_corpus"],
                       help="Datasets to analyze")
    
    args = parser.parse_args()
    
    all_results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            
            # Process each split
            for split_name, split_data in dataset.items():
                print(f"\nSplit: {split_name}")
                df = pd.DataFrame(split_data)
                
                # Define language columns based on dataset
                if "konkani" in dataset_name.lower():
                    language_columns = ['eng', 'hin', 'mar', 'gom']
                elif "arabic" in dataset_name.lower():
                    # For Arabic dataset, extract languages from translation dict
                    if 'translation' in df.columns:
                        # Get all language codes from the first translation dict
                        first_translation = df['translation'].iloc[0]
                        language_columns = list(first_translation.keys())
                        print(f"Found language codes: {language_columns}")
                        
                        # Create separate columns for each language
                        for lang in language_columns:
                            df[f"{lang}_text"] = df['translation'].apply(
                                lambda x: x.get(lang, '') if isinstance(x, dict) else ''
                            )
                        language_columns = [f"{lang}_text" for lang in language_columns]
                    else:
                        print("No translation column found, skipping...")
                        continue
                else:
                    print(f"Unknown dataset structure: {dataset_name}")
                    continue
                
                # Calculate similarities
                similarities = calculate_language_similarity(
                    df, language_columns, f"{dataset_name}_{split_name}"
                )
                
                all_results[f"{dataset_name}_{split_name}"] = {
                    "dataset": dataset_name,
                    "split": split_name,
                    "language_columns": language_columns,
                    "similarities": similarities
                }
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Print summary table
    for dataset_key, results in all_results.items():
        print(f"\n{dataset_key}:")
        print("-" * 40)
        similarities = results["similarities"]
        
        if similarities:
            # Create a simple table
            print(f"{'Language Pair':<25} {'Mean':<8} {'Std':<8} {'Count':<6}")
            print("-" * 50)
            
            for pair, stats in similarities.items():
                print(f"{pair:<25} {stats['mean']:<8.4f} {stats['std']:<8.4f} {stats['count']:<6}")
        else:
            print("No similarities calculated")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
