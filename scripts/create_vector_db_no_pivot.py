#!/usr/bin/env python3
"""
Create vector database for NO-PIVOT experiments.

This embeds the SOURCE text (not pivot) to enable retrieval of similar
source→target examples WITHOUT using a pivot language.

For Konkani: Embeds English (source) for English→Konkani translation
For Arabic: Embeds English (source) for English→Tunisian translation
"""
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import lancedb
import argparse

def create_vector_db_no_pivot(dataset_name, source, target, db_name="translations_no_pivot"):
    """
    Create vector database using SOURCE language (no pivot).
    
    Args:
        dataset_name: HuggingFace dataset name
        source: Source language column (e.g., 'eng' for English)
        target: Target language column (e.g., 'gom' for Konkani)
        db_name: Database name
    """
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])
    
    # Handle nested structure (Arabic) vs flat structure (Konkani)
    if 'translation' in df.columns:
        # Arabic dataset: nested structure
        # Map 3-letter codes to 2-letter codes used in the dataset
        code_mapping = {
            'eng': 'en',
            'tun': 'tn',
            'msa': 'msa'
        }
        source_key = code_mapping.get(source, source)
        target_key = code_mapping.get(target, target)
        
        print(f"📦 Detected nested dataset structure (Arabic)")
        print(f"   Using translation['{source_key}'] → translation['{target_key}']")
        
        new_df = pd.DataFrame({
            source: df['translation'].apply(lambda x: x.get(source_key, '')),
            target: df['translation'].apply(lambda x: x.get(target_key, ''))
        })
    else:
        # Konkani dataset: flat structure
        print(f"📦 Detected flat dataset structure (Konkani)")
        required_cols = [source, target]
        new_df = df[required_cols].copy()
    
    # Filter out rows with empty strings or NaN values
    mask = (new_df.fillna("").astype(str) == "").any(axis=1)
    new_df = new_df[~mask]
    
    print(f"Dataset: {dataset_name}")
    print(f"Languages: {source} -> {target} (NO PIVOT)")
    print(f"Rows: {len(new_df)}")
    
    # Create embeddings using SOURCE text (not pivot!)
    embed_model = SentenceTransformer("all-MiniLM-L12-v2")
    texts = new_df[source].tolist()
    embeddings = embed_model.encode(texts)
    
    # Prepare data for LanceDB
    data = []
    for i, row in new_df.iterrows():
        data.append({
            "text": row[source],  # SOURCE text for embedding
            "vector": embeddings[i].tolist(),
            source: row[source],
            target: row[target]
        })
    
    db = lancedb.connect(db_name)
    table_name = f"translations_{target}_no_pivot"
    
    tbl = db.create_table(table_name, data, mode='overwrite')
    
    print(f"✅ Created table '{table_name}' with {len(tbl)} entries")
    print(f"   Embeddings based on: {source} (source language)")
    print(f"   Direct translation: {source} → {target} (no pivot)")
    return tbl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create vector database for no-pivot experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Konkani (English→Konkani, no Marathi pivot)
  python scripts/create_vector_db_no_pivot.py \\
      --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \\
      --source "eng" \\
      --target "gom" \\
      --db "konkani_no_pivot_db"
  
  # Tunisian Arabic (English→Tunisian, no MSA pivot)
  python scripts/create_vector_db_no_pivot.py \\
      --dataset "predictionguard/arabic_acl_corpus" \\
      --source "eng" \\
      --target "tun" \\
      --db "arabic_no_pivot_db"
        """
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--source", required=True, help="Source language column")
    parser.add_argument("--target", required=True, help="Target language column")
    parser.add_argument("--db", default="translations_no_pivot", help="Database name")
    
    args = parser.parse_args()
    
    create_vector_db_no_pivot(args.dataset, args.source, args.target, args.db)

