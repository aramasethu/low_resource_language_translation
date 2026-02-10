#!/usr/bin/env python3
"""
Create vector database for Arabic dataset (nested structure)

The Arabic dataset has a nested 'translation' column with:
- en: English
- msa: Modern Standard Arabic
- tn: Tunisian Arabic
- eg: Egyptian Arabic
- jo: Jordanian Arabic
- pa: Palestinian Arabic
- sy: Syrian Arabic
"""
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import lancedb
import argparse

def create_vector_db_arabic(dataset_name, db_name="arabic_translations"):
    """Create vector database for Arabic dataset with nested structure."""
    
    dataset = load_dataset(dataset_name)
    
    # Extract the nested translation dictionaries
    train_data = []
    for item in dataset['train']:
        translation_dict = item['translation']
        train_data.append({
            'en': translation_dict.get('en', ''),
            'msa': translation_dict.get('msa', ''),
            'tn': translation_dict.get('tn', ''),
            'eg': translation_dict.get('eg', ''),
            'jo': translation_dict.get('jo', ''),
            'pa': translation_dict.get('pa', ''),
            'sy': translation_dict.get('sy', '')
        })
    
    df = pd.DataFrame(train_data)
    
    # For the experiments, we use: msa (pivot), en (source), tn (target)
    required_cols = ['msa', 'en', 'tn']
    new_df = df[required_cols].copy()
    
    # Filter out rows with empty strings or NaN values
    mask = (new_df.fillna("").astype(str) == "").any(axis=1)
    new_df = new_df[~mask]
    
    print(f"Dataset: {dataset_name}")
    print(f"Languages: msa (pivot) -> en (source) -> tn (target)")
    print(f"Rows after filtering: {len(new_df)}")
    
    # Create embeddings
    embed_model = SentenceTransformer("all-MiniLM-L12-v2")
    texts = new_df['msa'].tolist()
    embeddings = embed_model.encode(texts)
    
    # Prepare data for LanceDB
    data = []
    for i, row in new_df.iterrows():
        data.append({
            "text": row['msa'],
            "vector": embeddings[i].tolist(),
            "en": row['en'],
            "tn": row['tn']
        })
    
    # Create database
    db = lancedb.connect(db_name)
    table_name = "translations_tn"
    
    tbl = db.create_table(table_name, data, mode='overwrite')
    
    print(f"Created table '{table_name}' with {len(tbl)} entries")
    print(f"Database location: {db_name}/")
    return tbl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector database for Arabic dataset")
    parser.add_argument("--dataset", default="predictionguard/arabic_acl_corpus", 
                       help="Dataset name")
    parser.add_argument("--db", default="arabic_translations", 
                       help="Database name")
    
    args = parser.parse_args()
    
    create_vector_db_arabic(args.dataset, args.db)
    
    print("\nVector database created successfully!")
    print("You can now run the ablation study with:")
    print(f"  --db \"{args.db}\"")
    print(f"  --pivot \"msa\"")
    print(f"  --source \"en\"")
    print(f"  --target \"tn\"")

