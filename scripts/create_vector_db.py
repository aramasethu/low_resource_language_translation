#!/usr/bin/env python3
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import lancedb
import argparse

def create_vector_db(dataset_name, pivot, source, target, db_name="translations_db"):
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])
    required_cols = [pivot, source, target]
    new_df = df[required_cols].copy()
    mask = (new_df.fillna("").astype(str) == "").any(axis=1)
    new_df = new_df[~mask]
    
    print(f"Dataset: {dataset_name}")
    print(f"Languages: {pivot} -> {source} -> {target}")
    print(f"Rows: {len(new_df)}")
    embed_model = SentenceTransformer("all-MiniLM-L12-v2")
    texts = new_df[pivot].tolist()
    embeddings = embed_model.encode(texts)
    
    data = []
    for i, row in new_df.iterrows():
        data.append({
            "text": row[pivot],
            "vector": embeddings[i].tolist(),
            source: row[source],
            target: row[target]
        })
    db = lancedb.connect(db_name)
    table_name = f"translations_{target}"
    
    tbl = db.create_table(table_name, data, mode='overwrite')
    
    print(f"Created table '{table_name}' with {len(tbl)} entries")
    return tbl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector database for translation")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., predictionguard/english-hindi-marathi-konkani-corpus)")
    parser.add_argument("--pivot", required=True, help="Pivot language column (e.g., eng, hin)")
    parser.add_argument("--source", required=True, help="Source language column (e.g., mar, ara)")
    parser.add_argument("--target", required=True, help="Target language column (e.g., gom, tun)")
    parser.add_argument("--db", default="translations_db", help="Database name")
    
    args = parser.parse_args()
    
    create_vector_db(args.dataset, args.pivot, args.source, args.target, args.db)