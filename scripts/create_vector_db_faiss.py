#!/usr/bin/env python3
import pandas as pd
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import argparse
import os

def create_vector_db(dataset_name, source_lang, pivot_lang, target_lang, db_path="translations_db"):
    """
    Creates and saves a FAISS vector database from a Hugging Face dataset.
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])

    # Clean the dataframe
    required_cols = [source_lang, pivot_lang, target_lang]
    df = df.dropna(subset=required_cols)
    df = df[(df[required_cols] != '').all(axis=1)]

    print(f"Processing {len(df)} rows for the vector database.")

    # Create LangChain Document objects
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row[source_lang],
            metadata={
                "pivot": row[pivot_lang],
                "target": row[target_lang]
            }
        )
        documents.append(doc)

    print("Loading embedding model: hkunlp/instructor-base")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    print(f"Creating FAISS vector database at: {db_path}")
    # This creates the vector store from the documents and embeddings
    vector_db = FAISS.from_documents(documents, embeddings)

    # Save the vector database locally
    vector_db.save_local(db_path)
    print(f"Vector database saved successfully to {db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a FAISS vector database for translation examples.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name from Hugging Face.")
    parser.add_argument("--source_lang", required=True, help="Column name for the source language (e.g., 'eng').")
    parser.add_argument("--pivot_lang", required=True, help="Column name for the pivot language (e.g., 'mar').")
    parser.add_argument("--target_lang", required=True, help="Column name for the target language (e.g., 'gom').")
    parser.add_argument("--vector_db_path", default="data/translations_db", help="Path to save the FAISS database.")
    
    args = parser.parse_args()
    
    # Ensure the save directory exists
    os.makedirs(args.vector_db_path, exist_ok=True)
    
    create_vector_db(args.dataset_name, args.source_lang, args.pivot_lang, args.target_lang, args.vector_db_path)
