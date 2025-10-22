# -*- coding: utf-8 -*-
"""
This script is used for running inference using a trained model, with support for
both semantic and random few-shot example retrieval strategies.
"""

# Import necessary libraries
import argparse
import json
import re
import pandas as pd
import torch
import random
import numpy as np
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
# Preferred newer package (langchain-huggingface)
from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings as _HuggingFaceInstructEmbeddings
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Function to parse command-line arguments
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on a given dataset using a specified model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unbabel/TowerInstruct-7B-v0.2",
        help="The name of the model to use for inference.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset on Hugging Face.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to save the output file (in JSON format).",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
        help="The type of prompt to use for inference.",
    )
    parser.add_argument(
        "--num_fs_examples",
        type=int,
        default=5,
        help="The number of few-shot examples to use.",
    )
    parser.add_argument(
        "--vector_db_path",
        type=str,
        help="The path to the vector database for semantic retrieval.",
    )
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        choices=["semantic", "random"],
        default="semantic",
        help="Strategy to retrieve few-shot examples.",
    )
    parser.add_argument("--source_lang", type=str, required=True, help="Column name for the source language (e.g., 'eng').")
    parser.add_argument("--pivot_lang", type=str, required=True, help="Column name for the pivot language (e.g., 'mar').")
    parser.add_argument("--target_lang", type=str, required=True, help="Column name for the target language (e.g., 'gom').")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible random few-shot sampling (and optional generation).",
    )
    return parser.parse_args()

# Function to create a chat prompt
def _create_chat_prompt(
    instruction: str,
    source_text: str,
    pivot_text: str,
    target_text: str,
) -> str:
    """Creates a chat prompt in the ChatML format."""
    return (
        f"<|im_start|>user\n{instruction}\n\n"
        f"Original (English): {source_text}\n\n"
        f"Translation (Marathi): {pivot_text}\n\n"
        f"Post-edited (Konkani): <|im_end|>\n"
        f"<|im_start|>assistant\n{target_text}<|im_end|>"
    )


# Function to get the prompt for inference
def get_prompt(
    test_set_instance: pd.Series,
    prompt_type: str,
    num_fs_examples: int,
    args: argparse.Namespace,
    vector_db: FAISS = None,
    fs_examples_df: pd.DataFrame = None,
) -> str:
    """Generates the prompt for a given test set instance."""
    instruction = "APE is a task designed to enhance the quality of the translation by performing only minor adjustments to fix any existing translation mistakes. If the translation is already correct, you should retain it as is."
    
    source_text = test_set_instance[args.source_lang]
    pivot_text = test_set_instance[args.pivot_lang]

    if prompt_type == "zero_shot":
        return (
            f"<|im_start|>user\n{instruction}\n\n"
            f"Original (English): {source_text}\n\n"
            f"Translation (Marathi): {pivot_text}\n\n"
            f"Post-edited (Konkani): <|im_end|>\n<|im_start|>assistant\n"
        )
    
    elif prompt_type == "few_shot":
        docs = []
        if args.retrieval_strategy == "semantic":
            if vector_db is None:
                raise ValueError("Vector DB must be provided for semantic retrieval.")
            query = source_text
            docs = vector_db.similarity_search(query, k=num_fs_examples)

        elif args.retrieval_strategy == "random":
            if fs_examples_df is None:
                raise ValueError("Few-shot example DataFrame must be provided for random retrieval.")
            # Reproducible sampling with a fixed seed; cap n to available rows
            n = min(num_fs_examples, len(fs_examples_df))
            random_examples = fs_examples_df.sample(n=n, random_state=args.random_seed)
            for _, row in random_examples.iterrows():
                doc = Document(
                    page_content=row[args.source_lang],
                    metadata={
                        "pivot": row[args.pivot_lang],
                        "target": row[args.target_lang],
                    },
                )
                docs.append(doc)

        fs_prompt = ""
        for doc in docs:
            fs_prompt += _create_chat_prompt(
                instruction,
                doc.page_content,
                doc.metadata["pivot"],
                doc.metadata["target"],
            )

        fs_prompt += (
            f"<|im_start|>user\n{instruction}\n\n"
            f"Original (English): {source_text}\n\n"
            f"Translation (Marathi): {pivot_text}\n\n"
            f"Post-edited (Konkani): <|im_end|>\n<|im_start|>assistant\n"
        )
        return fs_prompt
    
    raise ValueError(f"Invalid prompt type: {prompt_type}")


# Main function
def main():
    """Main function for the script."""
    args = parse_args()

    # Seed Python, NumPy, and Torch to improve reproducibility, especially for random sampling
    try:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    except Exception:
        pass

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        dtype=torch.float16,
    )

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    test_set = pd.DataFrame(dataset['test'])

    vector_db = None
    fs_examples_df = None

    if args.prompt_type == "few_shot":
        if args.retrieval_strategy == "semantic":
            print("Using SEMANTIC retrieval strategy.")
            if not args.vector_db_path:
                raise ValueError("Vector DB path is required for semantic retrieval.")
            print(f"Loading vector database from: {args.vector_db_path}")
            # Helper to create embeddings with multiple fallbacks and helpful errors
            def _get_embeddings(model_name: str = "hkunlp/instructor-base"):
                """Return an embeddings instance using available HuggingFace classes.

                Preference order:
                  1. langchain-huggingface.HuggingFaceEmbeddings (recommended)
                  2. langchain.embeddings.HuggingFaceInstructEmbeddings (legacy)

                Raises a clear ImportError if required packages are missing.
                """
                if _HuggingFaceEmbeddings is not None:
                    try:
                        return _HuggingFaceEmbeddings(model_name=model_name)
                    except TypeError:
                        # Some versions take different init args
                        return _HuggingFaceEmbeddings(model_name)

                if _HuggingFaceInstructEmbeddings is not None:
                    try:
                        return _HuggingFaceInstructEmbeddings(model_name=model_name)
                    except Exception as e:
                        # Surface a helpful message if InstructorEmbedding runtime deps are missing
                        if "InstructorEmbedding" in str(e) or "InstructorEmbedding" in repr(e):
                            raise ImportError(
                                "InstructorEmbedding dependencies not found. Install them with:"
                                "\n    pip install InstructorEmbedding"
                                "\nOr install the newer package: pip install -U langchain-huggingface"
                            ) from e
                        raise

                # If none of the embedding classes are available, instruct the user how to fix
                raise ImportError(
                    "No suitable HuggingFace embedding class is available.\n"
                    "Recommended fixes:\n"
                    " 1) Install the maintained integration: pip install -U langchain-huggingface\n"
                    " 2) (Legacy) Install InstructorEmbedding if you want the old class: pip install InstructorEmbedding\n"
                )

            embeddings = _get_embeddings(model_name="hkunlp/instructor-base")
            vector_db = FAISS.load_local(args.vector_db_path, embeddings, allow_dangerous_deserialization=True)
        
        elif args.retrieval_strategy == "random":
            print("Using RANDOM retrieval strategy.")
            print("Loading 'train' split for random sampling.")
            fs_examples_df = pd.DataFrame(dataset['train'])
            # Clean the dataframe
            required_cols = [args.source_lang, args.pivot_lang, args.target_lang]
            fs_examples_df = fs_examples_df.dropna(subset=required_cols)
            fs_examples_df = fs_examples_df[(fs_examples_df[required_cols] != '').all(axis=1)]


    print("Running inference...")
    outputs = []
    for _, row in tqdm(test_set.iterrows(), total=len(test_set)):
        prompt = get_prompt(row, args.prompt_type, args.num_fs_examples, args, vector_db, fs_examples_df)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            top_k=50,
            top_p=0.75,
        )
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        match = re.search(r"<\|im_start\|>assistant\n(.*)", output, re.DOTALL)
        assistant_response = match.group(1).strip() if match else ""
        
        outputs.append({
            "source": row[args.source_lang],
            "pivot": row[args.pivot_lang],
            "target": row[args.target_lang],
            "prediction": assistant_response,
        })

    print(f"Saving outputs to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

    print("Inference complete.")

if __name__ == "__main__":
    main()

