# -*- coding: utf-8 -*-
"""
This script is used for running inference using a trained model, with support for
both semantic and random few-shot example retrieval strategies.
"""

# Import necessary libraries
import argparse
import json
import re
import os
import sys
import pandas as pd
import torch
import random
import numpy as np
import gc
import time
from datetime import datetime
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
import sacrebleu

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available. Install with: pip install wandb")

# Logging function for real-time output
def log(message, level="INFO", flush=True):
    """Print timestamped log message with immediate flush."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_colors = {
        "INFO": "",
        "SUCCESS": "✅ ",
        "WARNING": "⚠️  ",
        "ERROR": "❌ ",
        "PROGRESS": "📊 "
    }
    prefix = level_colors.get(level, "")
    print(f"[{timestamp}] {prefix}{message}")
    if flush:
        sys.stdout.flush()

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
    # W&B logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="sampling-ablation-study", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    
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


# Metric calculation functions
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

def calculate_chrf_plus_plus(references, hypotheses):
    """Calculate chrF++ score."""
    formatted_references = [[ref] for ref in references]
    chrf_pp = sacrebleu.corpus_chrf(hypotheses, formatted_references, word_order=2)
    return chrf_pp.score


# Main function
def main():
    """Main function for the script."""
    args = parse_args()
    
    start_time = time.time()

    # Seed Python, NumPy, and Torch to improve reproducibility, especially for random sampling
    try:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    except Exception:
        pass
    
    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        log("--wandb flag set but wandb not installed!", "WARNING")
        log("Install with: pip install wandb", "WARNING")
        use_wandb = False
    
    if use_wandb:
        # Extract model shorthand for run name
        model_short = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
        run_name = args.wandb_run_name or f"{args.retrieval_strategy}_k{args.num_fs_examples}_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log("📊 Initializing Weights & Biases...", "INFO")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model_name,
                "dataset": args.dataset_name,
                "retrieval_strategy": args.retrieval_strategy,
                "num_fs_examples": args.num_fs_examples,
                "prompt_type": args.prompt_type,
                "random_seed": args.random_seed,
                "source_lang": args.source_lang,
                "pivot_lang": args.pivot_lang,
                "target_lang": args.target_lang,
                "cuda_available": torch.cuda.is_available(),
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "dtype": "float16",
            },
            tags=["sampling-ablation", args.retrieval_strategy, f"k={args.num_fs_examples}"]
        )
        log("W&B initialized successfully", "SUCCESS")

    # GPU information
    log("=" * 80, "INFO")
    log("🔧 GPU CONFIGURATION", "INFO")
    log("=" * 80, "INFO")
    log(f"CUDA available: {torch.cuda.is_available()}", "INFO")
    if torch.cuda.is_available():
        log(f"Number of GPUs: {torch.cuda.device_count()}", "INFO")
        log(f"Current device: {torch.cuda.current_device()}", "INFO")
        log(f"Device name: {torch.cuda.get_device_name(0)}", "INFO")
        log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", "INFO")
    else:
        log("CUDA not available! Running on CPU (will be very slow)", "WARNING")
    log("=" * 80, "INFO")
    
    log(f"📥 Loading model: {args.model_name}", "INFO")
    log("   This may take a few minutes...", "INFO")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",  # Automatically distributes across available GPUs
        torch_dtype=torch.float16,  # Use FP16 for faster GPU inference
    )
    model_load_time = time.time() - model_load_start
    log(f"Model loaded in {model_load_time/60:.1f} minutes", "SUCCESS")
    log(f"   Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}", "INFO")

    log(f"📚 Loading dataset: {args.dataset_name}", "INFO")
    dataset = load_dataset(args.dataset_name)
    test_set = pd.DataFrame(dataset['test'])
    log(f"Dataset loaded: {len(test_set)} test samples", "SUCCESS")

    vector_db = None
    fs_examples_df = None

    if args.prompt_type == "few_shot":
        if args.retrieval_strategy == "semantic":
            log("🔍 Using SEMANTIC retrieval strategy", "INFO")
            if not args.vector_db_path:
                raise ValueError("Vector DB path is required for semantic retrieval.")
            log(f"Loading vector database from: {args.vector_db_path}", "INFO")
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
            # SECURITY: Only load vector DBs from fully trusted sources.
            # We trust this data as it was created by our own script (create_vector_db_faiss.py)
            vector_db = FAISS.load_local(
                args.vector_db_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            log("Vector database loaded successfully", "SUCCESS")
        
        elif args.retrieval_strategy == "random":
            log("🎲 Using RANDOM retrieval strategy", "INFO")
            log("Loading 'train' split for random sampling", "INFO")
            fs_examples_df = pd.DataFrame(dataset['train'])
            # Clean the dataframe
            required_cols = [args.source_lang, args.pivot_lang, args.target_lang]
            fs_examples_df = fs_examples_df.dropna(subset=required_cols)
            fs_examples_df = fs_examples_df[(fs_examples_df[required_cols] != '').all(axis=1)]
            log(f"Loaded {len(fs_examples_df)} training examples for random sampling", "SUCCESS")


    log("=" * 80, "INFO")
    log("🚀 Starting inference...", "INFO")
    log(f"   Strategy: {args.retrieval_strategy}", "INFO")
    log(f"   Prompt type: {args.prompt_type}", "INFO")
    log(f"   Few-shot examples (k): {args.num_fs_examples}", "INFO")
    log(f"   Test samples: {len(test_set)}", "INFO")
    log("=" * 80, "INFO")
    inference_start = time.time()
    outputs = []
    
    # Progress reporting intervals
    report_interval = max(1, len(test_set) // 20)  # Report every 5% or at least every sample
    
    for idx, row in enumerate(test_set.iterrows()):
        _, row = row
        sample_start = time.time()
        
        # Progress logging
        if idx == 0 or (idx + 1) % report_interval == 0 or idx == len(test_set) - 1:
            progress_pct = ((idx + 1) / len(test_set)) * 100
            elapsed = time.time() - inference_start
            avg_time = elapsed / (idx + 1) if idx > 0 else 0
            eta = avg_time * (len(test_set) - idx - 1) if idx > 0 else 0
            log(f"Progress: {idx + 1}/{len(test_set)} ({progress_pct:.1f}%) | "
                f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m", "PROGRESS")
        
        prompt = get_prompt(row, args.prompt_type, args.num_fs_examples, args, vector_db, fs_examples_df)
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                top_k=50,
                top_p=0.75,
            )
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract the assistant's response by removing the prompt
        # The decoded output contains: prompt + "assistant\n" + actual_translation
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if output.startswith(prompt_decoded):
            assistant_response = output[len(prompt_decoded):].strip()
            # Remove "assistant" prefix if present
            if assistant_response.startswith("assistant"):
                assistant_response = assistant_response[len("assistant"):].strip()
        else:
            # Fallback: try to find anything after "assistant"
            if "assistant" in output:
                assistant_response = output.split("assistant", 1)[1].strip()
            else:
                assistant_response = ""
        
        outputs.append({
            "source": row[args.source_lang],
            "pivot": row[args.pivot_lang],
            "target": row[args.target_lang],
            "prediction": assistant_response,
        })
        
        # Show first few translations as examples
        if idx < 3:
            sample_time = time.time() - sample_start
            log(f"   Sample {idx + 1} completed in {sample_time:.1f}s", "INFO")
            log(f"   Source: {row[args.source_lang][:80]}...", "INFO")
            log(f"   Prediction: {assistant_response[:80]}...", "INFO")
    
    inference_time = time.time() - inference_start
    log(f"Inference completed in {inference_time/60:.1f} minutes", "SUCCESS")
    log(f"   Average time per sample: {inference_time/len(test_set):.2f}s", "INFO")

    # Calculate metrics
    log("📊 Calculating evaluation metrics...", "INFO")
    references = [item["target"] for item in outputs]
    hypotheses = [item["prediction"] for item in outputs]
    
    bleu_score = calculate_bleu(references, hypotheses)
    chrf_score = calculate_chrf(references, hypotheses)
    chrf_pp_score = calculate_chrf_plus_plus(references, hypotheses)
    
    log("=" * 80, "INFO")
    log("📈 EVALUATION RESULTS", "INFO")
    log("=" * 80, "INFO")
    log(f"BLEU Score:    {bleu_score:.2f}", "SUCCESS")
    log(f"chrF Score:    {chrf_score:.2f}", "SUCCESS")
    log(f"chrF++ Score:  {chrf_pp_score:.2f}", "SUCCESS")
    log("=" * 80, "INFO")
    
    # Save outputs with metrics
    log(f"💾 Saving outputs to: {args.output_path}", "INFO")
    results = {
        "config": {
            "model": args.model_name,
            "dataset": args.dataset_name,
            "retrieval_strategy": args.retrieval_strategy,
            "num_fs_examples": args.num_fs_examples,
            "prompt_type": args.prompt_type,
        },
        "metrics": {
            "bleu": float(bleu_score),
            "chrf": float(chrf_score),
            "chrf_pp": float(chrf_pp_score),
            "inference_time_minutes": inference_time / 60,
            "num_samples": len(outputs),
        },
        "outputs": outputs
    }
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    log(f"Results saved successfully", "SUCCESS")
    
    # Log to W&B
    if use_wandb:
        log("📊 Logging results to W&B...", "INFO")
        wandb.log({
            "bleu": bleu_score,
            "chrf": chrf_score,
            "chrf_pp": chrf_pp_score,
            "inference_time_minutes": inference_time / 60,
            "num_samples": len(outputs),
        })
        
        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Score"],
            data=[
                ["BLEU", f"{bleu_score:.2f}"],
                ["chrF", f"{chrf_score:.2f}"],
                ["chrF++", f"{chrf_pp_score:.2f}"],
                ["Samples", str(len(outputs))],
                ["Time (min)", f"{inference_time / 60:.1f}"]
            ]
        )
        wandb.log({"results_summary": summary_table})
        log(f"Results logged to W&B: {wandb.run.url}", "SUCCESS")
        wandb.finish()
    
    # Memory cleanup
    log("🧹 Cleaning up memory...", "INFO")
    del model
    del tokenizer
    if vector_db is not None:
        del vector_db
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log("Memory cleaned up", "SUCCESS")
    
    total_time = time.time() - start_time
    log("=" * 80, "INFO")
    log(f"🎉 COMPLETE! Total time: {total_time/60:.1f} minutes", "SUCCESS")
    log("=" * 80, "INFO")

if __name__ == "__main__":
    main()

