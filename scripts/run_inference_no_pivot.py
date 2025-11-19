#!/usr/bin/env python3
"""
Inference script for NO-PIVOT experiments.

This script translates directly from SOURCE to TARGET without using a pivot language.
Used for ablation study to compare pivot vs no-pivot approaches.
"""
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
import transformers
import torch
import lancedb
from sentence_transformers import SentenceTransformer
import argparse
import json
import sacrebleu
import time
import sys
import os
from datetime import datetime
import gc

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()

def load_and_flatten_dataset(dataset_name, split='test'):
    """Load dataset and automatically handle both flat and nested structures."""
    log(f"📚 Loading dataset: {dataset_name}", "INFO")
    dataset = load_dataset(dataset_name)
    
    test_data = dataset[split]
    
    if len(test_data) > 0:
        first_item = test_data[0]
        
        if 'translation' in first_item and isinstance(first_item['translation'], dict):
            # Nested structure (e.g., Arabic dataset)
            log("   Detected nested 'translation' structure - flattening...", "INFO")
            flattened = []
            for item in test_data:
                translation_dict = item['translation']
                flattened.append(translation_dict)
            test_df = pd.DataFrame(flattened)
            log(f"   Flattened nested structure", "INFO")
        else:
            # Flat structure (e.g., Konkani dataset)
            log("   Detected flat structure - using directly", "INFO")
            test_df = pd.DataFrame(test_data)
    else:
        test_df = pd.DataFrame(test_data)
    
    return test_df

def main():
    parser = argparse.ArgumentParser(
        description="NO-PIVOT inference script for ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Konkani translation (English→Konkani, NO Marathi pivot)
  python scripts/run_inference_no_pivot.py \\
    --dataset predictionguard/english-hindi-marathi-konkani-corpus \\
    --source eng --target gom \\
    --db konkani_no_pivot_db --num-examples 3

  # Arabic translation (English→Tunisian, NO MSA pivot)
  python scripts/run_inference_no_pivot.py \\
    --dataset predictionguard/arabic_acl_corpus \\
    --source eng --target tun \\
    --db arabic_no_pivot_db --num-examples 5
        """
    )
    
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--source", required=True, help="Source language column (e.g., eng)")
    parser.add_argument("--target", required=True, help="Target language column (e.g., gom)")
    parser.add_argument("--db", required=True, help="Vector database path")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--scores", required=True, help="Scores JSON file")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test samples (for debugging)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="low-resource-translation", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"no-pivot-{args.target}-k{args.num_examples}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        log("✅ W&B logging enabled", "SUCCESS")
    elif args.wandb and not WANDB_AVAILABLE:
        log("⚠️  W&B requested but not available - continuing without logging", "WARNING")
    
    log("="*80, "INFO")
    log("🚀 NO-PIVOT INFERENCE (Direct Source→Target Translation)", "INFO")
    log("="*80, "INFO")
    log(f"📊 Configuration:", "INFO")
    log(f"   Dataset: {args.dataset}", "INFO")
    log(f"   Model: {args.model}", "INFO")
    log(f"   Translation: {args.source.upper()} → {args.target.upper()} (NO PIVOT)", "INFO")
    log(f"   Few-shot examples: k={args.num_examples}", "INFO")
    log(f"   Batch size: {args.batch_size}", "INFO")
    
    # GPU status
    if torch.cuda.is_available():
        log(f"🚀 GPU: {torch.cuda.get_device_name(0)}", "INFO")
        log(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", "INFO")
    else:
        log("⚠️  GPU: Not available - using CPU", "WARNING")
    
    log("="*80, "INFO")
    
    # Load model
    log(f"🤖 Loading model: {args.model}", "INFO")
    model_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    pipe = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.1,
        top_p=0.95
    )
    
    model_time = time.time() - model_start
    log(f"✅ Model loaded in {model_time:.2f}s", "SUCCESS")
    
    if use_wandb:
        wandb.log({"model_load_time": model_time})
    
    # Load dataset
    test_df = load_and_flatten_dataset(args.dataset, split='test')
    log(f"✅ Dataset loaded: {len(test_df)} test samples", "SUCCESS")
    
    # Apply test limit if specified
    if args.test_limit is not None and args.test_limit < len(test_df):
        log(f"⚠️  TEST MODE: Limiting to first {args.test_limit} samples", "INFO")
        test_df = test_df.head(args.test_limit)
    
    log(f"   Columns: {list(test_df.columns)}", "INFO")
    
    # Map language codes if needed (for Arabic dataset which uses 2-letter codes)
    # Only map if the original column doesn't exist
    source_col = args.source
    target_col = args.target
    
    if args.source not in test_df.columns:
        # Try mapping 3-letter to 2-letter codes (e.g., eng→en for Arabic)
        code_mapping = {'eng': 'en', 'tun': 'tn'}
        if args.source in code_mapping:
            source_col = code_mapping[args.source]
            log(f"   Mapped {args.source} → {source_col}", "INFO")
    
    if args.target not in test_df.columns:
        # Try mapping 3-letter to 2-letter codes
        code_mapping = {'eng': 'en', 'tun': 'tn'}
        if args.target in code_mapping:
            target_col = code_mapping[args.target]
            log(f"   Mapped {args.target} → {target_col}", "INFO")
    
    # Validate required columns
    required_cols = [source_col, target_col]
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        log(f"❌ ERROR: Required columns not found: {missing_cols}", "ERROR")
        log(f"   Available columns: {list(test_df.columns)}", "ERROR")
        log(f"   Attempted to use: source={source_col}, target={target_col}", "ERROR")
        sys.exit(1)
    
    log(f"   Using columns: source={source_col}, target={target_col}", "INFO")
    
    # Setup database and embeddings (only if needed)
    if args.num_examples > 0:
        log(f"🗄️  Connecting to vector database: {args.db}", "INFO")
        db = lancedb.connect(args.db)
        log(f"📊 Loading embedding model: all-MiniLM-L12-v2", "INFO")
        embed_model = SentenceTransformer("all-MiniLM-L12-v2")
        
        def embed(text):
            return embed_model.encode(text)
        
        log(f"✅ Vector database ready for k={args.num_examples} few-shot retrieval", "SUCCESS")
    else:
        log("⚠️  Zero-shot mode: No vector database needed (k=0)", "INFO")
    
    # Setup prompts - NOTE: Direct source→target translation (NO PIVOT)
    USER_PREFIX = f'Translate the following text from {args.source.upper()} to {args.target.upper()}. Only return the {args.target.upper()} translation and nothing else.'
    USER_MIDDLE = '\nSource: '
    USER_SUFFIX = f'\nTranslation: '
    
    def translate_prompt_no_pivot(source_text):
        """
        Build prompt for direct source→target translation (no pivot).
        
        Args:
            source_text: Text in source language
            
        Returns:
            str: Formatted prompt with optional few-shot examples
        """
        messages = []
        
        # Add few-shot examples if requested
        if args.num_examples > 0:
            # Search for similar SOURCE examples (not pivot!)
            table_name = f"translations_{target_col}_no_pivot"
            table = db.open_table(table_name)
            results = table.search(embed(source_text)).limit(10).to_pandas()
            results = results[results['text'] != source_text]
            results = results[results['text'] != ""]
            # Note: Vector DB stores with original column names (args.source/target)
            results = results[results[args.source] != ""]
            results = results[results[args.target] != ""]
            results.dropna(inplace=True)
            results.sort_values(by="_distance", ascending=True, inplace=True)
            
            # Add k examples
            num_examples = args.num_examples
            for i in range(0, num_examples+1):
                if i != 0 and i <= len(results):
                    try:
                        # Example is SOURCE→TARGET (no pivot!)
                        # Use original column names from vector DB
                        messages.append({
                            "role": "user",
                            "content": USER_PREFIX + USER_MIDDLE + results[args.source].values[i-1] + USER_SUFFIX
                        })
                        messages.append({
                            "role": "assistant",
                            "content": results[args.target].values[i-1]
                        })
                    except Exception as e:
                        log(f"Warning: Error adding example {i}: {e}", "WARNING")
        
        # Add the current translation request
        messages.append({
            "role": "user",
            "content": USER_PREFIX + USER_MIDDLE + source_text + USER_SUFFIX
        })
        
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Test prompt generation
    log("🧪 Testing prompt generation with first sample...", "INFO")
    prompt = translate_prompt_no_pivot(test_df[source_col].values[0])
    log(f"   Prompt length: {len(prompt)} characters", "INFO")
    if args.num_examples > 0:
        log(f"   Includes {args.num_examples} few-shot examples (SOURCE→TARGET, no pivot)", "INFO")
    
    # Generate all prompts
    test_df['prompt'] = ""
    test_df['response'] = ""
    
    log("🔧 Generating prompts for all samples...", "INFO")
    prompts = []
    for i, row in test_df.iterrows():
        prompt = translate_prompt_no_pivot(row[source_col])
        prompts.append(prompt)
        test_df.at[i, 'prompt'] = prompt
    log(f"✅ Generated {len(prompts)} prompts", "SUCCESS")
    
    # Run batch inference
    log("="*80, "INFO")
    log(f"🔄 Starting batch inference on {len(test_df)} samples...", "INFO")
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    log(f"   Processing {num_batches} batches", "INFO")
    log("="*80, "INFO")
    
    inference_start = time.time()
    successful = 0
    failed = 0
    
    # Process in batches
    for batch_idx in range(0, len(prompts), args.batch_size):
        batch_start_time = time.time()
        batch_prompts = prompts[batch_idx:batch_idx + args.batch_size]
        batch_indices = list(range(batch_idx, min(batch_idx + args.batch_size, len(prompts))))
        
        batch_num = batch_idx // args.batch_size + 1
        log(f"📦 Batch {batch_num}/{num_batches} (samples {batch_idx+1}-{min(batch_idx+args.batch_size, len(prompts))})", "INFO")
        
        try:
            outputs = pipe(batch_prompts, batch_size=len(batch_prompts))
            
            for idx, output in zip(batch_indices, outputs):
                try:
                    generated_text = output[0]['generated_text']
                    test_df.at[idx, 'response'] = generated_text
                    successful += 1
                except Exception as e:
                    log(f"   ⚠️  Failed to extract response for sample {idx}: {e}", "WARNING")
                    test_df.at[idx, 'response'] = ""
                    failed += 1
            
            batch_time = time.time() - batch_start_time
            samples_per_sec = len(batch_prompts) / batch_time
            log(f"   ✅ Completed in {batch_time:.2f}s ({samples_per_sec:.2f} samples/sec)", "SUCCESS")
            
        except Exception as e:
            log(f"   ❌ Batch failed: {e}", "ERROR")
            for idx in batch_indices:
                test_df.at[idx, 'response'] = ""
                failed += 1
    
    inference_time = time.time() - inference_start
    log("="*80, "INFO")
    log(f"✅ Inference complete!", "SUCCESS")
    log(f"   Total time: {inference_time:.2f}s", "INFO")
    log(f"   Successful: {successful}/{len(test_df)}", "INFO")
    log(f"   Failed: {failed}/{len(test_df)}", "INFO")
    log("="*80, "INFO")
    
    if use_wandb:
        wandb.log({
            "inference_time": inference_time,
            "successful_translations": successful,
            "failed_translations": failed
        })
    
    # Calculate metrics
    log("📊 Calculating metrics...", "INFO")
    
    references = test_df[target_col].tolist()
    hypotheses = test_df['response'].tolist()
    
    # BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    bleu_score = bleu.score
    
    # chrF
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    chrf_score = chrf.score
    
    # chrF++
    chrf_plus_plus = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    chrf_plus_plus_score = chrf_plus_plus.score
    
    scores = {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "chrf++": chrf_plus_plus_score,
        "num_samples": len(test_df),
        "successful": successful,
        "failed": failed,
        "inference_time": inference_time,
        "config": {
            "model": args.model,
            "source": args.source,
            "target": args.target,
            "num_examples": args.num_examples,
            "pivot": "NONE (direct translation)",
            "batch_size": args.batch_size
        }
    }
    
    log("="*80, "INFO")
    log("📈 RESULTS (NO-PIVOT BASELINE):", "INFO")
    log(f"   BLEU:     {bleu_score:.2f}", "INFO")
    log(f"   chrF:     {chrf_score:.2f}", "INFO")
    log(f"   chrF++:   {chrf_plus_plus_score:.2f}", "INFO")
    log("="*80, "INFO")
    
    if use_wandb:
        wandb.log(scores)
    
    # Save results
    log(f"💾 Saving results to: {args.output}", "INFO")
    test_df.to_csv(args.output, index=False)
    
    log(f"💾 Saving scores to: {args.scores}", "INFO")
    with open(args.scores, 'w') as f:
        json.dump(scores, f, indent=2)
    
    log("="*80, "INFO")
    log("✅ NO-PIVOT EXPERIMENT COMPLETE!", "SUCCESS")
    log("="*80, "INFO")
    
    # Cleanup
    del pipe
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

