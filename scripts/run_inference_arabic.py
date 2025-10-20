#!/usr/bin/env python3
"""
Inference script specifically for Arabic dataset with nested structure.
This handles the nested 'translation' column properly.
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
from datetime import datetime

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

def flatten_arabic_dataset(dataset):
    """Flatten the nested translation structure in Arabic dataset."""
    flattened = []
    for item in dataset:
        translation_dict = item['translation']
        flattened.append({
            'en': translation_dict.get('en', ''),
            'msa': translation_dict.get('msa', ''),
            'tn': translation_dict.get('tn', ''),
            'eg': translation_dict.get('eg', ''),
            'jo': translation_dict.get('jo', ''),
            'pa': translation_dict.get('pa', ''),
            'sy': translation_dict.get('sy', '')
        })
    return pd.DataFrame(flattened)

def main():
    parser = argparse.ArgumentParser(description="Inference for Arabic translation")
    parser.add_argument("--dataset", default="predictionguard/arabic_acl_corpus", help="Dataset name")
    parser.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.1", help="Model name")
    parser.add_argument("--pivot", default="msa", help="Pivot language (default: msa)")
    parser.add_argument("--source", default="en", help="Source language (default: en)")
    parser.add_argument("--target", default="tn", help="Target language (default: tn)")
    parser.add_argument("--db", default="arabic_translations", help="Database name")
    parser.add_argument("--output", default="arabic_results.csv", help="Output CSV file")
    parser.add_argument("--scores", default="arabic_scores.json", help="Scores JSON file")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of few-shot examples to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="low-resource-translation", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        log("⚠️  WARNING: --wandb flag set but wandb not installed!", "WARNING")
        log("   Install with: pip install wandb", "WARNING")
        use_wandb = False
    
    if use_wandb:
        run_name = args.wandb_run_name or f"arabic_inference_{args.target}_k{args.num_examples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log(f"📊 Initializing Weights & Biases...", "INFO")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset": args.dataset,
                "model": args.model,
                "pivot": args.pivot,
                "source": args.source,
                "target": args.target,
                "num_examples": args.num_examples,
                "batch_size": args.batch_size,
                "language_pair": "arabic"
            },
            tags=["inference", "arabic", args.target, f"k={args.num_examples}"]
        )
        log("✅ W&B initialized successfully", "SUCCESS")
    
    log("="*80, "INFO")
    log("🚀 ARABIC TRANSLATION INFERENCE", "INFO")
    log("="*80, "INFO")
    log(f"Dataset: {args.dataset}", "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Languages: {args.pivot} (pivot) -> {args.source} (source) -> {args.target} (target)", "INFO")
    log(f"Few-shot examples: k={args.num_examples}", "INFO")
    log(f"Batch size: {args.batch_size}", "INFO")
    log(f"Database: {args.db}", "INFO")
    log(f"Output: {args.output}", "INFO")
    log("="*80, "INFO")
    
    # Setup
    import os
    log(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", "INFO")
    log(f"🔧 CUDA available: {torch.cuda.is_available()}", "INFO")
    if torch.cuda.is_available():
        log(f"🔧 Number of GPUs visible: {torch.cuda.device_count()}", "INFO")
        log(f"🔧 Current device: {torch.cuda.current_device()}", "INFO")
        log(f"🔧 Device name: {torch.cuda.get_device_name(0)}", "INFO")
    
    log("📥 Loading tokenizer...", "INFO")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log("✅ Tokenizer loaded", "SUCCESS")
    
    # Detect device - respects CUDA_VISIBLE_DEVICES
    device = 0 if torch.cuda.is_available() else -1
    
    log("🤖 Loading model...", "INFO")
    log("   This may take 10-15 minutes if downloading for the first time (~14 GB)", "INFO")
    model_load_start = time.time()
    
    pipeline_model = transformers.pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16,
        batch_size=args.batch_size
    )
    
    model_load_time = time.time() - model_load_start
    log(f"✅ Model loaded in {model_load_time:.1f}s ({model_load_time/60:.1f} minutes)", "SUCCESS")
    log(f"📍 Pipeline using device: {device}", "INFO")
    
    if use_wandb:
        wandb.log({"model_load_time_minutes": model_load_time/60})
    
    # Load and flatten data
    log(f"📚 Loading dataset: {args.dataset}", "INFO")
    dataset = load_dataset(args.dataset)
    test_df = flatten_arabic_dataset(dataset['test'])
    
    log(f"✅ Dataset loaded: {len(test_df)} test samples", "SUCCESS")
    log(f"   Columns: {test_df.columns.tolist()}", "INFO")
    
    if use_wandb:
        wandb.log({"test_samples": len(test_df)})
    
    # Setup database and embeddings (only if needed)
    if args.num_examples > 0:
        log(f"🔍 Loading vector database: {args.db}", "INFO")
        db = lancedb.connect(args.db)
        embed_model = SentenceTransformer("all-MiniLM-L12-v2")
        
        def embed(text):
            return embed_model.encode(text)
        log("✅ Vector database ready", "SUCCESS")
    else:
        log("⏭️  Skipping vector database setup (num_examples = 0)", "INFO")
    
    # Setup prompts
    USER_PREFIX = f'Translate the following source text from {args.pivot.upper()} to {args.target.upper()}. Only return the {args.target.upper()} translation and nothing else.'
    USER_MIDDLE = '\nSource: '
    USER_SUFFIX = f'\nTranslation: '
    
    def translate_prompt_with_ft(pivot_text, source_text):
        messages = []
        
        # Only use vector database if we need examples
        if args.num_examples > 0:
            # Pull the most similar examples
            table = db.open_table(f"translations_{args.target}")
            results = table.search(embed(pivot_text)).limit(10).to_pandas()
            results = results[results['text'] != pivot_text]
            results = results[results['text'] != ""]
            results = results[results[args.source] != ""]
            results = results[results[args.target] != ""]
            results.dropna(inplace=True)
            results.sort_values(by="_distance", ascending=True, inplace=True)

            num_examples = args.num_examples
            for i in range(0, num_examples+1):
                if i != 0 and i <= len(results):
                    try:
                        messages.append({
                            "role": "user",
                            "content": USER_PREFIX + results['text'].values[i-1] + USER_MIDDLE + results[args.source].values[i-1] + USER_SUFFIX
                        })
                        messages.append({
                            "role": "assistant", 
                            "content": results[args.target].values[i-1]
                        })
                    except Exception as e:
                        print(f"Error adding example: {e}")

        # Add the current context (always needed)
        messages.append({
            "role": "user",
            "content": USER_PREFIX + pivot_text + USER_MIDDLE + source_text + USER_SUFFIX
        })

        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate prompts for all samples
    log("📝 Generating prompts for all test samples...", "INFO")
    prompts = []
    for i, row in test_df.iterrows():
        prompt = translate_prompt_with_ft(row[args.pivot], row[args.source])
        prompts.append(prompt)
        test_df.at[i, 'prompt'] = prompt
    
    log(f"✅ Generated {len(prompts)} prompts", "SUCCESS")
    log("", "INFO")
    log("Sample prompt (first 500 chars):", "INFO")
    log(prompts[0][:500] + "...", "INFO")
    log("", "INFO")
    
    # Run inference in batches
    log("="*80, "INFO")
    log("🔄 STARTING INFERENCE", "INFO")
    log("="*80, "INFO")
    
    test_df['response'] = ""
    inference_start = time.time()
    
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    log(f"Processing {len(prompts)} samples in {num_batches} batches (batch_size={args.batch_size})", "INFO")
    
    for batch_idx in range(0, len(prompts), args.batch_size):
        batch_num = batch_idx // args.batch_size + 1
        batch_end = min(batch_idx + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_idx:batch_end]
        
        batch_start = time.time()
        
        try:
            results = pipeline_model(
                batch_prompts,
                do_sample=True,
                temperature=0.1,
                num_return_sequences=1,
                max_new_tokens=200,
                return_full_text=False,
                top_k=50,
                top_p=0.75,
            )
            
            # Store results
            for i, result in enumerate(results):
                global_idx = batch_idx + i
                test_df.at[global_idx, 'response'] = result[0]['generated_text']
            
            batch_time = time.time() - batch_start
            samples_per_sec = len(batch_prompts) / batch_time
            
            # Progress logging
            if batch_num % max(1, num_batches // 10) == 0 or batch_num == num_batches:
                elapsed = time.time() - inference_start
                progress_pct = (batch_end / len(prompts)) * 100
                log(f"Batch {batch_num}/{num_batches} | Samples: {batch_end}/{len(prompts)} ({progress_pct:.1f}%) | "
                    f"Batch time: {batch_time:.1f}s | Speed: {samples_per_sec:.2f} samples/s", "INFO")
                
                # Log example translation
                if batch_num == 1:
                    log("", "INFO")
                    log("Example translation from first batch:", "INFO")
                    log(f"  Source: {test_df.at[batch_idx, args.source][:100]}...", "INFO")
                    log(f"  Reference: {test_df.at[batch_idx, args.target][:100]}...", "INFO")
                    log(f"  Generated: {test_df.at[batch_idx, 'response'][:100]}...", "INFO")
                    log("", "INFO")
                
                # W&B logging
                if use_wandb and batch_num % max(1, num_batches // 10) == 0:
                    elapsed = time.time() - inference_start
                    wandb.log({
                        "samples_processed": batch_end,
                        "progress_pct": batch_end / len(test_df) * 100,
                        "avg_time_per_sample": elapsed / batch_end,
                        "samples_per_second": samples_per_sec
                    })
        
        except Exception as e:
            log(f"❌ Error in batch {batch_num}: {e}", "ERROR")
            # Fill with empty responses for failed batch
            for i in range(len(batch_prompts)):
                global_idx = batch_idx + i
                test_df.at[global_idx, 'response'] = ""
    
    inference_time = time.time() - inference_start
    log("="*80, "INFO")
    log(f"✅ Inference complete in {inference_time:.1f}s ({inference_time/60:.1f} minutes)", "SUCCESS")
    log(f"   Average: {inference_time/len(prompts):.2f}s per sample", "INFO")
    log("="*80, "INFO")
    
    # Save results
    log(f"💾 Saving results to: {args.output}", "INFO")
    test_df.to_csv(args.output, index=False)
    log(f"✅ Results saved", "SUCCESS")
    
    # Calculate scores
    log("="*80, "INFO")
    log("📊 CALCULATING METRICS", "INFO")
    log("="*80, "INFO")
    
    def calculate_bleu(references, hypotheses):
        formatted_references = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(hypotheses, formatted_references)
        return bleu.score

    def calculate_chrf(references, hypotheses):
        formatted_references = [[ref] for ref in references]
        chrf = sacrebleu.corpus_chrf(hypotheses, formatted_references)
        return chrf.score

    def calculate_chrf_plus_plus(references, hypotheses):
        formatted_references = [[ref] for ref in references]
        chrf_pp = sacrebleu.corpus_chrf(hypotheses, formatted_references, beta=2, word_order=2)
        return chrf_pp.score

    references = test_df[args.target].tolist()
    hypotheses = test_df['response'].tolist()

    bleu_score = calculate_bleu(references, hypotheses)
    chrf_score = calculate_chrf(references, hypotheses)
    chrf_pp_score = calculate_chrf_plus_plus(references, hypotheses)
    
    log(f"   BLEU:   {bleu_score:.2f}", "INFO")
    log(f"   chrF:   {chrf_score:.2f}", "INFO")
    log(f"   chrF++: {chrf_pp_score:.2f}", "INFO")
    log("="*80, "INFO")

    # Save scores
    log(f"💾 Saving scores to: {args.scores}", "INFO")
    with open(args.scores, "w") as f:
        json.dump({
            "BLEU Score": bleu_score,
            "Normalized BLEU Score": bleu_score / 100,
            "chrF Score": chrf_score,
            "CHRF++ Score": chrf_pp_score
        }, f)
    log(f"✅ Scores saved", "SUCCESS")
    
    # Log to W&B
    if use_wandb:
        wandb.log({
            "final/bleu": bleu_score,
            "final/chrf": chrf_score,
            "final/chrfpp": chrf_pp_score,
            "final/inference_time_minutes": inference_time / 60,
            "final/samples_per_second": len(test_df) / inference_time
        })
        
        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Score"],
            data=[
                ["BLEU", bleu_score],
                ["chrF", chrf_score],
                ["chrF++", chrf_pp_score],
                ["Inference Time (min)", inference_time / 60],
                ["Samples/sec", len(test_df) / inference_time]
            ]
        )
        wandb.log({"results_summary": summary_table})
        
        log(f"📊 Results logged to W&B: {wandb.run.url}", "INFO")
        wandb.finish()
    
    total_time = time.time() - start_time
    log("="*80, "INFO")
    log("🎉 TRANSLATION COMPLETE!", "INFO")
    log("="*80, "INFO")
    log(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)", "INFO")
    log(f"Output: {args.output}", "INFO")
    log(f"Scores: {args.scores}", "INFO")
    log("="*80, "INFO")

if __name__ == "__main__":
    main()

