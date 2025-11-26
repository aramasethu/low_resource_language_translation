#!/usr/bin/env python3
"""
Unified inference script for low-resource translation.
Handles both flat datasets (e.g., Konkani) and nested datasets (e.g., Arabic).
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

# Try to import new metrics (MetricX and COMET)
try:
    from metricx import MetricX
    METRICX_AVAILABLE = True
except ImportError:
    METRICX_AVAILABLE = False

try:
    from unbabel_comet import load_from_checkpoint as unbabel_load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

def load_and_flatten_dataset(dataset_name, split='test'):
    """
    Load dataset and automatically handle both flat and nested structures.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to load (default: 'test')
    
    Returns:
        pd.DataFrame: Flattened dataframe
    """
    log(f"📚 Loading dataset: {dataset_name}", "INFO")
    dataset = load_dataset(dataset_name)
    
    # Get the test split
    test_data = dataset[split]
    
    # Check if the dataset has a nested 'translation' column
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
        # Empty dataset
        test_df = pd.DataFrame(test_data)
    
    return test_df

def main():
    parser = argparse.ArgumentParser(
        description="Unified inference script for low-resource translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Konkani translation
  python scripts/run_inference_unified.py \\
    --dataset predictionguard/english-hindi-marathi-konkani-corpus \\
    --pivot hin --source mar --target gom \\
    --db translations_db --num-examples 3

  # Arabic translation
  python scripts/run_inference_unified.py \\
    --dataset pierrebarbera/tunisian_msa_arabizi \\
    --pivot msa --source en --target tn \\
    --db arabic_translations --num-examples 3
        """
    )
    
    # Dataset and model
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., predictionguard/english-hindi-marathi-konkani-corpus)")
    parser.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.1", help="Model name")
    
    # Language configuration
    parser.add_argument("--pivot", required=True, help="Pivot language column (e.g., 'hin' for Hindi, 'msa' for MSA)")
    parser.add_argument("--source", required=True, help="Source language column (e.g., 'mar' for Marathi, 'en' for English)")
    parser.add_argument("--target", required=True, help="Target language column (e.g., 'gom' for Konkani, 'tn' for Tunisian)")
    
    # Database and output
    parser.add_argument("--db", required=True, help="Vector database name (e.g., 'translations_db', 'arabic_translations')")
    parser.add_argument("--output", default="translated_results.csv", help="Output CSV file")
    parser.add_argument("--scores", default="scores.json", help="Scores JSON file")
    
    # Inference parameters
    parser.add_argument("--num-examples", type=int, default=5, help="Number of few-shot examples to use (0 for zero-shot)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for GPU inference (default: 8)")
    parser.add_argument("--max-new-tokens", type=int, default=600, help="Maximum new tokens to generate (default: 600)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit number of test samples to process (for testing/validation)")
    
    # W&B logging
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
        run_name = args.wandb_run_name or f"inference_{args.target}_k{args.num_examples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "max_new_tokens": args.max_new_tokens
            },
            tags=["inference", args.target, f"k={args.num_examples}"]
        )
        log("✅ W&B initialized successfully", "SUCCESS")
    
    log("="*80, "INFO")
    log("🚀 TRANSLATION INFERENCE", "INFO")
    log("="*80, "INFO")
    log(f"Dataset: {args.dataset}", "INFO")
    log(f"Model: {args.model}", "INFO")
    log(f"Languages: {args.pivot} (pivot) -> {args.source} (source) -> {args.target} (target)", "INFO")
    log(f"Few-shot examples: k={args.num_examples}", "INFO")
    log(f"Batch size: {args.batch_size}", "INFO")
    log(f"Max new tokens: {args.max_new_tokens}", "INFO")
    log(f"Database: {args.db}", "INFO")
    log(f"Output: {args.output}", "INFO")
    log("="*80, "INFO")
    
    # CUDA info
    log(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", "INFO")
    log(f"🔧 CUDA available: {torch.cuda.is_available()}", "INFO")
    if torch.cuda.is_available():
        log(f"🔧 Number of GPUs visible: {torch.cuda.device_count()}", "INFO")
        log(f"🔧 Current device: {torch.cuda.current_device()}", "INFO")
        log(f"🔧 Device name: {torch.cuda.get_device_name(0)}", "INFO")
    
    # Setup
    log("📥 Loading tokenizer...", "INFO")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log("✅ Tokenizer loaded", "SUCCESS")
    
    log("🤖 Loading model...", "INFO")
    log("   This may take 10-15 minutes if downloading for the first time (~14 GB)", "INFO")
    model_load_start = time.time()
    
    pipeline_model = transformers.pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        device=0,
        torch_dtype=torch.float16,
        batch_size=args.batch_size
    )
    
    model_load_time = time.time() - model_load_start
    log(f"✅ Model loaded in {model_load_time/60:.1f} minutes", "SUCCESS")
    log(f"   Using batch size: {args.batch_size} for GPU inference", "INFO")
    
    if use_wandb:
        wandb.log({"model_load_time_minutes": model_load_time/60})
    
    # Load and flatten data (handles both flat and nested structures)
    test_df = load_and_flatten_dataset(args.dataset, split='test')
    log(f"✅ Dataset loaded: {len(test_df)} test samples", "SUCCESS")
    
    # Apply test limit if specified (for validation/testing)
    if args.test_limit is not None and args.test_limit < len(test_df):
        log(f"⚠️  TEST MODE: Limiting to first {args.test_limit} samples (from {len(test_df)})", "INFO")
        test_df = test_df.head(args.test_limit)
    
    log(f"   Columns: {list(test_df.columns)}", "INFO")
    
    # Validate that required columns exist
    required_cols = [args.pivot, args.source, args.target]
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        log(f"❌ ERROR: Required columns not found in dataset: {missing_cols}", "ERROR")
        log(f"   Available columns: {list(test_df.columns)}", "ERROR")
        sys.exit(1)
    
    if use_wandb:
        wandb.log({"test_samples": len(test_df)})
    
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
    
    # Setup prompts
    USER_PREFIX = f'Translate the following source text from {args.pivot.title()} to {args.target.title()}. Only return the {args.target.title()} translation and nothing else.'
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
                        log(f"Warning: Error adding example {i}: {e}", "WARNING")

        # Add the current context (always needed)
        messages.append({
            "role": "user",
            "content": USER_PREFIX + pivot_text + USER_MIDDLE + source_text + USER_SUFFIX
        })

        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Try it out for a sample
    log("🧪 Testing prompt generation with first sample...", "INFO")
    prompt = translate_prompt_with_ft(test_df[args.pivot].values[0], test_df[args.source].values[0])
    log(f"   Prompt length: {len(prompt)} characters", "INFO")
    if args.num_examples > 0:
        log(f"   Includes {args.num_examples} few-shot examples", "INFO")
    
    # Process all test data
    test_df['prompt'] = ""
    test_df['response'] = ""
    
    # Generate all prompts first
    log("🔧 Generating prompts for all samples...", "INFO")
    prompts = []
    for i, row in test_df.iterrows():
        prompt = translate_prompt_with_ft(row[args.pivot], row[args.source])
        prompts.append(prompt)
        test_df.at[i, 'prompt'] = prompt
    log(f"✅ Generated {len(prompts)} prompts", "SUCCESS")
    
    log("="*80, "INFO")
    log(f"🔄 Starting batch inference on {len(test_df)} samples (batch size: {args.batch_size})...", "INFO")
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    log(f"   Processing {num_batches} batches", "INFO")
    log("="*80, "INFO")
    
    inference_start = time.time()
    successful = 0
    failed = 0
    
    # Process in batches
    for batch_idx in range(0, len(prompts), args.batch_size):
        batch_num = batch_idx // args.batch_size + 1
        batch_end = min(batch_idx + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_idx:batch_end]
        batch_size_actual = len(batch_prompts)
        
        batch_start = time.time()
        
        # Log progress
        samples_done = batch_idx
        log(f"📦 Processing batch {batch_num}/{num_batches} (samples {batch_idx+1}-{batch_end}/{len(prompts)}) [{samples_done/len(prompts)*100:.1f}%]", "INFO")
        
        try:
            # Process batch
            with torch.no_grad():
                results = pipeline_model(
                    batch_prompts,
                    do_sample=True,
                    temperature=0.1,
                    num_return_sequences=1,
                    max_new_tokens=args.max_new_tokens,
                    return_full_text=False,
                    top_k=50,
                    top_p=0.75,
                )
            
            # Store results
            for i, result in enumerate(results):
                test_df.at[batch_idx + i, 'response'] = result[0]['generated_text']
                successful += 1
            
            batch_time = time.time() - batch_start
            samples_per_sec = batch_size_actual / batch_time
            log(f"   ✅ Batch {batch_num} completed in {batch_time:.1f}s ({samples_per_sec:.1f} samples/sec)", "SUCCESS")
            
            # Log first batch with more detail
            if batch_num == 1:
                log(f"      Example translation:", "INFO")
                log(f"      Source: {test_df.iloc[0][args.source][:60]}...", "INFO")
                log(f"      Translation: {results[0][0]['generated_text'][:60]}...", "INFO")
            
            # Log to wandb periodically
            if use_wandb and batch_num % max(1, num_batches // 10) == 0:
                elapsed = time.time() - inference_start
                wandb.log({
                    "samples_processed": batch_end,
                    "progress_pct": batch_end / len(test_df) * 100,
                    "avg_time_per_sample": elapsed / batch_end,
                    "samples_per_second": samples_per_sec
                })
            
            torch.cuda.empty_cache()
                
        except Exception as e:
            failed += batch_size_actual
            log(f"   ❌ Error in batch {batch_num}: {e}", "ERROR")
            # Fill with empty strings for failed batch
            for i in range(batch_size_actual):
                test_df.at[batch_idx + i, 'response'] = ""
    
    inference_time = time.time() - inference_start
    
    log("="*80, "INFO")
    log(f"✅ Inference completed!", "SUCCESS")
    log(f"   Successful: {successful}/{len(test_df)}", "SUCCESS")
    if failed > 0:
        log(f"   Failed: {failed}/{len(test_df)}", "WARNING")
    log(f"   Total time: {inference_time/60:.1f} minutes", "INFO")
    log(f"   Avg time per sample: {inference_time/len(test_df):.1f} seconds", "INFO")
    log("="*80, "INFO")
    
    # Save results
    log(f"💾 Saving results to: {args.output}", "INFO")
    test_df.to_csv(args.output, index=False)
    log(f"✅ Results saved successfully", "SUCCESS")
    
    # Calculate scores
    log("📊 Calculating evaluation metrics...", "INFO")
    
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

    def calculate_comet(references, hypotheses, sources, model_path=None):
        if not COMET_AVAILABLE:
            log("COMET not available (unbabel_comet not installed)", "WARNING")
            return None
        
        try:
            if model_path is None:
                model_path = "Unbabel/wmt22-comet-da"
            
            model = unbabel_load_from_checkpoint(model_path)
            
            # Prepare data in COMET format (source, hypothesis, reference)
            data = []
            for src, hyp, ref in zip(sources, hypotheses, references):
                data.append({
                    "src": src,
                    "mt": hyp,
                    "ref": ref
                })
            
            # Calculate scores
            scores, comet_score = model.predict(data, batch_size=32, gpus=1)
            return comet_score
        except Exception as e:
            log(f"Error calculating COMET score: {e}", "ERROR")
            return None

    def calculate_metricx(references, hypotheses, model_variant="GLOBAL"):
        if not METRICX_AVAILABLE:
            log("MetricX not available (metricx not installed)", "WARNING")
            return None
        
        try:
            # Initialize MetricX with specified variant
            metricx = MetricX(variant=model_variant)
            
            # Prepare data in MetricX format (references and hypotheses)
            scores = []
            for ref, hyp in zip(references, hypotheses):
                score = metricx.score(reference=ref, translation=hyp)
                scores.append(score)
            
            # Return average score
            if scores:
                avg_score = sum(scores) / len(scores)
                return avg_score
            return None
        except Exception as e:
            log(f"Error calculating MetricX score: {e}", "ERROR")
            return None

    # Get raw data
    raw_references = test_df[args.target].tolist()
    raw_hypotheses = test_df['response'].tolist()
    raw_sources = test_df[args.source].tolist()
    
    # Filter out NaN values and strip whitespace from hypotheses
    # NaN values can cause BLEU calculation to fail (especially if at position 0)
    valid_indices = []
    for i, (ref, hyp) in enumerate(zip(raw_references, raw_hypotheses)):
        if pd.notna(ref) and pd.notna(hyp) and str(hyp).strip() != '' and str(hyp) != 'nan':
            valid_indices.append(i)
    
    references = [str(raw_references[i]) for i in valid_indices]
    hypotheses = [str(raw_hypotheses[i]).strip() for i in valid_indices]  # Strip leading/trailing whitespace
    sources = [str(raw_sources[i]) for i in valid_indices]
    
    filtered_count = len(raw_references) - len(references)
    if filtered_count > 0:
        log(f"⚠️  Filtered {filtered_count} samples with NaN/empty responses for evaluation", "WARNING")
    log(f"📊 Evaluating on {len(references)} valid samples", "INFO")

    bleu_score = calculate_bleu(references, hypotheses)
    chrf_score = calculate_chrf(references, hypotheses)
    chrf_pp_score = calculate_chrf_plus_plus(references, hypotheses)
    
    log("="*80, "INFO")
    log("📈 EVALUATION RESULTS", "INFO")
    log("="*80, "INFO")
    log(f"BLEU Score:    {bleu_score:.2f}", "SUCCESS")
    log(f"chrF Score:    {chrf_score:.2f}", "SUCCESS")
    log(f"chrF++ Score:  {chrf_pp_score:.2f}", "SUCCESS")
    
    # Calculate COMET score
    log("🔄 Calculating COMET score...", "INFO")
    comet_score = calculate_comet(references, hypotheses, sources)
    if comet_score is not None:
        log(f"COMET Score:   {comet_score:.4f}", "SUCCESS")
    else:
        log("COMET Score:   Failed to calculate", "WARNING")
        comet_score = None
    
    # Calculate MetricX score
    log("🔄 Calculating MetricX score...", "INFO")
    metricx_score = calculate_metricx(references, hypotheses)
    if metricx_score is not None:
        log(f"MetricX Score: {metricx_score:.4f}", "SUCCESS")
    else:
        log("MetricX Score: Failed to calculate", "WARNING")
        metricx_score = None
    
    log("="*80, "INFO")

    # Save scores
    scores_dict = {
        "BLEU Score": bleu_score,
        "Normalized BLEU Score": bleu_score / 100,
        "chrF Score": chrf_score,
        "CHRF++ Score": chrf_pp_score,
        "COMET Score": float(comet_score) if comet_score is not None else None,
        "MetricX Score": float(metricx_score) if metricx_score is not None else None,
        "inference_time_minutes": inference_time / 60,
        "samples_processed": successful,
        "samples_failed": failed,
        "k": args.num_examples
    }
    
    with open(args.scores, "w") as f:
        json.dump(scores_dict, f, indent=2)
    log(f"💾 Scores saved to: {args.scores}", "INFO")
    
    # Log to wandb
    if use_wandb:
        wandb_metrics = {
            "final/bleu": bleu_score,
            "final/chrf": chrf_score,
            "final/chrfpp": chrf_pp_score,
            "final/inference_time_minutes": inference_time / 60,
            "final/samples_processed": successful,
            "final/samples_failed": failed,
            "final/success_rate": successful / len(test_df) * 100
        }
        
        # Add COMET and MetricX if available
        if comet_score is not None:
            wandb_metrics["final/comet"] = comet_score
        if metricx_score is not None:
            wandb_metrics["final/metricx"] = metricx_score
        
        wandb.log(wandb_metrics)
        
        # Create a summary table
        summary_data = [
            ["BLEU", f"{bleu_score:.2f}"],
            ["chrF", f"{chrf_score:.2f}"],
            ["chrF++", f"{chrf_pp_score:.2f}"]
        ]
        
        if comet_score is not None:
            summary_data.append(["COMET", f"{comet_score:.4f}"])
        if metricx_score is not None:
            summary_data.append(["MetricX", f"{metricx_score:.4f}"])
        
        summary_data.extend([
            ["Samples", str(successful)],
            ["Time (min)", f"{inference_time / 60:.1f}"]
        ])
        
        summary_table = wandb.Table(
            columns=["Metric", "Score"],
            data=summary_data
        )
        wandb.log({"results_summary": summary_table})
        
        log(f"📊 Results logged to W&B: {wandb.run.url}", "INFO")
        wandb.finish()
    
    log("🧹 Cleaning up memory...", "INFO")
    del pipeline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log("✅ Memory cleaned up", "SUCCESS")
    
    total_time = time.time() - start_time
    log("="*80, "INFO")
    log(f"🎉 INFERENCE COMPLETE!", "SUCCESS")
    log(f"   Total elapsed time: {total_time/60:.1f} minutes", "INFO")
    log("="*80, "INFO")

if __name__ == "__main__":
    main()

