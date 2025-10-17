#!/usr/bin/env python3
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

def main():
    parser = argparse.ArgumentParser(description="Inference for translation")
    parser.add_argument("--dataset", default="predictionguard/english-hindi-marathi-konkani-corpus", help="Dataset name")
    parser.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.1", help="Model name")
    parser.add_argument("--pivot", default="hin", help="Pivot language column")
    parser.add_argument("--source", default="mar", help="Source language column") 
    parser.add_argument("--target", default="gom", help="Target language column")
    parser.add_argument("--db", default="translations_db", help="Database name")
    parser.add_argument("--output", default="translated_results.csv", help="Output CSV file")
    parser.add_argument("--scores", default="scores.json", help="Scores JSON file")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of few-shot examples to use")
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
                "num_examples": args.num_examples
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
    log(f"Database: {args.db}", "INFO")
    log(f"Output: {args.output}", "INFO")
    log("="*80, "INFO")
    
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
        torch_dtype=torch.float16
    )
    
    model_load_time = time.time() - model_load_start
    log(f"✅ Model loaded in {model_load_time/60:.1f} minutes", "SUCCESS")
    
    if use_wandb:
        wandb.log({"model_load_time_minutes": model_load_time/60})
    
    # Load data
    log(f"📚 Loading dataset: {args.dataset}", "INFO")
    dataset = load_dataset(args.dataset)
    test_df = pd.DataFrame(dataset['test'])
    log(f"✅ Dataset loaded: {len(test_df)} test samples", "SUCCESS")
    log(f"   Columns: {list(test_df.columns)}", "INFO")
    
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
                    except:
                        print(results)

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
    
    log("="*80, "INFO")
    log(f"🔄 Starting inference on {len(test_df)} samples...", "INFO")
    log("="*80, "INFO")
    
    inference_start = time.time()
    successful = 0
    failed = 0
    
    for i, row in test_df.iterrows():
        sample_start = time.time()
        
        # Log progress every 10 samples or for first 5
        if i < 5 or (i + 1) % 10 == 0:
            log(f"📝 Processing sample {i+1}/{len(test_df)} ({(i+1)/len(test_df)*100:.1f}%)", "INFO")
        
        prompt = translate_prompt_with_ft(row[args.pivot], row[args.source])
        test_df.at[i, 'prompt'] = prompt
        
        try:
            mt = pipeline_model(
                prompt,
                do_sample=True,
                temperature=0.1,
                num_return_sequences=1,
                max_new_tokens=200,
                return_full_text=False,
                top_k=50,
                top_p=0.75,
            )
            test_df.at[i, 'response'] = mt[0]['generated_text']
            successful += 1
            
            sample_time = time.time() - sample_start
            
            # Log first few samples with more detail
            if i < 3:
                log(f"   ✅ Sample {i+1} completed in {sample_time:.1f}s", "SUCCESS")
                log(f"      Source: {row[args.source][:50]}...", "INFO")
                log(f"      Translation: {mt[0]['generated_text'][:50]}...", "INFO")
            
            # Log to wandb periodically
            if use_wandb and (i + 1) % 10 == 0:
                wandb.log({
                    "samples_processed": i + 1,
                    "progress_pct": (i + 1) / len(test_df) * 100,
                    "avg_time_per_sample": (time.time() - inference_start) / (i + 1)
                })
                
        except Exception as e:
            failed += 1
            log(f"   ❌ Error in sample {i+1}: {e}", "ERROR")
            test_df.at[i, 'response'] = ""
    
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

    references = test_df[args.target].tolist()
    hypotheses = test_df['response'].tolist()

    bleu_score = calculate_bleu(references, hypotheses)
    chrf_score = calculate_chrf(references, hypotheses)
    chrf_pp_score = calculate_chrf_plus_plus(references, hypotheses)
    
    log("="*80, "INFO")
    log("📈 EVALUATION RESULTS", "INFO")
    log("="*80, "INFO")
    log(f"BLEU Score:    {bleu_score:.2f}", "SUCCESS")
    log(f"chrF Score:    {chrf_score:.2f}", "SUCCESS")
    log(f"chrF++ Score:  {chrf_pp_score:.2f}", "SUCCESS")
    log("="*80, "INFO")

    # Save scores
    scores_dict = {
        "BLEU Score": bleu_score,
        "Normalized BLEU Score": bleu_score / 100,
        "chrF Score": chrf_score,
        "CHRF++ Score": chrf_pp_score,
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
        wandb.log({
            "final/bleu": bleu_score,
            "final/chrf": chrf_score,
            "final/chrfpp": chrf_pp_score,
            "final/inference_time_minutes": inference_time / 60,
            "final/samples_processed": successful,
            "final/samples_failed": failed,
            "final/success_rate": successful / len(test_df) * 100
        })
        
        # Create a summary table
        summary_table = wandb.Table(
            columns=["Metric", "Score"],
            data=[
                ["BLEU", bleu_score],
                ["chrF", chrf_score],
                ["chrF++", chrf_pp_score],
                ["Samples", successful],
                ["Time (min)", inference_time / 60]
            ]
        )
        wandb.log({"results_summary": summary_table})
        
        log(f"📊 Results logged to W&B: {wandb.run.url}", "INFO")
        wandb.finish()
    
    total_time = time.time() - start_time
    log("="*80, "INFO")
    log(f"🎉 INFERENCE COMPLETE!", "SUCCESS")
    log(f"   Total elapsed time: {total_time/60:.1f} minutes", "INFO")
    log("="*80, "INFO")

if __name__ == "__main__":
    main()