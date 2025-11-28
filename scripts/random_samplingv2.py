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
import numpy as np

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
    parser.add_argument("--sampling", default="semantic", choices=["semantic", "random"], help="Sampling strategy: semantic or random")

    args = parser.parse_args()
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    pipeline_model = transformers.pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        device=0,
        torch_dtype=torch.float16
    )
    
    # Load data
    dataset = load_dataset(args.dataset)
    test_df = pd.DataFrame(dataset['test'])
    print(f"Test data shape: {test_df.shape}")
    print(test_df.head())
    
    # Setup database and embeddings (only if needed)
    db = None
    embed_model = None
    if args.num_examples > 0 and args.sampling == "semantic":
        db = lancedb.connect(args.db)
        embed_model = SentenceTransformer("all-MiniLM-L12-v2")
        
        def embed(text):
            return embed_model.encode(text)
    else:
        print(f"Skipping vector DB setup (sampling={args.sampling})")
    
    # Common formatting
    USER_PREFIX = f"Translate the following source text from {args.pivot.title()} to {args.target.title()}. Only return the {args.target.title()} translation and nothing else."
    USER_MIDDLE = "\nSource: "
    USER_SUFFIX = "\nTranslation: "

    # --------------------------
    # Semantic few-shot builder
    # --------------------------
    def translate_prompt_with_ft(pivot_text, source_text):
        messages = []
        if args.num_examples > 0 and db is not None:
            table = db.open_table(f"translations_{args.target}")
            results = table.search(embed(pivot_text)).limit(10).to_pandas()
            results = results[results['text'] != pivot_text]
            results = results[(results['text'] != "") & (results[args.source] != "") & (results[args.target] != "")]
            results.dropna(inplace=True)
            results.sort_values(by="_distance", ascending=True, inplace=True)

            num_examples = min(args.num_examples, len(results))
            for i in range(num_examples):
                messages.append({
                    "role": "user",
                    "content": USER_PREFIX + results['text'].values[i] + USER_MIDDLE + results[args.source].values[i] + USER_SUFFIX
                })
                messages.append({
                    "role": "assistant", 
                    "content": results[args.target].values[i]
                })

        messages.append({
            "role": "user",
            "content": USER_PREFIX + pivot_text + USER_MIDDLE + source_text + USER_SUFFIX
        })

        return tokenizer.apply_chat_template(messages, tokenize=False)

    # --------------------------
    # Random few-shot builder
    # --------------------------
    def translate_prompt_with_random(pivot_text, source_text):
        table = db.open_table(f"translations_{args.target}") if db else None
        full_data = table.to_pandas() if table else pd.DataFrame()
        
        # Fallback: skip if DB not available
        if full_data.empty:
            print("Warning: No database table found for random sampling. Using single example only.")
            return tokenizer.apply_chat_template([{
                "role": "user",
                "content": USER_PREFIX + pivot_text + USER_MIDDLE + source_text + USER_SUFFIX
            }], tokenize=False)

        # Clean data
        full_data = full_data[(full_data['text'] != "") & (full_data[args.source] != "") & (full_data[args.target] != "")]
        full_data.dropna(inplace=True)

        # Random sample
        examples = full_data.sample(n=min(args.num_examples, len(full_data)), random_state=np.random.randint(0, 10000))

        messages = []
        for i in range(len(examples)):
            messages.append({
                "role": "user",
                "content": USER_PREFIX + examples['text'].values[i] + USER_MIDDLE + examples[args.source].values[i] + USER_SUFFIX
            })
            messages.append({
                "role": "assistant",
                "content": examples[args.target].values[i]
            })

        # Add target prompt
        messages.append({
            "role": "user",
            "content": USER_PREFIX + pivot_text + USER_MIDDLE + source_text + USER_SUFFIX
        })

        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Try one sample
    if args.sampling == "semantic":
        sample_prompt = translate_prompt_with_ft(test_df[args.pivot].values[0], test_df[args.source].values[0])
    else:
        sample_prompt = translate_prompt_with_random(test_df[args.pivot].values[0], test_df[args.source].values[0])

    print("Sample prompt:", sample_prompt)
    
    # Process test data
    test_df['prompt'] = ""
    test_df['response'] = ""
    
    for i, row in test_df.iterrows():
        if args.sampling == "semantic":
            prompt = translate_prompt_with_ft(row[args.pivot], row[args.source])
        else:
            prompt = translate_prompt_with_random(row[args.pivot], row[args.source])

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
            print(f"Generated text for row {i}: {mt[0]['generated_text']}")
        except Exception as e:
            print(f"Error generating text for row {i}: {e}")
    
    # Save results
    test_df.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")
    
    # Scoring
    def calculate_bleu(refs, hyps):
        formatted = [[r] for r in refs]
        return sacrebleu.corpus_bleu(hyps, formatted).score

    def calculate_chrf(refs, hyps):
        formatted = [[r] for r in refs]
        return sacrebleu.corpus_chrf(hyps, formatted).score

    def calculate_chrf_pp(refs, hyps):
        formatted = [[r] for r in refs]
        return sacrebleu.corpus_chrf(hyps, formatted, beta=2, word_order=2).score

    refs = test_df[args.target].tolist()
    hyps = test_df['response'].tolist()

    bleu = calculate_bleu(refs, hyps)
    chrf = calculate_chrf(refs, hyps)
    chrfpp = calculate_chrf_pp(refs, hyps)

    print(f"BLEU Score: {bleu}")
    print(f"chrF Score: {chrf}")
    print(f"CHRF++ Score: {chrfpp}")

    with open(args.scores, "w") as f:
        json.dump({
            "BLEU Score": bleu,
            "Normalized BLEU Score": bleu / 100,
            "chrF Score": chrf,
            "CHRF++ Score": chrfpp
        }, f)

if __name__ == "__main__":
    main()