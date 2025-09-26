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
    if args.num_examples > 0:
        db = lancedb.connect(args.db)
        embed_model = SentenceTransformer("all-MiniLM-L12-v2")
        
        def embed(text):
            return embed_model.encode(text)
    else:
        print("Skipping vector database setup (num_examples = 0)")
    
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
    prompt = translate_prompt_with_ft(test_df[args.pivot].values[0], test_df[args.source].values[0])
    print("Sample prompt:", prompt)
    
    # Process all test data
    test_df['prompt'] = ""
    test_df['response'] = ""
    
    for i, row in test_df.iterrows():
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
            print(f"Generated text for row {i}: {mt[0]['generated_text']}")
        except Exception as e:
            print(f"Error in generating text for row {i}: {e}")
    
    # Save results
    test_df.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")
    
    # Calculate scores
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
    
    print(f"BLEU Score: {bleu_score}")
    print(f"chrF Score: {chrf_score}")
    print(f"CHRF++ Score: {chrf_pp_score}")

    # Save scores
    with open(args.scores, "w") as f:
        json.dump({
            "BLEU Score": bleu_score,
            "Normalized BLEU Score": bleu_score / 100,
            "chrF Score": chrf_score,
            "CHRF++ Score": chrf_pp_score
        }, f)

if __name__ == "__main__":
    main()