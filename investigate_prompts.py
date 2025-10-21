#!/usr/bin/env python3
"""
Investigate prompt construction for different k values.
"""
import pandas as pd
import json

print("="*80)
print("INVESTIGATING PROMPT CONSTRUCTION")
print("="*80)

# Load results for different k values
k_values = [0, 1, 3, 5, 7, 10]

for k in k_values:
    result_file = f"ablation_results/arabic_full/k_{k}/results_k{k}.csv"
    try:
        df = pd.read_csv(result_file)
        
        if 'prompt' in df.columns:
            print(f"\n{'='*80}")
            print(f"k={k} PROMPT ANALYSIS")
            print(f"{'='*80}")
            
            # Get first prompt
            first_prompt = df.iloc[0]['prompt']
            
            # Count number of user/assistant pairs (indicates few-shot examples)
            user_count = first_prompt.count('<|im_start|>user')
            assistant_count = first_prompt.count('<|im_start|>assistant')
            
            print(f"Prompt length: {len(first_prompt)} characters")
            print(f"Number of <|im_start|>user blocks: {user_count}")
            print(f"Number of <|im_start|>assistant blocks: {assistant_count}")
            print(f"Expected few-shot examples: {k}")
            print(f"Actual few-shot examples in prompt: {assistant_count}")
            
            if assistant_count != k:
                print(f"⚠️  MISMATCH: Expected {k} examples, found {assistant_count}!")
            
            # Show first 1000 chars of prompt
            print(f"\nFirst 1500 characters of prompt:")
            print("-" * 80)
            print(first_prompt[:1500])
            print("-" * 80)
            
            # Show last 500 chars (the actual query)
            print(f"\nLast 500 characters of prompt (actual query):")
            print("-" * 80)
            print(first_prompt[-500:])
            print("-" * 80)
            
            # Check if prompts vary across samples
            unique_prompts = df['prompt'].nunique()
            total_samples = len(df)
            print(f"\nUnique prompts: {unique_prompts} out of {total_samples} samples")
            
            if unique_prompts < total_samples:
                print(f"⚠️  WARNING: {total_samples - unique_prompts} duplicate prompts!")
        else:
            print(f"\n⚠️  k={k}: No 'prompt' column found")
            
    except Exception as e:
        print(f"\n❌ k={k}: Error - {e}")

print("\n" + "="*80)
print("CHECKING VECTOR DATABASE RETRIEVAL")
print("="*80)

# Test retrieval with actual query
try:
    import lancedb
    from sentence_transformers import SentenceTransformer
    
    db = lancedb.connect("arabic_translations")
    table = db.open_table("translations_tn")
    embed_model = SentenceTransformer("all-MiniLM-L12-v2")
    
    # Get a test query from the test set
    from datasets import load_dataset
    dataset = load_dataset("predictionguard/arabic_acl_corpus")
    test_item = dataset['test'][0]
    test_msa = test_item['translation']['msa']
    
    print(f"\nTest query (MSA): {test_msa[:100]}...")
    
    # Try retrieving with different limits
    for limit in [1, 3, 5, 10]:
        query_embedding = embed_model.encode(test_msa)
        results = table.search(query_embedding).limit(limit + 5).to_pandas()
        
        # Filter out exact matches and empty
        results = results[results['text'] != test_msa]
        results = results[results['text'] != ""]
        results = results[results['en'] != ""]
        results = results[results['tn'] != ""]
        results.dropna(inplace=True)
        results.sort_values(by="_distance", ascending=True, inplace=True)
        
        print(f"\nRequested k={limit}, Retrieved {len(results)} results after filtering")
        print(f"Top 3 distances: {results['_distance'].head(3).tolist()}")
        
        if len(results) < limit:
            print(f"⚠️  WARNING: Requested {limit} examples but only got {len(results)}!")
        
        # Show first result
        if len(results) > 0:
            first = results.iloc[0]
            print(f"First result:")
            print(f"  Distance: {first['_distance']:.4f}")
            print(f"  MSA: {first['text'][:60]}...")
            print(f"  EN:  {first['en'][:60]}...")
            print(f"  TN:  {first['tn'][:60]}...")

except Exception as e:
    print(f"❌ Error testing retrieval: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)
print("""
Key observations:
1. Vector DB has 900 entries - well populated
2. Test set has 100 samples - all valid
3. k=0 and k=5 produce different outputs (71% different)
4. But scores are identical - suggests outputs are of similar quality

Next: Check if prompts actually contain the expected number of examples
""")

