#!/usr/bin/env python3
"""
Check token lengths for Konkani dataset.
"""
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

print("="*80)
print("KONKANI DATASET TOKEN LENGTH ANALYSIS")
print("="*80)

# Load tokenizer
model = "Unbabel/TowerInstruct-7B-v0.1"
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model)
print("✅ Tokenizer loaded")

# Load Konkani dataset
dataset = load_dataset("ai4bharat/IN22-Conv", "kok")
test_df = pd.DataFrame(dataset['test'])

# Check token lengths for Konkani (target)
kok_token_lengths = []
for kok_text in test_df['kok']:
    tokens = tokenizer.encode(kok_text, add_special_tokens=False)
    kok_token_lengths.append(len(tokens))

print(f"\n📊 Konkani (Target) Token Lengths:")
print(f"   Mean: {sum(kok_token_lengths)/len(kok_token_lengths):.1f} tokens")
print(f"   Max: {max(kok_token_lengths)} tokens")
print(f"   Min: {min(kok_token_lengths)} tokens")
print(f"   Median: {sorted(kok_token_lengths)[len(kok_token_lengths)//2]} tokens")

# Check how many exceed 200
over_200 = [x for x in kok_token_lengths if x > 200]
print(f"\n⚠️  Translations > 200 tokens: {len(over_200)}/{len(kok_token_lengths)} ({len(over_200)/len(kok_token_lengths)*100:.1f}%)")

if len(over_200) > 0:
    print(f"   🔴 These translations will be TRUNCATED by max_new_tokens=200!")
    print(f"   Longest: {max(over_200)} tokens")
    print(f"   RECOMMENDATION: Increase max_new_tokens to at least {max(kok_token_lengths) + 50}")
else:
    print(f"   ✅ max_new_tokens=200 is sufficient")

# Show distribution
print("\n📊 Token Length Distribution:")
bins = [0, 50, 100, 150, 200, 250, 300, 500, 1000]
for i in range(len(bins)-1):
    count = len([x for x in kok_token_lengths if bins[i] <= x < bins[i+1]])
    if count > 0:
        pct = count/len(kok_token_lengths)*100
        bar = "█" * int(pct/2)
        print(f"   {bins[i]:4d}-{bins[i+1]:4d}: {count:3d} ({pct:5.1f}%) {bar}")

print("\n" + "="*80)

