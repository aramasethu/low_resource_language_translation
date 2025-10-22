#!/usr/bin/env python3
"""
Check if max_new_tokens=200 is limiting translations.
"""
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

print("="*80)
print("CHECKING TOKEN LENGTHS")
print("="*80)

# Load tokenizer
model = "Unbabel/TowerInstruct-7B-v0.1"
print(f"\nLoading tokenizer for {model}...")
tokenizer = AutoTokenizer.from_pretrained(model)
print("✅ Tokenizer loaded")

# Check model's max context length
if hasattr(tokenizer, 'model_max_length'):
    print(f"📏 Model max context length: {tokenizer.model_max_length} tokens")

print("\n" + "="*80)
print("ARABIC DATASET ANALYSIS")
print("="*80)

# Load Arabic dataset
dataset = load_dataset("predictionguard/arabic_acl_corpus")
test_data = dataset['test']

# Flatten
flattened = []
for item in test_data:
    translation_dict = item['translation']
    flattened.append({
        'en': translation_dict.get('en', ''),
        'msa': translation_dict.get('msa', ''),
        'tn': translation_dict.get('tn', ''),
    })

test_df = pd.DataFrame(flattened)

# Check token lengths for Tunisian Arabic (target)
tn_token_lengths = []
for tn_text in test_df['tn']:
    tokens = tokenizer.encode(tn_text, add_special_tokens=False)
    tn_token_lengths.append(len(tokens))

print(f"\n📊 Tunisian Arabic (Target) Token Lengths:")
print(f"   Mean: {sum(tn_token_lengths)/len(tn_token_lengths):.1f} tokens")
print(f"   Max: {max(tn_token_lengths)} tokens")
print(f"   Min: {min(tn_token_lengths)} tokens")
print(f"   Median: {sorted(tn_token_lengths)[len(tn_token_lengths)//2]} tokens")

# Check how many exceed 200
over_200 = [x for x in tn_token_lengths if x > 200]
print(f"\n⚠️  Translations > 200 tokens: {len(over_200)}/{len(tn_token_lengths)} ({len(over_200)/len(tn_token_lengths)*100:.1f}%)")

if len(over_200) > 0:
    print(f"   These translations will be TRUNCATED by max_new_tokens=200!")
    print(f"   Longest: {max(over_200)} tokens")
    
# Show distribution
print("\n📊 Token Length Distribution:")
bins = [0, 50, 100, 150, 200, 250, 300, 500, 1000]
for i in range(len(bins)-1):
    count = len([x for x in tn_token_lengths if bins[i] <= x < bins[i+1]])
    pct = count/len(tn_token_lengths)*100
    bar = "█" * int(pct/2)
    print(f"   {bins[i]:4d}-{bins[i+1]:4d}: {count:3d} ({pct:5.1f}%) {bar}")

# Check prompt lengths for different k values
print("\n" + "="*80)
print("PROMPT LENGTH ANALYSIS")
print("="*80)

# Simulate prompts for k=0, 5, 10
from random import sample

# Create sample messages for k=10 (worst case)
USER_PREFIX = "Translate the following source text from MSA to TN. Only return the TN translation and nothing else."
USER_MIDDLE = "\nSource: "
USER_SUFFIX = "\nTranslation: "

# Get first test sample
test_msa = test_df['msa'].iloc[0]
test_en = test_df['en'].iloc[0]

# Simulate k=0 prompt
messages_k0 = [
    {"role": "user", "content": USER_PREFIX + test_msa + USER_MIDDLE + test_en + USER_SUFFIX}
]
prompt_k0 = tokenizer.apply_chat_template(messages_k0, tokenize=False)
tokens_k0 = tokenizer.encode(prompt_k0)

# Simulate k=5 prompt (5 examples)
messages_k5 = []
for i in range(5):
    messages_k5.append({
        "role": "user",
        "content": USER_PREFIX + test_df['msa'].iloc[i] + USER_MIDDLE + test_df['en'].iloc[i] + USER_SUFFIX
    })
    messages_k5.append({
        "role": "assistant",
        "content": test_df['tn'].iloc[i]
    })
messages_k5.append({
    "role": "user",
    "content": USER_PREFIX + test_msa + USER_MIDDLE + test_en + USER_SUFFIX
})
prompt_k5 = tokenizer.apply_chat_template(messages_k5, tokenize=False)
tokens_k5 = tokenizer.encode(prompt_k5)

# Simulate k=10 prompt (10 examples)
messages_k10 = []
for i in range(10):
    messages_k10.append({
        "role": "user",
        "content": USER_PREFIX + test_df['msa'].iloc[i] + USER_MIDDLE + test_df['en'].iloc[i] + USER_SUFFIX
    })
    messages_k10.append({
        "role": "assistant",
        "content": test_df['tn'].iloc[i]
    })
messages_k10.append({
    "role": "user",
    "content": USER_PREFIX + test_msa + USER_MIDDLE + test_en + USER_SUFFIX
})
prompt_k10 = tokenizer.apply_chat_template(messages_k10, tokenize=False)
tokens_k10 = tokenizer.encode(prompt_k10)

print(f"\nPrompt Token Counts:")
print(f"   k=0:  {len(tokens_k0):4d} tokens")
print(f"   k=5:  {len(tokens_k5):4d} tokens")
print(f"   k=10: {len(tokens_k10):4d} tokens")

if hasattr(tokenizer, 'model_max_length'):
    max_len = tokenizer.model_max_length
    if max_len and max_len < 1000000:  # Ignore if it's set to infinity
        print(f"\n⚠️  Model max context: {max_len} tokens")
        if len(tokens_k10) > max_len:
            print(f"   🔴 k=10 prompt ({len(tokens_k10)} tokens) EXCEEDS context limit!")
        else:
            print(f"   ✅ All prompts fit within context limit")

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

if len(over_200) > 0:
    print(f"\n🔴 ISSUE FOUND: max_new_tokens=200 is TOO LOW!")
    print(f"   {len(over_200)} out of {len(tn_token_lengths)} ({len(over_200)/len(tn_token_lengths)*100:.1f}%) reference translations")
    print(f"   are longer than 200 tokens and will be truncated.")
    print(f"\n   This will HURT BLEU scores because generated translations")
    print(f"   will be incomplete compared to references.")
    print(f"\n   RECOMMENDATION: Increase max_new_tokens to at least {max(tn_token_lengths) + 50}")
else:
    print(f"\n✅ max_new_tokens=200 is sufficient")
    print(f"   All reference translations are under 200 tokens.")

print("\n" + "="*80)

